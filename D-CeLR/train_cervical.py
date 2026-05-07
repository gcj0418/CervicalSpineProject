#!/usr/bin/env python

from __future__ import print_function

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

import config.config as cfg
from data.load_test_cervical import TestData
from data.load_train_cervical import TrainData
from net.ceph_reg_refine_net import get_model
from net.reg_loss import rcal_loss
from utils import cal_acc, decode_reg


class IOStream:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()


def read_list_file(path):
    files = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                files.append(line)
    return files


def load_resolution_map(path):
    if not path:
        return {}
    resolution_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            try:
                if len(row) >= 3:
                    resolution_map[name] = np.array([float(row[1]), float(row[2])], dtype=np.float32)
                else:
                    resolution_map[name] = float(row[1])
            except ValueError:
                continue
    return resolution_map


def load_sample_weight_map(path, max_weight):
    if not path:
        return {}
    sample_weight_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            try:
                weight = float(row[1])
            except ValueError:
                continue
            if not name:
                continue
            weight = max(1.0, min(weight, max_weight))
            sample_weight_map[name] = weight
    return sample_weight_map


def train(args, io):
    cfg.PointNms = args.point_nms
    cfg.CLASS_NUMS = args.point_nms

    train_files = read_list_file(args.train_list)
    test_files = read_list_file(args.test_list)
    resolution_map = load_resolution_map(args.resolution_csv)
    sample_weight_map = load_sample_weight_map(args.sample_weight_csv, args.max_sample_weight)

    train_loader = DataLoader(
        TrainData(
            train_files,
            return_sample_name=bool(sample_weight_map),
            use_bbox_crop=not args.train_full_image,
        ),
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        TestData(test_files, default_resolution=args.default_resolution, resolution_map=resolution_map),
        num_workers=0,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    model = get_model(num_layers=34, heads={"hm": 1, "class": cfg.PointNms}, NLayer1=2, NLayer2=4)
    model.cuda()

    if args.init_model:
        ckpt = torch.load(args.init_model, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Handle encoder_embed size mismatch across different landmark counts
        if "encoder_embed" in state_dict:
            src = state_dict["encoder_embed"]  # [1, N_src, C]
            dst_shape = model.encoder_embed.shape  # [1, N_dst, C]
            if src.shape != dst_shape:
                dst = torch.zeros(dst_shape, dtype=src.dtype, device=src.device)
                n_min = min(src.shape[1], dst_shape[1])
                dst[:, :n_min, :] = src[:, :n_min, :]
                state_dict["encoder_embed"] = dst
                io.cprint(
                    f"adapted encoder_embed from {tuple(src.shape)} to {tuple(dst_shape)} (copied first {n_min} landmarks)"
                )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        io.cprint(
            f"loaded init model: {args.init_model}, missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cos":
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=0.5e-6, last_epoch=-1)
    else:
        scheduler = StepLR(opt, step_size=20, gamma=0.7)

    scaler = GradScaler()
    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        tick = time.time()

        if args.scheduler == "cos":
            scheduler.step()
        else:
            if opt.param_groups[0]["lr"] > 1e-5:
                scheduler.step()

        for idx, batch_data in enumerate(train_loader):
            sample_weight = 1.0
            if sample_weight_map:
                train_data, hotmap, hot_mapl, offestxy, mask_, label_re, sample_name = batch_data
                if isinstance(sample_name, (list, tuple)):
                    sample_name = sample_name[0]
                sample_weight = float(sample_weight_map.get(sample_name, 1.0))
            else:
                train_data, hotmap, hot_mapl, offestxy, mask_, label_re = batch_data

            train_data = train_data.cuda().float()
            hot_mapl = hot_mapl.cuda().float()
            label_re = label_re.cuda().float()

            opt.zero_grad()
            with autocast():
                outputs, inint_coords, prehotmap = model(train_data)
                inint_loss_ = model.Initloss(inint_coords, label_re)
                loss = 0.0
                for i in range(len(outputs)):
                    loss = loss + model.loss(outputs[i], label_re)
                loss_ml_, loss_dl_, _, _ = rcal_loss(prehotmap, hot_mapl)
                loss = inint_loss_ + loss + loss_ml_ + loss_dl_
                loss = loss * sample_weight

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()
            if (idx + 1) % max(args.log_interval, 1) == 0:
                io.cprint(
                    f"epoch {epoch+1}/{args.epochs}, step {idx+1}/{len(train_loader)}, "
                    f"lr={opt.param_groups[0]['lr']:.6e}, loss={train_loss / args.log_interval:.6f}, "
                    f"time={time.time() - tick:.2f}s"
                )
                train_loss = 0.0
                tick = time.time()

        model.eval()
        all_counts = []
        total_masks = 0
        with torch.no_grad():
            for row_img, test_data, label_coords_, scalek, _, resol in test_loader:
                test_data = test_data.cuda().float()
                scalek = scalek.squeeze().numpy()
                resol = resol.squeeze().numpy()

                outputs, _, _ = model(test_data)
                pred = outputs[-1][:, :, :2].clip(0, 0.999)
                key_points, mask_ = decode_reg(pred)
                counts, _ = cal_acc(
                    torch.squeeze(row_img).numpy(),
                    key_points,
                    mask_,
                    label_coords_.squeeze().numpy(),
                    scalek,
                    resol,
                )
                all_counts.append(counts)
                total_masks += np.sum(mask_)

        all_counts = np.asarray(all_counts, dtype=np.float32)
        total_points = max(len(all_counts) * cfg.PointNms, 1)
        acc2 = float(np.sum(all_counts < cfg.ERROR_RANGE[0]) / total_points)
        io.cprint(
            f"eval epoch {epoch+1}/{args.epochs}: 2mm={acc2:.6f}, "
            f"2.5mm={np.sum(all_counts < cfg.ERROR_RANGE[1]) / total_points:.6f}, "
            f"3mm={np.sum(all_counts < cfg.ERROR_RANGE[2]) / total_points:.6f}, "
            f"4mm={np.sum(all_counts < cfg.ERROR_RANGE[3]) / total_points:.6f}, masks={int(total_masks)}"
        )

        if acc2 > best_acc:
            best_acc = acc2
            torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.output_dir, "best.pth"))

        if (epoch + 1) % args.save_interval == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.output_dir, f"epoch_{epoch+1}.pth"))

    io.cprint(f"training done, best 2mm acc={best_acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D-CeLR for cervical landmark regression")
    parser.add_argument("--train_list", type=str, required=True, help="Path to train.txt generated by converter")
    parser.add_argument("--test_list", type=str, required=True, help="Path to test.txt generated by converter")
    parser.add_argument("--point_nms", type=int, default=56, help="Number of landmarks")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scheduler", type=str, default="cos", choices=["cos", "step"])
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--default_resolution", type=float, default=1.0, help="Pixel spacing for metric reporting")
    parser.add_argument("--resolution_csv", type=str, default="", help="Optional CSV file mapping sample name to spacing")
    parser.add_argument("--sample_weight_csv", type=str, default="", help="Optional CSV mapping sample name to training loss weight")
    parser.add_argument("--max_sample_weight", type=float, default=3.0, help="Upper bound for per-sample loss weight")
    parser.add_argument("--init_model", type=str, default="", help="Optional checkpoint to initialize model weights")
    parser.add_argument(
        "--train_full_image",
        action="store_true",
        help="Use full-image resize for training instead of bbox-based random crop",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--output_dir", type=str, default="outputs/cervical")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    io = IOStream(os.path.join(args.output_dir, "train.log"))
    io.cprint(str(args))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by current D-CeLR model implementation")

    train(args, io)
