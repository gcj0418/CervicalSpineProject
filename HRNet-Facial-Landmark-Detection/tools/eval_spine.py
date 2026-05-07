#!/usr/bin/env python3

import argparse
import os
import pprint
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lib.models as models
from lib.config import config, update_config
from lib.core.evaluation import decode_preds
from lib.core.spine_evaluation import load_resolution_map, get_sample_resolution, pair_landmarks, point_error_mm
from lib.datasets import get_dataset
from lib.utils import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HRNet on spine landmarks')
    parser.add_argument('--cfg', required=True, type=str, help='experiment configuration filename')
    parser.add_argument('--model-file', required=True, type=str, help='model parameters')
    parser.add_argument('--resolution-csv', default='', type=str, help='spacing CSV with sample, spacing_x, spacing_y')
    parser.add_argument('--default-resolution', default=1.0, type=float, help='fallback spacing if a sample is missing from the CSV')
    parser.add_argument('--matching-mode', default='hungarian', type=str, help='landmark matching mode')
    parser.add_argument('--dump-sample-csv', default='', type=str, help='optional per-sample metric CSV')

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'eval')
    final_output_path = Path(final_output_dir)
    logs_dir = final_output_path / 'logs'
    predictions_dir = final_output_path / 'predictions'
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (final_output_path / 'config.yaml').write_text(config.dump(), encoding='utf-8')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    state_dict = torch.load(args.model_file, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
    else:
        # state_dict is a full model object (e.g. model_best.pth from save_checkpoint)
        model.module.load_state_dict(state_dict.state_dict())

    dataset_type = get_dataset(config)
    test_loader = DataLoader(
        dataset=dataset_type(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    resolution_map = load_resolution_map(args.resolution_csv)
    all_errors_mm = []
    per_sample_rows = []
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(test_loader.dataset), num_classes, 2))

    model.eval()
    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(test_loader):
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)

            batch_size = score_map.size(0)
            for n in range(batch_size):
                img_id = meta['img_id'][n]
                gt_landmarks = meta['pts'][n].cpu().numpy()
                pr_landmarks = preds[n].cpu().numpy()
                sample_resolution = get_sample_resolution(resolution_map, img_id, args.default_resolution)
                res_arr = np.asarray(sample_resolution, dtype=np.float32).reshape(-1)
                pr_matched, gt_matched, dists = pair_landmarks(
                    pr_landmarks,
                    gt_landmarks,
                    sample_resolution,
                    matching_mode=args.matching_mode,
                )
                sample_mean_mm = float(np.mean(dists)) if dists.size else 0.0
                all_errors_mm.extend(dists.tolist())
                spacing_x = float(res_arr[0]) if res_arr.size >= 1 else float(args.default_resolution)
                spacing_y = float(res_arr[1]) if res_arr.size >= 2 else spacing_x
                per_sample_rows.append((img_id, sample_mean_mm, len(dists), spacing_x, spacing_y))
                predictions[meta['index'][n], :, :] = preds[n, :, :]

    errors = np.asarray(all_errors_mm, dtype=np.float32)
    mean_error = float(errors.mean()) if errors.size else 0.0
    thresholds = [2.0, 2.5, 3.0, 4.0]
    accs = {thr: float((errors <= thr).mean()) if errors.size else 0.0 for thr in thresholds}

    msg = (
        'Spine eval mean_error:{:.4f} acc@2:{:.4f} acc@2.5:{:.4f} acc@3:{:.4f} acc@4:{:.4f}'
        .format(mean_error, accs[2.0], accs[2.5], accs[3.0], accs[4.0])
    )
    logger.info(msg)

    torch.save(predictions, os.path.join(str(predictions_dir), 'predictions.pth'))

    comparison_md = logs_dir / 'comparison_table.md'
    with comparison_md.open('w', encoding='utf-8') as f:
        f.write('| dataset | model | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n')
        f.write('| --- | --- | ---: | ---: | ---: | ---: | ---: |\n')
        f.write(
            '| {dataset} | {model} | {mean:.4f} | {a2:.4f} | {a25:.4f} | {a3:.4f} | {a4:.4f} |\n'.format(
                dataset=config.DATASET.DATASET,
                model=os.path.basename(args.model_file),
                mean=mean_error,
                a2=accs[2.0],
                a25=accs[2.5],
                a3=accs[3.0],
                a4=accs[4.0],
            )
        )
    logger.info('saved comparison table to {}'.format(comparison_md))

    if args.dump_sample_csv:
        import csv
        csv_path = args.dump_sample_csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample', 'mean_mm', 'num_points', 'spacing_x', 'spacing_y'])
            for row in per_sample_rows:
                writer.writerow(row)
        logger.info('saved per-sample csv to {}'.format(csv_path))


if __name__ == '__main__':
    main()
