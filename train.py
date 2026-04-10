"""Training entrypoint for cervical spine keypoint detection.

This script trains the keypoint branch of the multi-task model using the
current dataset and preprocessing pipeline. It keeps the implementation
lightweight and CPU-friendly, while still providing:

- train/val/test dataset splits
- checkpoint saving
- CSV metric logging
- resume support
- quick dry-run options for validation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import CervicalSpineDataset, collate_fn_cervical
from model import CervicalSpineKeypointModel, CervicalSpineLegacyKeypointModel, count_parameters


@dataclass
class TrainingConfig:
    data_dir: str = "data/"
    output_dir: str = "outputs/training"
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_size: float = 0.7
    val_size: float = 0.15
    seed: int = 42
    num_workers: int = 0
    target_size: int = 512
    augmentation: bool = True
    device: str = "auto"
    resume: Optional[str] = None
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    save_every: int = 1
    patience: int = 10
    grad_clip_norm: float = 1.0
    heatmap_size: int = 256
    heatmap_sigma: float = 2.5
    require_num_keypoints: Optional[int] = 56
    skip_final_test: bool = False
    hard_case_csv: Optional[str] = None
    hard_case_mpe_threshold: float = 35.0
    hard_case_weight: float = 3.0
    keypoint_weight_indices: Optional[str] = None
    keypoint_weight_factor: float = 2.0
    model_type: str = "heatmap"


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train cervical spine keypoint detector")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="outputs/training")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--augmentation", action="store_true", help="Enable training augmentation")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable training augmentation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--heatmap_size", type=int, default=256)
    parser.add_argument("--heatmap_sigma", type=float, default=2.5)
    parser.add_argument(
        "--require_num_keypoints",
        type=int,
        default=56,
        help="Keep only samples with exactly this number of keypoints. Set <=0 to disable.",
    )
    parser.add_argument(
        "--allow_variable_keypoints",
        action="store_true",
        help="Disable keypoint-count filtering and train on all samples.",
    )
    parser.add_argument(
        "--skip_final_test",
        action="store_true",
        help="Skip final test-set evaluation for faster smoke runs.",
    )
    parser.add_argument(
        "--hard_case_csv",
        default=None,
        help="Path to case-level metrics CSV with columns case_name, mean_pixel_error.",
    )
    parser.add_argument(
        "--hard_case_mpe_threshold",
        type=float,
        default=35.0,
        help="Mark cases with mean_pixel_error >= threshold as hard cases.",
    )
    parser.add_argument(
        "--hard_case_weight",
        type=float,
        default=3.0,
        help="Sampling weight multiplier for hard cases.",
    )
    parser.add_argument(
        "--keypoint_weight_indices",
        default=None,
        help="Comma-separated keypoint indices to upweight, e.g. '54,26,21,25'.",
    )
    parser.add_argument(
        "--keypoint_weight_factor",
        type=float,
        default=2.0,
        help="Loss weight factor for selected keypoint indices.",
    )
    parser.add_argument(
        "--model_type",
        default="heatmap",
        choices=["heatmap", "legacy_regression"],
        help="Training model type. Use legacy_regression to reproduce old baseline behavior.",
    )

    args = parser.parse_args()
    augmentation = args.augmentation
    if args.no_augmentation:
        augmentation = False

    require_num_keypoints = None if args.allow_variable_keypoints or args.require_num_keypoints <= 0 else int(args.require_num_keypoints)

    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        num_workers=args.num_workers,
        target_size=args.target_size,
        augmentation=augmentation,
        device=args.device,
        resume=args.resume,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        save_every=args.save_every,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        require_num_keypoints=require_num_keypoints,
        skip_final_test=args.skip_final_test,
        hard_case_csv=args.hard_case_csv,
        hard_case_mpe_threshold=args.hard_case_mpe_threshold,
        hard_case_weight=args.hard_case_weight,
        keypoint_weight_indices=args.keypoint_weight_indices,
        keypoint_weight_factor=args.keypoint_weight_factor,
        model_type=args.model_type,
    )


def parse_index_csv(value: Optional[str]) -> List[int]:
    if value is None:
        return []
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    indices: List[int] = []
    for token in tokens:
        try:
            indices.append(int(token))
        except ValueError:
            continue
    return sorted(set(index for index in indices if index >= 0))


def load_hard_cases_from_csv(csv_path: Optional[str], mpe_threshold: float) -> Set[str]:
    if not csv_path:
        return set()

    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] hard_case_csv not found: {path}")
        return set()

    hard_cases: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = (row.get("case_name") or "").strip()
            if not case_name:
                continue
            try:
                mpe = float(row.get("mean_pixel_error", "nan"))
            except ValueError:
                continue
            if math.isfinite(mpe) and mpe >= mpe_threshold:
                hard_cases.add(case_name)

    print(f"Loaded hard cases: {len(hard_cases)} from {path}")
    return hard_cases


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dirs(output_dir: Path) -> Dict[str, Path]:
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    visualizations_dir = output_dir / "visualizations"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_dir,
        "checkpoints": checkpoints_dir,
        "logs": logs_dir,
        "visualizations": visualizations_dir,
    }


def build_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    filter_enabled = config.require_num_keypoints is not None

    train_dataset = CervicalSpineDataset(
        data_dir=config.data_dir,
        split="train",
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
        augmentation=config.augmentation,
        target_size=config.target_size,
        required_num_keypoints=config.require_num_keypoints,
        skip_invalid_keypoint_count=filter_enabled,
    )
    val_dataset = CervicalSpineDataset(
        data_dir=config.data_dir,
        split="val",
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
        augmentation=False,
        target_size=config.target_size,
        required_num_keypoints=config.require_num_keypoints,
        skip_invalid_keypoint_count=filter_enabled,
    )
    test_dataset = CervicalSpineDataset(
        data_dir=config.data_dir,
        split="test",
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
        augmentation=False,
        target_size=config.target_size,
        required_num_keypoints=config.require_num_keypoints,
        skip_invalid_keypoint_count=filter_enabled,
    )

    hard_cases = load_hard_cases_from_csv(config.hard_case_csv, config.hard_case_mpe_threshold)
    train_sampler = None
    train_shuffle = True
    if hard_cases and config.hard_case_weight > 1.0:
        sample_weights: List[float] = []
        hard_count = 0
        for split_index in range(len(train_dataset)):
            actual_idx = int(train_dataset.indices[split_index])
            case_name = train_dataset.file_list[actual_idx][2]
            is_hard = case_name in hard_cases
            if is_hard:
                hard_count += 1
            sample_weights.append(float(config.hard_case_weight if is_hard else 1.0))

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
        print(
            f"Hard-case sampler enabled: hard_in_train={hard_count}/{len(sample_weights)}, "
            f"weight={config.hard_case_weight}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_cervical,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_cervical,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_cervical,
    )

    return train_loader, val_loader, test_loader


def make_keypoint_mask(targets: torch.Tensor) -> torch.Tensor:
    # Padding rows are all-zero in collate_fn_cervical; use them as invalid rows.
    return (targets.abs().sum(dim=-1) > 0).float()


def align_keypoint_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align prediction/target lengths for datasets with variable keypoint counts.

    Model output is fixed-size (e.g. 56), while dataset rows may be larger after
    collation when some samples contain more points. Keep only the shared prefix
    length so loss/metrics are well-defined.
    """
    shared_points = min(predictions.shape[1], targets.shape[1])
    if shared_points <= 0:
        raise ValueError("No shared keypoints available between predictions and targets")

    return (
        predictions[:, :shared_points, :],
        targets[:, :shared_points, :],
        mask[:, :shared_points],
    )


def generate_target_heatmaps(
    targets: torch.Tensor,
    mask: torch.Tensor,
    heatmap_size: Tuple[int, int],
    image_size: Tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Generate Gaussian target heatmaps from keypoint coordinates."""
    batch_size, num_keypoints, _ = targets.shape
    hm_h, hm_w = heatmap_size
    img_h, img_w = image_size

    yy = torch.arange(hm_h, device=targets.device, dtype=torch.float32).view(1, 1, hm_h, 1)
    xx = torch.arange(hm_w, device=targets.device, dtype=torch.float32).view(1, 1, 1, hm_w)

    center_y = targets[..., 0] / max(img_h - 1, 1) * (hm_h - 1)
    center_x = targets[..., 1] / max(img_w - 1, 1) * (hm_w - 1)

    center_y = center_y.view(batch_size, num_keypoints, 1, 1)
    center_x = center_x.view(batch_size, num_keypoints, 1, 1)

    dist2 = (yy - center_y).pow(2) + (xx - center_x).pow(2)
    heatmaps = torch.exp(-dist2 / (2.0 * sigma * sigma))
    heatmaps = heatmaps * mask.view(batch_size, num_keypoints, 1, 1)
    return heatmaps


def keypoint_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    image_size: Tuple[int, int],
    keypoint_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    heatmap_size = (int(logits.shape[-2]), int(logits.shape[-1]))
    target_heatmaps = generate_target_heatmaps(
        targets=targets,
        mask=mask,
        heatmap_size=heatmap_size,
        image_size=image_size,
        sigma=sigma,
    )

    per_pixel_loss = F.mse_loss(torch.sigmoid(logits), target_heatmaps, reduction="none")
    masked_loss = per_pixel_loss * mask.unsqueeze(-1).unsqueeze(-1)

    if keypoint_weights is not None:
        active_weights = keypoint_weights[: logits.shape[1]].view(1, -1, 1, 1)
        masked_loss = masked_loss * active_weights
        weighted_mask = mask * keypoint_weights[: logits.shape[1]].view(1, -1)
        normalizer = weighted_mask.sum().clamp_min(1.0) * logits.shape[-2] * logits.shape[-1]
        return masked_loss.sum() / normalizer

    normalizer = mask.sum().clamp_min(1.0) * logits.shape[-2] * logits.shape[-1]
    return masked_loss.sum() / normalizer


def build_keypoint_weights(config: TrainingConfig, num_keypoints: int, device: torch.device) -> Optional[torch.Tensor]:
    selected_indices = parse_index_csv(config.keypoint_weight_indices)
    if not selected_indices:
        return None

    weights = torch.ones(num_keypoints, device=device, dtype=torch.float32)
    for idx in selected_indices:
        if idx < num_keypoints:
            weights[idx] = float(config.keypoint_weight_factor)

    return weights


def keypoint_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    per_dim = F.mse_loss(predictions, targets, reduction="none")
    per_point = per_dim.sum(dim=-1)
    masked = per_point * mask
    normalizer = mask.sum().clamp_min(1.0)
    return masked.sum() / normalizer


@torch.no_grad()
def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    valid = mask.bool()
    if valid.sum().item() == 0:
        return {"loss": 0.0, "mean_pixel_error": 0.0, "pck_10px": 0.0}

    # Use x/y for pixel-space evaluation; z is not meaningful for 2D visual alignment.
    pred_xy = predictions[..., :2]
    target_xy = targets[..., :2]
    distance = torch.linalg.norm(pred_xy - target_xy, dim=-1)

    valid_distance = distance[valid]
    mean_pixel_error = valid_distance.mean().item()
    pck_10px = (valid_distance <= 10.0).float().mean().item()

    return {
        "mean_pixel_error": mean_pixel_error,
        "pck_10px": pck_10px,
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    heatmap_sigma: float,
    grad_clip_norm: float,
    keypoint_weights: Optional[torch.Tensor],
    model_type: str,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    total_metrics = {"mean_pixel_error": 0.0, "pck_10px": 0.0}

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device=device, dtype=torch.float32)
        targets = batch["keypoints"].to(device=device, dtype=torch.float32)
        mask = make_keypoint_mask(targets)

        if training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        predictions = outputs["keypoints"]
        predictions, targets, mask = align_keypoint_tensors(predictions, targets, mask)
        if model_type == "legacy_regression":
            loss = keypoint_regression_loss(
                predictions=predictions,
                targets=targets,
                mask=mask,
            )
        else:
            heatmap_logits = outputs["keypoint_heatmaps"]
            heatmap_logits = heatmap_logits[:, : predictions.shape[1], :, :]
            loss = keypoint_heatmap_loss(
                logits=heatmap_logits,
                targets=targets,
                mask=mask,
                sigma=heatmap_sigma,
                image_size=images.shape[-2:],
                keypoint_weights=keypoint_weights,
            )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        metrics = compute_metrics(predictions.detach(), targets, mask)
        total_loss += loss.item()
        total_metrics["mean_pixel_error"] += metrics["mean_pixel_error"]
        total_metrics["pck_10px"] += metrics["pck_10px"]
        total_batches += 1

        if max_batches is not None and batch_index >= max_batches:
            break

    if total_batches == 0:
        return {"loss": 0.0, "mean_pixel_error": 0.0, "pck_10px": 0.0}

    return {
        "loss": total_loss / total_batches,
        "mean_pixel_error": total_metrics["mean_pixel_error"] / total_batches,
        "pck_10px": total_metrics["pck_10px"] / total_batches,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: TrainingConfig,
    history: list,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": asdict(config),
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, float, list]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    history = list(checkpoint.get("history", []))
    return start_epoch, best_val_loss, history


def write_history_csv(history: list, csv_path: Path) -> None:
    if not history:
        return

    fieldnames = sorted({key for record in history for key in record.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    output_dirs = prepare_output_dirs(Path(config.output_dir))
    device = resolve_device(config.device)

    train_loader, val_loader, test_loader = build_dataloaders(config)

    if config.model_type == "legacy_regression":
        model = CervicalSpineLegacyKeypointModel(
            num_keypoints=56,
            keypoint_dims=3,
            in_channels=1,
        ).to(device)
        keypoint_weights = None
    else:
        model = CervicalSpineKeypointModel(
            num_keypoints=56,
            keypoint_dims=3,
            in_channels=1,
            heatmap_size=(config.heatmap_size, config.heatmap_size),
        ).to(device)
        keypoint_weights = build_keypoint_weights(config, num_keypoints=56, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    start_epoch = 1
    best_val_loss = math.inf
    history: list = []

    if config.resume:
        resume_path = Path(config.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_val_loss, history = load_checkpoint(resume_path, model, optimizer)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    config_path = output_dirs["root"] / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    metrics_csv = output_dirs["logs"] / "metrics.csv"
    latest_checkpoint = output_dirs["checkpoints"] / "latest.pth"
    best_checkpoint = output_dirs["checkpoints"] / "best.pth"

    print("=" * 80)
    print("Cervical spine training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model type: {config.model_type}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"Augmentation:  {config.augmentation}")
    if config.require_num_keypoints is None:
        print("Keypoint filter: disabled (variable keypoint counts enabled)")
    else:
        print(f"Keypoint filter: keep exactly {config.require_num_keypoints} points")
    if keypoint_weights is not None:
        selected = parse_index_csv(config.keypoint_weight_indices)
        print(f"Keypoint weighting: indices={selected}, factor={config.keypoint_weight_factor}")
    elif config.model_type == "legacy_regression":
        print("Keypoint weighting: disabled (legacy_regression mode)")
    else:
        print("Keypoint weighting: disabled")
    if config.hard_case_csv:
        print(
            f"Hard-case sampling config: csv={config.hard_case_csv}, "
            f"threshold={config.hard_case_mpe_threshold}, weight={config.hard_case_weight}"
        )
    print(f"Output dir:    {output_dirs['root']}")

    patience_counter = 0
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            heatmap_sigma=config.heatmap_sigma,
            grad_clip_norm=config.grad_clip_norm,
            keypoint_weights=keypoint_weights,
            model_type=config.model_type,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            device=device,
            heatmap_sigma=config.heatmap_sigma,
            grad_clip_norm=config.grad_clip_norm,
            keypoint_weights=keypoint_weights,
            model_type=config.model_type,
            max_batches=config.max_val_batches,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mean_pixel_error": train_metrics["mean_pixel_error"],
            "train_pck_10px": train_metrics["pck_10px"],
            "val_loss": val_metrics["loss"],
            "val_mean_pixel_error": val_metrics["mean_pixel_error"],
            "val_pck_10px": val_metrics["pck_10px"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mpe={val_metrics['mean_pixel_error']:.2f}px | "
            f"val_pck@10={val_metrics['pck_10px']:.3f}"
        )

        save_checkpoint(
            checkpoint_path=latest_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config,
            history=history,
        )

        if epoch % config.save_every == 0:
            periodic_checkpoint = output_dirs["checkpoints"] / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                checkpoint_path=periodic_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
                history=history,
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(
                checkpoint_path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
                history=history,
            )
            print(f"  New best checkpoint saved: {best_checkpoint.name}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered after {config.patience} epochs without improvement.")
                break

        write_history_csv(history, metrics_csv)

    write_history_csv(history, metrics_csv)

    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if config.skip_final_test:
        print("=" * 80)
        print("Skip final test metrics (--skip_final_test enabled)")
        print("=" * 80)
        return

    test_metrics = run_epoch(
        model=model,
        dataloader=test_loader,
        optimizer=None,
        device=device,
        heatmap_sigma=config.heatmap_sigma,
        grad_clip_norm=config.grad_clip_norm,
        keypoint_weights=keypoint_weights,
        model_type=config.model_type,
        max_batches=None,
    )
    print("=" * 80)
    print("Final test metrics")
    print("=" * 80)
    print(f"test_loss={test_metrics['loss']:.4f}")
    print(f"test_mean_pixel_error={test_metrics['mean_pixel_error']:.2f}px")
    print(f"test_pck@10={test_metrics['pck_10px']:.3f}")


if __name__ == "__main__":
    main()