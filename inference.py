"""Inference and evaluation script for cervical spine keypoint detection.

Features:
- Single-image inference
- Dataset split inference (train/val/test)
- Optional metric computation when ground-truth is available
- Visualization and prediction export for acceptance checks
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import read_keypoints_from_json
from dataset import CervicalSpineDataset, collate_fn_cervical, load_nifti_image
from model import CervicalSpineKeypointModel, CervicalSpineLegacyKeypointModel
from preprocess import ImagePreprocessor

matplotlib.use("Agg")
matplotlib.rcParams.update(
	{
		"font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
		"axes.unicode_minus": False,
	}
)

import matplotlib.pyplot as plt


@dataclass
class InferenceConfig:
	model_path: str
	output_dir: str = "outputs/inference"
	device: str = "auto"
	data_dir: str = "data/"
	split: str = "test"
	image_path: Optional[str] = None
	json_path: Optional[str] = None
	batch_size: int = 8
	num_workers: int = 0
	target_size: int = 512
	max_batches: Optional[int] = None
	max_visualizations: int = 20
	seed: int = 42
	train_size: float = 0.7
	val_size: float = 0.15
	decode_beta: float = 5.0
	heatmap_size_override: Optional[int] = None


def parse_args() -> InferenceConfig:
	parser = argparse.ArgumentParser(description="Inference for cervical spine keypoint model")
	parser.add_argument("--model_path", required=True)
	parser.add_argument("--output_dir", default="outputs/inference")
	parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument("--data_dir", default="data/")
	parser.add_argument("--split", default="test", choices=["train", "val", "test"])
	parser.add_argument("--image_path", default=None)
	parser.add_argument("--json_path", default=None)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--target_size", type=int, default=512)
	parser.add_argument("--max_batches", type=int, default=None)
	parser.add_argument("--max_visualizations", type=int, default=20)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--train_size", type=float, default=0.7)
	parser.add_argument("--val_size", type=float, default=0.15)
	parser.add_argument("--decode_beta", type=float, default=5.0)
	parser.add_argument("--heatmap_size_override", type=int, default=None)
	args = parser.parse_args()

	return InferenceConfig(
		model_path=args.model_path,
		output_dir=args.output_dir,
		device=args.device,
		data_dir=args.data_dir,
		split=args.split,
		image_path=args.image_path,
		json_path=args.json_path,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		target_size=args.target_size,
		max_batches=args.max_batches,
		max_visualizations=args.max_visualizations,
		seed=args.seed,
		train_size=args.train_size,
		val_size=args.val_size,
		decode_beta=args.decode_beta,
		heatmap_size_override=args.heatmap_size_override,
	)


def resolve_device(device_name: str) -> torch.device:
	if device_name == "cpu":
		return torch.device("cpu")
	if device_name == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dirs(output_dir: Path) -> Dict[str, Path]:
	predictions_dir = output_dir / "predictions"
	visualizations_dir = output_dir / "visualizations"
	logs_dir = output_dir / "logs"

	predictions_dir.mkdir(parents=True, exist_ok=True)
	visualizations_dir.mkdir(parents=True, exist_ok=True)
	logs_dir.mkdir(parents=True, exist_ok=True)

	return {
		"root": output_dir,
		"predictions": predictions_dir,
		"visualizations": visualizations_dir,
		"logs": logs_dir,
	}


def build_model_and_load_checkpoint(
	model_path: Path,
	device: torch.device,
	decode_beta: float,
	heatmap_size_override: Optional[int] = None,
) -> torch.nn.Module:
	if not model_path.exists():
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	checkpoint = torch.load(model_path, map_location="cpu")

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
	else:
		state_dict = checkpoint

	is_legacy_regression = any("keypoint_head.mlp" in key for key in state_dict.keys())

	if is_legacy_regression:
		print("Detected legacy regression checkpoint. Using compatible legacy model.")
		model = CervicalSpineLegacyKeypointModel(
			num_keypoints=56,
			keypoint_dims=3,
			in_channels=1,
		).to(device)
		incompatible = model.load_state_dict(state_dict, strict=False)
		if incompatible.missing_keys:
			print(f"Warning: missing keys while loading checkpoint: {incompatible.missing_keys}")
		if incompatible.unexpected_keys:
			print(f"Warning: unexpected keys while loading checkpoint: {incompatible.unexpected_keys}")
		model.eval()
		return model

	heatmap_size = 256
	if isinstance(checkpoint, dict):
		cfg = checkpoint.get("config", {})
		if isinstance(cfg, dict):
			heatmap_size = int(cfg.get("heatmap_size", 256))

	if heatmap_size_override is not None:
		heatmap_size = int(heatmap_size_override)

	model = CervicalSpineKeypointModel(
		num_keypoints=56,
		keypoint_dims=3,
		in_channels=1,
		heatmap_size=(heatmap_size, heatmap_size),
		decode_beta=decode_beta,
	).to(device)

	incompatible = model.load_state_dict(state_dict, strict=False)

	if incompatible.missing_keys:
		print(f"Warning: missing keys while loading checkpoint: {incompatible.missing_keys}")
	if incompatible.unexpected_keys:
		print(f"Warning: unexpected keys while loading checkpoint: {incompatible.unexpected_keys}")

	model.eval()
	return model


def align_keypoint_tensors(
	predictions: torch.Tensor,
	targets: torch.Tensor,
	mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	shared_points = min(predictions.shape[1], targets.shape[1])
	if shared_points <= 0:
		raise ValueError("No shared keypoints between predictions and targets")
	return predictions[:, :shared_points, :], targets[:, :shared_points, :], mask[:, :shared_points]


def compute_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
	mask = (targets.abs().sum(dim=-1) > 0)
	predictions, targets, mask = align_keypoint_tensors(predictions, targets, mask.float())
	valid = mask.bool()
	if valid.sum().item() == 0:
		return {"mean_pixel_error": 0.0, "pck_10px": 0.0}

	pred_xy = predictions[..., :2]
	target_xy = targets[..., :2]
	distance = torch.linalg.norm(pred_xy - target_xy, dim=-1)
	valid_distance = distance[valid]

	return {
		"mean_pixel_error": float(valid_distance.mean().item()),
		"pck_10px": float((valid_distance <= 10.0).float().mean().item()),
	}


def save_prediction(case_name: str, prediction: np.ndarray, output_dir: Path) -> Path:
	out_path = output_dir / f"{case_name}_keypoints.npy"
	np.save(out_path, prediction)
	return out_path


def save_single_visualization(
	image_2d: np.ndarray,
	pred_kps: np.ndarray,
	output_path: Path,
	gt_kps: Optional[np.ndarray] = None,
	title: Optional[str] = None,
) -> None:
	fig = plt.figure(figsize=(8, 8))
	plt.imshow(image_2d, cmap="gray")
	plt.scatter(pred_kps[:, 1], pred_kps[:, 0], c="lime", s=14, alpha=0.8, label="pred")
	if gt_kps is not None and gt_kps.shape[0] > 0:
		plt.scatter(gt_kps[:, 1], gt_kps[:, 0], c="red", s=10, alpha=0.7, label="gt")
		plt.legend(loc="upper right")
	if title:
		plt.title(title)
	plt.tight_layout()
	fig.savefig(output_path, dpi=120)
	plt.close(fig)


@torch.no_grad()
def run_split_inference(
	model: torch.nn.Module,
	config: InferenceConfig,
	device: torch.device,
	output_dirs: Dict[str, Path],
) -> None:
	dataset = CervicalSpineDataset(
		data_dir=config.data_dir,
		split=config.split,
		train_size=config.train_size,
		val_size=config.val_size,
		seed=config.seed,
		augmentation=False,
		target_size=config.target_size,
	)
	loader = DataLoader(
		dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
		pin_memory=torch.cuda.is_available(),
		collate_fn=collate_fn_cervical,
	)

	rows: List[Dict[str, float]] = []
	case_rows: List[Dict[str, float]] = []
	vis_count = 0
	per_kp_sum_dist: Dict[int, float] = {}
	per_kp_count: Dict[int, int] = {}
	per_kp_pck_count: Dict[int, int] = {}

	for batch_idx, batch in enumerate(loader, start=1):
		images = batch["image"].to(device=device, dtype=torch.float32)
		outputs = model(images)
		pred = outputs["keypoints"].detach().cpu()
		gt = batch["keypoints"].float()
		metrics = compute_batch_metrics(pred, gt)

		mask = (gt.abs().sum(dim=-1) > 0).float()
		pred_aligned, gt_aligned, mask_aligned = align_keypoint_tensors(pred, gt, mask)
		distance = torch.linalg.norm(pred_aligned[..., :2] - gt_aligned[..., :2], dim=-1)

		num_keypoints = int(distance.shape[1])
		for kp_idx in range(num_keypoints):
			valid_kp = mask_aligned[:, kp_idx].bool()
			if valid_kp.any():
				kp_dist = distance[:, kp_idx][valid_kp]
				per_kp_sum_dist[kp_idx] = per_kp_sum_dist.get(kp_idx, 0.0) + float(kp_dist.sum().item())
				per_kp_count[kp_idx] = per_kp_count.get(kp_idx, 0) + int(kp_dist.numel())
				per_kp_pck_count[kp_idx] = per_kp_pck_count.get(kp_idx, 0) + int((kp_dist <= 10.0).sum().item())

		rows.append(
			{
				"batch": float(batch_idx),
				"mean_pixel_error": metrics["mean_pixel_error"],
				"pck_10px": metrics["pck_10px"],
			}
		)

		pred_np = pred.numpy()
		gt_np = gt.numpy()
		img_np = batch["image"].numpy()[:, 0, :, :]

		for i, case_name in enumerate(batch["case_names"]):
			case_pred = pred_np[i]
			save_prediction(case_name, case_pred, output_dirs["predictions"])

			valid_case = mask_aligned[i].bool()
			if valid_case.any():
				case_dist = distance[i][valid_case]
				case_rows.append(
					{
						"case_name": case_name,
						"valid_keypoints": float(case_dist.numel()),
						"mean_pixel_error": float(case_dist.mean().item()),
						"pck_10px": float((case_dist <= 10.0).float().mean().item()),
					}
				)

			if vis_count < config.max_visualizations:
				vis_path = output_dirs["visualizations"] / f"{case_name}_pred.png"
				save_single_visualization(
					image_2d=img_np[i],
					pred_kps=case_pred,
					gt_kps=gt_np[i][: case_pred.shape[0]],
					output_path=vis_path,
					title=f"{case_name} ({config.split})",
				)
				vis_count += 1

		if config.max_batches is not None and batch_idx >= config.max_batches:
			break

		if batch_idx % 10 == 0:
			print(f"Processed {batch_idx} batches...")

	metrics_csv = output_dirs["logs"] / f"{config.split}_metrics.csv"
	with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["batch", "mean_pixel_error", "pck_10px"])
		writer.writeheader()
		writer.writerows(rows)

	case_metrics_csv = output_dirs["logs"] / f"{config.split}_case_metrics.csv"
	case_rows_sorted = sorted(case_rows, key=lambda x: x["mean_pixel_error"], reverse=True)
	with open(case_metrics_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["case_name", "valid_keypoints", "mean_pixel_error", "pck_10px"])
		writer.writeheader()
		writer.writerows(case_rows_sorted)

	keypoint_rows: List[Dict[str, float]] = []
	for kp_idx in sorted(per_kp_count.keys()):
		count = per_kp_count[kp_idx]
		if count <= 0:
			continue
		keypoint_rows.append(
			{
				"keypoint_index": float(kp_idx),
				"valid_count": float(count),
				"mean_pixel_error": float(per_kp_sum_dist[kp_idx] / count),
				"pck_10px": float(per_kp_pck_count[kp_idx] / count),
			}
		)

	keypoint_rows_sorted = sorted(keypoint_rows, key=lambda x: x["mean_pixel_error"], reverse=True)
	keypoint_metrics_csv = output_dirs["logs"] / f"{config.split}_keypoint_metrics.csv"
	with open(keypoint_metrics_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["keypoint_index", "valid_count", "mean_pixel_error", "pck_10px"])
		writer.writeheader()
		writer.writerows(keypoint_rows_sorted)

	if rows:
		avg_mpe = float(np.mean([r["mean_pixel_error"] for r in rows]))
		avg_pck = float(np.mean([r["pck_10px"] for r in rows]))
	else:
		avg_mpe = 0.0
		avg_pck = 0.0

	summary = {
		"split": config.split,
		"num_batches": len(rows),
		"avg_mean_pixel_error": avg_mpe,
		"avg_pck_10px": avg_pck,
		"output_predictions": str(output_dirs["predictions"]),
		"output_visualizations": str(output_dirs["visualizations"]),
	}
	summary_path = output_dirs["logs"] / f"{config.split}_summary.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	print("=" * 80)
	print(f"Split inference done: {config.split}")
	print(f"Batches: {len(rows)}")
	print(f"Average MPE: {avg_mpe:.2f}px")
	print(f"Average PCK@10: {avg_pck:.4f}")
	print(f"Metrics CSV: {metrics_csv}")
	print(f"Case metrics CSV: {case_metrics_csv}")
	print(f"Keypoint metrics CSV: {keypoint_metrics_csv}")
	print(f"Summary JSON: {summary_path}")


@torch.no_grad()
def run_single_image_inference(
	model: torch.nn.Module,
	config: InferenceConfig,
	device: torch.device,
	output_dirs: Dict[str, Path],
) -> None:
	if config.image_path is None:
		raise ValueError("--image_path is required for single-image inference")

	image_path = Path(config.image_path)
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	img_nib = load_nifti_image(str(image_path))
	image = img_nib.get_fdata().astype(np.float32)
	preprocessor = ImagePreprocessor(target_size=config.target_size)
	dummy_kps = np.zeros((0, 3), dtype=np.float32)
	image_tensor, _ = preprocessor.preprocess(image, dummy_kps)

	image_tensor = image_tensor.to(device=device, dtype=torch.float32)
	outputs = model(image_tensor)
	pred = outputs["keypoints"][0].detach().cpu().numpy()

	case_name = image_path.stem.replace(".nii", "")
	pred_path = save_prediction(case_name, pred, output_dirs["predictions"])

	gt_kps = None
	if config.json_path:
		json_path = Path(config.json_path)
		if json_path.exists():
			gt_kps, _ = read_keypoints_from_json(json_path, img_nib)
			gt_kps = preprocessor.transform_keypoints(gt_kps, (config.target_size / image.shape[0], config.target_size / image.shape[1]))

	img_2d = image_tensor[0, 0].detach().cpu().numpy()
	vis_path = output_dirs["visualizations"] / f"{case_name}_pred.png"
	save_single_visualization(
		image_2d=img_2d,
		pred_kps=pred,
		gt_kps=gt_kps,
		output_path=vis_path,
		title=f"Single image inference: {case_name}",
	)

	summary = {
		"case_name": case_name,
		"image_path": str(image_path),
		"prediction_path": str(pred_path),
		"visualization_path": str(vis_path),
		"num_pred_keypoints": int(pred.shape[0]),
	}
	summary_path = output_dirs["logs"] / f"{case_name}_summary.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	print("=" * 80)
	print("Single-image inference done")
	print(f"Prediction: {pred_path}")
	print(f"Visualization: {vis_path}")
	print(f"Summary: {summary_path}")


def main() -> None:
	config = parse_args()
	device = resolve_device(config.device)
	output_dirs = prepare_output_dirs(Path(config.output_dir))
	model = build_model_and_load_checkpoint(
		Path(config.model_path),
		device,
		config.decode_beta,
		heatmap_size_override=config.heatmap_size_override,
	)

	with open(output_dirs["logs"] / "inference_config.json", "w", encoding="utf-8") as f:
		json.dump(asdict(config), f, indent=2, ensure_ascii=False)

	print("=" * 80)
	print("Cervical spine inference")
	print("=" * 80)
	print(f"Device: {device}")
	print(f"Model checkpoint: {config.model_path}")
	print(f"Output root: {output_dirs['root']}")

	if config.image_path:
		run_single_image_inference(model, config, device, output_dirs)
	else:
		run_split_inference(model, config, device, output_dirs)


if __name__ == "__main__":
	main()
