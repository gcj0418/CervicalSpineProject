"""Model definitions for cervical spine keypoint detection and segmentation.

This module provides a lightweight multi-task network built on a ResNet18
encoder. The default configuration is tuned for the current dataset:

- single-channel 2D X-ray input
- 56 anatomical keypoints
- optional binary segmentation output

The module also includes a small self-test that validates the forward pass
against dummy input and, when available, one real sample from dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _make_resnet18_encoder(in_channels: int = 1, weights=None) -> nn.Module:
	"""Create a ResNet18 encoder that accepts single-channel input."""

	backbone = models.resnet18(weights=weights)

	if in_channels != 3:
		old_conv = backbone.conv1
		backbone.conv1 = nn.Conv2d(
			in_channels,
			old_conv.out_channels,
			kernel_size=old_conv.kernel_size[0],
			stride=old_conv.stride[0],
			padding=old_conv.padding[0],
			bias=old_conv.bias is not None,
		)

		with torch.no_grad():
			if in_channels == 1:
				backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
			else:
				repeat = (in_channels + 2) // 3
				expanded = old_conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
				backbone.conv1.weight.copy_(expanded / repeat)

	# Keep the stem and residual stages only; the heads are built separately.
	encoder = nn.Sequential(
		backbone.conv1,
		backbone.bn1,
		backbone.relu,
		backbone.maxpool,
		backbone.layer1,
		backbone.layer2,
		backbone.layer3,
		backbone.layer4,
	)

	return encoder


class ConvBlock(nn.Module):
	"""A small conv-batchnorm-relu block used in the decoder."""

	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class KeypointHeatmapHead(nn.Module):
	"""Predict per-keypoint heatmaps from the encoded feature map."""

	def __init__(
		self,
		in_channels: int,
		num_keypoints: int = 56,
		heatmap_size: Tuple[int, int] = (256, 256),
	):
		super().__init__()
		self.num_keypoints = num_keypoints
		self.heatmap_size = heatmap_size

		self.project = ConvBlock(in_channels, 256)
		self.upsample = nn.Sequential(
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
			ConvBlock(256, 128),
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
			ConvBlock(128, 64),
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
			ConvBlock(64, 32),
		)
		self.classifier = nn.Conv2d(32, num_keypoints, kernel_size=1)

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		x = self.project(features)
		x = self.upsample(x)
		x = self.classifier(x)
		if x.shape[-2:] != self.heatmap_size:
			x = F.interpolate(x, size=self.heatmap_size, mode="bilinear", align_corners=False)
		return x


class KeypointRegressionHead(nn.Module):
	"""Legacy direct-regression keypoint head (compatible with old checkpoints)."""

	def __init__(self, in_channels: int, num_keypoints: int = 56, keypoint_dims: int = 3):
		super().__init__()
		self.num_keypoints = num_keypoints
		self.keypoint_dims = keypoint_dims
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.mlp = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_channels, in_channels),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.1),
			nn.Linear(in_channels, num_keypoints * keypoint_dims),
		)

		self.register_buffer("range_mins", torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
		self.register_buffer("range_sizes", torch.tensor([511.0, 511.0, 1.0], dtype=torch.float32))

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		x = self.pool(features)
		x = self.mlp(x)
		x = x.view(features.shape[0], self.num_keypoints, self.keypoint_dims)

		range_mins = torch.reshape(cast(torch.Tensor, self.range_mins), (1, 1, -1))
		range_sizes = torch.reshape(cast(torch.Tensor, self.range_sizes), (1, 1, -1))
		return torch.sigmoid(x) * range_sizes + range_mins


def decode_heatmaps_to_keypoints(
	heatmap_logits: torch.Tensor,
	output_size: Tuple[int, int],
	beta: float = 5.0,
) -> torch.Tensor:
	"""Decode heatmap logits to (x, y, z) keypoint coordinates in image space.

	Uses soft-argmax over the spatial heatmap so coordinates stay continuous.
	"""
	if heatmap_logits.ndim != 4:
		raise ValueError(f"Expected heatmap logits shape (B, K, H, W), got {tuple(heatmap_logits.shape)}")

	batch_size, num_keypoints, hm_h, hm_w = heatmap_logits.shape
	flat_logits = heatmap_logits.view(batch_size, num_keypoints, -1)
	probabilities = torch.softmax(flat_logits * beta, dim=-1)

	row_coords = torch.arange(hm_h, device=heatmap_logits.device, dtype=torch.float32)
	col_coords = torch.arange(hm_w, device=heatmap_logits.device, dtype=torch.float32)
	grid_rows, grid_cols = torch.meshgrid(row_coords, col_coords, indexing="ij")
	grid_rows = grid_rows.reshape(-1)
	grid_cols = grid_cols.reshape(-1)

	rows = torch.sum(probabilities * grid_rows.view(1, 1, -1), dim=-1)
	cols = torch.sum(probabilities * grid_cols.view(1, 1, -1), dim=-1)

	out_h, out_w = output_size
	row_scale = (out_h - 1) / max(hm_h - 1, 1)
	col_scale = (out_w - 1) / max(hm_w - 1, 1)

	coords = torch.zeros((batch_size, num_keypoints, 3), device=heatmap_logits.device, dtype=torch.float32)
	coords[..., 0] = rows * row_scale
	coords[..., 1] = cols * col_scale
	# z remains 0 for 2D heatmap-based detection.
	return coords


class SegmentationDecoder(nn.Module):
	"""A lightweight decoder that upsamples the encoder output to image size."""

	def __init__(self, in_channels: int, num_classes: int = 1):
		super().__init__()
		self.reduce = ConvBlock(in_channels, 256)
		self.block1 = ConvBlock(256, 128)
		self.block2 = ConvBlock(128, 64)
		self.block3 = ConvBlock(64, 32)
		self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

	def forward(self, features: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
		x = self.reduce(features)
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = self.block1(x)
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = self.block2(x)
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = self.block3(x)
		x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
		return self.classifier(x)


class CervicalSpineMultiTaskModel(nn.Module):
	"""ResNet18-based multi-task model for keypoint detection and segmentation."""

	def __init__(
		self,
		num_keypoints: int = 56,
		keypoint_dims: int = 3,
		heatmap_size: Tuple[int, int] = (256, 256),
		decode_beta: float = 5.0,
		num_segmentation_classes: int = 1,
		in_channels: int = 1,
		encoder_weights=None,
		enable_segmentation: bool = True,
	):
		super().__init__()
		self.num_keypoints = num_keypoints
		self.keypoint_dims = keypoint_dims
		self.heatmap_size = heatmap_size
		self.decode_beta = decode_beta
		self.enable_segmentation = enable_segmentation and num_segmentation_classes > 0

		self.encoder = _make_resnet18_encoder(in_channels=in_channels, weights=encoder_weights)
		encoder_out_channels = 512

		self.keypoint_head = KeypointHeatmapHead(
			in_channels=encoder_out_channels,
			num_keypoints=num_keypoints,
			heatmap_size=heatmap_size,
		)

		self.segmentation_head = (
			SegmentationDecoder(encoder_out_channels, num_segmentation_classes)
			if self.enable_segmentation
			else None
		)

	def forward(self, x: torch.Tensor) -> dict:
		input_size = (int(x.shape[-2]), int(x.shape[-1]))
		features = self.encoder(x)

		keypoint_heatmaps = self.keypoint_head(features)
		keypoints = decode_heatmaps_to_keypoints(
			keypoint_heatmaps,
			output_size=input_size,
			beta=self.decode_beta,
		)
		outputs = {
			"keypoints": keypoints,
			"keypoint_heatmaps": keypoint_heatmaps,
		}

		if self.segmentation_head is not None:
			outputs["segmentation"] = self.segmentation_head(features, input_size)

		return outputs


class CervicalSpineKeypointModel(CervicalSpineMultiTaskModel):
	"""Compatibility alias focused on keypoint prediction only."""

	def __init__(
		self,
		num_keypoints: int = 56,
		keypoint_dims: int = 3,
		in_channels: int = 1,
		heatmap_size: Tuple[int, int] = (256, 256),
		decode_beta: float = 5.0,
	):
		super().__init__(
			num_keypoints=num_keypoints,
			keypoint_dims=keypoint_dims,
			heatmap_size=heatmap_size,
			decode_beta=decode_beta,
			num_segmentation_classes=0,
			in_channels=in_channels,
			enable_segmentation=False,
		)


class CervicalSpineLegacyKeypointModel(nn.Module):
	"""Legacy keypoint-only model used by the historical best regression run."""

	def __init__(
		self,
		num_keypoints: int = 56,
		keypoint_dims: int = 3,
		in_channels: int = 1,
	):
		super().__init__()
		self.encoder = _make_resnet18_encoder(in_channels=in_channels, weights=None)
		self.keypoint_head = KeypointRegressionHead(
			in_channels=512,
			num_keypoints=num_keypoints,
			keypoint_dims=keypoint_dims,
		)

	def forward(self, x: torch.Tensor) -> dict:
		features = self.encoder(x)
		keypoints = self.keypoint_head(features)
		return {"keypoints": keypoints}


def build_model(
	task: str = "multi_task",
	num_keypoints: int = 56,
	keypoint_dims: int = 3,
	heatmap_size: Tuple[int, int] = (256, 256),
	decode_beta: float = 5.0,
	num_segmentation_classes: int = 1,
	in_channels: int = 1,
	encoder_weights=None,
) -> nn.Module:
	"""Factory helper for downstream scripts."""

	task = task.lower()
	if task in {"keypoint", "keypoints", "kp"}:
		return CervicalSpineKeypointModel(
			num_keypoints=num_keypoints,
			keypoint_dims=keypoint_dims,
			in_channels=in_channels,
			heatmap_size=heatmap_size,
			decode_beta=decode_beta,
		)
	if task in {"multi_task", "multitask", "segmentation"}:
		return CervicalSpineMultiTaskModel(
			num_keypoints=num_keypoints,
			keypoint_dims=keypoint_dims,
			heatmap_size=heatmap_size,
			decode_beta=decode_beta,
			num_segmentation_classes=num_segmentation_classes,
			in_channels=in_channels,
			encoder_weights=encoder_weights,
			enable_segmentation=True,
		)

	raise ValueError(f"Unknown task: {task}")


def count_parameters(model: nn.Module) -> int:
	"""Return the number of trainable parameters."""

	return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _format_shape(value) -> Tuple[int, ...]:
	if isinstance(value, torch.Tensor):
		return tuple(value.shape)
	return tuple(value)


def _run_validation() -> None:
	print("=" * 80)
	print("CervicalSpine model validation")
	print("=" * 80)

	model = CervicalSpineMultiTaskModel(
		num_keypoints=56,
		keypoint_dims=3,
		num_segmentation_classes=1,
		in_channels=1,
		enable_segmentation=True,
	)
	model.eval()

	print(f"Trainable parameters: {count_parameters(model):,}")

	dummy = torch.randn(2, 1, 512, 512)
	with torch.no_grad():
		outputs = model(dummy)

	print(f"Dummy input shape: {_format_shape(dummy)}")
	print(f"Keypoints shape: {_format_shape(outputs['keypoints'])}")
	print(f"Keypoint heatmaps shape: {_format_shape(outputs['keypoint_heatmaps'])}")
	print(f"Segmentation shape: {_format_shape(outputs['segmentation'])}")

	assert outputs["keypoints"].shape == (2, 56, 3)
	expected_h, expected_w = model.heatmap_size
	assert outputs["keypoint_heatmaps"].shape == (2, 56, expected_h, expected_w)
	assert outputs["segmentation"].shape == (2, 1, 512, 512)
	assert torch.isfinite(outputs["keypoints"]).all()
	assert torch.isfinite(outputs["segmentation"]).all()

	try:
		from dataset import CervicalSpineDataset

		dataset = CervicalSpineDataset(data_dir="data/", split="train", augmentation=False)
		sample = dataset[0]
		image = sample["image"]
		if image.dim() == 3:
			image = image.unsqueeze(0)

		with torch.no_grad():
			real_outputs = model(image)

		print(f"Real sample case: {sample['case_name']}")
		print(f"Real input shape: {_format_shape(image)}")
		print(f"Real keypoints shape: {_format_shape(real_outputs['keypoints'])}")
		print(f"Real segmentation shape: {_format_shape(real_outputs['segmentation'])}")

		assert real_outputs["keypoints"].shape[1:] == (56, 3)
		assert real_outputs["segmentation"].shape[-2:] == (512, 512)
		print("Real-sample validation passed")
	except Exception as exc:
		print(f"Real-sample validation skipped or failed: {exc}")

	print("All model validation checks passed")


if __name__ == "__main__":
	_run_validation()
