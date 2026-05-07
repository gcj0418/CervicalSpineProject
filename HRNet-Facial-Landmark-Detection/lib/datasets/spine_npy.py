from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from ..utils.transforms import crop, generate_target, transform_pixel


@dataclass
class SpineSample:
    image_path: Path
    label_path: Path
    img_id: str


def _strip_npy_suffix(path: Path) -> str:
    name = path.name
    if name.lower().endswith('m.npy'):
        return name[:-5]
    return path.stem


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_list_file(path_value: str, root_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root_dir / path


def _resolve_data_path(root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return root / path


def _bbox_center_scale(points: np.ndarray) -> tuple[torch.Tensor, float]:
    xmin = float(np.min(points[:, 0]))
    ymin = float(np.min(points[:, 1]))
    xmax = float(np.max(points[:, 0]))
    ymax = float(np.max(points[:, 1]))

    center_x = (xmin + xmax) * 0.5
    center_y = (ymin + ymax) * 0.5
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    scale = max(width, height) / 200.0 * 1.25
    return torch.tensor([center_x, center_y], dtype=torch.float32), float(scale)


class SpineNpy(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.transform = transform
        self.data_root = Path(cfg.DATASET.ROOT)
        self.project_root = _project_root()
        list_file = _resolve_list_file(cfg.DATASET.TRAINSET if is_train else cfg.DATASET.TESTSET, self.project_root)
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.max_points = cfg.MODEL.NUM_JOINTS
        self.flip = bool(getattr(cfg.DATASET, 'FLIP', False))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.samples = self._load_samples(list_file)

    def _load_samples(self, list_file: Path) -> list[SpineSample]:
        samples: list[SpineSample] = []
        with list_file.open('r', encoding='utf-8') as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                image_path = _resolve_data_path(self.project_root / self.data_root, rel)
                if not image_path.name.lower().endswith('m.npy'):
                    raise ValueError(f'Expected image npy path ending with m.npy, got {image_path}')
                label_path = image_path.with_name(image_path.name[:-5] + 'l.npy')
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                if not label_path.exists():
                    raise FileNotFoundError(label_path)
                samples.append(SpineSample(image_path=image_path, label_path=label_path, img_id=_strip_npy_suffix(image_path)))
        return samples

    def __len__(self):
        return len(self.samples)

    def load_image(self, index):
        image = np.load(self.samples[index].image_path)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        return image

    def load_gt_pts(self, index):
        pts = np.load(self.samples[index].label_path).astype(np.float32)
        if pts.shape[0] > self.max_points:
            pts = pts[:self.max_points, :]
        return pts

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.load_image(index)
        pts = self.load_gt_pts(index)

        if pts.shape[0] != self.max_points:
            raise ValueError(f'{sample.img_id}: expected {self.max_points} points, got {pts.shape[0]}')

        center, scale = _bbox_center_scale(pts)
        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor, 1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
            if self.flip:
                # Horizontal flip is intentionally disabled for the spine datasets by default.
                pass

        out_image = crop(image, center, scale, self.input_size, rot=r)
        nparts = pts.shape[0]
        target = np.zeros((nparts, self.output_size[0], self.output_size[1]), dtype=np.float32)
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, center, scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i] - 1, self.sigma, label_type=self.label_type)

        out_image = out_image.astype(np.float32)
        out_image = (out_image / 255.0 - self.mean) / self.std
        out_image = out_image.transpose([2, 0, 1])

        target = torch.tensor(target)
        tpts = torch.tensor(tpts)
        meta = {
            'index': index,
            'center': center,
            'scale': scale,
            'pts': torch.tensor(pts),
            'tpts': tpts,
            'img_id': sample.img_id,
            'image_path': str(sample.image_path),
        }

        return out_image, target, meta
