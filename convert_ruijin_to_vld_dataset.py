#!/usr/bin/env python3
"""Convert Ruijin anatomy_annotation cases to VLD dataset format.

Expected source layout:
  <src_root>/<case_id>/
    <case_id>.nii or <case_id>.nii.gz
    *.mrk.json
    *Scene.mrml (optional, used to preserve annotation order)

Output format expected by Vertebra-Landmark-Detection:
    <out_root>/data/{train,val,test}/*.png
  <out_root>/labels/{train,val,test}/*.mat   (key: p2, shape: [N, 2])

This script aggregates the per-vertebra Ruijin markups into a single point
array per case. In the current dataset layout, each case usually contains 13
or 14 markup files with 4 control points each, and some trailing files can be
empty or partially labeled. The converter skips those invalid files and keeps
the valid 4-point groups.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.io import savemat


@dataclass
class Sample:
    case_dir: Path
    image_path: Path
    scene_path: Path | None
    markup_paths: List[Path]
    case_name: str


def load_nifti_image(file_path: Path) -> Any:
    return nib.load(str(file_path))  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Ruijin dataset to VLD format")
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path("data") / "RUIJIN" / "anatomy_annotation",
        help="Source root that contains Ruijin case folders",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("Vertebra-Landmark-Detection") / "data_ruijin_vld",
        help="Output root for converted dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--expected_points",
        type=int,
        default=52,
        help="Required total landmark count; mismatched samples are skipped",
    )
    parser.add_argument(
        "--expected_points_per_file",
        type=int,
        default=4,
        help="Required landmark count per .mrk.json file",
    )
    parser.add_argument(
        "--keep_if_mismatch",
        action="store_true",
        help="Keep samples even if total landmark count mismatches expected_points",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--preview_root",
        type=Path,
        default=None,
        help="Parallel preview output root (defaults to <out_root>_preview)",
    )
    return parser.parse_args()


def natural_sort_key(text: str) -> Tuple[str, int, str]:
    match = re.match(r"^(.*?)(\d+)$", text.strip())
    if match:
        return (match.group(1), int(match.group(2)), "")
    return (text.strip(), 10**9, text.strip())


def case_sort_key(path: Path) -> Tuple[str, int, str]:
    return natural_sort_key(path.name)


def markup_sort_key(path: Path) -> Tuple[str, int, str]:
    name = path.name
    if name.lower().endswith(".mrk.json"):
        name = name[:-9]
    return natural_sort_key(name)


def find_case_dirs(src_root: Path) -> List[Path]:
    candidate_dirs = [src_root] + [p for p in src_root.rglob("*") if p.is_dir()]
    case_dirs: List[Path] = []

    for d in sorted(candidate_dirs, key=case_sort_key):
        nii_candidates = sorted(d.glob("*.nii.gz")) + sorted(d.glob("*.nii"))
        json_candidates = sorted(d.glob("*.mrk.json"), key=markup_sort_key)
        if nii_candidates and json_candidates:
            case_dirs.append(d)

    return case_dirs


def find_scene_path(case_dir: Path) -> Path | None:
    scene_candidates = sorted(case_dir.glob("*.mrml"))
    if not scene_candidates:
        return None
    preferred = next((p for p in scene_candidates if "scene" in p.name.lower()), scene_candidates[0])
    return preferred


def load_markup_order_from_scene(scene_path: Path) -> List[str]:
    content = scene_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r'fileName="([^"]+\.mrk\.json)"')
    ordered: List[str] = []
    seen: set[str] = set()

    for match in pattern.finditer(content):
        file_name = match.group(1)
        if file_name not in seen:
            ordered.append(file_name)
            seen.add(file_name)

    return ordered


def collect_markup_paths(case_dir: Path, scene_path: Path | None) -> List[Path]:
    all_markups = {p.name: p for p in sorted(case_dir.glob("*.mrk.json"), key=markup_sort_key)}

    if scene_path is not None:
        ordered_names = load_markup_order_from_scene(scene_path)
        ordered_paths = [all_markups[name] for name in ordered_names if name in all_markups]
        if ordered_paths:
            return ordered_paths

    return list(all_markups.values())


def find_image_path(case_dir: Path) -> Path:
    nii_candidates = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))
    if not nii_candidates:
        raise FileNotFoundError(f"No NIfTI file found in {case_dir}")
    return nii_candidates[0]


def read_points_from_slicer_json(json_path: Path, img_nib: Any) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    markups = data.get("markups", [])
    if not markups:
        return np.zeros((0, 2), dtype=np.float32)

    control_points = markups[0].get("controlPoints", [])
    if not control_points:
        return np.zeros((0, 2), dtype=np.float32)

    labels: List[str] = []
    world_xyz: List[List[float]] = []
    for cp in control_points:
        pos = cp.get("position", [0.0, 0.0, 0.0])
        lbl = cp.get("label", "")
        world_xyz.append([float(pos[0]), float(pos[1]), float(pos[2])])
        labels.append(str(lbl))

    pts = np.asarray(world_xyz, dtype=np.float32)

    if len(labels) == len(pts):
        order = sorted(range(len(labels)), key=lambda i: natural_sort_key(labels[i]))
        pts = pts[order]

    affine = np.asarray(img_nib.affine, dtype=np.float64).copy()
    flip_x = affine[0, 0] < 0
    flip_y = affine[1, 1] < 0
    flip_z = affine[2, 2] < 0
    affine[0, 0] = abs(affine[0, 0])
    affine[1, 1] = abs(affine[1, 1])
    affine[2, 2] = abs(affine[2, 2])

    inv_affine = np.linalg.inv(affine)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    voxel = (inv_affine @ homo.T).T[:, :3]

    shape = img_nib.shape
    if flip_x:
        voxel[:, 0] = shape[0] - 1 - voxel[:, 0]
    if flip_y:
        voxel[:, 1] = shape[1] - 1 - voxel[:, 1]
    if flip_z:
        voxel[:, 2] = shape[2] - 1 - voxel[:, 2]

    points_xy = np.stack([voxel[:, 1], voxel[:, 0]], axis=1).astype(np.float32)
    return points_xy


def extract_2d_slice(image_3d: np.ndarray) -> np.ndarray:
    if image_3d.ndim == 2:
        return image_3d.astype(np.float32)
    if image_3d.ndim == 3:
        if image_3d.shape[2] == 1:
            return image_3d[:, :, 0].astype(np.float32)
        z = image_3d.shape[2] // 2
        return image_3d[:, :, z].astype(np.float32)
    raise ValueError(f"Unsupported image shape: {image_3d.shape}")


def to_uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip(image, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return (img * 255.0).round().astype(np.uint8)


def rotate_image_clockwise_90(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image for rotation, got {image.shape}")
    return np.ascontiguousarray(np.rot90(image, k=3))


def rotate_points_clockwise_90(points_xy: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)

    height, width = image_shape
    rotated = points_xy.astype(np.float32).copy()
    original_x = rotated[:, 0].copy()
    original_y = rotated[:, 1].copy()
    rotated[:, 0] = (height - 1) - original_y
    rotated[:, 1] = original_x
    return rotated


def rotate_points_180(points_xy: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)

    height, width = image_shape
    rotated = points_xy.astype(np.float32).copy()
    rotated[:, 0] = (width - 1) - rotated[:, 0]
    rotated[:, 1] = (height - 1) - rotated[:, 1]
    return rotated


def draw_preview(image_u8: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    num_points = points_xy.shape[0]
    for idx, (x, y) in enumerate(points_xy):
        color = (0, 255, 255)
        cv2.circle(preview, (int(round(x)), int(round(y))), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            preview,
            str(idx + 1),
            (int(round(x)) + 4, int(round(y)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    num_groups = num_points // 4
    for i in range(num_groups):
        pts_4 = points_xy[4 * i : 4 * i + 4]
        if pts_4.shape[0] < 4:
            continue
        box_color = (0, 128 + (i * 7) % 127, 255 - (i * 11) % 127)
        for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
            p1 = tuple(int(round(v)) for v in pts_4[a])
            p2 = tuple(int(round(v)) for v in pts_4[b])
            cv2.line(preview, p1, p2, box_color, 2, lineType=cv2.LINE_AA)
    return preview


def ensure_dirs(out_root: Path) -> None:
    for sub in ("train", "val", "test"):
        (out_root / "data" / sub).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sub).mkdir(parents=True, exist_ok=True)


def ensure_preview_dirs(preview_root: Path) -> None:
    for sub in ("train", "val", "test"):
        (preview_root / sub).mkdir(parents=True, exist_ok=True)


def remove_legacy_image_outputs(img_out: Path) -> None:
    for suffix in (".jpg", ".jpeg", ".png"):
        legacy_path = img_out.with_suffix(suffix)
        if legacy_path != img_out and legacy_path.exists():
            legacy_path.unlink()


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    if n <= 0:
        return [], [], []
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def write_sample(
    sample: Sample,
    split: str,
    out_root: Path,
    preview_root: Path | None,
    expected_points: int,
    expected_points_per_file: int,
    keep_if_mismatch: bool,
    overwrite: bool,
) -> Tuple[bool, str]:
    img_nib = load_nifti_image(sample.image_path)
    image = img_nib.get_fdata().astype(np.float32)
    image2d = extract_2d_slice(image)

    if not sample.markup_paths:
        return False, "skip_no_markups"

    all_points: List[np.ndarray] = []
    skipped_files: List[str] = []
    for json_path in sample.markup_paths:
        pts_xy = read_points_from_slicer_json(json_path, img_nib)
        if pts_xy.shape[0] != expected_points_per_file:
            skipped_files.append(f"{json_path.name}:{pts_xy.shape[0]}")
            if keep_if_mismatch and pts_xy.shape[0] > 0:
                all_points.append(pts_xy)
            continue
        all_points.append(pts_xy)

    if not all_points:
        return False, "skip_no_points"

    pts_xy = np.concatenate(all_points, axis=0)
    if pts_xy.shape[0] != expected_points and not keep_if_mismatch:
        return False, f"skip_points_mismatch:{pts_xy.shape[0]}"

    rotated_image2d = rotate_image_clockwise_90(image2d)
    rotated_pts_xy = rotate_points_clockwise_90(pts_xy, image2d.shape)
    rotated_pts_xy = rotate_points_180(rotated_pts_xy, rotated_image2d.shape)

    img_name = f"{sample.case_name}.png"
    mat_name = f"{sample.case_name}.mat"
    img_out = out_root / "data" / split / img_name
    mat_out = out_root / "labels" / split / mat_name

    if (img_out.exists() or mat_out.exists()) and not overwrite:
        return False, "skip_exists"

    remove_legacy_image_outputs(img_out)
    image_u8 = to_uint8(rotated_image2d)
    Image.fromarray(image_u8).save(img_out, format="PNG")
    savemat(str(mat_out), {"p2": rotated_pts_xy})

    if preview_root is not None:
        preview_out = preview_root / split / f"{sample.case_name}.png"
        if not preview_out.exists() or overwrite:
            preview_image = draw_preview(image_u8, rotated_pts_xy)
            cv2.imwrite(str(preview_out), preview_image)

    if skipped_files:
        return True, f"ok_skipped_files:{','.join(skipped_files)}"
    return True, "ok"


def build_samples(src_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    case_dirs = find_case_dirs(src_root)

    for case_dir in case_dirs:
        image_path = find_image_path(case_dir)
        scene_path = find_scene_path(case_dir)
        markup_paths = collect_markup_paths(case_dir, scene_path)

        case_name = image_path.name
        if case_name.endswith(".nii.gz"):
            case_name = case_name[:-7]
        else:
            case_name = image_path.stem

        samples.append(
            Sample(
                case_dir=case_dir,
                image_path=image_path,
                scene_path=scene_path,
                markup_paths=markup_paths,
                case_name=case_name,
            )
        )

    return samples


def main() -> None:
    args = parse_args()
    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    preview_root = (
        args.preview_root.resolve() if args.preview_root is not None else out_root.parent / f"{out_root.name}_preview"
    )

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    ensure_dirs(out_root)
    ensure_preview_dirs(preview_root)

    samples = build_samples(src_root)
    if not samples:
        raise RuntimeError(f"No valid image+markup pairs found under {src_root}")

    train_idx, val_idx, test_idx = split_indices(len(samples), args.train_ratio, args.val_ratio, args.seed)
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}

    kept = 0
    skipped = 0
    details = []

    for split, ids in split_map.items():
        for i in ids:
            ok, status = write_sample(
                sample=samples[i],
                split=split,
                out_root=out_root,
                preview_root=preview_root,
                expected_points=int(args.expected_points),
                expected_points_per_file=int(args.expected_points_per_file),
                keep_if_mismatch=bool(args.keep_if_mismatch),
                overwrite=bool(args.overwrite),
            )
            if ok:
                kept += 1
            else:
                skipped += 1
            details.append(
                {
                    "split": split,
                    "case": samples[i].case_name,
                    "status": status,
                    "image": str(samples[i].image_path),
                    "scene": str(samples[i].scene_path) if samples[i].scene_path is not None else None,
                    "markups": [str(p) for p in samples[i].markup_paths],
                }
            )

    report = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "preview_root": str(preview_root),
        "total_pairs": len(samples),
        "kept": kept,
        "skipped": skipped,
        "expected_points": int(args.expected_points),
        "expected_points_per_file": int(args.expected_points_per_file),
        "keep_if_mismatch": bool(args.keep_if_mismatch),
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "details": details,
    }

    report_path = out_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Conversion finished")
    print("=" * 80)
    print(f"Source:        {src_root}")
    print(f"Output:        {out_root}")
    print(f"Total pairs:   {len(samples)}")
    print(f"Kept:          {kept}")
    print(f"Skipped:       {skipped}")
    print(f"Report:        {report_path}")
    print()
    print("Important: upstream VLD training code assumes 68 landmarks unless patched.")


if __name__ == "__main__":
    main()