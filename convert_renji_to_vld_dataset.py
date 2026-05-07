#!/usr/bin/env python3
"""Convert RENJI-style NIfTI + Slicer JSON annotations to VLD dataset format.

Output format expected by Vertebra-Landmark-Detection:
    <out_root>/data/{train,val,test}/*.png
  <out_root>/labels/{train,val,test}/*.mat   (key: p2, shape: [N, 2])

Notes:
- The upstream VLD code assumes 68 landmarks (17 vertebrae x 4 corners).
- Your RENJI data appears to be mostly 56 landmarks.
- This script can still export 56-point samples, but upstream code needs adaptation
  before training/testing on 56 points.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import cv2
from PIL import Image
from scipy.io import savemat


@dataclass
class Sample:
    case_dir: Path
    image_path: Path
    json_path: Path
    case_name: str


def load_nifti_image(file_path: Path) -> Any:
    # nibabel's dynamic return typing is broader than what static checkers infer.
    return nib.load(str(file_path))  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RENJI dataset to VLD format")
    parser.add_argument("--src_root", type=Path, default=Path("data") / "RENJI", help="Source root (RENJI tree)")
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("Vertebra-Landmark-Detection") / "data_renji_vld",
        help="Output root for converted dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--expected_points",
        type=int,
        default=56,
        help="Required landmark count; mismatched samples are skipped",
    )
    parser.add_argument(
        "--keep_if_mismatch",
        action="store_true",
        help="Keep samples even if landmark count mismatches expected_points",
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


def natural_label_key(label: str) -> Tuple[str, int, str]:
    if not label:
        return ("", -1, "")
    label = label.strip()
    prefix = []
    digits = []
    for ch in label:
        if ch.isdigit():
            digits.append(ch)
        elif not digits:
            prefix.append(ch)
    if digits:
        return ("".join(prefix), int("".join(digits)), "")
    return (label, 10**9, label)


def is_under_renji_tree(path: Path) -> bool:
    return any(part.upper() == "RENJI" for part in path.parts)


def find_image_json_pairs(src_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    all_dirs = [src_root] + [p for p in src_root.rglob("*") if p.is_dir()]

    for d in sorted(all_dirs):
        if not is_under_renji_tree(d):
            continue
        nii_candidates = sorted(d.glob("*.nii.gz")) + sorted(d.glob("*.nii"))
        # Remove segmentation files, e.g. *.nii.gz.nii.seg.nrrd mis-detected by broad globs.
        nii_files = [p for p in nii_candidates if not p.name.endswith(".seg.nrrd")]
        if not nii_files:
            continue

        json_candidates = sorted(d.glob("*.json"))
        if not json_candidates:
            continue

        # Prefer F.mrk.json or first json in the directory.
        json_path = next((p for p in json_candidates if p.name.lower() == "f.mrk.json"), json_candidates[0])

        # Use first NIfTI in this folder as representative image.
        image_path = nii_files[0]
        case_name = image_path.name
        if case_name.endswith(".nii.gz"):
            case_name = case_name[:-7]
        else:
            case_name = image_path.stem

        samples.append(
            Sample(
                case_dir=d,
                image_path=image_path,
                json_path=json_path,
                case_name=case_name,
            )
        )

    return samples


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
        order = sorted(range(len(labels)), key=lambda i: natural_label_key(labels[i]))
        pts = pts[order]

    # Convert world (LPS) to voxel coordinates.
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

    # Voxel axis convention used in this workspace is (row, col, z).
    # Convert to image xy for VLD: x=col, y=row.
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
    """Rotate points from original image coords by 90 degrees clockwise.

    Original coords: (x, y) where x is column and y is row.
    For clockwise rotation:
      new_x = H - 1 - y
      new_y = x
    """
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
    """Rotate points by 180 degrees in image coordinates.

    new_x = W - 1 - x
    new_y = H - 1 - y
    """
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

    num_vertebra = num_points // 4
    for i in range(num_vertebra):
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
    keep_if_mismatch: bool,
    overwrite: bool,
) -> Tuple[bool, str]:
    img_nib = load_nifti_image(sample.image_path)
    image = img_nib.get_fdata().astype(np.float32)
    image2d = extract_2d_slice(image)

    pts_xy = read_points_from_slicer_json(sample.json_path, img_nib)
    if pts_xy.shape[0] != expected_points and not keep_if_mismatch:
        return False, f"skip_points_mismatch:{pts_xy.shape[0]}"

    rotated_image2d = rotate_image_clockwise_90(image2d)
    rotated_pts_xy = rotate_points_clockwise_90(pts_xy, image2d.shape)
    # Per preview verification, landmark group needs an additional 180-degree rotation.
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

    return True, "ok"


def main() -> None:
    args = parse_args()
    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    preview_root = (args.preview_root.resolve() if args.preview_root is not None else out_root.parent / f"{out_root.name}_preview")

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    ensure_dirs(out_root)
    ensure_preview_dirs(preview_root)
    samples = find_image_json_pairs(src_root)
    if not samples:
        raise RuntimeError(f"No valid image+json pairs found under {src_root}")

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
                keep_if_mismatch=bool(args.keep_if_mismatch),
                overwrite=bool(args.overwrite),
            )
            if ok:
                kept += 1
            else:
                skipped += 1
            details.append({
                "split": split,
                "case": samples[i].case_name,
                "status": status,
                "image": str(samples[i].image_path),
                "json": str(samples[i].json_path),
            })

    report = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "preview_root": str(preview_root),
        "total_pairs": len(samples),
        "kept": kept,
        "skipped": skipped,
        "expected_points": int(args.expected_points),
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
