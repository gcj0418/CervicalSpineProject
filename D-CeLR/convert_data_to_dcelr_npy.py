#!/usr/bin/env python3
"""Convert source medical datasets directly into D-CeLR npy format.

Supported sources:
- Ruijin anatomy annotation folders
- RENJI folders with NIfTI + Slicer json annotations

Output format:
  <out_root>/{train,val,test}/*m.npy
  <out_root>/{train,val,test}/*l.npy
  <out_root>/train.txt
  <out_root>/test.txt
  <out_root>/<split>_resolution.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np


@dataclass
class Sample:
    case_name: str
    image_path: Path
    json_paths: List[Path]
    scene_path: Path | None


def natural_sort_key(text: str) -> Tuple[str, int, str]:
    match = re.match(r"^(.*?)(\d+)$", text.strip())
    if match:
        return (match.group(1), int(match.group(2)), "")
    return (text.strip(), 10**9, text.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert source dataset directly to D-CeLR npy format")
    parser.add_argument("--dataset", type=str, required=True, choices=["ruijin", "renji"])
    parser.add_argument("--src_root", type=Path, default=None, help="Source root for selected dataset")
    parser.add_argument("--out_root", type=Path, required=True, help="Output root in D-CeLR/data")
    parser.add_argument("--split_report", type=Path, default=None, help="Optional conversion_report.json to reuse splits")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_points", type=int, default=None)
    parser.add_argument("--expected_points_per_file", type=int, default=4)
    parser.add_argument("--include_val_in_train", action="store_true", help="Merge val split into train.txt")
    parser.add_argument("--keep_if_mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_rotate", action="store_true", help="Disable legacy rotation (90cw + 180)")
    return parser.parse_args()


def load_nifti_image(path: Path) -> Any:
    return nib.load(str(path))  # type: ignore[attr-defined]


def find_scene_path(case_dir: Path) -> Path | None:
    scene_candidates = sorted(case_dir.glob("*.mrml"))
    if not scene_candidates:
        return None
    return next((p for p in scene_candidates if "scene" in p.name.lower()), scene_candidates[0])


def load_markup_order_from_scene(scene_path: Path) -> List[str]:
    content = scene_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r'fileName="([^"]+\.mrk\.json)"')
    ordered: List[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(content):
        name = match.group(1)
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def collect_ruijin_markups(case_dir: Path, scene_path: Path | None) -> List[Path]:
    all_markups = {p.name: p for p in sorted(case_dir.glob("*.mrk.json"), key=lambda p: natural_sort_key(p.stem))}
    if scene_path is not None:
        ordered_names = load_markup_order_from_scene(scene_path)
        ordered = [all_markups[name] for name in ordered_names if name in all_markups]
        if ordered:
            return ordered
    return list(all_markups.values())


def is_under_renji_tree(path: Path) -> bool:
    return any(part.upper() == "RENJI" for part in path.parts)


def find_ruijin_samples(src_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    case_dirs = [p for p in src_root.iterdir() if p.is_dir()]
    for case_dir in sorted(case_dirs, key=lambda p: natural_sort_key(p.name)):
        nii_files = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))
        if not nii_files:
            continue
        scene_path = find_scene_path(case_dir)
        markups = collect_ruijin_markups(case_dir, scene_path)
        if not markups:
            continue
        image_path = nii_files[0]
        case_name = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
        samples.append(Sample(case_name=case_name, image_path=image_path, json_paths=markups, scene_path=scene_path))
    return samples


def find_renji_samples(src_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    all_dirs = [src_root] + [p for p in src_root.rglob("*") if p.is_dir()]
    for d in sorted(all_dirs):
        if not is_under_renji_tree(d):
            continue
        nii_candidates = sorted(d.glob("*.nii.gz")) + sorted(d.glob("*.nii"))
        nii_files = [p for p in nii_candidates if not p.name.endswith(".seg.nrrd")]
        if not nii_files:
            continue

        json_candidates = sorted(d.glob("*.json"))
        if not json_candidates:
            continue

        json_path = next((p for p in json_candidates if p.name.lower() == "f.mrk.json"), json_candidates[0])
        image_path = nii_files[0]
        case_name = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
        samples.append(Sample(case_name=case_name, image_path=image_path, json_paths=[json_path], scene_path=None))

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

    return np.stack([voxel[:, 1], voxel[:, 0]], axis=1).astype(np.float32)


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
    return np.ascontiguousarray(np.rot90(image, k=3))


def rotate_points_clockwise_90(points_xy: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    height, _ = image_shape
    rotated = points_xy.astype(np.float32).copy()
    x = rotated[:, 0].copy()
    y = rotated[:, 1].copy()
    rotated[:, 0] = (height - 1) - y
    rotated[:, 1] = x
    return rotated


def rotate_points_180(points_xy: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    height, width = image_shape
    rotated = points_xy.astype(np.float32).copy()
    rotated[:, 0] = (width - 1) - rotated[:, 0]
    rotated[:, 1] = (height - 1) - rotated[:, 1]
    return rotated


def load_split_from_report(report_path: Path) -> Dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split_map: Dict[str, str] = {}
    for item in report.get("details", []):
        case = str(item.get("case", "")).strip()
        split = str(item.get("split", "")).strip()
        status = str(item.get("status", "")).strip().lower()
        if case and split and status.startswith("ok"):
            split_map[case] = split
    return split_map


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def rel_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.src_root is None:
        if args.dataset == "ruijin":
            src_root = (script_dir.parent / "data" / "RUIJIN" / "anatomy_annotation").resolve()
            expected_points = 52
        else:
            src_root = (script_dir.parent / "data" / "RENJI").resolve()
            expected_points = 56
    else:
        src_root = args.src_root.resolve()
        expected_points = 52 if args.dataset == "ruijin" else 56

    if args.expected_points is not None:
        expected_points = int(args.expected_points)

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ruijin":
        samples = find_ruijin_samples(src_root)
    else:
        samples = find_renji_samples(src_root)

    if not samples:
        raise RuntimeError(f"No valid samples found under: {src_root}")

    split_by_case: Dict[str, str] = {}
    if args.split_report:
        split_by_case = load_split_from_report(args.split_report.resolve())

    if split_by_case:
        split_map: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for idx, sample in enumerate(samples):
            if sample.case_name in split_by_case and split_by_case[sample.case_name] in split_map:
                split_map[split_by_case[sample.case_name]].append(idx)
    else:
        train_idx, val_idx, test_idx = split_indices(len(samples), args.train_ratio, args.val_ratio, args.seed)
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    train_list: List[str] = []
    test_list: List[str] = []
    resolution_rows: Dict[str, List[List[str]]] = {"train": [], "val": [], "test": []}

    kept = 0
    skipped = 0
    details: List[Dict[str, Any]] = []

    for split, indices in split_map.items():
        for i in indices:
            sample = samples[i]
            img_nib = load_nifti_image(sample.image_path)
            image = img_nib.get_fdata().astype(np.float32)
            image2d = extract_2d_slice(image)

            all_points: List[np.ndarray] = []
            skipped_files: List[str] = []
            for json_path in sample.json_paths:
                pts_xy = read_points_from_slicer_json(json_path, img_nib)
                if args.dataset == "ruijin" and pts_xy.shape[0] != args.expected_points_per_file:
                    skipped_files.append(f"{json_path.name}:{pts_xy.shape[0]}")
                    if args.keep_if_mismatch and pts_xy.shape[0] > 0:
                        all_points.append(pts_xy)
                    continue
                all_points.append(pts_xy)

            if not all_points:
                skipped += 1
                details.append({"case": sample.case_name, "split": split, "status": "skip_no_points"})
                continue

            pts_xy = np.concatenate(all_points, axis=0)
            if pts_xy.shape[0] != expected_points and not args.keep_if_mismatch:
                skipped += 1
                details.append(
                    {
                        "case": sample.case_name,
                        "split": split,
                        "status": f"skip_points_mismatch:{pts_xy.shape[0]}",
                    }
                )
                continue

            if not args.no_rotate:
                image2d = rotate_image_clockwise_90(image2d)
                pts_xy = rotate_points_clockwise_90(pts_xy, extract_2d_slice(image).shape)
                pts_xy = rotate_points_180(pts_xy, image2d.shape)

            image_u8 = to_uint8(image2d)
            image_rgb = np.stack([image_u8, image_u8, image_u8], axis=-1)

            m_path = out_root / split / f"{sample.case_name}m.npy"
            l_path = out_root / split / f"{sample.case_name}l.npy"

            if (m_path.exists() or l_path.exists()) and not args.overwrite:
                skipped += 1
                details.append({"case": sample.case_name, "split": split, "status": "skip_exists"})
                continue

            np.save(m_path, image_rgb)
            np.save(l_path, pts_xy.astype(np.float32))

            rel_m = rel_posix(m_path, script_dir)
            if split == "test":
                test_list.append(rel_m)
            elif split == "train" or (split == "val" and args.include_val_in_train):
                train_list.append(rel_m)

            zooms = img_nib.header.get_zooms()
            resolution_rows[split].append([sample.case_name, f"{float(zooms[0]):.4f}", f"{float(zooms[1]):.4f}"])

            kept += 1
            status = "ok"
            if skipped_files:
                status = f"ok_skipped_files:{','.join(skipped_files)}"
            details.append(
                {
                    "case": sample.case_name,
                    "split": split,
                    "status": status,
                    "image": str(sample.image_path),
                    "json_paths": [str(p) for p in sample.json_paths],
                }
            )

    train_list = sorted(train_list)
    test_list = sorted(test_list)

    with (out_root / "train.txt").open("w", encoding="utf-8") as f:
        for item in train_list:
            f.write(item + "\n")

    with (out_root / "test.txt").open("w", encoding="utf-8") as f:
        for item in test_list:
            f.write(item + "\n")

    for split in ("train", "val", "test"):
        rows = sorted(resolution_rows[split], key=lambda r: natural_sort_key(r[0]))
        with (out_root / f"{split}_resolution.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    report = {
        "dataset": args.dataset,
        "src_root": str(src_root),
        "out_root": str(out_root),
        "expected_points": expected_points,
        "include_val_in_train": bool(args.include_val_in_train),
        "used_split_report": str(args.split_report.resolve()) if args.split_report else None,
        "kept": kept,
        "skipped": skipped,
        "train_count": len(train_list),
        "test_count": len(test_list),
        "split_sizes": {k: len(v) for k, v in split_map.items()},
        "details": details,
    }

    report_path = out_root / "conversion_report_direct.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Direct conversion finished")
    print("=" * 80)
    print(f"Dataset:       {args.dataset}")
    print(f"Source:        {src_root}")
    print(f"Output:        {out_root}")
    print(f"Kept:          {kept}")
    print(f"Skipped:       {skipped}")
    print(f"Train list:    {len(train_list)}")
    print(f"Test list:     {len(test_list)}")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    main()
