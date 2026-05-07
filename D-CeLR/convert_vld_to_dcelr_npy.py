#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VLD jpg/mat dataset to D-CeLR npy pairs")
    parser.add_argument("--vld_root", type=Path, required=True, help="Path to VLD dataset root")
    parser.add_argument("--out_root", type=Path, required=True, help="Output root for D-CeLR npy data")
    parser.add_argument("--train_splits", type=str, default="train,val", help="Comma-separated split names for train list")
    parser.add_argument("--test_splits", type=str, default="test", help="Comma-separated split names for test list")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing npy files")
    return parser.parse_args()


def convert_split(vld_root: Path, out_root: Path, split: str, overwrite: bool):
    src_data = vld_root / "data" / split
    src_labels = vld_root / "labels" / split
    dst_split = out_root / split
    dst_split.mkdir(parents=True, exist_ok=True)

    converted = []
    if not src_data.exists() or not src_labels.exists():
        return converted

    for img_path in sorted(src_data.glob("*.jpg")):
        stem = img_path.stem
        mat_path = src_labels / f"{stem}.mat"
        if not mat_path.exists():
            continue

        m_path = dst_split / f"{stem}m.npy"
        l_path = dst_split / f"{stem}l.npy"

        if (not overwrite) and m_path.exists() and l_path.exists():
            converted.append(m_path)
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        mat = loadmat(str(mat_path))
        if "p2" not in mat:
            continue
        p2 = np.asarray(mat["p2"], dtype=np.float32)

        np.save(m_path, img)
        np.save(l_path, p2)
        converted.append(m_path)

    return converted


def write_list_file(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in items:
            f.write(str(p.as_posix()))
            f.write("\n")


def main():
    args = parse_args()
    train_splits = [s.strip() for s in args.train_splits.split(",") if s.strip()]
    test_splits = [s.strip() for s in args.test_splits.split(",") if s.strip()]

    all_train = []
    all_test = []

    for split in train_splits:
        all_train.extend(convert_split(args.vld_root, args.out_root, split, args.overwrite))

    for split in test_splits:
        all_test.extend(convert_split(args.vld_root, args.out_root, split, args.overwrite))

    write_list_file(args.out_root / "train.txt", sorted(all_train))
    write_list_file(args.out_root / "test.txt", sorted(all_test))

    print(f"train files: {len(all_train)}")
    print(f"test files: {len(all_test)}")
    print(f"output root: {args.out_root}")


if __name__ == "__main__":
    main()
