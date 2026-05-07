#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import nibabel as nib


def strip_nifti_suffix(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".nii.gz"):
        return name[:-7]
    if lowered.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def build_index(source_root: Path):
    index = {}
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if not (lowered.endswith(".nii.gz") or lowered.endswith(".nii")):
            continue
        sample_name = strip_nifti_suffix(path.name)
        index.setdefault(sample_name, []).append(path)
    return index


def read_sample_names(list_file: Path):
    names = []
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = Path(line).name.replace("m.npy", "")
            names.append(sample)
    return names


def get_spacing_xy(nifti_path: Path):
    img = nib.load(str(nifti_path))
    zooms = img.header.get_zooms()
    if len(zooms) >= 2:
        spacing_x = float(abs(zooms[1]))
        spacing_y = float(abs(zooms[0]))
    elif len(zooms) == 1:
        spacing_x = spacing_y = float(abs(zooms[0]))
    else:
        spacing_x = spacing_y = 1.0
    return spacing_x, spacing_y


def parse_args():
    parser = argparse.ArgumentParser(description="Build spacing CSV from source NIfTI files")
    parser.add_argument("--source_root", type=Path, required=True, help="Root folder containing original NIfTI files")
    parser.add_argument("--list_file", type=Path, required=True, help="D-CeLR train/test txt file to map names from")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--strict", action="store_true", help="Fail if any sample cannot be matched")
    return parser.parse_args()


def main():
    args = parse_args()
    sample_names = read_sample_names(args.list_file)
    index = build_index(args.source_root)

    rows = []
    missing = []
    for sample_name in sample_names:
        candidates = index.get(sample_name, [])
        if not candidates:
            missing.append(sample_name)
            continue
        spacing_x, spacing_y = get_spacing_xy(candidates[0])
        rows.append((sample_name, spacing_x, spacing_y, str(candidates[0])))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "spacing_x", "spacing_y", "source_path"])
        for row in rows:
            writer.writerow(row)

    print(f"matched = {len(rows)}")
    print(f"missing = {len(missing)}")
    print(f"csv = {args.out_csv}")
    if missing:
        print("missing samples:")
        for name in missing[:20]:
            print(name)
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
