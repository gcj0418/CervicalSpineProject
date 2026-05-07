#!/usr/bin/env python3
"""Separate skipped Renji cases from the original data tree.

This reads the VLD conversion report, finds all skipped Renji samples, and
copies their original case directories into a separate output tree while
preserving the source-relative folder structure.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate skipped Renji cases into an isolated tree")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("Vertebra-Landmark-Detection") / "data_renji_vld" / "conversion_report.json",
        help="Renji VLD conversion report containing skipped sample details",
    )
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path("data") / "RENJI",
        help="Original Renji data root",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("data") / "RENJI_skipped_vld",
        help="Destination root for the separated skipped cases",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the destination root before copying",
    )
    return parser.parse_args()


def load_skipped_entries(report_path: Path) -> List[Dict[str, object]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details = report.get("details", [])
    return [entry for entry in details if entry.get("status") != "ok"]


def collect_unique_case_dirs(skipped_entries: List[Dict[str, object]], src_root: Path) -> List[Tuple[Path, Dict[str, object]]]:
    unique: List[Tuple[Path, Dict[str, object]]] = []
    seen: set[Path] = set()

    for entry in skipped_entries:
        image_path = Path(str(entry["image"]))
        case_dir = image_path.parent
        try:
            rel_case_dir = case_dir.relative_to(src_root)
        except ValueError as exc:
            raise RuntimeError(f"Skipped case is outside source root: {case_dir}") from exc

        if rel_case_dir in seen:
            continue

        if not case_dir.exists():
            raise FileNotFoundError(f"Source case directory not found: {case_dir}")

        seen.add(rel_case_dir)
        unique.append((case_dir, entry))

    return unique


def copy_case_dirs(case_dirs: List[Tuple[Path, Dict[str, object]]], src_root: Path, out_root: Path) -> List[Dict[str, object]]:
    copied: List[Dict[str, object]] = []

    for case_dir, entry in case_dirs:
        rel_case_dir = case_dir.relative_to(src_root)
        dst_case_dir = out_root / rel_case_dir
        if dst_case_dir.exists():
            shutil.rmtree(dst_case_dir)
        shutil.copytree(case_dir, dst_case_dir)

        copied.append(
            {
                "case": entry.get("case"),
                "status": entry.get("status"),
                "split": entry.get("split"),
                "source_case_dir": str(case_dir),
                "destination_case_dir": str(dst_case_dir),
                "image": entry.get("image"),
                "json": entry.get("json"),
            }
        )

    return copied


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()

    if not report_path.exists():
        raise FileNotFoundError(f"Conversion report not found: {report_path}")
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    skipped_entries = load_skipped_entries(report_path)
    unique_case_dirs = collect_unique_case_dirs(skipped_entries, src_root)
    copied = copy_case_dirs(unique_case_dirs, src_root, out_root)

    report = {
        "report": str(report_path),
        "src_root": str(src_root),
        "out_root": str(out_root),
        "skipped_entries": len(skipped_entries),
        "unique_case_dirs": len(unique_case_dirs),
        "copied": copied,
    }

    manifest_path = out_root / "separated_skipped_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Skipped-case separation finished")
    print("=" * 80)
    print(f"Report:            {report_path}")
    print(f"Source root:       {src_root}")
    print(f"Output root:       {out_root}")
    print(f"Skipped entries:   {len(skipped_entries)}")
    print(f"Unique case dirs:  {len(unique_case_dirs)}")
    print(f"Manifest:          {manifest_path}")


if __name__ == "__main__":
    main()