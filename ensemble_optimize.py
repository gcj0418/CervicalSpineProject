#!/usr/bin/env python3
"""
Optimize ensemble weights via grid search and pairwise combinations.
Imports core functions from ensemble_spine.py.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import csv
from itertools import combinations

from ensemble_spine import (
    load_resolution_map, get_sample_resolution,
    load_hrnet_preds, load_vld_preds, load_dcelr_preds,
    point_error_mm, align_landmarks_hungarian, compute_metrics,
    normalize_key, align_samples, fuse_predictions
)


def evaluate_ensemble(method_preds, method_names, gt_dict, resolution_map, weights, mode, default_res=1.0):
    """Evaluate a single ensemble configuration. Returns metrics dict or None."""
    ref_idx = method_names.index("VLD") if "VLD" in method_names else 0
    all_errors = []
    for sample in sorted(gt_dict.keys()):
        if not all(sample in mp for mp in method_preds):
            continue
        ref_pts = method_preds[ref_idx][sample]
        aligned_preds = []
        for i, mp in enumerate(method_preds):
            pts = mp[sample]
            if i == ref_idx:
                aligned_preds.append(pts)
            else:
                aligned_preds.append(align_landmarks_hungarian(pts, ref_pts))
        fused = fuse_predictions(aligned_preds, weights=weights if mode == "weighted" else None, mode=mode)
        gt = gt_dict[sample]
        res = get_sample_resolution(resolution_map, sample, default_res)
        errors = point_error_mm(fused, gt, res)
        all_errors.extend(errors.tolist())
    if not all_errors:
        return None
    mean_err, accs = compute_metrics(all_errors)
    return {
        "mode": mode,
        "mean_error": mean_err,
        "acc@2": accs[2.0],
        "acc@2.5": accs[2.5],
        "acc@3": accs[3.0],
        "acc@4": accs[4.0],
    }


def run_pairwise_experiments(all_preds, all_names, gt_dict, resolution_map, default_res=1.0):
    """Run all C(n,2) pairwise combinations with mean and median fusion."""
    results = []
    n = len(all_names)
    for i, j in combinations(range(n), 2):
        names = [all_names[i], all_names[j]]
        preds = [all_preds[i], all_preds[j]]
        preds_aligned, common_samples = align_samples(preds, names)
        gt_common = {k: v for k, v in gt_dict.items() if k in common_samples}
        for mode in ["mean", "median"]:
            r = evaluate_ensemble(preds_aligned, names, gt_common, resolution_map, None, mode, default_res)
            if r:
                r["methods"] = "+".join(names)
                results.append(r)
                print(f"  [{r['methods']:20s} {mode:6s}] mean={r['mean_error']:.4f}  acc@2={r['acc@2']:.4f}")
    return results


def run_grid_search(all_preds, all_names, gt_dict, resolution_map, step=0.1, default_res=1.0):
    """Grid search over weights for all n methods."""
    results = []
    n = len(all_names)
    preds_aligned, common_samples = align_samples(all_preds, all_names)
    gt_common = {k: v for k, v in gt_dict.items() if k in common_samples}
    
    # Generate weight combinations (sum to 1, each >= step)
    def gen_weights(k, remaining, current):
        if k == 1:
            if remaining >= step - 1e-9 and remaining <= 1.0 + 1e-9:
                yield current + [remaining]
            return
        for w in np.arange(step, remaining - step + 1e-9, step):
            yield from gen_weights(k - 1, remaining - w, current + [round(w, 3)])
    
    best_mean = None
    best_acc2 = None
    total = 0
    for weights in gen_weights(n, 1.0, []):
        total += 1
        r = evaluate_ensemble(preds_aligned, all_names, gt_common, resolution_map, weights, "weighted", default_res)
        if r:
            r["weights"] = dict(zip(all_names, weights))
            results.append(r)
            if best_mean is None or r["mean_error"] < best_mean["mean_error"]:
                best_mean = r
            if best_acc2 is None or r["acc@2"] > best_acc2["acc@2"]:
                best_acc2 = r
    print(f"  Grid search: {total} combinations evaluated")
    return results, best_mean, best_acc2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["renji", "ruijin"], required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset == "renji":
        hrnet_path = "HRNet-Facial-Landmark-Detection/outputs/inference_renji_spine_renji_hrnet_w18_pretrained_e60/predictions/predictions_dict.pth"
        vld_path = "Vertebra-Landmark-Detection/outputs/inference_renji_e60_tta/predictions/predictions.pth"
        dcelr_dir = "D-CeLR/outputs/renji_ab_fullimg_e80_improved/eval_best/predictions"
        res_csv = "D-CeLR/data/renji_npy_direct/test_resolution.csv"
    else:
        hrnet_path = "HRNet-Facial-Landmark-Detection/outputs/inference_ruijin_spine_ruijin_hrnet_w18_pretrained_e60/predictions/predictions_dict.pth"
        vld_path = "Vertebra-Landmark-Detection/outputs/inference_ruijin_e60/predictions/predictions.pth"
        dcelr_dir = "D-CeLR/outputs/ruijin_ab_fullimg_e80_renji_init/eval_best/predictions"
        res_csv = "D-CeLR/data/ruijin_npy_direct/test_resolution.csv"

    print(f"=== Optimizing ensemble for {args.dataset.upper()} ===")
    resolution_map = load_resolution_map(res_csv)

    # Load all methods
    hrnet = load_hrnet_preds(hrnet_path)
    if not isinstance(hrnet, dict):
        raise ValueError("HRNet predictions must be dict format")
    vld = load_vld_preds(vld_path)
    dcelr = load_dcelr_preds(dcelr_dir)

    all_preds = [hrnet, vld, dcelr]
    all_names = ["HRNet", "VLD", "D-CeLR"]

    # Load GT from D-CeLR dir
    gt_dict = {}
    for p in sorted(Path(dcelr_dir).glob("*_gt.npy")):
        name = normalize_key(p.name.replace("_gt.npy", ""))
        gt_dict[name] = np.load(p)
    print(f"Loaded GT: {len(gt_dict)} samples")

    out_dir = Path(args.output_dir) / f"ensemble_optimize_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pairwise experiments
    print("\n--- Pairwise Combinations ---")
    pairwise_results = run_pairwise_experiments(all_preds, all_names, gt_dict, resolution_map)

    # 2. Grid search on all 3 methods
    print("\n--- Grid Search (3 methods) ---")
    grid_results, best_mean, best_acc2 = run_grid_search(all_preds, all_names, gt_dict, resolution_map, step=args.step)
    if best_mean:
        print(f"  Best mean_error: {best_mean['mean_error']:.4f} @ weights={best_mean['weights']}")
    if best_acc2:
        print(f"  Best acc@2:      {best_acc2['acc@2']:.4f} @ weights={best_acc2['weights']}")

    # 3. Also grid search on pairwise combos
    print("\n--- Grid Search (pairwise) ---")
    pairwise_grid = []
    for i, j in combinations(range(len(all_names)), 2):
        names = [all_names[i], all_names[j]]
        preds = [all_preds[i], all_preds[j]]
        print(f"  {names[0]} + {names[1]}:")
        _, bm, ba = run_grid_search(preds, names, gt_dict, resolution_map, step=args.step)
        if bm:
            bm["methods"] = "+".join(names)
            bm["optimized_for"] = "mean_error"
            pairwise_grid.append(bm)
            print(f"    Best mean: {bm['mean_error']:.4f} @ {bm['weights']}")
        if ba:
            ba["methods"] = "+".join(names)
            ba["optimized_for"] = "acc@2"
            pairwise_grid.append(ba)
            print(f"    Best acc2: {ba['acc@2']:.4f} @ {ba['weights']}")

    # Write results to markdown
    md_path = out_dir / "grid_search_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Ensemble Optimization Results – {args.dataset.upper()}\n\n")
        
        f.write("## Pairwise Combinations (mean / median)\n\n")
        f.write("| methods | mode | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in pairwise_results:
            f.write(f"| {r['methods']} | {r['mode']} | {r['mean_error']:.4f} | {r['acc@2']:.4f} | {r['acc@2.5']:.4f} | {r['acc@3']:.4f} | {r['acc@4']:.4f} |\n")
        
        f.write("\n## Grid Search Top-5 (3 methods, by mean_error)\n\n")
        top5_mean = sorted(grid_results, key=lambda x: x["mean_error"])[:5]
        f.write("| weights | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in top5_mean:
            wstr = ", ".join(f"{k}={v:.2f}" for k, v in r["weights"].items())
            f.write(f"| {wstr} | {r['mean_error']:.4f} | {r['acc@2']:.4f} | {r['acc@2.5']:.4f} | {r['acc@3']:.4f} | {r['acc@4']:.4f} |\n")
        
        f.write("\n## Grid Search Top-5 (3 methods, by acc@2)\n\n")
        top5_acc2 = sorted(grid_results, key=lambda x: -x["acc@2"])[:5]
        f.write("| weights | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in top5_acc2:
            wstr = ", ".join(f"{k}={v:.2f}" for k, v in r["weights"].items())
            f.write(f"| {wstr} | {r['mean_error']:.4f} | {r['acc@2']:.4f} | {r['acc@2.5']:.4f} | {r['acc@3']:.4f} | {r['acc@4']:.4f} |\n")
        
        f.write("\n## Pairwise Grid Search Best\n\n")
        f.write("| methods | optimize | weights | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in pairwise_grid:
            wstr = ", ".join(f"{k}={v:.2f}" for k, v in r["weights"].items())
            f.write(f"| {r['methods']} | {r['optimized_for']} | {wstr} | {r['mean_error']:.4f} | {r['acc@2']:.4f} | {r['acc@2.5']:.4f} | {r['acc@3']:.4f} | {r['acc@4']:.4f} |\n")

    print(f"\nSaved results to {md_path}")


if __name__ == "__main__":
    main()
