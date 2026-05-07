#!/usr/bin/env python3
"""
Generate publication-quality figures for spine landmark detection comparison.
Figures: error distribution, CDF, Bland-Altman, bar charts, per-sample ranking.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Use ensemble_spine utilities for consistent loading
sys.path.insert(0, os.getcwd())
from ensemble_spine import (
    load_resolution_map,
    load_vld_preds,
    load_hrnet_preds,
    load_dcelr_preds,
    normalize_key,
    align_samples,
    point_error_mm,
)

# Aesthetic settings
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

# Unified color palette (same as visualizations)
METHOD_COLORS = {
    "VLD": "#E74C3C",       # red
    "D-CeLR": "#3498DB",    # blue
    "HRNet": "#2ECC71",     # green
    "Ensemble": "#9B59B6",  # purple
}


def load_dataset_predictions(dataset_name):
    """Load all method predictions and GT for a dataset."""
    if dataset_name == "RENJI":
        vld_path = "Vertebra-Landmark-Detection/outputs/inference_renji_e60_tta/predictions/predictions.pth"
        hrnet_path = "HRNet-Facial-Landmark-Detection/outputs/inference_renji_spine_renji_hrnet_w18_pretrained_e60/predictions/predictions_dict.pth"
        dcelr_dir = "D-CeLR/outputs/renji_ab_fullimg_e80_improved/eval_best/predictions"
        res_csv = "D-CeLR/data/renji_npy_direct/test_resolution.csv"
    else:
        vld_path = "Vertebra-Landmark-Detection/outputs/inference_ruijin_e60/predictions/predictions.pth"
        hrnet_path = "HRNet-Facial-Landmark-Detection/outputs/inference_ruijin_spine_ruijin_hrnet_w18_pretrained_e60/predictions/predictions_dict.pth"
        dcelr_dir = "D-CeLR/outputs/ruijin_ab_fullimg_e80_renji_init/eval_best/predictions"
        res_csv = "D-CeLR/data/ruijin_npy_direct/test_resolution.csv"

    vld = load_vld_preds(vld_path)
    hrnet = load_hrnet_preds(hrnet_path)
    dcelr = load_dcelr_preds(dcelr_dir)

    method_preds = [vld, hrnet, dcelr]
    method_names = ["VLD", "HRNet", "D-CeLR"]
    method_preds, common_samples = align_samples(method_preds, method_names)

    # Load GT from D-CeLR
    gt_dict = {}
    for p in sorted(Path(dcelr_dir).glob("*_gt.npy")):
        name = normalize_key(p.name.replace("_gt.npy", ""))
        gt_dict[name] = np.load(p)

    resolution_map = load_resolution_map(res_csv)
    return method_preds, method_names, gt_dict, resolution_map, common_samples


def compute_all_errors(method_preds, method_names, gt_dict, resolution_map, common_samples):
    """Compute per-landmark errors (mm) for all methods."""
    errors = {name: [] for name in method_names}
    per_sample = {name: {} for name in method_names}

    for sample in common_samples:
        gt = gt_dict[sample]
        res = resolution_map.get(sample, 1.0)
        if isinstance(res, np.ndarray):
            res = res.tolist()

        for mp, name in zip(method_preds, method_names):
            pts = mp[sample]
            sample_errors = point_error_mm(pts, gt, res)
            errors[name].extend(sample_errors.tolist())
            per_sample[name][sample] = float(np.mean(sample_errors))

    return errors, per_sample


def fig_error_distribution(errors, dataset, out_dir):
    """Figure 1: KDE of error distributions."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ["VLD", "D-CeLR", "HRNet"]:
        data = np.array(errors[name])
        sns.kdeplot(data, ax=ax, label=name, color=METHOD_COLORS[name], fill=True, alpha=0.15, linewidth=2)
    ax.axvline(2.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="2 mm threshold")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="2.5 mm threshold")
    ax.set_xlabel("Localization Error (mm)")
    ax.set_ylabel("Density")
    ax.set_title(f"Error Distribution – {dataset}")
    ax.set_xlim(0, 15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_error_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {dataset}_error_distribution.png")


def fig_cdf(errors, dataset, out_dir):
    """Figure 2: Cumulative distribution function with SDR annotations."""
    fig, ax = plt.subplots(figsize=(6, 4))
    thresholds = np.linspace(0, 15, 500)

    for name in ["VLD", "D-CeLR", "HRNet"]:
        data = np.array(errors[name])
        cdf = [(data <= t).mean() for t in thresholds]
        ax.plot(thresholds, cdf, label=name, color=METHOD_COLORS[name], linewidth=2)

    for thr, ls in [(2.0, "--"), (2.5, ":"), (3.0, "-.")]:
        ax.axvline(thr, color="gray", linestyle=ls, linewidth=1, alpha=0.7)

    ax.set_xlabel("Localization Error (mm)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"CDF of Localization Error – {dataset}")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_cdf.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {dataset}_cdf.png")


def fig_bland_altman(errors_dict, dataset, out_dir):
    """Figure 3: Bland-Altman plots between method pairs."""
    pairs = [("VLD", "HRNet"), ("VLD", "D-CeLR"), ("HRNet", "D-CeLR")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (a, b) in zip(axes, pairs):
        errs_a = np.array(errors_dict[a])
        errs_b = np.array(errors_dict[b])
        # Use per-landmark errors; Bland-Altman on paired errors
        mean = (errs_a + errs_b) / 2
        diff = errs_a - errs_b
        md = np.mean(diff)
        sd = np.std(diff)

        ax.scatter(mean, diff, alpha=0.3, s=10, color=METHOD_COLORS["Ensemble"])
        ax.axhline(md, color="red", linestyle="--", linewidth=1.5)
        ax.axhline(md + 1.96 * sd, color="gray", linestyle=":", linewidth=1)
        ax.axhline(md - 1.96 * sd, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Mean Error (mm)")
        ax.set_ylabel(f"Difference ({a} − {b}) (mm)")
        ax.set_title(f"{a} vs {b}")
        ax.text(
            0.02, 0.98,
            f"Mean diff = {md:.3f}\nSD = {sd:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Bland-Altman Analysis – {dataset}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_bland_altman.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {dataset}_bland_altman.png")


def fig_bar_comparison(errors, dataset, out_dir):
    """Figure 4: Bar chart of mean error and SDR metrics."""
    methods = ["VLD", "D-CeLR", "HRNet"]
    metrics = {"mean": [], "acc2": [], "acc2_5": [], "acc3": [], "acc4": []}
    for m in methods:
        arr = np.array(errors[m])
        metrics["mean"].append(arr.mean())
        metrics["acc2"].append((arr <= 2.0).mean() * 100)
        metrics["acc2_5"].append((arr <= 2.5).mean() * 100)
        metrics["acc3"].append((arr <= 3.0).mean() * 100)
        metrics["acc4"].append((arr <= 4.0).mean() * 100)

    x = np.arange(len(methods))
    width = 0.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Mean error
    bars1 = ax1.bar(x, metrics["mean"], width=0.5, color=[METHOD_COLORS[m] for m in methods], edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Mean Error (mm)")
    ax1.set_title("Mean Localization Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # SDR
    ax2.bar(x - 1.5 * width, metrics["acc2"], width, label="≤2 mm", color="#E74C3C", edgecolor="black", alpha=0.85)
    ax2.bar(x - 0.5 * width, metrics["acc2_5"], width, label="≤2.5 mm", color="#E67E22", edgecolor="black", alpha=0.85)
    ax2.bar(x + 0.5 * width, metrics["acc3"], width, label="≤3 mm", color="#F1C40F", edgecolor="black", alpha=0.85)
    ax2.bar(x + 1.5 * width, metrics["acc4"], width, label="≤4 mm", color="#2ECC71", edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Successful Detection Rate (%)")
    ax2.set_title("SDR at Different Thresholds")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Quantitative Comparison – {dataset}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_bar_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {dataset}_bar_comparison.png")


def fig_per_sample_ranking(per_sample, errors, dataset, out_dir, topk=15):
    """Figure 5: Per-sample mean error ranking (worst samples)."""
    # Use VLD as reference for sample ordering
    ref_name = "VLD"
    samples_sorted = sorted(per_sample[ref_name].items(), key=lambda kv: kv[1], reverse=True)
    worst = samples_sorted[:topk]
    sample_names = [s[0] for s in worst]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sample_names))
    width = 0.25

    for i, name in enumerate(["VLD", "D-CeLR", "HRNet"]):
        vals = [per_sample[name].get(s, 0) for s in sample_names]
        ax.bar(x + i * width, vals, width, label=name, color=METHOD_COLORS[name], edgecolor="black", alpha=0.85)

    ax.set_ylabel("Mean Error (mm)")
    ax.set_title(f"Top-{topk} Worst Samples by {ref_name} Error – {dataset}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sample_names, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_worst_samples.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {dataset}_worst_samples.png")


def main():
    out_dir = Path("outputs/paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ["RENJI", "RUIJIN"]:
        print(f"\n=== Processing {dataset} ===")
        method_preds, method_names, gt_dict, resolution_map, common_samples = load_dataset_predictions(dataset)
        errors, per_sample = compute_all_errors(method_preds, method_names, gt_dict, resolution_map, common_samples)

        fig_error_distribution(errors, dataset, out_dir)
        fig_cdf(errors, dataset, out_dir)
        fig_bland_altman(errors, dataset, out_dir)
        fig_bar_comparison(errors, dataset, out_dir)
        fig_per_sample_ranking(per_sample, errors, dataset, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
