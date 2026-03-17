#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
Regenerate ALL KALAVU publication figures from existing JSON results.
Never re-runs experiments — reads JSON only.
At the end: git add figures/, commit, push.
"""

import os
import json
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
try:
    import figure_style
    figure_style.apply_style()
    COLORS = figure_style.COLORS
    clean_axes = figure_style.clean_axes
    label_bars = figure_style.label_bars
    cell_text_color = figure_style.cell_text_color
except ImportError:
    import matplotlib.pyplot as plt
    COLORS = {
        "base":       "#94a3b8",
        "code":       "#3b82f6",
        "science":    "#10b981",
        "fiction":    "#f59e0b",
        "weight_avg": "#8b5cf6",
        "moe":        "#ef4444",
        "monolithic": "#6b7280",
        "classifier": "#f97316",
        "multihead":  "#06b6d4",
        "wider":      "#84cc16",
        "freeze0":    "#3b82f6",
        "freeze4":    "#ef4444",
        "crossover":  "#10b981",
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    def clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="#000000", linewidth=0.5)

    def label_bars(ax, bars, fmt="{:.3f}", fontsize=8, offset_frac=0.003, colors=None):
        ylim = ax.get_ylim()
        offset = (ylim[1] - ylim[0]) * offset_frac
        for i, bar in enumerate(bars):
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + offset, fmt.format(val),
                    ha="center", va="bottom", fontsize=fontsize, color="#374151")

    def cell_text_color(bg_color_rgba):
        r, g, b = bg_color_rgba[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if luminance < 0.5 else "#1f2937"

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(BASE, "results")
FIG  = os.path.join(BASE, "figures")
FIG_P   = os.path.join(FIG, "pythia")
FIG_6B  = os.path.join(FIG, "pythia_6b")
os.makedirs(FIG_P,  exist_ok=True)
os.makedirs(FIG_6B, exist_ok=True)

def rpath(*parts):
    return os.path.join(RES, *parts)

def fpath(*parts):
    return os.path.join(FIG, *parts)

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# fig_fusion_comparison.png — 410M
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_fusion_comparison.png...")
try:
    d = load(rpath("pythia", "step5_final_summary.json"))
    seeds = ["42", "137", "2026"]

    base_vals  = [d["per_seed_fusion"][s]["eval_heldout"]["base"]["mixed"]      for s in seeds]
    best_vals  = [d["per_seed_fusion"][s]["eval_heldout"]["code_spec"]["mixed"] for s in seeds]
    # best individual is the minimum mixed loss among the 3 specialists
    best_ind   = []
    for s in seeds:
        eh = d["per_seed_fusion"][s]["eval_heldout"]
        best_ind.append(min(
            eh["code_spec"]["mixed"],
            eh["science_spec"]["mixed"],
            eh["fiction_spec"]["mixed"],
        ))
    wavg_vals  = [d["per_seed_fusion"][s]["eval_heldout"]["weight_avg"]["mixed"] for s in seeds]
    moe_vals   = [d["per_seed_fusion"][s]["eval_heldout"]["moe"]["mixed"]        for s in seeds]

    base_m  = float(np.mean(base_vals))
    best_m  = float(np.mean(best_ind))
    wavg_m  = float(np.mean(wavg_vals))
    moe_m   = float(np.mean(moe_vals))

    labels = ["Base", "Best Individual", "Weight Avg", "MoE"]
    vals   = [base_m, best_m, wavg_m, moe_m]
    colors = [COLORS["base"], COLORS["code"], COLORS["weight_avg"], COLORS["moe"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_title("Fusion Methods — Pythia-410M (3 seeds)")
    ax.set_ylabel("Held-Out Mixed Loss")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_fusion_comparison.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_fusion_comparison.png")
except Exception as e:
    print(f"WARNING: skipped fig_fusion_comparison: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_1b_fusion_comparison.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_1b_fusion_comparison.png...")
try:
    d = load(rpath("pythia", "pythia_1b", "main_result_summary.json"))
    seeds = ["42", "137", "2026"]

    base_m = d["base_held_out_losses"]["mixed"]
    best_ind = []
    wavg_vals = []
    moe_vals  = []
    for s in seeds:
        eh = d["per_seed_fusion"][s]["eval_heldout"]
        best_ind.append(min(
            eh["code_spec"]["mixed"],
            eh["science_spec"]["mixed"],
            eh["fiction_spec"]["mixed"],
        ))
        wavg_vals.append(eh["weight_avg"]["mixed"])
        moe_vals.append(eh["moe"]["mixed"])

    best_m = float(np.mean(best_ind))
    wavg_m = float(np.mean(wavg_vals))
    moe_m  = float(np.mean(moe_vals))

    labels = ["Base", "Best Individual", "Weight Avg", "MoE"]
    vals   = [base_m, best_m, wavg_m, moe_m]
    colors = [COLORS["base"], COLORS["code"], COLORS["weight_avg"], COLORS["moe"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_title("Fusion Methods — Pythia-1B (3 seeds)")
    ax.set_ylabel("Held-Out Mixed Loss")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_1b_fusion_comparison.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_1b_fusion_comparison.png")
except Exception as e:
    print(f"WARNING: skipped fig_1b_fusion_comparison: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: divergence heatmap
# ══════════════════════════════════════════════════════════════════════════════
def _draw_divergence_heatmap(ax, loss_matrix, title):
    row_labels = list(loss_matrix.keys())           # base, code, science, fiction
    col_labels = ["code", "science", "fiction"]

    data = np.array([[loss_matrix[r][c] for c in col_labels] for r in row_labels])

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([c.capitalize() for c in col_labels])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([r.capitalize() for r in row_labels])

    # white cell borders
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            val = data[y, x]
            rgba = im.cmap(im.norm(val))
            tc = cell_text_color(rgba)
            ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="medium")

    # draw grid lines to create white borders
    for y in np.arange(-0.5, data.shape[0], 1):
        ax.axhline(y, color="white", linewidth=1.5)
    for x in np.arange(-0.5, data.shape[1], 1):
        ax.axvline(x, color="white", linewidth=1.5)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Evaluation Domain")
    ax.set_ylabel("Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# fig_divergence_heatmap.png — 410M seed42
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_divergence_heatmap.png...")
try:
    d = load(rpath("pythia", "step3_divergence_check_seed42.json"))
    fig, ax = plt.subplots(figsize=(5, 4))
    _draw_divergence_heatmap(
        ax, d["loss_matrix"],
        "Specialist Divergence — Cross-Domain Losses (410M, seed=42)"
    )
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_divergence_heatmap.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_divergence_heatmap.png")
except Exception as e:
    print(f"WARNING: skipped fig_divergence_heatmap: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_1b_divergence_heatmap.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_1b_divergence_heatmap.png...")
try:
    d = load(rpath("pythia", "pythia_1b", "step3_divergence_check_seed42.json"))
    fig, ax = plt.subplots(figsize=(5, 4))
    _draw_divergence_heatmap(
        ax, d["loss_matrix"],
        "Specialist Divergence — Cross-Domain Losses (1B, seed=42)"
    )
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_1b_divergence_heatmap.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_1b_divergence_heatmap.png")
except Exception as e:
    print(f"WARNING: skipped fig_1b_divergence_heatmap: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: router distribution bar chart
# ══════════════════════════════════════════════════════════════════════════════
def _draw_router_distribution(ax, router_dist, title):
    domains  = ["code", "science", "fiction"]
    n_exp    = 3
    x        = np.arange(len(domains))
    width    = 0.25
    exp_colors = [COLORS["code"], COLORS["science"], COLORS["fiction"]]

    for ei in range(n_exp):
        vals = [router_dist[d][ei] for d in domains]
        bars = ax.bar(x + (ei - 1) * width, vals,
                      width=width * 0.9,
                      color=exp_colors[ei],
                      label=f"Expert {ei} ({domains[ei].capitalize()})",
                      edgecolor="white", linewidth=0.5)

    ax.axhline(1 / 3, color="#6b7280", linestyle="--", alpha=0.4,
               linewidth=1.2, label="Uniform (1/3)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_xlabel("Evaluation Domain")
    ax.set_ylabel("Mean Gate Weight")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    clean_axes(ax)


# ══════════════════════════════════════════════════════════════════════════════
# fig_router_distribution.png — 410M seed42
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_router_distribution.png...")
try:
    d = load(rpath("pythia", "step4_fusion_results_seed42.json"))
    router_dist = d.get("router_distribution") or d.get("gate_distribution")
    if router_dist is None:
        # try step5_final_summary per seed
        d2 = load(rpath("pythia", "step5_final_summary.json"))
        router_dist = d2["per_seed_fusion"]["42"]["router_distribution"]

    fig, ax = plt.subplots(figsize=(5, 4))
    _draw_router_distribution(ax, router_dist, "Router Gate Distribution (410M, seed=42)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_router_distribution.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_router_distribution.png")
except Exception as e:
    print(f"WARNING: skipped fig_router_distribution: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_1b_router_distribution.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_1b_router_distribution.png...")
try:
    d = load(rpath("pythia", "pythia_1b", "main_result_seed42.json"))
    router_dist = d.get("router_distribution") or d.get("gate_distribution")
    if router_dist is None:
        d2 = load(rpath("pythia", "pythia_1b", "main_result_summary.json"))
        router_dist = d2["per_seed_fusion"]["42"]["router_distribution"]

    fig, ax = plt.subplots(figsize=(5, 4))
    _draw_router_distribution(ax, router_dist, "Router Gate Distribution (1B, seed=42)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_1b_router_distribution.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_1b_router_distribution.png")
except Exception as e:
    print(f"WARNING: skipped fig_1b_router_distribution: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_ablation_freeze.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_ablation_freeze.png...")
try:
    d = load(rpath("pythia", "ablation_freeze_summary.json"))

    # Build x/y arrays from phase1_seed42_results
    phase1 = d["phase1_seed42_results"]
    xs = [r["freeze_layers"] for r in phase1]
    ys = [r["improvement_pct"] for r in phase1]

    # Error bars from multi_seed_results where available
    yerr = []
    for r in phase1:
        key = str(r["freeze_layers"])
        ms = d.get("multi_seed_results", {}).get(key)
        if ms and "std" in ms:
            yerr.append(ms["std"])
        else:
            yerr.append(0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(xs, ys, yerr=yerr,
                color=COLORS["moe"], marker="o", markersize=6,
                linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
                label="MoE improvement %")

    # Highlight freeze=4
    ax.axvline(x=4, color="#6b7280", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.annotate("freeze=4\n(default)", xy=(4, ys[xs.index(4)]),
                xytext=(5.5, ys[xs.index(4)] + 0.3),
                fontsize=8, color="#374151",
                arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1))

    ax.set_xlabel("Frozen Layers")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Freeze Depth Ablation (410M, seed=42)")
    ax.set_xticks(xs)
    clean_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_ablation_freeze.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_ablation_freeze.png")
except Exception as e:
    print(f"WARNING: skipped fig_ablation_freeze: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_ablation_router.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_ablation_router.png...")
try:
    d = load(rpath("pythia", "ablation_router_summary.json"))
    variants = d["variants"]

    labels = ["Uniform", "Simple Linear", "Two-Layer"]
    keys   = ["uniform", "simple_linear", "two_layer"]
    vals   = [variants[k]["improvement_pct"] for k in keys]
    bar_colors = [COLORS["base"], COLORS["moe"], COLORS["weight_avg"]]

    best_individual_pct = 0.0  # reference: 0% = same as best individual

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=bar_colors, width=0.55,
                  edgecolor="white", linewidth=0.8)

    # Best individual reference line at improvement=0 is not meaningful;
    # add a horizontal line at the simple_linear value as "strong baseline"
    ax.axhline(variants["simple_linear"]["improvement_pct"],
               color="#6b7280", linestyle="--", linewidth=1.2,
               alpha=0.5, label="Simple linear baseline")

    clean_axes(ax)
    label_bars(ax, bars, fmt="{:.2f}%")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Router Architecture Ablation (410M, seed=42)")
    ax.legend(fontsize=8)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_ablation_router.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_ablation_router.png")
except Exception as e:
    print(f"WARNING: skipped fig_ablation_router: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: benchmark grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def _draw_benchmarks(ax, model_variants, title):
    # Collect tasks that have valid accuracy for at least base and moe
    base_v = model_variants.get("base", {})
    moe_v  = model_variants.get("moe_fused", model_variants.get("moe", {}))
    mono_v = model_variants.get("monolithic", {})

    # Check if data has nested accuracy dicts (410m format) or flat % (1b format)
    sample_val = next(iter(base_v.values())) if base_v else None
    nested = isinstance(sample_val, dict)

    if nested:
        tasks = [t for t in base_v
                 if base_v[t].get("accuracy") is not None
                 and moe_v.get(t, {}).get("accuracy") is not None]
        base_scores = [base_v[t]["accuracy"] * 100 for t in tasks]
        moe_scores  = [moe_v[t]["accuracy"] * 100  for t in tasks]
        mono_scores = [mono_v.get(t, {}).get("accuracy", None) for t in tasks]
        mono_scores = [v * 100 if v is not None else None for v in mono_scores]
    else:
        tasks = [t for t in base_v if base_v.get(t) is not None and moe_v.get(t) is not None]
        base_scores = [float(base_v[t]) for t in tasks]
        moe_scores  = [float(moe_v[t])  for t in tasks]
        mono_scores = [float(mono_v[t]) if mono_v.get(t) is not None else None for t in tasks]

    x = np.arange(len(tasks))
    width = 0.28

    ax.bar(x - width, base_scores, width=width * 0.92,
           color=COLORS["base"],       label="Base",      edgecolor="white", linewidth=0.5)
    ax.bar(x,         moe_scores,  width=width * 0.92,
           color=COLORS["moe"],        label="MoE Fused", edgecolor="white", linewidth=0.5)

    valid_mono = [(i, v) for i, v in enumerate(mono_scores) if v is not None]
    if valid_mono:
        mono_x = [x[i] + width for i, _ in valid_mono]
        mono_y = [v for _, v in valid_mono]
        ax.bar(mono_x, mono_y, width=width * 0.92,
               color=COLORS["monolithic"], label="Monolithic", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    clean_axes(ax)


# ══════════════════════════════════════════════════════════════════════════════
# fig_benchmarks.png — 410M
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_benchmarks.png...")
try:
    d = load(rpath("pythia", "benchmarks_seed42.json"))
    fig, ax = plt.subplots(figsize=(9, 4))
    _draw_benchmarks(ax, d["model_variants"], "Downstream Benchmarks — Pythia-410M (seed=42)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_benchmarks.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_benchmarks.png")
except Exception as e:
    print(f"WARNING: skipped fig_benchmarks: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_benchmarks_1b.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_benchmarks_1b.png...")
try:
    d = load(rpath("pythia", "pythia_1b", "benchmarks_seed42.json"))
    # 1B has flat results dict
    fig, ax = plt.subplots(figsize=(9, 4))
    results = d["results"]
    # Build model_variants in nested format for reuse
    mv = {
        "base":      {t: {"accuracy": results["base"][t] / 100}
                      for t in d["benchmarks"] if t in results["base"]},
        "moe_fused": {t: {"accuracy": results["moe"][t] / 100}
                      for t in d["benchmarks"] if t in results.get("moe", {})},
        "monolithic": {t: {"accuracy": results["monolithic"][t] / 100}
                       for t in d["benchmarks"] if t in results.get("monolithic", {})},
    }
    _draw_benchmarks(ax, mv, "Downstream Benchmarks — Pythia-1B (seed=42)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_benchmarks_1b.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_benchmarks_1b.png")
except Exception as e:
    print(f"WARNING: skipped fig_benchmarks_1b: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: maturity curve
# ══════════════════════════════════════════════════════════════════════════════
def _extract_maturity_410m(d):
    xs, ys, errs = [], [], []
    for pt in d["curve"]:
        xs.append(pt["training_pct"])
        ms = pt.get("multiseed")
        if ms and "mean" in ms:
            ys.append(ms["mean"])
            errs.append(ms.get("std", 0.0))
        else:
            ys.append(pt["improvement_pct_seed42"])
            errs.append(0.0)
    return xs, ys, errs

def _extract_maturity_1b(d):
    xs, ys, errs = [], [], []
    for pt in d["checkpoints"]:
        xs.append(pt["training_pct"])
        ys.append(pt["improvement_pct"])
        errs.append(0.0)
    return xs, ys, errs


# ══════════════════════════════════════════════════════════════════════════════
# fig_maturity_curve_410m.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_maturity_curve_410m.png...")
try:
    d = load(rpath("pythia", "maturity_sweep_410m", "summary.json"))
    xs, ys, errs = _extract_maturity_410m(d)

    fig, ax = plt.subplots(figsize=(5, 4))
    # Green shaded KALAVU target zone (0–20%)
    ax.axvspan(0, 20, alpha=0.08, color=COLORS["science"], label="KALAVU target zone")
    ax.errorbar(xs, ys, yerr=errs,
                color=COLORS["moe"], marker="o", markersize=6,
                linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
                label="410M MoE improvement %")
    ax.set_xlabel("Base Model Training (%)")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Maturity Sweep — Pythia-410M")
    ax.legend(fontsize=8)
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_maturity_curve_410m.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_maturity_curve_410m.png")
except Exception as e:
    print(f"WARNING: skipped fig_maturity_curve_410m: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_maturity_curve_1b.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_maturity_curve_1b.png...")
try:
    d = load(rpath("pythia", "pythia_1b", "maturity_sweep", "summary.json"))
    xs, ys, errs = _extract_maturity_1b(d)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axvspan(0, 20, alpha=0.08, color=COLORS["science"], label="KALAVU target zone")
    ax.errorbar(xs, ys, yerr=errs,
                color=COLORS["moe"], marker="s", markersize=6,
                linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
                label="1B MoE improvement %")
    ax.set_xlabel("Base Model Training (%)")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Maturity Sweep — Pythia-1B")
    ax.legend(fontsize=8)
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_maturity_curve_1b.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_maturity_curve_1b.png")
except Exception as e:
    print(f"WARNING: skipped fig_maturity_curve_1b: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_maturity_curve_combined.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_maturity_curve_combined.png...")
try:
    d410 = load(rpath("pythia", "maturity_sweep_410m", "summary.json"))
    d1b  = load(rpath("pythia", "pythia_1b", "maturity_sweep", "summary.json"))
    xs_410, ys_410, errs_410 = _extract_maturity_410m(d410)
    xs_1b,  ys_1b,  errs_1b  = _extract_maturity_1b(d1b)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvspan(0, 20, alpha=0.08, color=COLORS["science"], label="KALAVU target zone")

    ax.errorbar(xs_410, ys_410, yerr=errs_410,
                color=COLORS["code"], marker="o", markersize=6,
                linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
                label="410M")
    ax.errorbar(xs_1b, ys_1b, yerr=errs_1b,
                color=COLORS["moe"], marker="s", markersize=6,
                linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
                label="1B")

    # Qwen data point at -1.0% improvement (not a KALAVU result — shown as reference)
    ax.scatter([100], [-1.0], color="#9ca3af", marker="D", s=60, zorder=5,
               label="Qwen-0.5B (ref, −1.0%)")

    ax.set_xlabel("Base Model Training (%)")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Maturity Sweep — 410M vs 1B")
    ax.legend(fontsize=8)
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_maturity_curve_combined.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_maturity_curve_combined.png")
except Exception as e:
    print(f"WARNING: skipped fig_maturity_curve_combined: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_maturity_curve_comparison_bar.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_maturity_curve_comparison_bar.png...")
try:
    d410 = load(rpath("pythia", "maturity_sweep_410m", "summary.json"))
    d1b  = load(rpath("pythia", "pythia_1b", "maturity_sweep", "summary.json"))

    # step10000 (7%) and step143000 (100%)
    curve_410 = {pt["step_n"]: pt for pt in d410["curve"]}
    curve_1b  = {pt["revision"]: pt for pt in d1b["checkpoints"]}

    val_410_early = curve_410.get(10000, {}).get("improvement_pct_seed42", 0)
    val_410_full  = curve_410.get(143000, {}).get("improvement_pct_seed42", 0)
    val_1b_early  = curve_1b.get("step10000",  {}).get("improvement_pct", 0)
    val_1b_full   = curve_1b.get("step143000", {}).get("improvement_pct", 0)

    labels = ["410M\nstep10k", "410M\nstep143k", "1B\nstep10k", "1B\nstep143k"]
    vals   = [val_410_early, val_410_full, val_1b_early, val_1b_full]
    colors = [COLORS["code"], COLORS["code"], COLORS["moe"], COLORS["moe"]]
    alphas = [0.6, 1.0, 0.6, 1.0]

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (lbl, val, col, alp) in enumerate(zip(labels, vals, colors, alphas)):
        ax.bar(i, val, color=col, alpha=alp, width=0.6, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    clean_axes(ax)
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Key Maturity Points — 410M vs 1B")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_maturity_curve_comparison_bar.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_maturity_curve_comparison_bar.png")
except Exception as e:
    print(f"WARNING: skipped fig_maturity_curve_comparison_bar: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_specialist_scaling.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_specialist_scaling.png...")
try:
    d = load(rpath("pythia", "five_domain", "summary.json"))
    agg = d["aggregate_scaling"]

    ns   = []
    ys   = []
    errs = []
    for key in sorted(agg.keys()):
        entry = agg[key]
        ns.append(entry["n_specialists"])
        ys.append(entry["improvement_mean_pct"])
        errs.append(entry["improvement_std_pct"])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(ns, ys, yerr=errs,
                color=COLORS["moe"], marker="o", markersize=7,
                linewidth=2.2, capsize=5, capthick=1.5, elinewidth=1.2)
    ax.set_xlabel("Number of Specialists")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Specialist Scaling (410M, 3 seeds)")
    ax.set_xticks(ns)
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_specialist_scaling.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_specialist_scaling.png")
except Exception as e:
    print(f"WARNING: skipped fig_specialist_scaling: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_monolithic_comparison.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_monolithic_comparison.png...")
try:
    d = load(rpath("pythia", "monolithic_baseline_summary.json"))
    base_loss  = d["base_losses"]["mixed"]
    mono_loss  = d["results"]["mean"]["monolithic_mixed"]
    moe_loss   = d["results"]["mean"]["moe_fused_mixed"]

    labels = ["Base", "Monolithic", "MoE Fused"]
    vals   = [base_loss, mono_loss, moe_loss]
    colors = [COLORS["base"], COLORS["monolithic"], COLORS["moe"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_ylabel("Held-Out Mixed Loss")
    ax.set_title("Monolithic Baseline vs MoE (410M)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_monolithic_comparison.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_monolithic_comparison.png")
except Exception as e:
    print(f"WARNING: skipped fig_monolithic_comparison: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_monolithic_trajectory.png
# No trajectory field — use same comparison bar chart fallback
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_monolithic_trajectory.png...")
try:
    d = load(rpath("pythia", "monolithic_baseline_summary.json"))

    # Look for trajectory data; fall back to per-seed comparison
    trajectory = d.get("trajectory") or d.get("curve")
    if trajectory:
        steps = [pt.get("step", i) for i, pt in enumerate(trajectory)]
        mono_curve = [pt.get("monolithic_loss", np.nan) for pt in trajectory]
        moe_curve  = [pt.get("moe_loss", np.nan) for pt in trajectory]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(steps, mono_curve, color=COLORS["monolithic"], label="Monolithic", linewidth=2)
        ax.plot(steps, moe_curve,  color=COLORS["moe"],        label="MoE Fused",  linewidth=2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mixed Loss")
        ax.set_title("Monolithic vs MoE — Training Trajectory")
        ax.legend(fontsize=8)
        clean_axes(ax)
    else:
        # Fallback: per-seed bar comparison
        per_seed = d.get("per_seed", [])
        base_loss = d["base_losses"]["mixed"]
        moe_loss  = d["results"]["mean"]["moe_fused_mixed"]
        seeds = [str(r["seed"]) for r in per_seed]
        mono_vals = [r["final_mixed"] for r in per_seed]
        x = np.arange(len(seeds))
        width = 0.3
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(x - width / 2, mono_vals, width=width * 0.9,
               color=COLORS["monolithic"], label="Monolithic", edgecolor="white")
        ax.axhline(moe_loss,  color=COLORS["moe"],  linestyle="--", linewidth=1.5, label="MoE (mean)")
        ax.axhline(base_loss, color=COLORS["base"], linestyle=":",  linewidth=1.5, label="Base")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed={s}" for s in seeds], fontsize=9)
        ax.set_ylabel("Mixed Loss")
        ax.set_title("Monolithic per Seed vs MoE (410M)")
        ax.legend(fontsize=8)
        clean_axes(ax)

    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_monolithic_trajectory.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_monolithic_trajectory.png")
except Exception as e:
    print(f"WARNING: skipped fig_monolithic_trajectory: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_domain_classifier.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_domain_classifier.png...")
try:
    d = load(rpath("pythia", "domain_classifier_baseline.json"))
    base_loss       = d["base_loss"]
    moe_loss        = d["moe_loss"]
    classifier_loss = d["classifier_loss"]

    labels = ["Base", "MoE Fused", "Domain Classifier"]
    vals   = [base_loss, moe_loss, classifier_loss]
    colors = [COLORS["base"], COLORS["moe"], COLORS["classifier"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_ylabel("Held-Out Mixed Loss")
    ax.set_title("Domain Classifier Baseline (410M, seed=42)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_domain_classifier.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_domain_classifier.png")
except Exception as e:
    print(f"WARNING: skipped fig_domain_classifier: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_hybrid_routing_0.png through fig_hybrid_routing_4.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_hybrid_routing_*.png...")
try:
    d = load(rpath("pythia", "hybrid_routing_analysis.json"))
    prompts = d["prompts"]

    expert_colors = [COLORS["code"], COLORS["science"], COLORS["fiction"]]
    expert_labels = ["Code", "Science", "Fiction"]

    for pi, prompt in enumerate(prompts):
        try:
            tokens       = prompt["tokens"]
            gate_weights = np.array(prompt["gate_weights"])  # shape (T, 3)
            n_tokens     = len(tokens)
            n_experts    = gate_weights.shape[1] if gate_weights.ndim == 2 else 3
            switches     = prompt.get("switches", 0)
            text_trunc   = prompt["text"][:50]

            fig, ax = plt.subplots(figsize=(max(6, n_tokens * 0.55), 3))

            # Stacked bar per token
            bottoms = np.zeros(n_tokens)
            for ei in range(n_experts):
                weights = gate_weights[:, ei] if gate_weights.ndim == 2 else np.zeros(n_tokens)
                ax.bar(range(n_tokens), weights, bottom=bottoms,
                       color=expert_colors[ei % len(expert_colors)],
                       label=expert_labels[ei % len(expert_labels)],
                       edgecolor="white", linewidth=0.3)
                bottoms += weights

            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Gate Weight")
            ax.set_xlabel("Token")
            ax.set_title(f"Token Routing — \"{text_trunc}\" (switches={switches})", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, loc="upper right")
            clean_axes(ax)
            fig.tight_layout(pad=1.5)
            out_path = fpath("pythia", f"fig_hybrid_routing_{pi}.png")
            fig.savefig(out_path)
            plt.close(fig)
            print(f"Saved: figures/pythia/fig_hybrid_routing_{pi}.png")
        except Exception as e2:
            print(f"WARNING: skipped fig_hybrid_routing_{pi}: {e2}")
except Exception as e:
    print(f"WARNING: skipped fig_hybrid_routing_*: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_multihead_baseline.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_multihead_baseline.png...")
try:
    d = load(rpath("pythia", "multihead_baseline.json"))
    base_loss      = d["base_loss"]
    moe_loss       = d["moe_loss"]
    multihead_loss = d["multihead_loss"]

    labels = ["Base", "MoE Fused", "Multi-Head"]
    vals   = [base_loss, moe_loss, multihead_loss]
    colors = [COLORS["base"], COLORS["moe"], COLORS["multihead"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_ylabel("Held-Out Mixed Loss")
    ax.set_title("Multi-Head Baseline (410M, seed=42)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_multihead_baseline.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_multihead_baseline.png")
except Exception as e:
    print(f"WARNING: skipped fig_multihead_baseline: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_wider_model_baseline.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_wider_model_baseline.png...")
try:
    d = load(rpath("pythia", "wider_model_baseline.json"))
    base_loss    = d["base_loss"]          # 1.4B base
    wider_loss   = d["wider_loss"]         # 1.4B fine-tuned (all-data monolithic)
    moe_imp_pct  = d["moe_410m_improvement_pct"]

    # MoE 410M equivalent loss: reconstruct from 410M base + improvement %
    d_sum = load(rpath("pythia", "step5_final_summary.json"))
    moe_410_loss = float(np.mean([
        d_sum["per_seed_fusion"][s]["eval_heldout"]["moe"]["mixed"]
        for s in ["42", "137", "2026"]
    ]))

    labels = ["1.4B Base", "1.4B Fine-tuned\n(6k steps)", f"MoE 410M\n(+{moe_imp_pct:.1f}%)"]
    vals   = [base_loss, wider_loss, moe_410_loss]
    colors = [COLORS["base"], COLORS["wider"], COLORS["moe"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_ylabel("Held-Out Mixed Loss")
    ax.set_title("Wider Model Baseline")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_wider_model_baseline.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_wider_model_baseline.png")
except Exception as e:
    print(f"WARNING: skipped fig_wider_model_baseline: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_training_duration_crossover.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_training_duration_crossover.png...")
try:
    d = load(rpath("pythia", "training_duration_crossover.json"))
    steps     = d["steps"]
    f0_imp    = d["freeze0_improvement"]
    f4_imp    = d["freeze4_improvement"]
    crossover = d.get("crossover_steps", None)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(steps, f0_imp, color=COLORS["freeze0"], linestyle="--",
            linewidth=2, marker="o", markersize=5, label="freeze=0")
    ax.plot(steps, f4_imp, color=COLORS["freeze4"], linestyle="-",
            linewidth=2, marker="s", markersize=5, label="freeze=4")

    if crossover:
        ax.axvline(crossover, color=COLORS["crossover"], linestyle=":",
                   linewidth=1.5, alpha=0.8)
        ax.annotate(f"crossover\n@{crossover} steps",
                    xy=(crossover, (max(f0_imp + f4_imp)) * 0.6),
                    xytext=(crossover * 1.2, (max(f0_imp + f4_imp)) * 0.6),
                    fontsize=8, color="#374151",
                    arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1))

    ax.set_xscale("log")
    ax.set_xlabel("Router Training Steps (log scale)")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Training Duration vs Fusion Improvement")
    ax.legend(fontsize=8)
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_training_duration_crossover.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_training_duration_crossover.png")
except Exception as e:
    print(f"WARNING: skipped fig_training_duration_crossover: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_hard_routing_verification.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_hard_routing_verification.png...")
try:
    d = load(rpath("pythia", "hard_routing_verification.json"))
    base_loss     = d["base_loss"]
    soft_loss     = d["soft_moe_loss"]
    hard_loss     = d["hard_routing_loss"]

    labels = ["Base", "Soft MoE", "Hard Routing"]
    vals   = [base_loss, soft_loss, hard_loss]
    colors = [COLORS["base"], COLORS["moe"], COLORS["classifier"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_ylabel("Held-Out Mixed Loss")
    ax.set_title("Hard Routing Verification (410M, seed=42)")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_hard_routing_verification.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_hard_routing_verification.png")
except Exception as e:
    print(f"WARNING: skipped fig_hard_routing_verification: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_mutual_information.png — heatmap of error correlations
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_mutual_information.png...")
try:
    d    = load(rpath("pythia", "mutual_information.json"))
    mat  = np.array(d["error_correlation_matrix"])
    labs = ["Code", "Science", "Fiction"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(labs)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labs)

    for y in range(3):
        for x in range(3):
            val  = mat[y, x]
            rgba = im.cmap(im.norm(val))
            tc   = cell_text_color(rgba)
            ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=tc, fontweight="medium")

    for y in np.arange(-0.5, 3, 1):
        ax.axhline(y, color="white", linewidth=1.5)
    for x in np.arange(-0.5, 3, 1):
        ax.axvline(x, color="white", linewidth=1.5)

    ax.set_title("Error Correlation Matrix (410M, seed=42)")
    ax.set_xlabel("Specialist")
    ax.set_ylabel("Specialist")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_mutual_information.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_mutual_information.png")
except Exception as e:
    print(f"WARNING: skipped fig_mutual_information: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_scale_ladder.png — improvement % vs model size
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_scale_ladder.png...")
try:
    d410 = load(rpath("pythia", "step5_final_summary.json"))
    d1b  = load(rpath("pythia", "pythia_1b", "main_result_summary.json"))
    # 6.9B: aggregate across seeds
    seeds_6b = ["42", "137", "2026"]
    imp_6b = []
    for s in seeds_6b:
        try:
            df = load(rpath("pythia_6b", f"step6_fusion_seed{s}.json"))
            imp_6b.append(df["improvement_pct"])
        except Exception:
            pass

    sizes   = [410e6, 1e9]
    means   = [d410["summary"]["improvement_mean_pct"],
               d1b["summary"]["improvement_mean_pct"]]
    stds    = [d410["summary"]["improvement_std_pct"],
               d1b["summary"]["improvement_std_pct"]]
    labels  = ["410M", "1B"]

    if imp_6b:
        sizes.append(6.9e9)
        means.append(float(np.mean(imp_6b)))
        stds.append(float(np.std(imp_6b)))
        labels.append("6.9B")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(sizes, means, yerr=stds,
                color=COLORS["moe"], marker="o", markersize=8,
                linewidth=2, capsize=5, capthick=1.5, elinewidth=1.2)

    for sz, m, lbl in zip(sizes, means, labels):
        ax.annotate(lbl, xy=(sz, m), xytext=(sz * 1.15, m + 0.05),
                    fontsize=9, color="#374151")

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (parameters, log scale)")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_title("Scale Ladder — MoE Improvement vs Model Size")
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_scale_ladder.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_scale_ladder.png")
except Exception as e:
    print(f"WARNING: skipped fig_scale_ladder: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_6b_summary.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_6b_summary.png...")
try:
    seeds = [42, 137, 2026]
    base_vals  = []
    best_vals  = []
    wavg_vals  = []
    moe_vals   = []

    for s in seeds:
        df = load(rpath("pythia_6b", f"step6_fusion_seed{s}.json"))
        eh = df["eval_heldout"]
        base_vals.append(eh["base"]["mixed"])
        best_vals.append(min(
            eh["code_spec"]["mixed"],
            eh["science_spec"]["mixed"],
            eh["fiction_spec"]["mixed"],
        ))
        wavg_vals.append(eh["weight_avg"]["mixed"])
        moe_vals.append(eh["moe"]["mixed"])

    labels = ["Base", "Best Individual", "Weight Avg", "MoE"]
    vals   = [np.mean(base_vals), np.mean(best_vals),
              np.mean(wavg_vals), np.mean(moe_vals)]
    colors = [COLORS["base"], COLORS["code"], COLORS["weight_avg"], COLORS["moe"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.8)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)
    clean_axes(ax)
    label_bars(ax, bars)
    ax.set_title("Fusion Methods — Pythia-6.9B (3 seeds)")
    ax.set_ylabel("Held-Out Mixed Loss")
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_6b_summary.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_6b_summary.png")
except Exception as e:
    print(f"WARNING: skipped fig_6b_summary: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_paper_hero.png — 2×2 hero figure
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_paper_hero.png...")
try:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # ── Top-left: maturity curves (410M + 1B) ─────────────────────────────────
    ax = axes[0, 0]
    try:
        d410 = load(rpath("pythia", "maturity_sweep_410m", "summary.json"))
        d1b  = load(rpath("pythia", "pythia_1b", "maturity_sweep", "summary.json"))
        xs_410, ys_410, errs_410 = _extract_maturity_410m(d410)
        xs_1b,  ys_1b,  _        = _extract_maturity_1b(d1b)
        ax.axvspan(0, 20, alpha=0.08, color=COLORS["science"])
        ax.errorbar(xs_410, ys_410, yerr=errs_410,
                    color=COLORS["code"], marker="o", markersize=4,
                    linewidth=1.8, capsize=3, label="410M", elinewidth=1.0)
        ax.plot(xs_1b, ys_1b, color=COLORS["moe"], marker="s", markersize=4,
                linewidth=1.8, label="1B")
        ax.set_xlabel("Base Training (%)", fontsize=9)
        ax.set_ylabel("Improvement (%)", fontsize=9)
        ax.set_title("(A) Maturity Sweep", fontsize=11)
        ax.legend(fontsize=8)
    except Exception as e_inner:
        ax.text(0.5, 0.5, f"No data\n{e_inner}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#6b7280")
    clean_axes(ax)

    # ── Top-right: specialist scaling ─────────────────────────────────────────
    ax = axes[0, 1]
    try:
        d = load(rpath("pythia", "five_domain", "summary.json"))
        agg = d["aggregate_scaling"]
        ns   = sorted([v["n_specialists"] for v in agg.values()])
        ys   = [agg[k]["improvement_mean_pct"] for k in sorted(agg)]
        errs = [agg[k]["improvement_std_pct"]  for k in sorted(agg)]
        ax.errorbar(ns, ys, yerr=errs,
                    color=COLORS["moe"], marker="o", markersize=5,
                    linewidth=2, capsize=4, elinewidth=1.0)
        ax.set_xlabel("N Specialists", fontsize=9)
        ax.set_ylabel("Improvement (%)", fontsize=9)
        ax.set_title("(B) Specialist Scaling", fontsize=11)
        ax.set_xticks(ns)
    except Exception as e_inner:
        ax.text(0.5, 0.5, f"No data\n{e_inner}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#6b7280")
    clean_axes(ax)

    # ── Bottom-left: freeze depth ablation ────────────────────────────────────
    ax = axes[1, 0]
    try:
        d     = load(rpath("pythia", "ablation_freeze_summary.json"))
        phase1 = d["phase1_seed42_results"]
        xs_f  = [r["freeze_layers"] for r in phase1]
        ys_f  = [r["improvement_pct"] for r in phase1]
        yerr_f = []
        for r in phase1:
            ms = d.get("multi_seed_results", {}).get(str(r["freeze_layers"]))
            yerr_f.append(ms["std"] if ms and "std" in ms else 0.0)
        ax.errorbar(xs_f, ys_f, yerr=yerr_f,
                    color=COLORS["moe"], marker="o", markersize=5,
                    linewidth=2, capsize=4, elinewidth=1.0)
        ax.axvline(4, color="#9ca3af", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Frozen Layers", fontsize=9)
        ax.set_ylabel("Improvement (%)", fontsize=9)
        ax.set_title("(C) Freeze Depth Ablation", fontsize=11)
        ax.set_xticks(xs_f)
    except Exception as e_inner:
        ax.text(0.5, 0.5, f"No data\n{e_inner}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#6b7280")
    clean_axes(ax)

    # ── Bottom-right: router architecture ablation ────────────────────────────
    ax = axes[1, 1]
    try:
        d = load(rpath("pythia", "ablation_router_summary.json"))
        variants = d["variants"]
        labels_r = ["Uniform", "Simple\nLinear", "Two-Layer"]
        keys_r   = ["uniform", "simple_linear", "two_layer"]
        vals_r   = [variants[k]["improvement_pct"] for k in keys_r]
        cols_r   = [COLORS["base"], COLORS["moe"], COLORS["weight_avg"]]
        bars = ax.bar(labels_r, vals_r, color=cols_r, width=0.55,
                      edgecolor="white", linewidth=0.8)
        ax.set_ylabel("Improvement (%)", fontsize=9)
        ax.set_title("(D) Router Architecture", fontsize=11)
        label_bars(ax, bars, fmt="{:.1f}%", fontsize=7)
    except Exception as e_inner:
        ax.text(0.5, 0.5, f"No data\n{e_inner}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#6b7280")
    clean_axes(ax)

    fig.suptitle("KALAVU: Cooperative MoE Fusion — Key Results", fontsize=13, y=1.01)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia", "fig_paper_hero.png"))
    plt.close(fig)
    print("Saved: figures/pythia/fig_paper_hero.png")
except Exception as e:
    print(f"WARNING: skipped fig_paper_hero: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# figures/pythia_6b/ — per-seed divergence, fusion, router distribution
# ══════════════════════════════════════════════════════════════════════════════
for seed in [42, 137, 2026]:
    # Divergence heatmap
    print(f"Generating fig_6b_divergence_heatmap_seed{seed}.png...")
    try:
        df = load(rpath("pythia_6b", f"step5_divergence_seed{seed}.json"))
        fig, ax = plt.subplots(figsize=(5, 4))
        _draw_divergence_heatmap(
            ax, df["loss_matrix"],
            f"Specialist Divergence — Cross-Domain Losses (6.9B, seed={seed})"
        )
        fig.tight_layout(pad=1.5)
        fig.savefig(fpath("pythia_6b", f"fig_6b_divergence_heatmap_seed{seed}.png"))
        plt.close(fig)
        print(f"Saved: figures/pythia_6b/fig_6b_divergence_heatmap_seed{seed}.png")
    except Exception as e:
        print(f"WARNING: skipped fig_6b_divergence_heatmap_seed{seed}: {e}")

    # Fusion comparison
    print(f"Generating fig_6b_fusion_comparison_seed{seed}.png...")
    try:
        df = load(rpath("pythia_6b", f"step6_fusion_seed{seed}.json"))
        eh = df["eval_heldout"]
        base_m  = eh["base"]["mixed"]
        best_m  = min(eh["code_spec"]["mixed"],
                      eh["science_spec"]["mixed"],
                      eh["fiction_spec"]["mixed"])
        wavg_m  = eh["weight_avg"]["mixed"]
        moe_m   = eh["moe"]["mixed"]

        labels = ["Base", "Best Individual", "Weight Avg", "MoE"]
        vals   = [base_m, best_m, wavg_m, moe_m]
        colors = [COLORS["base"], COLORS["code"], COLORS["weight_avg"], COLORS["moe"]]

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, vals, color=colors, width=0.6,
                      edgecolor="white", linewidth=0.8)
        ymin = min(vals) * 0.97
        ymax = max(vals) * 1.005
        ax.set_ylim(ymin, ymax)
        clean_axes(ax)
        label_bars(ax, bars)
        ax.set_title(f"Fusion Methods — Pythia-6.9B (seed={seed})")
        ax.set_ylabel("Held-Out Mixed Loss")
        fig.tight_layout(pad=1.5)
        fig.savefig(fpath("pythia_6b", f"fig_6b_fusion_comparison_seed{seed}.png"))
        plt.close(fig)
        print(f"Saved: figures/pythia_6b/fig_6b_fusion_comparison_seed{seed}.png")
    except Exception as e:
        print(f"WARNING: skipped fig_6b_fusion_comparison_seed{seed}: {e}")

    # Router distribution
    print(f"Generating fig_6b_router_distribution_seed{seed}.png...")
    try:
        df = load(rpath("pythia_6b", f"step6_fusion_seed{seed}.json"))
        router_dist = df.get("router_distribution") or df.get("gate_distribution")
        if router_dist is None:
            raise ValueError("No router_distribution field found")

        fig, ax = plt.subplots(figsize=(5, 4))
        _draw_router_distribution(
            ax, router_dist,
            f"Router Gate Distribution (6.9B, seed={seed})"
        )
        fig.tight_layout(pad=1.5)
        fig.savefig(fpath("pythia_6b", f"fig_6b_router_distribution_seed{seed}.png"))
        plt.close(fig)
        print(f"Saved: figures/pythia_6b/fig_6b_router_distribution_seed{seed}.png")
    except Exception as e:
        print(f"WARNING: skipped fig_6b_router_distribution_seed{seed}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_6b_maturity.png — step10000 vs step143000 for 6.9B
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_6b_maturity.png...")
try:
    d10k  = load(rpath("pythia_6b", "step6_fusion_seed42.json"))
    d143k = load(rpath("pythia_6b", "maturity_step143000_seed42.json"))

    revisions   = ["step10000\n(15% training)", "step143000\n(100% training)"]
    base_losses = [d10k["eval_heldout"]["base"]["mixed"],
                   d143k["eval_heldout"]["base"]["mixed"]]
    moe_losses  = [d10k["eval_heldout"]["moe"]["mixed"],
                   d143k["eval_heldout"]["moe"]["mixed"]]
    imps        = [d10k["improvement_pct"], d143k["improvement_pct"]]

    x = np.arange(len(revisions))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_base = ax.bar(x - w/2, base_losses, w, label="Base",
                       color=COLORS["base"], edgecolor="white")
    bars_moe  = ax.bar(x + w/2, moe_losses,  w, label="MoE Fusion",
                       color=COLORS["moe"],  edgecolor="white")

    for bar, imp in zip(bars_moe, imps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"+{imp:.2f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(revisions, fontsize=9)
    ax.set_ylabel("Held-Out Mixed Loss")
    ax.set_title("6.9B Maturity: step10000 vs step143000 (seed=42)")
    ax.legend(fontsize=9)
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia_6b", "fig_6b_maturity.png"))
    plt.close(fig)
    print("Saved: figures/pythia_6b/fig_6b_maturity.png")
except Exception as e:
    print(f"WARNING: skipped fig_6b_maturity: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# fig_6b_benchmarks.png — downstream task accuracy base vs MoE
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_6b_benchmarks.png...")
try:
    db      = load(rpath("pythia_6b", "benchmarks_seed42.json"))
    tasks   = ["hellaswag", "arc_easy", "lambada", "sciq", "winogrande"]
    base_sc = [db["results"]["base"][t] for t in tasks]
    moe_sc  = [db["results"]["moe"][t]  for t in tasks]
    labels_t = ["HellaSwag", "ARC-Easy", "LAMBADA", "SciQ", "WinoGrande"]

    x = np.arange(len(tasks))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, base_sc, w,
           label=f"Base ({db['results']['base']['average']:.1f}% avg)",
           color=COLORS["base"], edgecolor="white")
    bars_m = ax.bar(x + w/2, moe_sc, w,
                    label=f"MoE  ({db['results']['moe']['average']:.1f}% avg)",
                    color=COLORS["moe"],  edgecolor="white")

    for bar, bv, mv in zip(bars_m, base_sc, moe_sc):
        delta = mv - bv
        sign  = "+" if delta >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{sign}{delta:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_t, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("6.9B Downstream Benchmarks — Base vs MoE Fusion (seed=42)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(moe_sc + base_sc) * 1.15)
    clean_axes(ax)
    fig.tight_layout(pad=1.5)
    fig.savefig(fpath("pythia_6b", "fig_6b_benchmarks.png"))
    plt.close(fig)
    print("Saved: figures/pythia_6b/fig_6b_benchmarks.png")
except Exception as e:
    print(f"WARNING: skipped fig_6b_benchmarks: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Git: add all figures, commit, push
# ══════════════════════════════════════════════════════════════════════════════
print("\nCommitting figures to git...")
import subprocess

subprocess.run(["git", "add", "figures/"], check=True, cwd=BASE)
result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=BASE)
if result.returncode != 0:
    subprocess.run(
        ["git", "commit", "-m", "[kalavai] add 6.9B maturity + benchmark figures, update scale ladder"],
        check=True, cwd=BASE
    )
    subprocess.run(["git", "push"], check=True, cwd=BASE)
    print("[git] Committed and pushed.")
else:
    print("[git] No changes to commit.")

print("\nDone.")
