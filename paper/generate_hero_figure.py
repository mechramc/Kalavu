#!/usr/bin/env python3
"""
Generate the KALAVAI paper hero figure (4-panel).

Panel A: Scale comparison bars (410M, 1B, 6.9B)
Panel B: Training duration crossover
Panel C: Routing failure comparison (MoE vs Classifier vs Multi-head)
Panel D: Monolithic comparison (Base / Monolithic / Best Spec / Weight Avg / MoE)
"""
import json
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from figure_style import apply_style, COLORS, clean_axes, label_bars

apply_style()

# ── result paths ──────────────────────────────────────────────────────────────
RESULTS = REPO_ROOT / "results" / "pythia"
RESULTS_V2 = REPO_ROOT / "results" / "pythia" / "v2"
RESULTS_6B = REPO_ROOT / "results" / "pythia_6b"

# ── load data (corrected eval) ────────────────────────────────────────────────
# 410M: corrected per-domain equal-weight eval (3 seeds)
with open(RESULTS / "corrected_eval_42.json") as f:
    corr_410m_42 = json.load(f)
with open(RESULTS / "corrected_eval_137.json") as f:
    corr_410m_137 = json.load(f)
with open(RESULTS / "corrected_eval_2026.json") as f:
    corr_410m_2026 = json.load(f)

# 1B: corrected eval (3 seeds)
with open(RESULTS / "pythia_1b" / "corrected_eval_42.json") as f:
    corr_1b_42 = json.load(f)
with open(RESULTS / "pythia_1b" / "corrected_eval_137.json") as f:
    corr_1b_137 = json.load(f)
with open(RESULTS / "pythia_1b" / "corrected_eval_2026.json") as f:
    corr_1b_2026 = json.load(f)

# Training duration crossover (v2 corrected equal-weight eval)
with open(RESULTS_V2 / "crossover_v2.json") as f:
    crossover_raw = json.load(f)

with open(RESULTS / "domain_classifier_baseline.json") as f:
    classifier_data = json.load(f)

with open(RESULTS / "multihead_baseline.json") as f:
    multihead_data = json.load(f)

# ── compute derived values ────────────────────────────────────────────────────

# Panel A: scale comparison (corrected equal-weight eval)
imps_410m = [
    corr_410m_42["metrics"]["improvement_vs_spec"],
    corr_410m_137["metrics"]["improvement_vs_spec"],
    corr_410m_2026["metrics"]["improvement_vs_spec"],
]
imp_410m_mean = float(np.mean(imps_410m))
imp_410m_std  = float(np.std(imps_410m, ddof=1))

# 1B: 3 seeds (corrected)
imps_1b = [
    corr_1b_42["metrics"]["improvement_vs_spec"],
    corr_1b_137["metrics"]["improvement_vs_spec"],
    corr_1b_2026["metrics"]["improvement_vs_spec"],
]
imp_1b_mean = float(np.mean(imps_1b))
imp_1b_std  = float(np.std(imps_1b, ddof=1))

# 6.9B: corrected per-domain equal-weight, seeded shuffle fix, 3 seeds
imp_6b_mean = 6.53
imp_6b_std  = 0.024

# Panel B crossover: extract from v2 results list (improvement_vs_spec)
_results = crossover_raw["results"]
_f0 = [(r["steps"], r["improvement_vs_spec"]) for r in _results if r["freeze"] == 0]
_f4 = [(r["steps"], r["improvement_vs_spec"]) for r in _results if r["freeze"] == 4]
_f0.sort(); _f4.sort()
crossover_data = {
    "steps":               [s for s, _ in _f0],
    "freeze0_improvement": [v for _, v in _f0],
    "freeze4_improvement": [v for _, v in _f4],
    "crossover_steps":     5000,
}

# Panel D: monolithic comparison (corrected equal-weight, seed 42)
m42 = corr_410m_42["eval_matrix"]
base_loss       = m42["base"]["equal_weight_avg"]
best_spec_loss  = corr_410m_42["metrics"]["best_spec_equal_weight"]
weight_avg_loss = m42["weight_avg"]["equal_weight_avg"]
moe_loss_410m   = m42["moe"]["equal_weight_avg"]
monolithic_loss = corr_410m_42["metrics"]["monolithic_equal_weight"]

# ── figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("KALAVAI: Four Key Findings", fontsize=16, fontweight="bold", y=1.01)

ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Scale Comparison
# ─────────────────────────────────────────────────────────────────────────────
labels_a = ["410M", "1B", "6.9B"]
means_a  = [imp_410m_mean, imp_1b_mean, imp_6b_mean]
stds_a   = [imp_410m_std,  imp_1b_std,  imp_6b_std]
x_a = np.arange(len(labels_a))

bars_a = ax_a.bar(
    x_a, means_a,
    yerr=stds_a, capsize=6,
    color=COLORS["moe"], width=0.55, zorder=3,
    error_kw={"elinewidth": 1.5, "ecolor": "#374151"},
)

ax_a.set_xticks(x_a)
ax_a.set_xticklabels(labels_a)
ax_a.set_ylabel("Improvement over best specialist (%)")
ax_a.set_title("(A)  Scale Comparison: KALAVAI MoE vs Best Specialist")
clean_axes(ax_a)

# label bars with +X.X%
for bar, mean in zip(bars_a, means_a):
    ax_a.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(stds_a) * 1.1 + 0.1,
        f"+{mean:.1f}%",
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1f2937",
    )

ax_a.set_ylim(0, max(means_a) + max(stds_a) * 1.5 + 1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: Training Duration Crossover
# ─────────────────────────────────────────────────────────────────────────────
steps      = crossover_data["steps"]
freeze0    = crossover_data["freeze0_improvement"]
freeze4    = crossover_data["freeze4_improvement"]
crossover  = crossover_data["crossover_steps"]

ax_b.plot(steps, freeze0, color=COLORS["freeze0"], lw=2.0, marker="o", markersize=5,
          label="Freeze=0 (no anchor)")
ax_b.plot(steps, freeze4, color=COLORS["freeze4"], lw=2.0, marker="s", markersize=5,
          label="Freeze=4 layers")

ax_b.axvline(crossover, color=COLORS["crossover"], linestyle="--", lw=1.5, alpha=0.8)
y_annot = min(min(freeze0), min(freeze4)) + 0.5
ax_b.text(crossover * 1.12, y_annot,
          f"Crossover\n~{crossover // 1000}k steps",
          color=COLORS["crossover"], fontsize=9, va="bottom")

ax_b.set_xscale("log")
ax_b.set_xlabel("Training steps (log scale)")
ax_b.set_ylabel("MoE improvement over best specialist (%)")
ax_b.set_title("(B)  Training Duration Crossover: Freeze=0 vs Freeze=4")
ax_b.legend(loc="upper left")
clean_axes(ax_b)

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: Routing Failure Comparison
# ─────────────────────────────────────────────────────────────────────────────
routing_labels = [
    "MoE (ours)",
    "Classifier dispatch\n(99.3% acc.)",
    "Multi-head\n(same params)",
]
routing_values = [
    classifier_data["moe_improvement_pct"],
    classifier_data["classifier_improvement_pct"],
    multihead_data["multihead_improvement_pct"],
]
routing_colors = [COLORS["moe"], COLORS["classifier"], COLORS["multihead"]]

x_c = np.arange(len(routing_labels))
bars_c = ax_c.bar(
    x_c, routing_values,
    color=routing_colors, width=0.55, zorder=3,
)

ax_c.axhline(0, color="#374151", lw=1.0, linestyle="-")
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(routing_labels)
ax_c.set_ylabel("Improvement vs. base (%)")
ax_c.set_title("(C)  Routing Failure: Why Architecture Matters")

# accommodate negative bars
y_min = min(routing_values) * 1.2
y_max = max(routing_values) * 1.4
ax_c.set_ylim(y_min, y_max)
clean_axes(ax_c)

# label bars
for bar, val in zip(bars_c, routing_values):
    offset = 0.3 if val >= 0 else -1.2
    ax_c.text(
        bar.get_x() + bar.get_width() / 2,
        val + offset,
        f"{val:+.1f}%",
        ha="center", va="bottom" if val >= 0 else "top",
        fontsize=10, fontweight="bold", color="#1f2937",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: Monolithic Comparison
# ─────────────────────────────────────────────────────────────────────────────
mono_labels = ["Base", "Monolithic", "Best Specialist", "Weight Avg", "KALAVAI MoE"]
mono_values = [base_loss, monolithic_loss, best_spec_loss, weight_avg_loss, moe_loss_410m]
mono_colors = [
    COLORS["base"],
    COLORS["monolithic"],
    COLORS["code"],
    COLORS["weight_avg"],
    COLORS["moe"],
]

x_d = np.arange(len(mono_labels))
bars_d = ax_d.bar(
    x_d, mono_values,
    color=mono_colors, width=0.55, zorder=3,
)

ax_d.set_xticks(x_d)
ax_d.set_xticklabels(mono_labels, rotation=12, ha="right")
ax_d.set_ylabel("Equal-weight loss (lower is better)")
ax_d.set_title("(D)  Monolithic vs KALAVAI: Same Compute Budget (corrected eval)")
clean_axes(ax_d)

# label bars with loss values
y_range = max(mono_values) - min(mono_values)
for bar, val in zip(bars_d, mono_values):
    ax_d.text(
        bar.get_x() + bar.get_width() / 2,
        val + y_range * 0.015,
        f"{val:.3f}",
        ha="center", va="bottom", fontsize=9, color="#374151",
    )

ax_d.set_ylim(min(mono_values) * 0.97, max(mono_values) * 1.04)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()

out1 = REPO_ROOT / "figures" / "paper" / "fig_hero_4panel.png"
out2 = REPO_ROOT / "paper" / "figures" / "fig_hero_4panel.png"

fig.savefig(out1, dpi=300)
fig.savefig(out2, dpi=300)
plt.close(fig)

print(f"Saved: {out1}")
print(f"Saved: {out2}")
print(f"File sizes: {out1.stat().st_size // 1024} KB, {out2.stat().st_size // 1024} KB")
