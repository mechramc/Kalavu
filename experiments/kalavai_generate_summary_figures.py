#!/usr/bin/env python3
"""
Generate two summary figures:
  1. fig_6b_summary.png       — 6.9B fusion bar chart (averaged across 3 seeds)
  2. fig_scale_ladder.png     — improvement % vs model size (410M, 1B, 6.9B)

NOTE: Re-run with 2.8B data once kalavai_pythia_2b_experiment.py completes
to get the 4-point scale ladder.
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path("figures/pythia")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

# 6.9B per-seed results
seeds = [42, 137, 2026]
base_losses, best_ind_losses, weight_avg_losses, moe_losses, imps = [], [], [], [], []
for seed in seeds:
    d = json.loads(Path(f"results/pythia_6b/step6_fusion_seed{seed}.json").read_text())
    h = d["eval_heldout"]
    base_losses.append(h["base"]["mixed"])
    best_ind_losses.append(d["best_individual_mixed"])
    weight_avg_losses.append(h["weight_avg"]["mixed"])
    moe_losses.append(d["moe_mixed"])
    imps.append(d["improvement_pct"])

base_mean    = np.mean(base_losses)
bestind_mean = np.mean(best_ind_losses)
wavg_mean    = np.mean(weight_avg_losses)
moe_mean     = np.mean(moe_losses)
imp_mean     = np.mean(imps)
imp_std      = np.std(imps)

# 410M
d410  = json.loads(Path("results/pythia/step5_final_summary.json").read_text())
imp_410m     = d410["summary"]["improvement_mean_pct"]
std_410m     = d410["summary"]["improvement_std_pct"]

# 1B
d1b   = json.loads(Path("results/pythia/pythia_1b/main_result_summary.json").read_text())
imp_1b       = d1b["summary"]["improvement_mean_pct"]
std_1b       = d1b["summary"]["improvement_std_pct"]

# ── Figure 1: 6.9B summary bar chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

labels = ["Base\n(step10000)", "Best\nIndividual", "Weight\nAverage", "MoE Fusion"]
means  = [base_mean, bestind_mean, wavg_mean, moe_mean]
colors = ["#95a5a6", "#3498db", "#f39c12", "#9b59b6"]

y_min = min(means) * 0.985
y_max = max(means) * 1.008
bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5)
ax.set_ylim(y_min, y_max)

# Improvement annotations
for bar, val, ref in zip(bars, means, [None, base_mean, base_mean, base_mean]):
    if ref is not None:
        imp = (ref - val) / ref * 100
        sign = "+" if imp >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (y_max - y_min) * 0.003,
                f"{sign}{imp:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Held-Out Mixed Loss", fontsize=11)
ax.set_title(f"KALAVAI Fusion at Scale: Pythia-6.9B\n"
             f"MoE improvement = +{imp_mean:.2f}% ± {imp_std:.2f}% (3 seeds)", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
ax.text(0.98, 0.02, "Seeds: 42, 137, 2026 | step10000 | freeze=6/32",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="gray")

fig.tight_layout()
path1 = FIGURES_DIR / "fig_6b_summary.png"
fig.savefig(path1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path1}")

# ── Figure 2: Scale ladder (3-point) ─────────────────────────────────────────
# NOTE: Add 2.8B data point here once kalavai_pythia_2b_experiment.py completes.
# Expected 4th point: ~410M params * 6.9 = 2.8B, run same config on RunPod.

fig, ax = plt.subplots(figsize=(8, 5))

model_sizes  = [0.41,   1.0,    6.9]
improvements = [imp_410m, imp_1b, imp_mean]
stds         = [std_410m,  std_1b, imp_std]
labels_s     = ["Pythia-410M", "Pythia-1B", "Pythia-6.9B"]
colors_s     = ["#3498db", "#e74c3c", "#9b59b6"]

for x, y, e, label, c in zip(model_sizes, improvements, stds, labels_s, colors_s):
    ax.errorbar(x, y, yerr=e, fmt="o", color=c, markersize=10,
                capsize=5, linewidth=2, label=label)

ax.plot(model_sizes, improvements, "--", color="gray", alpha=0.5, linewidth=1.5)
ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

ax.set_xscale("log")
ax.set_xticks(model_sizes)
ax.set_xticklabels(["410M", "1B", "6.9B"])
ax.set_xlabel("Model Size (parameters, log scale)", fontsize=11)
ax.set_ylabel("MoE Fusion Improvement (%)", fontsize=11)
ax.set_title("KALAVAI Scale Ladder: Fusion Benefit vs Model Size\n"
             "(3-point — add 2.8B for 4-point version)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate each point
for x, y, label in zip(model_sizes, improvements, [f"+{imp_410m:.1f}%", f"+{imp_1b:.1f}%", f"+{imp_mean:.2f}%"]):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, 5), fontsize=9)

fig.tight_layout()
path2 = FIGURES_DIR / "fig_scale_ladder.png"
fig.savefig(path2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path2}")
print(f"\nScale ladder data:")
print(f"  410M:  +{imp_410m:.2f}% ± {std_410m:.3f}%")
print(f"  1B:    +{imp_1b:.2f}% ± {std_1b:.3f}%")
print(f"  6.9B:  +{imp_mean:.2f}% ± {imp_std:.3f}%")
print(f"\nNOTE: Re-run with 2.8B point once RunPod scale ladder completes.")
