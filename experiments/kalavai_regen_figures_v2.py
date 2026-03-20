#!/usr/bin/env python3
"""
Regenerate stale figures using corrected v2 evaluation data.
Reads from results/pythia/v2/ and results/pythia_6b/corrected_eval_6b_summary.json.
Saves to figures/pythia/ and docs/kalavai/figures/ (synced).
"""
import json, statistics, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).parent.parent
V2     = ROOT / "results" / "pythia" / "v2"
R6B    = ROOT / "results" / "pythia_6b"
FIGDIR = ROOT / "figures" / "pythia"
WEBDIR = ROOT / "docs" / "kalavai" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

COLORS = {
    "base": "#94a3b8", "moe": "#ef4444", "code": "#3b82f6",
    "science": "#10b981", "fiction": "#f59e0b",
    "weight_avg": "#8b5cf6", "freeze0": "#3b82f6", "freeze4": "#ef4444",
}

def load(p):
    with open(p) as f: return json.load(f)

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5)

def save(name):
    path = FIGDIR / name
    plt.savefig(path)
    plt.close()
    web = WEBDIR / name
    if web.exists() or WEBDIR.exists():
        import shutil; shutil.copy(path, WEBDIR / name)
    print(f"  Saved: {name}")

# Step → training % maps
PCT_410M = {"step5000": 3.5, "step10000": 7.0, "step20000": 14.0,
            "step50000": 35.0, "step100000": 70.0, "step143000": 100.0}
PCT_1B   = {"step5000": 3.5, "step20000": 14.0, "step50000": 35.0, "step143000": 100.0}


# ── 1. fig_training_duration_crossover ────────────────────────────────────────
print("fig_training_duration_crossover.png")
d = load(V2 / "crossover_v2.json")
steps_f0, imp_f0, steps_f4, imp_f4 = [], [], [], []
for r in d["results"]:
    if r["freeze"] == 0:
        steps_f0.append(r["steps"]); imp_f0.append(r["improvement_vs_spec"])
    else:
        steps_f4.append(r["steps"]); imp_f4.append(r["improvement_vs_spec"])

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(steps_f0, imp_f0, color=COLORS["freeze0"], linestyle="--",
        linewidth=2, marker="o", markersize=5, label="freeze=0")
ax.plot(steps_f4, imp_f4, color=COLORS["freeze4"], linestyle="-",
        linewidth=2, marker="s", markersize=5, label="freeze=4")
ax.axvline(5000, color="#10b981", linestyle=":", linewidth=1.5, alpha=0.8)
ax.annotate("crossover\n@5,000 steps",
            xy=(5000, max(imp_f0 + imp_f4) * 0.65),
            xytext=(7000, max(imp_f0 + imp_f4) * 0.65),
            fontsize=8, color="#374151",
            arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1))
ax.set_xscale("log")
ax.set_xlabel("Specialist Training Steps (log scale)")
ax.set_ylabel("Improvement vs. Best Specialist (%)")
ax.set_title("Training Duration vs Fusion Improvement\n(corrected equal-weight eval)")
ax.legend(fontsize=8)
clean_axes(ax)
fig.tight_layout(pad=1.5)
save("fig_training_duration_crossover.png")


# ── 2. fig_ablation_freeze ────────────────────────────────────────────────────
print("fig_ablation_freeze.png")
d = load(V2 / "freeze_ablation_v2.json")
# seed=42 results at 2k steps, sorted by freeze depth
seed42 = sorted([r for r in d["results"] if r.get("seed") == 42],
                key=lambda r: r["freeze"])
xs = [r["freeze"] for r in seed42]
ys = [r["improvement_vs_spec"] for r in seed42]

fig, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(xs, ys, yerr=[0]*len(xs),
            color=COLORS["moe"], marker="o", markersize=6,
            linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
            label="MoE improvement % (seed=42)")
ax.axvline(x=4, color="#6b7280", linestyle=":", linewidth=1.5, alpha=0.7)
ax.annotate("freeze=4\n(default)", xy=(4, ys[xs.index(4)]),
            xytext=(5.5, ys[xs.index(4)] + 0.3),
            fontsize=8, color="#374151",
            arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1))
ax.set_xlabel("Frozen Layers")
ax.set_ylabel("Improvement vs. Best Specialist (%)")
ax.set_title("Freeze Depth Ablation\n(410M, seed=42, 2,000 steps, corrected eval)")
ax.set_xticks(xs)
clean_axes(ax)
ax.legend(fontsize=8)
fig.tight_layout(pad=1.5)
save("fig_ablation_freeze.png")


# ── 3. fig_maturity_curve_410m ────────────────────────────────────────────────
print("fig_maturity_curve_410m.png")
d = load(V2 / "maturity_sweep_410m_v2.json")
by_step = collections.defaultdict(list)
for r in d["results"]:
    by_step[r["step"]].append(r["improvement_vs_spec"])
steps_sorted = sorted(by_step, key=lambda s: PCT_410M.get(s, 0))
xs = [PCT_410M[s] for s in steps_sorted]
ys = [statistics.mean(by_step[s]) for s in steps_sorted]
errs = [statistics.stdev(by_step[s]) if len(by_step[s]) > 1 else 0.0
        for s in steps_sorted]

fig, ax = plt.subplots(figsize=(5, 4))
ax.axvspan(0, 20, alpha=0.08, color=COLORS["science"], label="KALAVAI target zone")
ax.errorbar(xs, ys, yerr=errs, color=COLORS["moe"], marker="o", markersize=6,
            linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
            label="410M MoE improvement %")
ax.set_xlabel("Base Model Training (%)")
ax.set_ylabel("Improvement vs. Best Specialist (%)")
ax.set_title("Maturity Sweep — Pythia-410M\n(corrected equal-weight eval)")
ax.legend(fontsize=8)
clean_axes(ax)
fig.tight_layout(pad=1.5)
save("fig_maturity_curve_410m.png")


# ── 4. fig_maturity_curve_1b ──────────────────────────────────────────────────
print("fig_maturity_curve_1b.png")
d = load(V2 / "maturity_sweep_1b_v2.json")
by_step_1b = collections.defaultdict(list)
for r in d["results"]:
    by_step_1b[r["step"]].append(r["improvement_vs_spec"])
steps_1b = sorted(by_step_1b, key=lambda s: PCT_1B.get(s, 0))
xs_1b  = [PCT_1B[s] for s in steps_1b]
ys_1b  = [statistics.mean(by_step_1b[s]) for s in steps_1b]
errs_1b = [statistics.stdev(by_step_1b[s]) if len(by_step_1b[s]) > 1 else 0.0
           for s in steps_1b]

fig, ax = plt.subplots(figsize=(5, 4))
ax.axvspan(0, 20, alpha=0.08, color=COLORS["science"], label="KALAVAI target zone")
ax.errorbar(xs_1b, ys_1b, yerr=errs_1b, color=COLORS["moe"], marker="s", markersize=6,
            linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2,
            label="1B MoE improvement %")
# annotate the near-zero point
ax.annotate(f"step143k\n(+0.40%)", xy=(100, ys_1b[-1]),
            xytext=(75, ys_1b[-1] + 1.5), fontsize=8, color="#6b7280",
            arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1))
ax.set_xlabel("Base Model Training (%)")
ax.set_ylabel("Improvement vs. Best Specialist (%)")
ax.set_title("Maturity Sweep — Pythia-1B\n(corrected equal-weight eval)")
ax.legend(fontsize=8)
clean_axes(ax)
fig.tight_layout(pad=1.5)
save("fig_maturity_curve_1b.png")


# ── 5. fig_maturity_curve_combined ───────────────────────────────────────────
print("fig_maturity_curve_combined.png")
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(0, 20, alpha=0.08, color=COLORS["science"], label="KALAVAI target zone")
ax.errorbar(xs, ys, yerr=errs,
            color=COLORS["code"], marker="o", markersize=5, linewidth=2,
            capsize=4, capthick=1.5, elinewidth=1.2, label="410M")
ax.errorbar(xs_1b, ys_1b, yerr=errs_1b,
            color=COLORS["moe"], marker="s", markersize=5, linewidth=2,
            capsize=4, capthick=1.5, elinewidth=1.2, label="1B")
# Qwen reference point
ax.scatter([100], [1.06], color="#9ca3af", marker="D", s=60, zorder=5,
           label="Qwen-1.5B (+1.06%)")
ax.set_xlabel("Base Model Training (%)")
ax.set_ylabel("Improvement vs. Best Specialist (%)")
ax.set_title("Maturity Sweep — 410M vs 1B\n(corrected equal-weight eval)")
ax.legend(fontsize=8)
clean_axes(ax)
fig.tight_layout(pad=1.5)
save("fig_maturity_curve_combined.png")


# ── 6. fig_maturity_curve_comparison_bar ─────────────────────────────────────
print("fig_maturity_curve_comparison_bar.png")
# step10000 (7%) and step143000 (100%) for both models
early_410 = statistics.mean(by_step["step10000"])
late_410  = statistics.mean(by_step["step143000"])
early_1b  = statistics.mean(by_step_1b["step5000"])   # earliest for 1B
late_1b   = statistics.mean(by_step_1b["step143000"])

bar_labels = ["410M\nearly (7%)", "410M\nfull (100%)",
              "1B\nearly (3.5%)", "1B\nfull (100%)"]
bar_vals   = [early_410, late_410, early_1b, late_1b]
bar_colors = [COLORS["code"], COLORS["code"], COLORS["moe"], COLORS["moe"]]
bar_alphas = [0.9, 0.5, 0.9, 0.5]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(bar_labels, bar_vals, color=bar_colors,
              alpha=0.85, width=0.55, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylabel("Improvement vs. Best Specialist (%)")
ax.set_title("Maturity: Early vs. Fully-Trained Base\n(corrected equal-weight eval)")
ax.set_ylim(0, max(bar_vals) * 1.15)
clean_axes(ax)
fig.tight_layout(pad=1.5)
save("fig_maturity_curve_comparison_bar.png")


# ── 7. fig_scale_ladder ───────────────────────────────────────────────────────
print("fig_scale_ladder.png")
# Corrected equal-weight improvement values (authoritative)
SCALE = [
    (410e6, 7.72, 0.02, "410M"),
    (1e9,   7.49, 0.01, "1B"),
    (6.9e9, 6.53, 0.024, "6.9B"),
]
sizes = [s[0] for s in SCALE]
means = [s[1] for s in SCALE]
stds  = [s[2] for s in SCALE]
lbls  = [s[3] for s in SCALE]

fig, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(sizes, means, yerr=stds, color=COLORS["moe"], marker="o", markersize=8,
            linewidth=2, capsize=5, capthick=1.5, elinewidth=1.2)
for sz, m, lbl in zip(sizes, means, lbls):
    ax.annotate(lbl, xy=(sz, m), xytext=(sz * 1.15, m + 0.05),
                fontsize=9, color="#374151")
ax.set_xscale("log")
ax.set_xlabel("Model Size (parameters, log scale)")
ax.set_ylabel("Improvement vs. Best Specialist (%)")
ax.set_title("Scale Ladder — MoE Improvement vs Model Size\n(corrected equal-weight eval)")
clean_axes(ax)
fig.tight_layout(pad=1.5)
save("fig_scale_ladder.png")


# ── 8. fig_6b_summary ────────────────────────────────────────────────────────
print("fig_6b_summary.png")
d6b = load(R6B / "corrected_eval_6b_summary.json")
base_vals = [r["per_model_ew"]["base"]      for r in d6b["results"]]
best_vals = [r["best_spec_ew"]              for r in d6b["results"]]
wavg_vals = [r["per_model_ew"]["weight_avg"] for r in d6b["results"]]
moe_vals  = [r["moe_ew"]                    for r in d6b["results"]]

labels_6b = ["Base", "Best Specialist", "Weight Avg", "MoE"]
vals_6b   = [statistics.mean(base_vals), statistics.mean(best_vals),
             statistics.mean(wavg_vals), statistics.mean(moe_vals)]
errs_6b   = [0, 0, 0, statistics.stdev(moe_vals)]
colors_6b = [COLORS["base"], COLORS["code"], COLORS["weight_avg"], COLORS["moe"]]

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(labels_6b, vals_6b, color=colors_6b, width=0.6,
              edgecolor="white", linewidth=0.8,
              yerr=errs_6b, capsize=5, error_kw={"linewidth": 1.5})
ymin = min(vals_6b) * 0.985
ymax = max(vals_6b) * 1.005
ax.set_ylim(ymin, ymax)
clean_axes(ax)
base_ew = vals_6b[0]
for bar, v in zip(bars, vals_6b):
    imp = (base_ew - v) / base_ew * 100
    lbl = f"{v:.4f}"
    if abs(imp) > 0.01:
        lbl += f"\n({imp:+.2f}%)"
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (ymax - ymin) * 0.003,
            lbl, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_title("Fusion Methods — Pythia-6.9B (3 seeds)\n(corrected equal-weight eval)")
ax.set_ylabel("EW Loss (code+science+fiction)/3 — lower is better")
fig.tight_layout(pad=1.5)
save("fig_6b_summary.png")


# ── 9. fig_divergence_gain_regression ────────────────────────────────────────
print("fig_divergence_gain_regression.png  (paper/figures/)")
# Data points: (div%, gain%, label)
POINTS = [
    (3.16,  1.06,  "Qwen-1.5B"),
    (8.29,  6.53,  "Pythia-6.9B"),   # corrected
    (15.28, 7.49,  "Pythia-1B"),
    (15.65, 7.72,  "Pythia-410M"),
    (18.52, 10.17, "Private-domain"),
    (25.65, 21.76, "Cross-lingual"),
]
OOS = (15.71, 16.79, "Exp3 20-contributor\n(out-of-sample)")

divs  = [p[0] for p in POINTS]
gains = [p[1] for p in POINTS]

# OLS fit
x = np.array(divs); y = np.array(gains)
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(0, 30, 200)
y_line = m * x_line + b

# 95% prediction band
n = len(x)
x_mean = x.mean()
se_slope = np.sqrt(np.sum((y - (m*x+b))**2) / (n-2) / np.sum((x - x_mean)**2))
se_int = se_slope * np.sqrt(np.sum(x**2) / n)
y_err = 2.776 * np.sqrt(se_slope**2 * (x_line - x_mean)**2 + (np.sum((y-(m*x+b))**2)/(n-2)) * (1 + 1/n + (x_line-x_mean)**2/np.sum((x-x_mean)**2)))
ss_res = np.sum((y - (m*x+b))**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res/ss_tot

fig, ax = plt.subplots(figsize=(6, 5))
ax.fill_between(x_line, y_line - y_err, y_line + y_err,
                alpha=0.10, color=COLORS["moe"], label="95% prediction band")
ax.plot(x_line, y_line, color=COLORS["moe"], linewidth=2,
        label=f"gain = {m:.2f}×div {b:+.2f}  (R²={r2:.3f})")
for div, gain, lbl in POINTS:
    ax.scatter(div, gain, color=COLORS["code"], s=70, zorder=5)
    ax.annotate(lbl, xy=(div, gain), xytext=(div + 0.4, gain + 0.3),
                fontsize=8, color="#374151")
# OOS point
ax.scatter(OOS[0], OOS[1], color=COLORS["science"], marker="*", s=120,
           zorder=6, label="Exp 3 (out-of-sample)")
ax.annotate(OOS[2], xy=(OOS[0], OOS[1]),
            xytext=(OOS[0] + 0.5, OOS[1] - 2.0),
            fontsize=8, color="#059669")
ax.set_xlabel("Mean Specialist Divergence (%)")
ax.set_ylabel("Fusion Gain vs. Best Specialist (%)")
ax.set_title("Divergence–Gain Relationship\n(corrected equal-weight eval, 6.9B updated)")
ax.legend(fontsize=8)
clean_axes(ax)
fig.tight_layout(pad=1.5)
out = ROOT / "figures" / "paper" / "fig_divergence_gain_regression.png"
plt.savefig(out)
plt.close()
print(f"  Saved: figures/paper/fig_divergence_gain_regression.png")


# ── 10. fig_monolithic_comparison ────────────────────────────────────────────
print("fig_monolithic_comparison.png")
dmono = load(ROOT / "results" / "pythia" / "v2" / "monolithic_v2.json")
m = dmono["models"]
# MoE EW from corrected_eval_seed42 (seed 42, 3-domain equal-weight)
base_ew  = m["base"]["equal_weight_avg"]           # 2.6510
mono_ew  = m["monolithic_seed42"]["equal_weight_avg"]  # 2.2288
best_spec_ew = 2.403864                             # fiction_spec (best at seed42)
moe_ew   = 2.218327                                 # seed 42 corrected
wavg_ew  = m["weight_avg_seed42"]["equal_weight_avg"] if "weight_avg_seed42" in m else 2.485715

labels_m = ["Base", "Best Specialist", "Weight Avg", "Monolithic\n(6k steps)", "MoE Fused"]
vals_m   = [base_ew, best_spec_ew, wavg_ew, mono_ew, moe_ew]
colors_m = [COLORS["base"], COLORS["code"], COLORS["weight_avg"], "#6b7280", COLORS["moe"]]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels_m, vals_m, color=colors_m, width=0.55,
              edgecolor="white", linewidth=0.8)
ymin = min(vals_m) * 0.985; ymax = max(vals_m) * 1.005
ax.set_ylim(ymin, ymax)
clean_axes(ax)
for bar, v in zip(bars, vals_m):
    imp = (base_ew - v) / base_ew * 100
    lbl = f"{v:.4f}" + (f"\n({imp:+.1f}%)" if abs(imp) > 0.01 else "")
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (ymax-ymin)*0.003,
            lbl, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_ylabel("EW Loss (code+science+fiction)/3 — lower is better")
ax.set_title("Equal-Compute Comparison — Pythia-410M\n(corrected equal-weight eval, seed 42)")
fig.tight_layout(pad=1.5)
save("fig_monolithic_comparison.png")


print("\nAll stale figures regenerated.")
