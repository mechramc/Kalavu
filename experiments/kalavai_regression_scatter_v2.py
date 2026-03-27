"""
kalavai_regression_scatter_v2.py

Expanded divergence-gain regression scatter (n>=14).

Data sources:
  1. Main 6 points: results/analysis/regression_fit.json
  2. Crossover 8 points: results/pythia/crossover_regression_points.json
  3. 20-contributor 3 seeds: results/phase2/twenty_contributor/result_seed*.json

Outputs:
  - results/analysis/regression_scatter_v2.json  (all points + fit params)
  - paper/figures/fig_divergence_gain_scatter_v2.png
"""

import sys
import os
import json
import pathlib

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).resolve().parent.parent
MAIN_REGRESSION_JSON = REPO / "results" / "analysis" / "regression_fit.json"
CROSSOVER_JSON = REPO / "results" / "pythia" / "crossover_regression_points.json"
TWENTY_DIR = REPO / "results" / "phase2" / "twenty_contributor"
OUT_JSON = REPO / "results" / "analysis" / "regression_scatter_v2.json"
OUT_FIG = REPO / "paper" / "figures" / "fig_divergence_gain_scatter_v2.png"

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load main 6 points from regression_fit.json
# ---------------------------------------------------------------------------
with open(MAIN_REGRESSION_JSON, "r", encoding="utf-8") as f:
    fit_data = json.load(f)

main_points = []
for pt in fit_data["data"]:
    main_points.append({
        "label": pt["label"],
        "divergence_pct": float(pt["div"]),
        "gain_pct": float(pt["gain"]),
        "group": "main",
    })

print(f"[main] loaded {len(main_points)} points")
for pt in main_points:
    print(f"  {pt['label']:35s}  div={pt['divergence_pct']:6.2f}  gain={pt['gain_pct']:6.2f}")

# ---------------------------------------------------------------------------
# 2. Load crossover 8 points from crossover_regression_points.json
# ---------------------------------------------------------------------------
with open(CROSSOVER_JSON, "r", encoding="utf-8") as f:
    crossover_data = json.load(f)

crossover_points = []
for pt in crossover_data["regression_points"]:
    crossover_points.append({
        "label": pt["label"],
        "divergence_pct": float(pt["mean_divergence"]),
        "gain_pct": float(pt["gain_vs_spec_pct"]),
        "group": "crossover",
        "steps": pt["steps"],
    })

print(f"\n[crossover] loaded {len(crossover_points)} points")
for pt in crossover_points:
    print(f"  {pt['label']:35s}  div={pt['divergence_pct']:6.2f}  gain={pt['gain_pct']:6.2f}")

# ---------------------------------------------------------------------------
# 3. Load 20-contributor 3 seeds
# ---------------------------------------------------------------------------
twenty_seeds = [42, 137, 2026]
twenty_files = {
    42: "result_seed42_router_retry.json",
    137: "result_seed137.json",
    2026: "result_seed2026.json",
}

twenty_points = []
for seed in twenty_seeds:
    fpath = TWENTY_DIR / twenty_files[seed]
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data["metrics"]
    div = float(metrics["mean_divergence"])
    gain = float(metrics["improvement_vs_spec"])
    twenty_points.append({
        "label": f"20-contrib, seed={seed}",
        "divergence_pct": div,
        "gain_pct": gain,
        "group": "twenty_contributor",
        "seed": seed,
    })

print(f"\n[20-contributor] loaded {len(twenty_points)} points")
for pt in twenty_points:
    print(f"  {pt['label']:35s}  div={pt['divergence_pct']:6.2f}  gain={pt['gain_pct']:6.2f}")

# ---------------------------------------------------------------------------
# 4. Assemble all points
# ---------------------------------------------------------------------------
all_points = main_points + crossover_points + twenty_points
n = len(all_points)
print(f"\nTotal points: {n}")

divs = np.array([p["divergence_pct"] for p in all_points])
gains = np.array([p["gain_pct"] for p in all_points])

# ---------------------------------------------------------------------------
# 5. Linear regression
# ---------------------------------------------------------------------------
slope, intercept, r_value, p_value, std_err = stats.linregress(divs, gains)
r2 = r_value ** 2

formula = f"gain = {slope:.4f} * div + ({intercept:.4f})"
print(f"\nRegression: {formula}")
print(f"R² = {r2:.4f}   p = {p_value:.4e}   SE(slope) = {std_err:.4f}")

# ---------------------------------------------------------------------------
# 6. Save JSON
# ---------------------------------------------------------------------------
result = {
    "n_points": n,
    "formula": formula,
    "slope": slope,
    "intercept": intercept,
    "r2": r2,
    "p_value": p_value,
    "std_err_slope": std_err,
    "points": [
        {k: v for k, v in p.items()}
        for p in all_points
    ],
    "sources": {
        "main": str(MAIN_REGRESSION_JSON),
        "crossover": str(CROSSOVER_JSON),
        "twenty_contributor": str(TWENTY_DIR),
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved JSON -> {OUT_JSON}")

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

# Style config per group
GROUP_STYLE = {
    "main": {
        "color": "#1976D2",   # blue
        "marker": "o",
        "s": 90,
        "zorder": 4,
        "label": f"Main experiments (n={len(main_points)})",
        "alpha": 0.92,
    },
    "crossover": {
        "color": "#E65100",   # orange
        "marker": "^",
        "s": 75,
        "zorder": 3,
        "label": f"Crossover checkpoints (n={len(crossover_points)})",
        "alpha": 0.88,
    },
    "twenty_contributor": {
        "color": "#2E7D32",   # green
        "marker": "s",
        "s": 90,
        "zorder": 5,
        "label": f"20-contributor (n={len(twenty_points)})",
        "alpha": 0.95,
    },
}

# Scatter points
for group, style in GROUP_STYLE.items():
    pts = [p for p in all_points if p["group"] == group]
    xs = [p["divergence_pct"] for p in pts]
    ys = [p["gain_pct"] for p in pts]
    ax.scatter(
        xs, ys,
        color=style["color"],
        marker=style["marker"],
        s=style["s"],
        zorder=style["zorder"],
        alpha=style["alpha"],
        edgecolors="white",
        linewidths=0.5,
    )

# Regression line
x_min = divs.min() - 1.5
x_max = divs.max() + 1.5
x_line = np.linspace(x_min, x_max, 200)
y_line = slope * x_line + intercept
eq_str = (
    f"gain = {slope:.2f} × div + ({intercept:.2f})\n"
    f"$R^2$ = {r2:.3f},  $n$ = {n}"
)
ax.plot(x_line, y_line, color="#555555", linewidth=1.6, linestyle="--",
        label=eq_str, zorder=2)

# Axes labels
ax.set_xlabel("Mean Divergence from Base (%)", fontsize=12)
ax.set_ylabel("Gain vs. Best Specialist (%)", fontsize=12)
ax.set_title(f"Divergence Predicts Fusion Gain (n={n})", fontsize=13, fontweight="bold")

# Light grid
ax.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", alpha=0.6)
ax.set_axisbelow(True)

# Legend with custom markers for groups + regression line entry
handles = []
for group, style in GROUP_STYLE.items():
    h = plt.Line2D(
        [0], [0],
        marker=style["marker"],
        color="w",
        markerfacecolor=style["color"],
        markeredgecolor="white",
        markersize=9,
        label=style["label"],
    )
    handles.append(h)
# regression line
reg_line = plt.Line2D([0], [0], color="#555555", linewidth=1.6, linestyle="--", label=eq_str)
handles.append(reg_line)

ax.legend(handles=handles, fontsize=8.5, loc="upper left",
          framealpha=0.92, edgecolor="#CCCCCC")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Saved figure -> {OUT_FIG}")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("REGRESSION SUMMARY")
print("=" * 60)
print(f"  n         = {n}")
print(f"  formula   = {formula}")
print(f"  R²        = {r2:.4f}")
print(f"  p-value   = {p_value:.4e}")
print(f"  Groups: main={len(main_points)}, "
      f"crossover={len(crossover_points)}, "
      f"twenty_contributor={len(twenty_points)}")
print("=" * 60)
