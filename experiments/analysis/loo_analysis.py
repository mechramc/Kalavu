"""
Leave-One-Out (LOO) regression analysis on the 6-point divergence-gain dataset.

Dataset: mean specialist divergence (%) vs fusion gain over best specialist (%)
across 6 experimental conditions used in the KALAVAI paper crossover regression.

Fits OLS via closed-form (numpy polyfit) on 5-point subsets, predicts left-out
point, computes per-point residuals, and aggregates LOO-MAE statistics.

Saves results to results/analysis/loo_analysis.json.
"""

import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# (label, mean_specialist_divergence_pct, fusion_gain_pct)
DATA = [
    ("Qwen-1.5B",           3.16,  1.06),
    ("Pythia-6.9B",         8.73,  6.53),
    ("Pythia-1B",          15.28,  7.49),
    ("Pythia-410M",        15.65,  7.70),
    ("Exp2 (private)",     18.52, 10.17),
    ("Exp1 cross-lingual", 25.65, 21.76),  # 2-seed clean result
]

# Alternative cross-lingual value: 3-seed mean
CROSSLINGUAL_3SEED_GAIN = 16.55
CROSSLINGUAL_INDEX = 5  # 0-based index of the cross-lingual point

# Full 6-point OLS (reference)
FULL_SLOPE     =  0.82
FULL_INTERCEPT = -2.72
FULL_R2        =  0.856


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ols(xs: np.ndarray, ys: np.ndarray):
    """Return (slope, intercept) via least-squares closed form."""
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def predict(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def r_squared(xs: np.ndarray, ys: np.ndarray, slope: float, intercept: float) -> float:
    y_hat = slope * xs + intercept
    ss_res = float(np.sum((ys - y_hat) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# LOO analysis — primary dataset (2-seed cross-lingual = 21.76%)
# ---------------------------------------------------------------------------

labels = [d[0] for d in DATA]
divs   = np.array([d[1] for d in DATA])
gains  = np.array([d[2] for d in DATA])

n = len(DATA)
loo_records = []

for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False

    xs_train = divs[mask]
    ys_train = gains[mask]

    slope, intercept = ols(xs_train, ys_train)
    r2_loo = r_squared(xs_train, ys_train, slope, intercept)

    x_left  = divs[i]
    y_actual = gains[i]
    y_pred  = predict(slope, intercept, x_left)
    residual = y_actual - y_pred  # positive → actual above prediction

    loo_records.append({
        "label":            labels[i],
        "div_pct":          float(divs[i]),
        "gain_actual_pct":  float(y_actual),
        "loo_slope":        round(slope, 4),
        "loo_intercept":    round(intercept, 4),
        "loo_train_r2":     round(r2_loo, 4),
        "gain_predicted_pct": round(y_pred, 4),
        "residual_pct":     round(residual, 4),
        "abs_residual_pct": round(abs(residual), 4),
    })

abs_residuals = np.array([r["abs_residual_pct"] for r in loo_records])
loo_mae_all   = float(np.mean(abs_residuals))

# LOO-MAE excluding cross-lingual (5 English / professional conditions)
mask_5 = [i for i in range(n) if i != CROSSLINGUAL_INDEX]
loo_mae_5 = float(np.mean(abs_residuals[mask_5]))

# Largest residual
max_idx = int(np.argmax(abs_residuals))
max_record = loo_records[max_idx]

# ---------------------------------------------------------------------------
# Sensitivity analysis — 3-seed cross-lingual mean (16.55%)
# ---------------------------------------------------------------------------

data_3seed = list(DATA)
data_3seed[CROSSLINGUAL_INDEX] = (
    DATA[CROSSLINGUAL_INDEX][0],
    DATA[CROSSLINGUAL_INDEX][1],
    CROSSLINGUAL_3SEED_GAIN,
)
gains_3seed = np.array([d[2] for d in data_3seed])

loo_records_3seed = []
for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False

    xs_train  = divs[mask]
    ys_train  = gains_3seed[mask]

    slope, intercept = ols(xs_train, ys_train)
    r2_loo    = r_squared(xs_train, ys_train, slope, intercept)

    x_left    = divs[i]
    y_actual  = gains_3seed[i]
    y_pred    = predict(slope, intercept, x_left)
    residual  = y_actual - y_pred

    loo_records_3seed.append({
        "label":              data_3seed[i][0],
        "div_pct":            float(divs[i]),
        "gain_actual_pct":    float(y_actual),
        "loo_slope":          round(slope, 4),
        "loo_intercept":      round(intercept, 4),
        "loo_train_r2":       round(r2_loo, 4),
        "gain_predicted_pct": round(y_pred, 4),
        "residual_pct":       round(residual, 4),
        "abs_residual_pct":   round(abs(residual), 4),
    })

abs_residuals_3seed = np.array([r["abs_residual_pct"] for r in loo_records_3seed])
loo_mae_3seed       = float(np.mean(abs_residuals_3seed))
loo_mae_5_3seed     = float(np.mean(abs_residuals_3seed[mask_5]))

# ---------------------------------------------------------------------------
# Full-fit global OLS R² (verification)
# ---------------------------------------------------------------------------
slope_full, intercept_full = ols(divs, gains)
r2_full = r_squared(divs, gains, slope_full, intercept_full)

# ---------------------------------------------------------------------------
# Assemble JSON output
# ---------------------------------------------------------------------------
output = {
    "description": "Leave-one-out regression analysis on 6-point divergence–gain dataset",
    "dataset_note": "2-seed clean result for Exp1 cross-lingual = 21.76%",
    "full_ols": {
        "slope":     round(slope_full, 4),
        "intercept": round(intercept_full, 4),
        "r2":        round(r2_full, 4),
    },
    "loo_primary": {
        "crosslingual_gain_used_pct": 21.76,
        "per_condition": loo_records,
        "loo_mae_all_6_pct":        round(loo_mae_all, 4),
        "loo_mae_5_excl_crosslingual_pct": round(loo_mae_5, 4),
        "largest_residual": {
            "label":        max_record["label"],
            "residual_pct": max_record["residual_pct"],
        },
    },
    "loo_sensitivity_3seed": {
        "crosslingual_gain_used_pct": CROSSLINGUAL_3SEED_GAIN,
        "note": "Cross-lingual point replaced with 3-seed mean (16.55%); all other points identical",
        "per_condition": loo_records_3seed,
        "loo_mae_all_6_pct":        round(loo_mae_3seed, 4),
        "loo_mae_5_excl_crosslingual_pct": round(loo_mae_5_3seed, 4),
    },
}

out_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "analysis"
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "loo_analysis.json")

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {out_path}\n")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
HEADER = f"{'Condition':<26} {'Div%':>6} {'Actual':>8} {'Pred':>8} {'Resid':>8} {'|Resid|':>8}"
SEP    = "-" * len(HEADER)

print("=" * len(HEADER))
print("LOO REGRESSION ANALYSIS — KALAVAI divergence–gain dataset")
print("=" * len(HEADER))
print(f"\nFull OLS (all 6): gain = {intercept_full:+.4f} + {slope_full:.4f} × div   R²={r2_full:.3f}")
print(f"  (Paper values:  gain = -2.72        + 0.82         × div   R²=0.856)\n")

print("PRIMARY DATASET  (cross-lingual = 21.76%, 2-seed clean)")
print(SEP)
print(HEADER)
print(SEP)
for r in loo_records:
    flag = "  <-- max" if r["label"] == max_record["label"] else ""
    print(
        f"{r['label']:<26} {r['div_pct']:>6.2f} "
        f"{r['gain_actual_pct']:>8.2f} "
        f"{r['gain_predicted_pct']:>8.2f} "
        f"{r['residual_pct']:>+8.2f} "
        f"{r['abs_residual_pct']:>8.2f}"
        f"{flag}"
    )
print(SEP)
print(f"  LOO-MAE (all 6 conditions)          : {loo_mae_all:.4f} pp")
print(f"  LOO-MAE (5 excl. cross-lingual)      : {loo_mae_5:.4f} pp")

print()
print("SENSITIVITY: cross-lingual = 16.55% (3-seed mean)")
print(SEP)
print(HEADER)
print(SEP)
for r in loo_records_3seed:
    print(
        f"{r['label']:<26} {r['div_pct']:>6.2f} "
        f"{r['gain_actual_pct']:>8.2f} "
        f"{r['gain_predicted_pct']:>8.2f} "
        f"{r['residual_pct']:>+8.2f} "
        f"{r['abs_residual_pct']:>8.2f}"
    )
print(SEP)
print(f"  LOO-MAE (all 6 conditions)          : {loo_mae_3seed:.4f} pp")
print(f"  LOO-MAE (5 excl. cross-lingual)      : {loo_mae_5_3seed:.4f} pp")

print()
print("LARGEST LOO RESIDUAL (primary dataset):")
print(f"  Condition : {max_record['label']}")
print(f"  Actual    : {max_record['gain_actual_pct']:.2f}%")
print(f"  Predicted : {max_record['gain_predicted_pct']:.2f}%")
print(f"  Residual  : {max_record['residual_pct']:+.2f}pp")
print()
print("RECOMMENDED PAPER SENTENCE:")
paper_mae = loo_mae_5
paper_mae_rounded = round(paper_mae * 2) / 2  # round to nearest 0.5pp
print(
    f"  \"The divergence–gain relationship functions as an empirical heuristic, "
    f"validated by leave-one-out cross-validation with a mean prediction error of "
    f"±{paper_mae:.1f}pp across the five English and professional-domain conditions "
    f"(LOO-MAE={paper_mae:.2f}pp; the sixth, cross-lingual condition contributes a "
    f"larger residual of {abs_residuals[CROSSLINGUAL_INDEX]:.2f}pp, reflecting "
    f"its linguistic distributional shift).\""
)
print("=" * len(HEADER))
