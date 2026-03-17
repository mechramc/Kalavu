#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
"""
KALAVAI: Paper Hero Figure
=========================
2x2 subplot figure using real data from all completed experiments.
Reads actual JSON formats from each results file.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS = Path("results/pythia")
FIGS    = Path("figures/pythia")
FIGS.mkdir(parents=True, exist_ok=True)


def load(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    C_410M  = "#2980b9"
    C_1B    = "#e74c3c"
    C_SCALE = "#8e44ad"
    C_FREEZE= "#e67e22"
    C_BARS  = ["#95a5a6", "#f39c12", "#2980b9", "#8e44ad"]

    # ── Panel A: Maturity curves (410M + 1B) ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    ms410 = load(RESULTS / "maturity_sweep_410m/summary.json")
    if ms410:
        curve = ms410["curve"]
        xs   = [c["training_pct"] for c in curve]
        # Use multiseed mean where available, else seed42 value
        ys   = [c["multiseed"]["mean"] if c.get("multiseed") else c["improvement_pct_seed42"]
                for c in curve]
        errs = [c["multiseed"]["std"] if c.get("multiseed") else 0 for c in curve]
        ax1.errorbar(xs, ys, yerr=errs, fmt="o-", color=C_410M,
                     linewidth=2, markersize=7, capsize=4, label="Pythia-410M")

    ms1b = load(RESULTS / "pythia_1b/maturity_sweep/summary.json")
    main1b = load(RESULTS / "pythia_1b/main_result_summary.json")
    if ms1b:
        checkpoints = ms1b["checkpoints"]
        xs1b = [c["training_pct"] for c in checkpoints]
        ys1b = [c["improvement_pct"] for c in checkpoints]
        # Insert step10000 from main 1B experiment
        if main1b:
            imp10k = main1b.get("summary", {}).get("improvement_mean_pct")
            if imp10k:
                xs1b.append(7.0)
                ys1b.append(imp10k)
        pairs = sorted(zip(xs1b, ys1b))
        xs1b = [p[0] for p in pairs]
        ys1b = [p[1] for p in pairs]
        ax1.plot(xs1b, ys1b, "s-", color=C_1B, linewidth=2, markersize=7, label="Pythia-1B")

    # Qwen reference point
    ax1.scatter([100], [-1.0], marker="D", color="#7f8c8d", s=90, zorder=5,
                label="Qwen-1.5B (full train)")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.axvspan(0, 20, alpha=0.07, color="green")
    ax1.text(10, ax1.get_ylim()[0] + 0.5 if ax1.get_ylim()[0] < -2 else -1.5,
             "KALAVAI\ntarget", ha="center", fontsize=7, color="green", alpha=0.7)
    ax1.set_xlabel("Base model training progress (%)", fontsize=10)
    ax1.set_ylabel("Fusion improvement over base (%)", fontsize=10)
    ax1.set_title("A  Maturity Curves: When Does Fusion Help?", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel B: Specialist count scaling ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    fd = load(RESULTS / "five_domain/summary.json")
    if fd:
        scaling = fd["aggregate_scaling"]
        ns    = sorted(int(k.split("_")[0]) for k in scaling)
        means = [scaling[f"{n}_specialists"]["improvement_mean_pct"] for n in ns]
        stds  = [scaling[f"{n}_specialists"]["improvement_std_pct"] for n in ns]
        ax2.errorbar(ns, means, yerr=stds, fmt="o-", color=C_SCALE,
                     linewidth=2.5, markersize=9, capsize=5,
                     markerfacecolor="white", markeredgewidth=2)
        ax2.set_xticks(ns)
        ax2.set_xlim(1.5, 5.5)
        for n, m in zip(ns, means):
            ax2.annotate(f"+{m:.1f}%", (n, m), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Number of specialists fused", fontsize=10)
    ax2.set_ylabel("Fusion improvement over base (%)", fontsize=10)
    ax2.set_title("B  Specialist Count Scaling (Pythia-410M)", fontweight="bold", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ── Panel C: Freeze depth sweep ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    fz = load(RESULTS / "ablation_freeze_summary.json")
    if fz:
        phase1 = fz["phase1_seed42_results"]
        ms_r   = fz.get("multi_seed_results", {})
        fzs  = [r["freeze_layers"] for r in phase1]
        imps = [r["improvement_pct"] for r in phase1]
        errs = [ms_r.get(str(f), {}).get("std", 0) for f in fzs]
        ax3.errorbar(fzs, imps, yerr=errs, fmt="o-", color=C_FREEZE,
                     linewidth=2, markersize=7, capsize=4)
        ax3.set_xticks(fzs)
        ax3.axvline(4, color="green", linestyle=":", linewidth=1.5, alpha=0.8,
                    label="Main exp (freeze=4)")
        ax3.legend(fontsize=8)
        # Annotate spread
        spread = imps[0] - imps[-1]
        ax3.annotate(f"{spread:.1f}pp spread\n→ robust to\nfreeze depth",
                     xy=(fzs[-1], imps[-1]), xytext=(6, imps[-1]+0.8),
                     fontsize=7.5, color="#7f8c8d",
                     arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    ax3.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Frozen layers (out of 24)", fontsize=10)
    ax3.set_ylabel("Fusion improvement over base (%)", fontsize=10)
    ax3.set_title("C  Freeze Depth Robustness (Pythia-410M)", fontweight="bold", fontsize=11)
    ax3.grid(True, alpha=0.3)

    # ── Panel D: Router ablation ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    ra = load(RESULTS / "ablation_router_summary.json")
    if ra:
        variants = ra["variants"]
        best_ind_loss = ra["best_individual_mixed"]
        labels = ["Best\nIndividual", "Uniform\nRouting", "Simple\nLinear", "2-Layer\nMLP"]
        losses = [
            best_ind_loss,
            variants["uniform"]["mixed_loss"],
            variants["simple_linear"]["mixed_loss"],
            variants["two_layer"]["mixed_loss"],
        ]
        imps = [
            0.0,
            variants["uniform"]["improvement_pct"],
            variants["simple_linear"]["improvement_pct"],
            variants["two_layer"]["improvement_pct"],
        ]
        y_min = min(losses) * 0.993
        y_max = max(losses) * 1.007
        bars = ax4.bar(labels, losses, color=C_BARS, alpha=0.85, width=0.55,
                       edgecolor="white", linewidth=0.5)
        ax4.set_ylim(y_min, y_max)
        for bar, imp in zip(bars, imps):
            tag = f"+{imp:.1f}%" if imp > 0.01 else ("base" if imp == 0 else f"{imp:.1f}%")
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (y_max - y_min) * 0.006,
                     tag, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        # Bracket showing learned == learned
        x2 = bars[2].get_x() + bars[2].get_width() / 2
        x3 = bars[3].get_x() + bars[3].get_width() / 2
        y_br = y_min + (y_max - y_min) * 0.03
        ax4.annotate("", xy=(x3, y_br), xytext=(x2, y_br),
                     arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
        ax4.text((x2 + x3) / 2, y_br + (y_max - y_min) * 0.01,
                 "identical", ha="center", fontsize=7.5, color="#555")
    ax4.set_ylabel("Held-Out Mixed Loss (lower=better)", fontsize=10)
    ax4.set_title("D  Router Architecture Ablation (Pythia-410M)", fontweight="bold", fontsize=11)
    ax4.grid(True, axis="y", alpha=0.3)

    fig.suptitle("KALAVAI: Cooperative LLM Fusion — Complete Results",
                 fontsize=14, fontweight="bold", y=1.01)

    out = FIGS / "fig_paper_hero.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
