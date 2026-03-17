#!/usr/bin/env python3
"""
Shared publication-quality figure style for KALAVU.
Import this at the top of any figure-generating script.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "base":       "#94a3b8",   # slate gray
    "code":       "#3b82f6",   # blue
    "science":    "#10b981",   # emerald
    "fiction":    "#f59e0b",   # amber
    "weight_avg": "#8b5cf6",   # violet
    "moe":        "#ef4444",   # red (protagonist)
    "monolithic": "#6b7280",   # gray
    "classifier": "#f97316",   # orange
    "multihead":  "#06b6d4",   # cyan
    "wider":      "#84cc16",   # lime
    "freeze0":    "#3b82f6",   # blue
    "freeze4":    "#ef4444",   # red
    "crossover":  "#10b981",   # emerald
}

FONT_FAMILY = ["Inter", "Helvetica Neue", "DejaVu Sans", "Arial", "sans-serif"]

# ── Apply global rcParams ──────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "font.family":           "sans-serif",
        "font.sans-serif":       FONT_FAMILY,
        "font.size":             10,
        "axes.titlesize":        14,
        "axes.titleweight":      "medium",
        "axes.labelsize":        11,
        "axes.labelweight":      "normal",
        "xtick.labelsize":       10,
        "ytick.labelsize":       10,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             True,
        "grid.alpha":            0.15,
        "grid.linestyle":        "-",
        "grid.color":            "#000000",
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
        "savefig.facecolor":     "white",
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.15,
        "legend.framealpha":     0.9,
        "legend.edgecolor":      "#e2e8f0",
        "legend.fontsize":       9,
    })

# ── Helper: remove top/right spines on an axis ────────────────────────────────
def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#000000", linewidth=0.5)

# ── Helper: add value labels on top of bars ───────────────────────────────────
def label_bars(ax, bars, fmt="{:.3f}", fontsize=8, offset_frac=0.003, colors=None):
    ylim = ax.get_ylim()
    offset = (ylim[1] - ylim[0]) * offset_frac
    for i, bar in enumerate(bars):
        val = bar.get_height()
        color = colors[i] if colors else bar.get_facecolor()
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + offset,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=fontsize, color="#374151")

# ── Helper: auto-contrast text for heatmap cells ──────────────────────────────
def cell_text_color(bg_color_rgba):
    r, g, b = bg_color_rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "#1f2937"
