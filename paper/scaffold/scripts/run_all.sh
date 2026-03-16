#!/bin/bash
# =============================================================================
# Run ALL paper experiments
# Estimated total time: ~8-12 hours on RTX 5090
# =============================================================================

set -e

echo "============================================"
echo "KALAVU Paper: Full Experiment Suite"
echo "============================================"
echo "Start time: $(date)"
echo ""

# --- Phase 1: Synthetic Experiments (Table 1) ---
echo "=== Phase 1: Synthetic Experiments ==="
echo "Running 3M and 25M parameter synthetic experiments..."

for SEED in 42 137 2026; do
    echo "-- Seed $SEED --"

    # 3M params (n_layers=8, d_model=256)
    python -m kalavu.train --config configs/synthetic/2mod_3M.yaml --seed $SEED

    # 25M params (n_layers=12, d_model=512)
    python -m kalavu.train --config configs/synthetic/2mod_25M.yaml --seed $SEED

    # 25M params, 5 domains
    python -m kalavu.train --config configs/synthetic/5mod_25M.yaml --seed $SEED
done
echo "Phase 1 complete: $(date)"

# --- Phase 2: Real Model Experiments (Table 2) ---
echo ""
echo "=== Phase 2: Real Model Experiments ==="

for SEED in 42 137 2026; do
    echo "-- Seed $SEED --"

    # 2 domains: math + science (strongest result)
    python -m kalavu.train_hf --config configs/real/qwen_2mod_math_science.yaml --seed $SEED

    # 5 domains
    python -m kalavu.train_hf --config configs/real/qwen_5mod.yaml --seed $SEED

    # Monolithic baseline (equal compute, mixed data)
    python -m kalavu.train_hf --config configs/real/qwen_monolithic_baseline.yaml --seed $SEED
done
echo "Phase 2 complete: $(date)"

# --- Phase 3: Freeze Ablation (Figure 2) ---
echo ""
echo "=== Phase 3: Freeze Depth Ablation ==="
bash scripts/run_freeze_ablation.sh
echo "Phase 3 complete: $(date)"

# --- Phase 4: BTX Comparison ---
echo ""
echo "=== Phase 4: BTX Baseline Comparison ==="
for SEED in 42 137 2026; do
    python -m kalavu.train_hf --config configs/real/qwen_btx_baseline.yaml --seed $SEED
done
echo "Phase 4 complete: $(date)"

# --- Phase 5: Fusion + Evaluation ---
echo ""
echo "=== Phase 5: Fusion and Benchmark Evaluation ==="
echo "Running lm-eval on all model variants..."
bash scripts/run_eval.sh
echo "Phase 5 complete: $(date)"

# --- Phase 6: Generate Figures ---
echo ""
echo "=== Phase 6: Generate Figures ==="
python scripts/generate_figures.py

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo ""
echo "Results: results/"
echo "Figures: figures/"
echo "============================================"
