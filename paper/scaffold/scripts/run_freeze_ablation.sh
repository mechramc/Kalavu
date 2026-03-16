#!/bin/bash
# =============================================================================
# Freeze Depth Ablation (Figure 2 in paper)
# Sweeps freeze_layers from 0 to 8, runs 3 seeds each
# Total: 7 freeze depths × 3 seeds × 2 domains = 42 training runs
# Estimated time: ~4-6 hours on RTX 5090
# =============================================================================

set -e

CONFIG="configs/real/qwen_freeze_ablation.yaml"
RESULTS_DIR="results/real/freeze_ablation"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "KALAVU Paper: Freeze Depth Ablation"
echo "============================================"
echo ""

for FREEZE in 0 1 2 3 4 6 8; do
    for SEED in 42 137 2026; do
        echo "--- freeze_layers=$FREEZE, seed=$SEED ---"

        # Create a temporary config with the current freeze depth
        TMP_CONFIG="/tmp/kalavu_freeze_${FREEZE}_seed_${SEED}.yaml"
        python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['alignment']['freeze_layers'] = $FREEZE
cfg['alignment']['seed'] = $SEED
cfg['name'] = f'freeze_ablation_f{$FREEZE}'
with open('$TMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
"

        # Train all domain specialists
        python -m kalavu.train_hf --config "$TMP_CONFIG" --seed $SEED

        rm "$TMP_CONFIG"
    done
done

echo ""
echo "============================================"
echo "All training runs complete. Running fusion + evaluation..."
echo "============================================"
python scripts/fuse_and_eval_ablation.py
