# KALAVU: Pythia Ablation Experiments
# Router architecture ablation + Freeze depth sweep

## Context

The Pythia-410M experiment produced +14.2% ± 0.0% held-out improvement with:
- 3 domains (code, science, fiction)
- freeze_layers=4 out of 24
- 2-layer router: Linear(1024→256) → ReLU → Linear(256→3)
- 2000 training steps per specialist

We need two ablations before the paper is complete:
1. **Router ablation**: Does the 2-layer router matter, or does a simple linear router work equally well?
2. **Freeze depth sweep**: What's the optimal number of frozen layers?

## EXECUTION RULES (same as before)
1. Execute ONE ablation at a time. Verify results before starting the next.
2. Git commit after each completed ablation.
3. If anything fails or produces unexpected results, STOP and report.
4. Use the naming convention below strictly.

## File Naming

```
kalavu_pythia_ablation_router.py         # Ablation 1: router architecture
kalavu_pythia_ablation_freeze.py         # Ablation 2: freeze depth sweep

results/pythia/
  ablation_router_summary.json           # Router ablation aggregated results
  ablation_freeze_summary.json           # Freeze depth aggregated results

figures/pythia/
  fig_ablation_router.png                # Bar chart comparing router architectures
  fig_ablation_freeze.png                # Line plot: improvement vs freeze depth
```

---

## Ablation 1: Router Architecture

### Purpose
The main experiment used a 2-layer router (Linear→ReLU→Linear). We need to verify the result isn't dependent on router complexity. Test 3 router architectures:

### Router Variants
```python
# Variant A: Simple linear (single layer, no nonlinearity)
router_simple = nn.Linear(hidden_size, 3, bias=False)
# This is what the original spec called for and what synthetic experiments used

# Variant B: 2-layer (what the main experiment used)
router_2layer = nn.Sequential(
    nn.Linear(hidden_size, 256),
    nn.ReLU(),
    nn.Linear(256, 3)
)

# Variant C: No learned router — uniform averaging (1/3 each)
# This is equivalent to weight averaging but computed at inference time
# router_uniform = lambda x: torch.ones(x.shape[0], 3, device=x.device) / 3.0
```

### Setup
- Reuse the SAME specialist checkpoints from the main experiment (seed=42 only — don't retrain)
- If specialist checkpoints weren't saved to disk, retrain seed=42 specialists once with the same config and save them this time
- Train ONLY the router (all specialist weights frozen) for each variant
- Router training: 500 steps, batch_size=4, lr=1e-3, mixed domain data
- Evaluate each variant on held-out mixed eval

### Output Table
```
ROUTER ABLATION (Pythia-410M, seed=42, freeze=4)
Router Type          Mixed Loss   Improvement   Router Distribution
Simple Linear(1024,3)   X.XXXX     +XX.X%       [code:X.XX sci:X.XX fic:X.XX]
2-Layer (main exp)      X.XXXX     +XX.X%       [code:X.XX sci:X.XX fic:X.XX]  
Uniform (no router)     X.XXXX     +XX.X%       [0.33 0.33 0.33]
Best individual         X.XXXX     (baseline)
```

### Expected Outcome
If specialists genuinely diverged (which the hard switching proves), the simple linear router should produce similar results to the 2-layer router. Uniform averaging should be worse but still positive (weight averaging was +4.0% in the main experiment). This confirms the result is about the frozen-layer fusion mechanism, not about router architecture.

### Figure: `fig_ablation_router.png`
- 4 bars: best individual, uniform, simple linear, 2-layer
- Y-axis: held-out mixed loss (start y-axis at a value that shows the differences clearly)
- Title: "Router Architecture Ablation (Pythia-410M)"
- Annotate improvement % above each bar

### Save to `results/pythia/ablation_router_summary.json`:
```json
{
  "experiment": "router_ablation",
  "base_model": "pythia-410m-step10000",
  "seed": 42,
  "freeze_layers": 4,
  "best_individual_mixed": ...,
  "variants": {
    "uniform": {"mixed_loss": ..., "improvement_pct": ..., "gate_distribution": ...},
    "simple_linear": {"mixed_loss": ..., "improvement_pct": ..., "gate_distribution": ...},
    "two_layer": {"mixed_loss": ..., "improvement_pct": ..., "gate_distribution": ...}
  }
}
```

**Git commit:** `[kalavu] ablation: router architecture — simple={X.X}% vs 2-layer={X.X}%`

---

## Ablation 2: Freeze Depth Sweep

### Purpose
The main experiment froze 4/24 layers. How does fusion improvement change with freeze depth? This is Figure 2 in the paper — potentially the most important figure.

### Sweep Values
```python
freeze_layers_values = [0, 2, 4, 6, 8, 12]
# 0  = no frozen layers (pure MoE, no shared backbone — BTX-style baseline)
# 2  = minimal sharing (~8%)
# 4  = main experiment value (~17%)
# 6  = moderate sharing (25%)
# 8  = heavy sharing (33%)
# 12 = half the model frozen (50%)
```

### Setup
- For EACH freeze depth:
  - Load fresh base model (pythia-410m step10000)
  - Freeze the specified number of layers
  - Train 3 specialists (code, science, fiction) for 2000 steps each
  - Train MoE router (use the SIMPLE linear router from Ablation 1 — we want to isolate freeze depth, not router architecture)
  - Evaluate on held-out mixed eval
- Run with seed=42 only for the sweep (3 seeds × 6 depths = 18 specialist trainings is too much)
- THEN run the best 2 freeze depths (besides 4, which we already have) with all 3 seeds for error bars

### Important: freeze_layers=0 is the BTX Baseline
When freeze_layers=0, NO layers are shared. The specialists diverge maximally. The router has to reconcile completely different representations. This is approximately what BTX (Branch-Train-Mix) does. If frozen layers help, freeze=0 should perform WORSE than freeze=4. This is the central claim of the paper.

### Output Table
```
FREEZE DEPTH SWEEP (Pythia-410M, seed=42, simple linear router)
Freeze   Shared%   Mixed Loss   Improvement   Divergence Gap
0        0%        X.XXXX       +XX.X%        X.XXXX
2        8%        X.XXXX       +XX.X%        X.XXXX
4        17%       X.XXXX       +XX.X%        X.XXXX  (main experiment)
6        25%       X.XXXX       +XX.X%        X.XXXX
8        33%       X.XXXX       +XX.X%        X.XXXX
12       50%       X.XXXX       +XX.X%        X.XXXX
```

### Figure: `fig_ablation_freeze.png`
- X-axis: number of frozen layers (0, 2, 4, 6, 8, 12)
- Primary Y-axis (left): improvement % over best individual (line plot with markers)
- Secondary Y-axis (right): divergence gap between specialists (dotted line)
- Highlight the freeze=4 point as "main experiment"
- Mark freeze=0 as "BTX baseline (no frozen layers)"
- Title: "Fusion Improvement vs Freeze Depth (Pythia-410M)"
- If you ran 3 seeds on the best depths, add error bars on those points

### Interpretation Guide
**Expected pattern**: improvement should increase from freeze=0 to some optimum, then decrease.
- Too few frozen layers (0-2): specialists diverge too much, representations are incompatible, router can't reconcile → lower improvement
- Optimal freeze depth (4-6): enough shared structure for fusibility, enough trainable capacity for specialization → peak improvement
- Too many frozen layers (8-12): specialists can't diverge enough, fusion is redundant → lower improvement (approaching weight averaging)

If freeze=0 beats freeze=4, our thesis is wrong — frozen layers don't help. Report honestly.

### Save to `results/pythia/ablation_freeze_summary.json`:
```json
{
  "experiment": "freeze_depth_ablation",
  "base_model": "pythia-410m-step10000",
  "seed": 42,
  "router_type": "simple_linear",
  "results": [
    {"freeze_layers": 0, "mixed_loss": ..., "improvement_pct": ..., "divergence_gap": ...},
    {"freeze_layers": 2, "mixed_loss": ..., "improvement_pct": ..., "divergence_gap": ...},
    ...
  ],
  "multi_seed_results": {
    "freeze_N": {"seeds": [42, 137, 2026], "mean": ..., "std": ...},
    ...
  }
}
```

**Git commit:** `[kalavu] ablation: freeze depth sweep — optimal={N} layers at {X.X}%`

---

## Execution Order

1. **Ablation 1 (router)** — fast, reuses existing specialists, ~15 min
   - Git commit when done
2. **Ablation 2 (freeze sweep seed=42)** — trains 6×3=18 specialists, ~2-3 hours
   - Git commit after EACH freeze depth completes (so nothing is lost)
3. **Ablation 2 (multi-seed on best depths)** — trains 2×3×3=18 more specialists
   - Git commit after each depth×seed completes

Total estimated time: 3-4 hours on RTX 5090.

## Do NOT
- Do not retrain specialists unnecessarily — reuse checkpoints from the main experiment where freeze=4/seed=42 matches
- Do not use the 2-layer router for the freeze sweep — use simple Linear(1024,3) to isolate the freeze depth variable
- Do not skip freeze=0 — it's the BTX baseline comparison, critical for the paper
- Do not run all 3 seeds on all 6 freeze depths — that's 54 specialist trainings. Run seed=42 for the sweep, then 3 seeds on the top 2 depths only.
- Do not modify the specialist training config (lr, steps, data) — only freeze depth and router architecture change
- Do not combine commits. One commit per completed sub-experiment.
