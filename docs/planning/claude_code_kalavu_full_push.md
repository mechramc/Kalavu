# KALAVU: The Full Push — All Remaining Experiments
# Run after monolithic baseline and downstream benchmarks are complete

## Context

Completed experiments:
- Synthetic 25M: +60.7% ± 0.7% held-out (3 seeds)
- Pythia-410M 3-domain: +14.2% ± 0.0% held-out (3 seeds)
- Qwen-1.5B code+fiction: -1.0% ± 0.0% held-out (3 seeds)
- Router ablation: simple linear = 2-layer = +14.2% (confound resolved)
- Freeze depth sweep: 0→12, monotonic decline, +14.9% to +12.4%
- Monolithic baseline: [FILL IN WHEN DONE]
- Downstream benchmarks: [FILL IN WHEN DONE]

What remains: 4 experiments that turn this from an arxiv report into a top-venue paper.

## MASTER EXECUTION RULES

1. Run experiments in the EXACT order listed (1 → 2 → 3 → 4)
2. Git commit after EACH sub-experiment with descriptive message
3. Save ALL checkpoints to disk — we lost files once, never again
4. If any experiment produces unexpected results, STOP and report before continuing
5. All eval is on HELD-OUT data. Never report in-distribution numbers.
6. Reuse data loading, eval functions, and tokenization from kalavu_pythia_experiment.py — do not rewrite these

## File Naming Convention (strict)

```
kalavu_pythia_maturity_sweep.py          # Experiment 1
kalavu_pythia_1b_experiment.py           # Experiment 2
kalavu_pythia_1b_maturity_sweep.py       # Experiment 3
kalavu_pythia_5domain_experiment.py      # Experiment 4

results/pythia/
  maturity_sweep_410m/
    checkpoint_{STEP}_seed{N}.json       # Per-checkpoint per-seed results
    summary.json                         # Aggregated curve data
  
  pythia_1b/
    main_result_seed{N}.json             # 3-domain main result per seed
    main_result_summary.json             # Aggregated
    maturity_sweep/
      checkpoint_{STEP}_seed42.json
      summary.json
  
  five_domain/
    result_seed{N}.json
    summary.json

figures/pythia/
  fig_maturity_curve_410m.png            # THE key figure — improvement vs base maturity
  fig_maturity_curve_1b.png              # Same curve at 1B scale
  fig_maturity_curve_combined.png        # Both curves on one plot
  fig_1b_fusion_comparison.png           # 1B bar chart (like 410M version)
  fig_1b_divergence_heatmap.png          # 1B heatmap
  fig_1b_router_distribution.png         # 1B router gates
  fig_specialist_scaling.png             # Improvement vs number of specialists
  fig_paper_hero.png                     # Combined overview figure for paper

checkpoints/pythia/
  maturity_sweep_410m/                   # Specialist checkpoints at each maturity
  pythia_1b/                             # 1B specialist checkpoints
  five_domain/                           # 5-domain specialist checkpoints
```

---

## Experiment 1: Maturity Sweep (Pythia-410M)
### Time estimate: ~2.5 hours
### Purpose: Turn 3 anecdotal data points into a proper scaling law

Train 3 specialists and fuse at each of 6 Pythia training checkpoints, measuring how fusion improvement changes as the base model matures.

### Checkpoints to Test
```python
checkpoints = [
    ("step5000", 5000),     # ~3.5% through training — very early
    ("step10000", 10000),   # ~7% — our main experiment
    ("step20000", 20000),   # ~14%
    ("step50000", 50000),   # ~35%
    ("step100000", 100000), # ~70%
    ("step143000", 143000), # ~100% — fully trained
]
```

Each checkpoint is loaded via:
```python
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-410m",
    revision="step5000",  # or step10000, step20000, etc.
    torch_dtype=torch.bfloat16
)
```

### For EACH checkpoint:

**Phase A — Single seed (seed=42):**
1. Load base model at that checkpoint
2. Evaluate base model on held-out code, science, fiction, mixed
3. Freeze first 4 layers (same as main experiment)
4. Train 3 specialists (code, science, fiction) for 2000 steps each
   - Same training config as main experiment (lr=2e-5, batch=2, grad_accum=4, bf16)
   - Same data (same packed chunks — generate once, reuse across all checkpoints)
5. Run divergence check
6. Train simple linear router (500 steps, batch=4, lr=1e-3)
7. Evaluate all variants on held-out mixed
8. Record improvement %
9. Save specialist checkpoints to `checkpoints/pythia/maturity_sweep_410m/step{N}/`
10. Save results to `results/pythia/maturity_sweep_410m/checkpoint_{STEP}_seed42.json`

**Git commit after each checkpoint:** `[kalavu] maturity sweep 410m: step={STEP} improvement={X.X}%`

**Phase B — Multi-seed on interesting points:**
After all 6 checkpoints complete with seed=42, identify the 2 most interesting results (e.g., peak improvement and the point where improvement drops to near-zero). Run seeds [137, 2026] on those 2 checkpoints only.

**Git commit:** `[kalavu] maturity sweep 410m: multi-seed on step={STEP1} and step={STEP2}`

### Expected Pattern
```
step5000  (3.5%):   +20-30% (very plastic, specialists diverge easily)
step10000 (7%):     +14.2%  (known result)
step20000 (14%):    +8-12%  (more rigid, less room to specialize)
step50000 (35%):    +3-6%   (substantial prior knowledge)
step100000 (70%):   +0-2%   (nearly fully trained)
step143000 (100%):  -1-0%   (matches Qwen pattern — fully trained)
```

If the curve is NOT monotonically decreasing, that's a surprising and important finding. Report it.

### KEY FIGURE: `fig_maturity_curve_410m.png`
- X-axis: Training checkpoint (step number or % of full training)
- Y-axis: Fusion improvement % on held-out mixed eval
- Single line with markers at each checkpoint
- Error bars on the multi-seed points
- Horizontal dashed line at 0% (break-even)
- Mark the main experiment point (step10000)
- Title: "Fusion improvement vs base model maturity (Pythia-410M)"
- This is potentially the paper's most important figure. Make it clean and readable.

### JSON format for summary:
```json
{
  "experiment": "maturity_sweep_410m",
  "model": "pythia-410m",
  "freeze_layers": 4,
  "specialist_steps": 2000,
  "domains": ["code", "science", "fiction"],
  "curve": [
    {
      "revision": "step5000",
      "training_pct": 3.5,
      "base_mixed_loss": ...,
      "fused_mixed_loss": ...,
      "improvement_pct": ...,
      "seeds": [42],
      "std": null
    },
    ...
  ]
}
```

---

## Experiment 2: Pythia-1B Main Result
### Time estimate: ~3.5 hours
### Purpose: Prove the mechanism scales beyond 410M

### Setup
```python
model_id = "EleutherAI/pythia-1b"
revision = "step10000"          # Same relative maturity as 410M experiment
freeze_layers = 4               # 4 out of 16 layers (25%)
learning_rate = 2e-5
max_steps = 2000
batch_size = 2
gradient_accumulation = 4       # Effective batch = 8
precision = "bf16"
max_length = 512
```

### Architecture Notes
Pythia-1B: 16 layers, hidden_size=2048, 8 heads, vocab_size=50304.
Freezing 4/16 = 25% (vs 4/24 = 17% at 410M). This is slightly more shared proportionally, which is fine.

### Freezing
```python
model.gpt_neox.embed_in.requires_grad_(False)
for i in range(freeze_layers):
    model.gpt_neox.layers[i].requires_grad_(False)
```

### Domains
Same 3 domains as 410M: code (code_search_net python), science (allenai/sciq with support field), fiction (pg19 first 5000 chars).

IMPORTANT: Regenerate packed training chunks with the 1B tokenizer (same tokenizer as 410M — Pythia uses the same tokenizer across sizes, so chunks should be identical. Verify this.)

### Router
```python
router = nn.Linear(2048, 3, bias=False)  # Simple linear, hidden_size=2048
# Router training: 500 steps, batch=4, lr=1e-3
```

### Run with 3 seeds: [42, 137, 2026]

### For each seed:
1. Load fresh pythia-1b at step10000
2. Evaluate base on held-out
3. Freeze 4 layers
4. Train code specialist (2000 steps) → save checkpoint
5. Train science specialist → save checkpoint
6. Train fiction specialist → save checkpoint
7. Divergence check (STOP if fails)
8. Weight averaging
9. MoE router training + fusion
10. Evaluate all variants on held-out
11. Save results + checkpoints

**Git commit after each seed:** `[kalavu] pythia-1b seed={N}: improvement={X.X}%`

### Output Table
```
PYTHIA-1B MAIN RESULT (3 seeds)
Model                    Code    Science   Fiction   Mixed    Average
Base model              X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (code)       X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (science)    X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (fiction)    X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Weight averaged         X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
MoE fused               X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX

Improvement: +X.X% ± X.X%
```

### Figures (same format as 410M)
- `fig_1b_fusion_comparison.png` — bar chart
- `fig_1b_divergence_heatmap.png` — cross-domain loss matrix
- `fig_1b_router_distribution.png` — gate weights per domain

### CRITICAL: Compare with 410M
Print a cross-scale comparison:
```
SCALE COMPARISON (step10000, 3 domains, freeze=4)
Model Size    Improvement    Std
410M          +14.2%         ±0.0%
1B            +X.X%          ±X.X%
```

If 1B improvement is HIGHER than 410M, the mechanism scales favorably. If LOWER, it may be that 4/16 frozen layers (25%) is too much at 1B — note this for future ablation.

---

## Experiment 3: Pythia-1B Maturity Sweep
### Time estimate: ~5 hours
### Purpose: Confirm the maturity scaling law holds at 1B

### Checkpoints
```python
checkpoints_1b = [
    ("step5000", 5000),     # ~3.5%
    ("step20000", 20000),   # ~14%
    ("step50000", 50000),   # ~35%
    ("step143000", 143000), # ~100%
]
```

4 checkpoints (not 6 — we can interpolate, and 1B training is slower).
step10000 is already done in Experiment 2.

### Run with seed=42 only for the sweep

Same setup as Experiment 2 but loading different revisions.

**Git commit after each checkpoint:** `[kalavu] 1b maturity sweep: step={STEP} improvement={X.X}%`

### KEY FIGURE: `fig_maturity_curve_combined.png`
This is the hero figure of the paper. Two curves on one plot:
- Blue line with circles: Pythia-410M maturity curve (6 points from Experiment 1)
- Red line with squares: Pythia-1B maturity curve (5 points: 4 from Experiment 3 + 1 from Experiment 2)
- X-axis: % of full training (0% to 100%)
- Y-axis: Fusion improvement % on held-out mixed
- Horizontal dashed line at 0%
- Shaded region or annotation showing "KALAVU's target regime" in the early-training zone
- Title: "Fusion improvement vs base model maturity"
- Secondary annotation: mark where Qwen (-1.0%) would fall if plotted (100% trained, different model family — show as a separate marker with a note)

If both curves show the same declining pattern, you have a scaling law that holds across model sizes. That's a publishable finding on its own.

---

## Experiment 4: 5-Specialist Scaling (Pythia-410M)
### Time estimate: ~2 hours
### Purpose: Show improvement scales with number of specialists

### Domains (5 total)
```python
domains = {
    "code": {
        "dataset": "code_search_net", "config": "python", "split": "train",
        "text_fn": lambda x: x["whole_func_string"]
    },
    "science": {
        "dataset": "allenai/sciq", "split": "train",
        "text_fn": lambda x: x["support"] + "\n" + x["question"] + "\n" + x["correct_answer"]
    },
    "fiction": {
        "dataset": "emozilla/pg19", "split": "train",
        "text_fn": lambda x: x["text"][:5000]
    },
    "math": {
        "dataset": "gsm8k", "config": "main", "split": "train",
        "text_fn": lambda x: x["question"] + "\n" + x["answer"]
    },
    "multilingual": {
        "dataset": "mc4", "config": "es", "split": "train",
        "text_fn": lambda x: x["text"][:3000],
        "description": "Spanish text — genuinely foreign to English-primary Pythia"
    },
}
```

Use Spanish from mc4 for the multilingual domain. Pythia was trained primarily on English (The Pile), so Spanish text is genuinely out-of-distribution — the specialist should diverge dramatically.

If mc4 Spanish is too slow to download, use any readily available non-English dataset. The key is that it's a language the model barely knows.

### Setup
```python
model_id = "EleutherAI/pythia-410m"
revision = "step10000"
freeze_layers = 4
# Same training config as main experiment
# 5 specialists × 2000 steps each
```

### Router
```python
router = nn.Linear(1024, 5, bias=False)  # 5 experts
# Router training: 500 steps, batch=4, lr=1e-3, mixed data from all 5 domains
```

### Run with 3 seeds: [42, 137, 2026]

### Output — Specialist Count Scaling
Also compute 2-specialist and 4-specialist fusions from subsets of the same trained specialists (reuse checkpoints, just train different routers):

```python
# After training all 5 specialists, compute fusion for subsets:
subsets = {
    "2 specialists": ["code", "fiction"],           # Most divergent pair
    "3 specialists": ["code", "science", "fiction"], # Main experiment domains
    "4 specialists": ["code", "science", "fiction", "math"],
    "5 specialists": ["code", "science", "fiction", "math", "multilingual"],
}
# For each subset: train a router with len(subset) experts, evaluate on held-out mixed
```

This is cheap — you already have the specialist checkpoints, you just need to train 3 additional routers (~5 min each).

### Output Table
```
SPECIALIST COUNT SCALING (Pythia-410M step10000, 3 seeds)
Specialists   Domains                              Improvement    Std
2             code + fiction                        +X.X%          ±X.X%
3             code + science + fiction              +14.2%         ±0.0%  (known)
4             code + science + fiction + math       +X.X%          ±X.X%
5             code + sci + fic + math + multilingual +X.X%        ±X.X%
```

### KEY FIGURE: `fig_specialist_scaling.png`
- X-axis: Number of specialists (2, 3, 4, 5)
- Y-axis: Fusion improvement %
- Line plot with error bars
- Title: "Fusion improvement vs number of cooperative specialists"
- This figure directly supports the "20 people, 20 GPUs" product thesis

**Git commit after completion:** `[kalavu] 5-domain: 2/3/4/5 specialist scaling = {results}`

---

## Final: Generate Paper Hero Figure

After ALL experiments complete, generate one combined overview figure.

### `fig_paper_hero.png`
A 2×2 subplot figure:
- Top-left: Maturity curves (410M + 1B) — from Experiment 1+3
- Top-right: Specialist count scaling — from Experiment 4
- Bottom-left: Freeze depth sweep — from existing results
- Bottom-right: Router ablation (uniform vs linear vs 2-layer) — from existing results

Each subplot should be self-contained with its own title, axis labels, and legend.
Overall figure title: "KALAVU: Cooperative LLM Fusion — Key Results"
Save as high-resolution PNG (300 DPI) suitable for paper submission.

Also generate individual high-res versions of each subplot for the paper's main figures.

**Git commit:** `[kalavu] all experiments complete — hero figure generated`

---

## Final Summary Print

After everything completes, print the complete picture:

```
KALAVU COMPLETE EXPERIMENT INVENTORY
=====================================

MECHANISM VALIDATION
  Synthetic 25M (held-out):           +60.7% ± 0.7%   [3 seeds]
  Pythia-410M 3-domain (held-out):    +14.2% ± 0.0%   [3 seeds]
  Pythia-1B 3-domain (held-out):      +X.X% ± X.X%    [3 seeds]
  Qwen-1.5B code+fiction (held-out):  -1.0% ± 0.0%    [3 seeds]

MATURITY SCALING LAW (Pythia-410M)
  step5000 (3.5%):    +X.X%
  step10000 (7%):     +14.2%
  step20000 (14%):    +X.X%
  step50000 (35%):    +X.X%
  step100000 (70%):   +X.X%
  step143000 (100%):  +X.X%

MATURITY SCALING LAW (Pythia-1B)
  step5000 (3.5%):    +X.X%
  step10000 (7%):     +X.X%
  step20000 (14%):    +X.X%
  step50000 (35%):    +X.X%
  step143000 (100%):  +X.X%

SPECIALIST COUNT SCALING (410M)
  2 specialists:      +X.X% ± X.X%
  3 specialists:      +14.2% ± 0.0%
  4 specialists:      +X.X% ± X.X%
  5 specialists:      +X.X% ± X.X%

ABLATIONS
  Freeze depth (0-12):     14.9% → 12.4% (monotonic decline)
  Router architecture:     simple=14.2%, 2-layer=14.2%, uniform=6.7%
  Monolithic baseline:     +X.X% vs fused +14.2% (equal compute)

DOWNSTREAM BENCHMARKS (Pythia-1B)
  [results from earlier run]

Total experiments: XX specialist trainings, XX fusion evaluations
Total compute time: ~15 hours on RTX 5090
All results reproducible from committed scripts with fixed seeds.
```

**Final git commit:** `[kalavu] experiment campaign complete — ready for paper writing`

## Do NOT
- Do not skip any experiment. Run all 4 in order.
- Do not change training config between experiments — consistency is critical for cross-experiment comparisons.
- Do not forget to save checkpoints. Every specialist checkpoint must be on disk.
- Do not report in-distribution numbers anywhere. Held-out only.
- Do not use LoRA. Full fine-tuning of unfrozen layers only.
- Do not use padded tokenization. Packed only.
- Do not strip the support field from SciQ.
- Do not proceed past a failed divergence check.
- Do not combine git commits. One commit per meaningful result.
- Do not run the hero figure generation until ALL experiments are done.
