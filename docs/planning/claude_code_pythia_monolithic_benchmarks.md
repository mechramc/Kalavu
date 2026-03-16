# KALAVU: Monolithic Baseline + Downstream Benchmarks

## Context

We have the core results:
- MoE fused (3 specialists): +14.2% on held-out mixed eval
- Router ablation: simple linear matches 2-layer (confound resolved)
- Freeze depth sweep: running now

Two experiments remain before paper writing:
1. **Monolithic baseline**: Does specialist-then-fuse beat generalist training given equal compute?
2. **Downstream benchmarks**: Do the improvements show up on actual tasks, not just eval loss?

## EXECUTION RULES
1. Execute one experiment at a time.
2. Git commit after each.
3. Stop and report if anything unexpected happens.

---

## Experiment 1: Monolithic Baseline

### Purpose

The strongest objection a reviewer can raise: "Why not just train one model on all the data instead of training 3 specialists and fusing them?"

The monolithic baseline answers this directly. Take the same total compute budget (3 specialists × 2000 steps = 6000 total training steps) and train a SINGLE model on mixed-domain data for 6000 steps. If the fused model beats the monolithic model, specialist-then-fuse is a better use of compute than generalist continued pre-training.

### Script: `kalavu_pythia_monolithic_baseline.py`

### Setup
```python
model_id = "EleutherAI/pythia-410m"
revision = "step10000"
freeze_layers = 4              # Same as specialist experiment
learning_rate = 2e-5           # Same
max_steps = 6000               # 3x specialist steps = equal total compute
batch_size = 2
gradient_accumulation = 4      # Effective batch = 8
precision = "bf16"
max_length = 512
seed = 42                      # Same seed
warmup_ratio = 0.1
weight_decay = 0.1
gradient_clip = 1.0
```

### Training Data
Mix all 3 domain training sets together. Interleave chunks:
- Take the same packed 512-token training chunks used for code, science, and fiction specialists
- Combine into one dataset, shuffle
- The monolithic model sees ALL domain data, while each specialist only saw its own domain
- This is actually an ADVANTAGE for the monolithic model — it has access to more diverse data

### Evaluation
Evaluate the monolithic model on the SAME held-out eval sets:
- Code held-out
- Science held-out
- Fiction held-out
- Mixed held-out

Also evaluate at intermediate steps for a training curve:
- Eval every 500 steps (steps 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000)
- Log held-out loss on all domains at each checkpoint

### Run with 3 seeds: [42, 137, 2026]

### Output Table
```
MONOLITHIC BASELINE (Pythia-410M, equal compute comparison)
Model                     Code    Science   Fiction   Mixed    Average
Base model               X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Monolithic (6000 steps)  X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Best individual spec.    X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Weight averaged          X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
MoE fused                X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX

Monolithic vs Base:          +X.X%
MoE fused vs Base:           +14.2%
MoE fused vs Monolithic:     +X.X%  ← THE KEY NUMBER
```

### Figure: `fig_monolithic_comparison.png`
- Bar chart: 5 bars (base, monolithic, best individual, weight avg, MoE fused)
- Y-axis: held-out mixed loss
- Annotate the gap between monolithic and MoE fused
- Title: "Equal-Compute Comparison: Monolithic vs Specialist Fusion"
- Add error bars from 3 seeds

### Figure: `fig_monolithic_trajectory.png`
- X-axis: training steps (0 to 6000)
- Lines:
  - Gray dashed: base model (horizontal)
  - Orange: monolithic model held-out mixed loss over training
  - Purple dashed horizontal: MoE fused result (constant line — fusion happens after training, not during)
- Shows at what step the monolithic model catches the fused result (if ever)
- Title: "Monolithic Training Trajectory vs Fusion Result"

### Save Results
```
results/pythia/
  monolithic_baseline_seed42.json
  monolithic_baseline_seed137.json
  monolithic_baseline_seed2026.json
  monolithic_baseline_summary.json

figures/pythia/
  fig_monolithic_comparison.png
  fig_monolithic_trajectory.png
```

### JSON format for summary:
```json
{
  "experiment": "monolithic_baseline",
  "total_steps": 6000,
  "equivalent_compute": "3 specialists × 2000 steps",
  "seeds": [42, 137, 2026],
  "results": {
    "mean": {
      "monolithic_mixed": ...,
      "moe_fused_mixed": ...,
      "fused_vs_monolithic_pct": ...
    },
    "std": {
      "fused_vs_monolithic_pct": ...
    }
  }
}
```

**Git commit:** `[kalavu] monolithic baseline: fused vs monolithic = {X.X}% ± {X.X}%`

---

## Experiment 2: Downstream Benchmarks

### Purpose

Eval loss is the right metric for a mechanism paper, but reviewers want to see task performance too. Run standard benchmarks on all model variants.

### Script: `kalavu_pythia_benchmarks.py`

### Important Caveat

Pythia-410M at step 10,000 is very early in training. It will score near-zero on hard benchmarks like GSM8K or HumanEval. That's fine — we're measuring RELATIVE improvement between model variants, not absolute capability. Even if all models score 5% on GSM8K, if the fused model scores 7% vs specialists at 5%, that's a 40% relative improvement.

Choose benchmarks where even a weak model shows SOME signal. Skip benchmarks where all variants score 0%.

### Benchmarks to Run (via lm-eval harness)

```python
benchmarks = [
    "hellaswag",          # Sentence completion — even weak models show signal
    "arc_easy",           # Easy science QA — more signal than arc_challenge
    "piqa",               # Physical intuition — good for weak models
    "winogrande",         # Coreference resolution
    "lambada_openai",     # Word prediction — directly related to language modeling
    "sciq",               # Science QA — directly matches our science domain
]
```

### Installation
```bash
pip install lm-eval
```

### Models to Evaluate
For seed=42 only (benchmarks are expensive):
1. Base model (pythia-410m step10000)
2. Code specialist
3. Science specialist
4. Fiction specialist
5. Weight averaged
6. MoE fused
7. Monolithic (from Experiment 1)

### Running lm-eval

For standard models (base, specialists, weight-averaged, monolithic):
```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path},dtype=bfloat16",
    tasks=benchmarks,
    batch_size=8,
    device="cuda",
)
```

For the MoE fused model, lm-eval won't load it directly because it's a custom architecture. Two options:

**Option A (preferred):** Write a thin wrapper that makes the MoE model look like a HuggingFace model to lm-eval. It needs to implement `forward()` that returns logits given input_ids. Register it as a custom model with lm-eval.

**Option B (simpler):** Skip lm-eval for the MoE model. Instead, compute accuracy manually on a subset of each benchmark:
```python
# For each benchmark question:
# 1. Tokenize the question + each answer choice
# 2. Run through MoE model, get log-likelihood of each choice
# 3. Pick the choice with highest log-likelihood
# 4. Check if correct
```

Use whichever approach is faster to implement. The MoE number is the one that matters most.

### Output Table
```
DOWNSTREAM BENCHMARKS (Pythia-410M, seed=42)
Model             HellaSwag  ARC-Easy  PIQA  WinoGrande  LAMBADA  SciQ   Avg
Base              XX.X%      XX.X%     XX.X% XX.X%       XX.X%    XX.X%  XX.X%
Code spec.        XX.X%      XX.X%     XX.X% XX.X%       XX.X%    XX.X%  XX.X%
Science spec.     XX.X%      XX.X%     XX.X% XX.X%       XX.X%    XX.X%  XX.X%
Fiction spec.     XX.X%      XX.X%     XX.X% XX.X%       XX.X%    XX.X%  XX.X%
Weight avg        XX.X%      XX.X%     XX.X% XX.X%       XX.X%    XX.X%  XX.X%
MoE fused         XX.X%      XX.X%     XX.X% XX.X%       XX.X%    XX.X%  XX.X%
Monolithic        XX.X%      XX.X%     XX.X% XX.X%       XX.X%    XX.X%  XX.X%
```

### If All Scores Are Near Random

If the base model scores near random chance on ALL benchmarks (25% for 4-choice, 50% for binary), the model is too weak for task benchmarks. In that case:

1. Report this honestly: "Pythia-410M at step 10,000 is too early in training for meaningful downstream task evaluation. We report eval loss as the primary metric."
2. Skip this table in the main paper — mention it in the appendix.
3. This is not a problem for the paper. The mechanism is validated by eval loss; downstream benchmarks are supplementary.

Do NOT cherry-pick benchmarks where the fused model happens to do well while dropping benchmarks where it doesn't.

### Save Results
```
results/pythia/
  benchmarks_seed42.json

figures/pythia/
  fig_benchmarks.png       # Grouped bar chart if scores show signal
                            # Skip this figure if all scores are near-random
```

### JSON format:
```json
{
  "experiment": "downstream_benchmarks",
  "seed": 42,
  "model_variants": {
    "base": {"hellaswag": ..., "arc_easy": ..., ...},
    "code_specialist": {...},
    "science_specialist": {...},
    "fiction_specialist": {...},
    "weight_averaged": {...},
    "moe_fused": {...},
    "monolithic": {...}
  },
  "note": "Pythia-410M at step10000 — scores may be near-random on harder tasks"
}
```

**Git commit:** `[kalavu] downstream benchmarks: avg fused={X.X}% vs base={X.X}%`

---

## Execution Order

1. **Monolithic baseline** (~45 min for 3 seeds at 6000 steps each)
   - Git commit when done
2. **Downstream benchmarks** (~30-60 min depending on lm-eval)
   - Git commit when done

Run these AFTER the freeze depth sweep completes (or in parallel if the freeze sweep is still running — these use different scripts and don't conflict).

## Do NOT
- Do not give the monolithic model more or fewer steps than 6000. Equal compute is the whole point.
- Do not change any hyperparameters between monolithic and specialist training (except max_steps).
- Do not cherry-pick benchmarks. Run all listed benchmarks, report all results.
- Do not skip the monolithic trajectory figure — showing where the monolithic model is at step 2000 (when specialists finish) vs step 6000 (equal compute) vs the fused result is a powerful visual.
- Do not panic if benchmark scores are near-random. Report honestly and let eval loss carry the argument.
