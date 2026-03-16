# KALAVU: Pythia Early-Checkpoint Experiment
# The Missing Middle of the Paper

## Why This Experiment

We have two validated results:
- **Synthetic (random init)**: +60.7% held-out improvement. Proves the fusion mechanism works.
- **Qwen (fully pre-trained)**: -1.0% held-out improvement. Proves that fine-tuning a model that already knows the domains doesn't produce enough divergence for fusion to help.

The missing piece: **a real model with real data that's early enough in training for specialists to genuinely diverge.** That's what this experiment provides.

We use EleutherAI's Pythia-410M at an early training checkpoint (step 10,000 out of 143,000). The model knows basic English syntax but has shallow domain knowledge — exactly the "baby LLM" scenario that matches KALAVU's real use case: cooperative training from an early-stage base model.

Pythia is Apache 2.0 licensed, checkpoints at every 1000 steps are on Hugging Face, and hundreds of papers have used it. Reviewers can reproduce everything.

## IMPORTANT: Execution Rules

1. **Execute ONE step at a time.** Do not start Step N+1 until Step N is verified.
2. **Git commit after every completed step.** Use clean commit messages: `[kalavu] step N: description`
3. **If any step fails or produces unexpected results, STOP and report.** Do not try to fix and continue silently.
4. **All output files use the naming convention below.** No ad-hoc names.

## File Naming Convention

```
kalavu_pythia_experiment.py              # Main experiment script

results/pythia/
  step1_base_eval.json                   # Step 1: base model evaluation
  step3_divergence_check_seed42.json     # Step 3: per-seed divergence check
  step3_divergence_check_seed137.json
  step3_divergence_check_seed2026.json
  step4_fusion_results_seed42.json       # Step 4: per-seed fusion results
  step4_fusion_results_seed137.json
  step4_fusion_results_seed2026.json
  step5_final_summary.json               # Step 5: aggregated 3-seed summary

figures/pythia/
  fig_training_curves_seed42.png         # Loss curves during specialist training
  fig_divergence_heatmap.png             # Specialist cross-domain loss matrix
  fig_fusion_comparison.png              # Bar chart: base vs specialists vs fused
  fig_router_distribution.png            # Router expert selection per domain
```

## Step 0: Setup and Verify Environment

```python
# Verify these work before anything else:
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "EleutherAI/pythia-410m"
revision = "step10000"  # Early checkpoint — ~7% through training

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16)

print(f"Model loaded: {model_id} at {revision}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Layers: {model.config.num_hidden_layers}")
print(f"Hidden size: {model.config.hidden_size}")
```

Expected: ~410M params, 24 layers, hidden_size=1024, vocab_size=50304.

If `step10000` doesn't load or seems too untrained (loss > 8 on eval), try `step20000` instead. The goal is a model that can form coherent English but doesn't have deep domain knowledge.

**Git commit:** `[kalavu] step 0: verify pythia-410m step10000 loads correctly`

## Step 1: Evaluate Base Model on Domain Data

Before any training, measure the base model's loss on each domain. This is our baseline.

### Domains (3 domains for this experiment)

```python
domains = {
    "code": {
        "dataset": "code_search_net",
        "config": "python",
        "split": "train",
        "text_fn": lambda x: x["whole_func_string"],
        "description": "Python source code"
    },
    "science": {
        "dataset": "allenai/sciq",
        "split": "train",
        "text_fn": lambda x: x["support"] + "\n" + x["question"] + "\n" + x["correct_answer"],
        "description": "Science passages + Q&A"
    },
    "fiction": {
        "dataset": "emozilla/pg19",
        "split": "train",
        "text_fn": lambda x: x["text"][:5000],  # First 5000 chars per book to avoid huge sequences
        "description": "Project Gutenberg literary fiction"
    },
}
```

### Data Preparation
- For each domain: load dataset, extract text using `text_fn`, concatenate all texts into one long string
- Tokenize with Pythia's tokenizer
- Pack into 512-token chunks (NO padding)
- Split: first 80% = train, next 10% = in-distribution eval, last 10% = HELD-OUT eval
- The held-out 10% must NEVER be seen during training — this is what we report in the paper
- Keep at least 2000 training chunks per domain (if a domain has less, increase text extraction)

### Evaluation
Compute cross-entropy loss of the base model on ALL held-out eval splits:
- Base model on code held-out
- Base model on science held-out  
- Base model on fiction held-out
- Base model on mixed held-out (interleaved chunks from all 3 domains)

Save to `results/pythia/step1_base_eval.json`:
```json
{
  "model": "pythia-410m",
  "revision": "step10000",
  "eval_type": "held-out",
  "base_loss": {
    "code": ...,
    "science": ...,
    "fiction": ...,
    "mixed": ...
  },
  "data_stats": {
    "code": {"train_chunks": ..., "eval_chunks": ..., "held_out_chunks": ...},
    "science": {"train_chunks": ..., "eval_chunks": ..., "held_out_chunks": ...},
    "fiction": {"train_chunks": ..., "eval_chunks": ..., "held_out_chunks": ...}
  }
}
```

**Sanity check:** Base model loss should be moderate (3-6 range). If it's >10, the model is too undertrained — try `step20000`. If it's <2 on any domain, the model already knows that domain too well — try `step5000`.

**Git commit:** `[kalavu] step 1: base model eval on 3 domains — code/science/fiction`

## Step 2: Train Specialists

Train 3 specialist models, one per domain. Each starts from the same base checkpoint with the same frozen layers.

### Training Config
```python
model_id = "EleutherAI/pythia-410m"
revision = "step10000"
freeze_layers = 4          # Freeze first 4 out of 24 layers (~17%)
learning_rate = 2e-5
max_steps = 2000           # More steps than Qwen — this model has more to learn
batch_size = 2
gradient_accumulation = 4  # Effective batch = 8
precision = "bf16"
max_length = 512
warmup_ratio = 0.1         # 10% warmup
weight_decay = 0.1
gradient_clip = 1.0
optimizer = "adamw"        # betas=(0.9, 0.95)
```

### Freezing
```python
# Freeze embedding layer
model.gpt_neox.embed_in.requires_grad_(False)

# Freeze first N transformer layers
for i in range(freeze_layers):
    model.gpt_neox.layers[i].requires_grad_(False)

# Leave unfrozen: layers 4-23, final layer norm, lm_head (embed_out)
```

### Training Loop
- Standard language modeling: predict next token, cross-entropy loss
- Only compute loss on non-padding tokens (but with packed tokenization there's no padding)
- Log training loss every 50 steps
- Save checkpoint at end of training

### For EACH seed in [42, 137, 2026]:
1. Load base model from `step10000` (fresh load each time)
2. Freeze layers
3. Train code specialist for 2000 steps on code training data
4. Save checkpoint: `checkpoints/pythia/code_specialist_seed{seed}.pt`
5. Load base model again (fresh)
6. Freeze layers
7. Train science specialist for 2000 steps on science training data
8. Save checkpoint: `checkpoints/pythia/science_specialist_seed{seed}.pt`
9. Load base model again (fresh)
10. Freeze layers
11. Train fiction specialist for 2000 steps on fiction training data
12. Save checkpoint: `checkpoints/pythia/fiction_specialist_seed{seed}.pt`

**Save training curves** for seed=42 to `figures/pythia/fig_training_curves_seed42.png`:
- 3 lines (code, science, fiction specialist training loss over steps)
- Title: "Specialist Training Loss (Pythia-410M, step10000, seed=42)"

**Git commit after EACH seed completes:** `[kalavu] step 2: trained 3 specialists seed={seed}`

## Step 3: Divergence Check

For each seed, verify that specialists actually diverged on HELD-OUT data.

### Compute cross-domain loss matrix
For each specialist, evaluate on ALL domain held-out sets:

```
                    Code eval   Science eval   Fiction eval
Base model          B_code      B_sci          B_fic
Code specialist     C_code      C_sci          C_fic
Science specialist  S_code      S_sci          S_fic
Fiction specialist  F_code      F_sci          F_fic
```

### Divergence Checks (ALL must pass):
1. Each specialist beats base on its own domain: `C_code < B_code`, `S_sci < B_sci`, `F_fic < B_fic`
2. Each specialist is NOT better than base on ALL other domains (at least one cross-domain loss >= base)
3. The divergence gap (max specialist own-domain improvement - min cross-domain change) > 0.1

### If ANY check fails for ANY seed: STOP and report. Do not proceed to fusion.

Save per-seed results to `results/pythia/step3_divergence_check_seed{seed}.json`:
```json
{
  "seed": 42,
  "cross_domain_matrix": {
    "base": {"code": ..., "science": ..., "fiction": ...},
    "code_specialist": {"code": ..., "science": ..., "fiction": ...},
    "science_specialist": {"code": ..., "science": ..., "fiction": ...},
    "fiction_specialist": {"code": ..., "science": ..., "fiction": ...}
  },
  "checks": {
    "code_beats_base_on_code": true/false,
    "science_beats_base_on_science": true/false,
    "fiction_beats_base_on_fiction": true/false,
    "divergence_gap": ...,
    "all_pass": true/false
  }
}
```

**Save figure** for seed=42: `figures/pythia/fig_divergence_heatmap.png`
- Heatmap of the cross-domain matrix (4 rows × 3 columns)
- Color scale: green = lower loss (better), red = higher loss (worse)
- Title: "Cross-Domain Loss Matrix (Pythia-410M, seed=42)"

**Git commit:** `[kalavu] step 3: divergence check — {PASS/FAIL} for all seeds`

## Step 4: Fusion

### Method 1: Weight Averaging
Average the unfrozen layers across all 3 specialists. Frozen layers are identical by construction.

```python
fused_avg = deep_copy(specialist_code)
for layer_idx in range(freeze_layers, num_layers):
    for param_name in layer_params:
        fused_avg[param_name] = (
            specialist_code[param_name] +
            specialist_science[param_name] +
            specialist_fiction[param_name]
        ) / 3.0
# Also average final layer norm and lm_head if they were trainable
```

### Method 2: MoE Routing
Learned router with 3 experts.

```python
router = nn.Linear(hidden_size, 3, bias=False)  # hidden_size = 1024 for pythia-410m
```

Router architecture:
- Input: output of last frozen layer (layer 3), detached from gradient graph
- Output: softmax weights over 3 specialists
- Forward: run frozen layers → router → run all 3 specialist unfrozen layers → weighted sum → final norm → lm_head

Router training:
```python
router_steps = 500       # More steps since 3 experts
router_batch_size = 4
router_lr = 1e-3
# Train on mixed data: equal chunks from all 3 domains, shuffled
```

### Evaluation — HELD-OUT ONLY
For each model variant, compute loss on held-out eval sets:
- Base model
- Code specialist
- Science specialist
- Fiction specialist
- Weight averaged
- MoE fused

Report:
```
FUSION RESULTS (held-out eval, seed=XX)
Model                    Code    Science   Fiction   Mixed    Average
Base model              X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (code)       X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (science)    X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (fiction)    X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Weight averaged         X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
MoE fused               X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX

Improvement over best individual (mixed held-out): XX.X%
```

Save per-seed to `results/pythia/step4_fusion_results_seed{seed}.json`.

**Save figure** (seed=42): `figures/pythia/fig_fusion_comparison.png`
- Grouped bar chart: 6 model variants × mixed held-out loss
- Highlight the best fused result
- Title: "Fusion Results on Held-Out Mixed Eval (Pythia-410M)"

**Save figure** (seed=42): `figures/pythia/fig_router_distribution.png`
- For each domain's eval data, show what % of tokens the router sends to each expert
- 3 groups (code tokens, science tokens, fiction tokens) × 3 bars (expert 0/1/2)
- Title: "MoE Router Expert Selection by Domain"

**Git commit after each seed:** `[kalavu] step 4: fusion results seed={seed} — improvement={X.X}%`

## Step 5: Final Summary

Aggregate across all 3 seeds. Compute mean ± std.

Save to `results/pythia/step5_final_summary.json`:
```json
{
  "experiment": "pythia-410m-step10000-3domain",
  "freeze_layers": 4,
  "training_steps": 2000,
  "seeds": [42, 137, 2026],
  "held_out_results": {
    "mean": {
      "base_mixed": ...,
      "best_individual_mixed": ...,
      "weight_avg_mixed": ...,
      "moe_fused_mixed": ...,
      "improvement_pct": ...
    },
    "std": {
      "improvement_pct": ...
    },
    "per_seed": [...]
  }
}
```

Print the final summary:
```
KALAVU PYTHIA EXPERIMENT — FINAL RESULTS
=========================================
Model: pythia-410m at step 10000 (early training)
Domains: code, science, fiction
Freeze: 4/24 layers
Training: 2000 steps per specialist

Held-out mixed eval (mean ± std across 3 seeds):
  Base model:           X.XXXX ± X.XXXX
  Best individual:      X.XXXX ± X.XXXX
  Weight averaged:      X.XXXX ± X.XXXX
  MoE fused:            X.XXXX ± X.XXXX
  Improvement:          X.X% ± X.X%

PAPER NARRATIVE:
  Synthetic (zero prior):     +60.7% ± 0.7%   ← mechanism works
  Pythia (early training):    +X.X% ± X.X%    ← transfers to real models
  Qwen (fully trained):      -1.0% ± 0.0%     ← diminishes with base knowledge
```

**Git commit:** `[kalavu] step 5: final summary — pythia improvement={X.X}% ± {X.X}%`

## Step 6: If Results Are Negative

If the Pythia experiment also shows ≤0% improvement:

1. Try `step3000` instead of `step10000` (even earlier = less domain knowledge)
2. Try `freeze_layers=2` instead of 4 (more trainable capacity)
3. Try `max_steps=5000` (more training for specialists)

But do NOT run these automatically. Report the negative result first. We will decide the next move.

## Do NOT
- Do not skip the divergence check (Step 3). If specialists don't diverge, fusion can't help.
- Do not use in-distribution eval. ALL reported numbers must be held-out.
- Do not use LoRA. Full fine-tuning of unfrozen layers only.
- Do not use padded tokenization. Packed only.
- Do not combine multiple steps into one git commit. One commit per completed step.
- Do not proceed past a failed step without reporting.
- Do not change the file naming convention.
- Do not strip the `support` field from SciQ.
