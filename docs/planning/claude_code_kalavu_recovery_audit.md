# KALAVU: Reconstruct Lost Qwen Experiment Scripts + Full Audit

## What Was Lost

Two files were permanently lost when Claude Code wiped the drive:

1. `kalavu_qwen_experiment.py` — the original 2-domain Qwen experiment
2. `kalavu_qwen_experiment_5domain.py` — the 5-domain variant

Both produced verified results that are central to the paper. Check if results JSONs still exist in `kalavu_experiment_results/` or `kalavu_checkpoints*/` first — they may contain enough info to validate the reconstruction.

---

## File 1: `kalavu_qwen_experiment.py`

This script fine-tunes Qwen2.5-1.5B on two domains (math + science), then fuses via MoE routing.

### Architecture & Setup
```python
model_id = "Qwen/Qwen2.5-1.5B"
freeze_layers = 2  # first 2 transformer blocks frozen
learning_rate = 2e-5
max_steps = 200
batch_size = 2
gradient_accumulation_steps = 2  # effective batch = 4
precision = "bf16"
max_length = 512
seed = 42
```

### Data Loading — CRITICAL DETAILS
The original used **packed tokenization** (not padded):
- **Math domain**: GSM8K dataset (`gsm8k`, `main` split). Used `question + "\n" + answer` concatenated.
- **Science domain**: SciQ dataset (`allenai/sciq`, `train` split). Used `support + "\n" + question + "\n" + correct_answer` — **THE FULL `support` FIELD** (long scientific passages). This is what created real domain divergence.
- All texts concatenated into one long string per domain, tokenized, then split into packed 512-token chunks with NO padding.
- 90% train / 10% eval split on the chunked data.

**WARNING**: A previous ablation attempt failed (0% improvement) because it stripped the `support` field from SciQ and used padded tokenization instead of packed. The `support` field is what makes science data actually diverge from math data. Do NOT repeat that mistake.

### Training
- Full fine-tuning of unfrozen layers (NOT LoRA — LoRA at r=16 was tried and produced insufficient divergence)
- AdamW optimizer, betas=(0.9, 0.95)
- Linear warmup (first 10% of steps) then cosine decay
- Gradient clipping at 1.0
- Each specialist trains independently from the same frozen Qwen base

### Freezing Mechanism
Freeze the first N transformer blocks of the Qwen model. For Qwen2.5-1.5B, the model has `model.layers` (a list of transformer blocks). Freeze layers 0 and 1. Also freeze the embedding layers (`model.embed_tokens`). Leave `model.norm` (final RMSNorm) and `lm_head` trainable (or tied to embeddings — check Qwen's config).

### Fusion — Two Methods
1. **Weight averaging**: Average unfrozen layer weights between the two specialists. Frozen layers are already identical by construction.
2. **MoE routing**: Learned router on shared frozen backbone output.
   - Router: `nn.Linear(hidden_size, 2, bias=False)` operating on the output of the last frozen layer
   - The forward pass: run input through shared frozen layers → router decides weights → run through specialist A's unfrozen layers AND specialist B's unfrozen layers → weighted combination → final norm → lm_head
   - Router training: 300 steps, batch_size=4, lr=1e-3, on mixed data from both domains
   - Soft routing: `weights = softmax(router(x.detach()))`, weighted sum of specialist outputs
   - `x.detach()` on router input — don't backprop routing decisions through the frozen backbone

### Evaluation
Compute cross-entropy loss for: base model, math specialist, science specialist, weight-averaged, MoE-fused.
Eval on: math eval split, science eval split, mixed eval (interleaved chunks from both).
Use same packed tokenization for eval as for training.

### Known Results (verified, ground truth for validation)
```
Model                    Math     Science    Mixed    Average
Base Qwen2.5-1.5B      1.5663    1.5663    1.5663    1.5663
Math specialist         0.8569    1.6024    1.1776    1.2123
Science specialist      1.5990    1.0069    1.2549    1.2869
Weight averaged         1.1259    1.2014    1.1254    1.1509
MoE fused              0.8343    1.0299    0.8789    0.9144
Improvement: +17.15% over best individual on mixed eval (in-distribution)
```

**IMPORTANT NOTE**: These are IN-DISTRIBUTION results (eval on 10% of training data). On truly held-out data, the improvement was ~0% because 200 steps at 2e-5 memorizes the training distribution rather than producing generalizable specialization. This is a known finding, not a bug.

### Checkpoint Saving
Save specialist checkpoints and the router to `kalavu_checkpoints/`. Save results to a JSON file.

---

## File 2: `kalavu_qwen_experiment_5domain.py`

Same structure as the 2-domain version but with 5 domains.

### Domains
```python
domains = {
    "math": {
        "dataset": "gsm8k", "config": "main", "split": "train",
        "text_fn": lambda x: x["question"] + "\n" + x["answer"]
    },
    "science": {
        "dataset": "allenai/sciq", "split": "train",
        "text_fn": lambda x: x["support"] + "\n" + x["question"] + "\n" + x["correct_answer"]
    },
    "code": {
        "dataset": "code_search_net", "config": "python", "split": "train",
        "text_fn": lambda x: x["whole_func_string"]
    },
    "legal": {
        # Check kalavu_checkpoints_5domain/ for the exact dataset used.
        # Likely pile-of-law, lexlms/lex_files, or multi_legal_pile
        "text_fn": lambda x: x["text"]
    },
    "fiction": {
        # Likely emozilla/pg19 or similar
        "text_fn": lambda x: x["text"]
    },
}
```

If `kalavu_checkpoints_5domain/` has a config or results JSON that records which datasets were used, use those exact datasets. Otherwise use reasonable alternatives and note the substitution.

### Training
Same setup as 2-domain: freeze_layers=2, lr=2e-5, 200 steps per specialist, full fine-tuning, packed tokenization.

### Fusion
MoE router with 5 experts:
- `router = nn.Linear(hidden_size, 5, bias=False)`
- Router training: 300 steps on mixed data from all 5 domains
- Soft routing: softmax over 5 experts, weighted sum

### Known Results (in-distribution)
```
5-domain results:
  MoE fused mixed loss: 0.8165
  Best individual mixed loss: 1.3221
  Improvement: +38.24%
```

Scaling pattern: 2 overlapping domains (+0.9%) → 2 divergent domains (+17%) → 5 domains (+38%).

---

## Step 2: Full Repo Audit

After reconstruction, do a complete inventory:

```bash
echo "=== Python files ==="
find . -name "*.py" | sort

echo "=== JSON results ==="
find . -name "*.json" | sort

echo "=== Checkpoints ==="
ls -la kalavu_checkpoints/ kalavu_checkpoints_5domain/ 2>/dev/null

echo "=== Markdown/docs ==="
find . -name "*.md" | sort
```

### Files that MUST exist for the paper:
- [ ] `kalavu_experiment.py` — synthetic proof-of-concept (should be on GitHub from before)
- [ ] `kalavu_held_out_validation.py` — held-out validation (+60.7% result, just recovered)
- [ ] `kalavu_qwen_experiment.py` — 2-domain Qwen (RECONSTRUCT NOW)
- [ ] `kalavu_qwen_experiment_5domain.py` — 5-domain Qwen (RECONSTRUCT NOW)
- [ ] `kalavu_qwen_divergent_domains.py` — code vs fiction, 1000 steps (currently running)
- [ ] Results JSONs for each experiment

### Check the running experiment:
The `kalavu_qwen_divergent_domains.py` background job should still be running or may have finished. Check:
1. Is it still running?
2. If finished: did the divergence verification pass? (Specialists must beat base on own domain AND be worse than base on cross-domain, on held-out data)
3. If divergence passed: what is the fusion improvement on held-out mixed eval?

---

## Step 3: Report

Print:
```
KALAVU REPO STATUS
==================
Files present:          [list every .py file]
Files recovered:        [list what was recovered from session memory]
Files reconstructed:    [list what was rebuilt from this spec]
Files still missing:    [anything else]

EXPERIMENT RESULTS INVENTORY
============================
Synthetic 25M (held-out):     +60.7% ± 0.7% [3 seeds] — kalavu_held_out_validation.py
Qwen 2-domain (in-dist):     +17.15% — kalavu_qwen_experiment.py
Qwen 5-domain (in-dist):     +38.24% — kalavu_qwen_experiment_5domain.py  
Qwen divergent (held-out):   [PENDING or result] — kalavu_qwen_divergent_domains.py

NEXT ACTIONS
============
1. [whatever is most urgent based on current state]
```

## Do NOT
- Do not modify `kalavu_experiment.py` (original synthetic script) — ground truth
- Do not modify `kalavu_held_out_validation.py` — just recovered, produces +60.7%
- Do not modify `kalavu_qwen_divergent_domains.py` — currently running
- Do not use LoRA. Full fine-tuning only.
- Do not use padded tokenization. Packed only.
- Do not strip the `support` field from SciQ.
- Do not change learning rate from 2e-5 for the Qwen experiments.
