# KALAVU: Final Experiments — 1B Benchmarks + 6.9B Scale Validation

## Two scripts, two machines

**Script 1:** `kalavu_pythia_1b_benchmarks.py` — Run on your RTX 5090 locally
**Script 2:** `kalavu_pythia_6b_experiment.py` — Run on RunPod A100 80GB

---

## Script 1: Pythia-1B Downstream Benchmarks
### Machine: Local RTX 5090
### Time: ~1-2 hours
### Purpose: Fix the N/A benchmark table with a model that actually shows signal

The 410M@step10000 benchmark table was all N/As because the model is too early in training. Pythia-1B@step10000 is larger and should show real signal on easier benchmarks. We also run the fully trained Pythia-1B@step143000 variants to get the strongest benchmark numbers.

### Models to Evaluate
Use the specialist checkpoints and fused model from the existing Pythia-1B experiment (seed=42):
1. Base model (pythia-1b step10000)
2. Code specialist
3. Science specialist
4. Fiction specialist
5. Weight averaged
6. MoE fused
7. Monolithic (train one if not already saved — 6000 steps on mixed data, same as 410M monolithic)

### Benchmarks
```python
benchmarks = {
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "method": "log_likelihood_completion",
        "description": "Sentence completion — 4 choices",
        "random_chance": 0.25
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc", "config": "ARC-Easy",
        "method": "log_likelihood_completion",
        "description": "Easy science QA — 4 choices",
        "random_chance": 0.25
    },
    "lambada": {
        "dataset": "EleutherAI/lambada_openai",
        "method": "log_likelihood_last_word",
        "description": "Predict the last word — tests language modeling directly",
        "random_chance": 0.0
    },
    "sciq": {
        "dataset": "allenai/sciq",
        "method": "log_likelihood_completion",
        "description": "Science QA — directly related to our science domain",
        "random_chance": 0.25
    },
    "winogrande": {
        "dataset": "allenai/winogrande", "config": "winogrande_xl",
        "method": "log_likelihood_completion",
        "description": "Coreference resolution — 2 choices",
        "random_chance": 0.50
    },
}
```

### Evaluation Method (Option B — manual log-likelihood)

For each benchmark question with multiple choices:
```python
def evaluate_multiple_choice(model, tokenizer, context, choices, device):
    """
    For each choice, compute log-likelihood of the choice given the context.
    Return the index of the choice with highest log-likelihood.
    """
    best_ll = float('-inf')
    best_idx = 0
    for i, choice in enumerate(choices):
        input_text = context + choice
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        context_len = context_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab)
        
        # Compute log-likelihood of the completion tokens only
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        target_ids = input_ids[0, 1:]
        
        # Only score tokens AFTER the context
        completion_log_prob = log_probs[context_len-1:, :].gather(
            1, target_ids[context_len-1:].unsqueeze(1)
        ).sum().item()
        
        if completion_log_prob > best_ll:
            best_ll = completion_log_prob
            best_idx = i
    
    return best_idx
```

For LAMBADA (last word prediction):
```python
def evaluate_lambada(model, tokenizer, text, device):
    """
    Given full text, check if the model's top prediction for the last token
    matches the actual last token.
    """
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
    target = tokens[-1]
    
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
    
    predicted = logits.argmax().item()
    return predicted == target
```

### For the MoE model
The MoE model is a custom class. Load it the same way you load it for eval loss — it takes input_ids and returns logits. The benchmark functions above only need model(input_ids) → logits, which your MoE class already provides.

### Sample Size
500 examples per benchmark, sampled from the test/validation split. Use the same 500 examples for ALL model variants so scores are directly comparable.

### Output Table
```
DOWNSTREAM BENCHMARKS — Pythia-1B@step10000 (seed=42)
Model             HellaSwag  ARC-Easy  LAMBADA  SciQ   WinoGrande  Avg
Base              XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
Code spec.        XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
Science spec.     XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
Fiction spec.     XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
Weight avg        XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
MoE fused         XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
Monolithic        XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
Random chance     25.0%      25.0%     0.0%     25.0%  50.0%       —
```

### Figure: `figures/pythia/fig_benchmarks_1b.png`
Grouped bar chart — 7 model variants × 5 benchmarks. Dashed horizontal lines at random chance for each benchmark. 

### Save Results
```
results/pythia/pythia_1b/benchmarks_seed42.json
```

**Git commit:** `[kalavu] 1B benchmarks: MoE avg={X.X}% vs base avg={X.X}%`

---

## Script 2: Pythia-6.9B Scale Validation
### Machine: RunPod A100 80GB
### Time: ~12-15 hours
### Purpose: Close the scale gap — prove the mechanism works at a scale reviewers respect

### RunPod Setup Instructions
```bash
# After SSH-ing into the RunPod A100 instance:

# 1. Clone the repo
git clone https://github.com/mechramc/Kalavu.git
cd Kalavu

# 2. Install dependencies
pip install transformers datasets torch accelerate

# 3. Verify GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
# Should print: True, A100-SXM4-80GB (or similar), 80.0GB

# 4. Run the experiment
python kalavu_pythia_6b_experiment.py 2>&1 | tee experiment_log.txt

# 5. When done, push results
git add -A
git commit -m "[kalavu] pythia-6.9B: full experiment complete"
git push

# 6. SHUT DOWN THE POD (stop billing)
```

### Model Details
```python
model_id = "EleutherAI/pythia-6.9b"
revision = "step10000"  # Same relative maturity as 410M and 1B experiments

# Architecture: 32 layers, hidden_size=4096, 32 heads, vocab_size=50304
# Memory: ~14GB in bf16 for weights alone
# With optimizer states + activations: ~45-55GB for training
# A100 80GB handles this comfortably
```

### Memory Management
```python
import torch

# Load in bf16
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Let accelerate handle placement
)

# If memory is tight during training, enable gradient checkpointing:
model.gradient_checkpointing_enable()
```

### Training Config
```python
freeze_layers = 6               # 6 out of 32 = 19% (similar ratio to 4/24 at 410M)
learning_rate = 1e-5            # Lower LR for larger model — 2e-5 may be too aggressive
max_steps = 1000                # Fewer steps — larger model learns faster per step
batch_size = 1                  # Smaller batch to fit in memory
gradient_accumulation = 8       # Effective batch = 8
precision = "bf16"
max_length = 512
warmup_ratio = 0.1
weight_decay = 0.1
gradient_clip = 1.0
```

Note: LR is 1e-5 (not 2e-5) because larger models are more sensitive to learning rate. If specialists diverge too fast or loss spikes, drop to 5e-6. If specialists don't diverge enough after 1000 steps, increase to 2e-5 or increase steps to 2000.

### Domains
Same 3 domains as all previous experiments:
```python
domains = {
    "code": {"dataset": "code_search_net", "config": "python", "split": "train",
             "text_fn": lambda x: x["whole_func_string"]},
    "science": {"dataset": "allenai/sciq", "split": "train",
                "text_fn": lambda x: x["support"] + "\n" + x["question"] + "\n" + x["correct_answer"]},
    "fiction": {"dataset": "emozilla/pg19", "split": "train",
                "text_fn": lambda x: x["text"][:5000]},
}
```

### Freezing
```python
# Pythia-6.9B uses the same GPT-NeoX architecture
model.gpt_neox.embed_in.requires_grad_(False)
for i in range(freeze_layers):
    model.gpt_neox.layers[i].requires_grad_(False)

# Verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
```

### Router
```python
router = nn.Linear(4096, 3, bias=False)  # hidden_size=4096 for 6.9B
# Training: 500 steps, batch=4, lr=1e-3
```

### Experiment Structure — COMMIT AFTER EACH STEP

The experiment is structured to produce useful results even if the pod gets interrupted.

**Phase 1: Base eval + single seed (seed=42)** (~5 hours)

```
Step 1: Load base model, evaluate on held-out data for all 3 domains + mixed
        → Save results/pythia_6b/step1_base_eval.json
        → Git commit + push

Step 2: Train code specialist (1000 steps)
        → Save checkpoint to checkpoints/pythia_6b/code_specialist_seed42.pt
        → Git commit + push

Step 3: Train science specialist (1000 steps)  
        → Save checkpoint
        → Git commit + push

Step 4: Train fiction specialist (1000 steps)
        → Save checkpoint
        → Git commit + push

Step 5: Divergence check
        → Save results/pythia_6b/step5_divergence_seed42.json
        → Git commit + push
        → IF DIVERGENCE FAILS: STOP. Report. Do not proceed.

Step 6: Weight averaging + MoE router training + eval
        → Save results/pythia_6b/step6_fusion_seed42.json
        → Git commit + push
        → PRINT THE KEY NUMBER: improvement % on held-out mixed
```

**Phase 2: Two more seeds (seed=137, seed=2026)** (~8 hours)

Same as Phase 1 but for seeds 137 and 2026. Commit after each specialist.

**Phase 3: Maturity check at step143000** (~2 hours)

Run one seed (42) at the fully trained checkpoint to see if the mechanism still works:
```python
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision="step143000", ...)
```
Train 3 specialists, fuse, evaluate. This gives us the fully-trained data point for the 6.9B maturity curve.

**Phase 4: Downstream benchmarks on 6.9B** (~1 hour)

Run the same 5 benchmarks as Script 1 on the 6.9B variants (seed=42 only). At 6.9B and step10000, the model should show meaningful signal on HellaSwag, ARC-Easy, and LAMBADA.

### Output Tables

**Main result:**
```
PYTHIA-6.9B CORE RESULT (step10000, 3 seeds)
Model                    Code    Science   Fiction   Mixed    Average
Base model              X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (code)       X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (science)    X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Specialist (fiction)    X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
Weight averaged         X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
MoE fused               X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX

Improvement: +X.X% ± X.X%
```

**Cross-scale comparison:**
```
SCALE COMPARISON (step10000, 3 domains, ~19% frozen)
Model Size    Parameters    Improvement    Std
410M          410M          +14.2%         ±0.016%
1B            1B            +14.8%         ±0.00%
6.9B          6.9B          +X.X%          ±X.X%
```

**Maturity check:**
```
6.9B MATURITY (seed=42)
Checkpoint       Training %    Improvement
step10000        7%            +X.X%
step143000       100%          +X.X%
```

**Benchmarks:**
```
DOWNSTREAM BENCHMARKS — Pythia-6.9B@step10000 (seed=42)
Model             HellaSwag  ARC-Easy  LAMBADA  SciQ   WinoGrande  Avg
Base              XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
MoE fused         XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
Monolithic*       XX.X%      XX.X%     XX.X%    XX.X%  XX.X%       XX.X%
```
*If time permits, train a 3000-step monolithic on 6.9B for the benchmark comparison. Skip if running low on pod time.

### Figures
```
figures/pythia_6b/
  fig_6b_fusion_comparison.png
  fig_6b_divergence_heatmap.png
  fig_6b_router_distribution.png
  fig_scale_comparison.png           # Bar chart: 410M vs 1B vs 6.9B improvement
```

### The Scale Comparison Figure: `fig_scale_comparison.png`
This is the figure that closes the scale argument:
- X-axis: Model size (410M, 1B, 6.9B)
- Y-axis: MoE improvement % on held-out mixed
- Three bars with error bars from 3 seeds
- Title: "Fusion improvement across model scales"
- If improvement holds steady or increases with scale, the mechanism is validated for larger models

### Save All Results
```
results/pythia_6b/
  step1_base_eval.json
  step5_divergence_seed42.json
  step5_divergence_seed137.json
  step5_divergence_seed2026.json
  step6_fusion_seed42.json
  step6_fusion_seed137.json
  step6_fusion_seed2026.json
  maturity_step143000_seed42.json
  benchmarks_seed42.json
  summary.json                       # Aggregated 3-seed result
```

### Critical: Before Shutting Down the Pod

```bash
# Make sure EVERYTHING is pushed
git add -A
git status  # Verify no uncommitted changes
git commit -m "[kalavu] pythia-6.9B: all experiments complete"
git push

# Double-check by listing remote
git log --oneline -5

# THEN shut down the pod
```

### If the Pod Gets Interrupted

Because we commit after every step, you can restart from wherever you left off. The script should detect existing checkpoints and skip completed steps:

```python
import os

def step_completed(result_path):
    return os.path.exists(result_path)

if not step_completed("results/pythia_6b/step1_base_eval.json"):
    run_base_eval()

if not step_completed("checkpoints/pythia_6b/code_specialist_seed42.pt"):
    train_code_specialist(seed=42)

# ... etc
```

### Time Budget
```
Phase 1 (seed=42):                    ~5 hours
Phase 2 (seeds 137, 2026):           ~8 hours  
Phase 3 (maturity at step143000):    ~2 hours
Phase 4 (benchmarks):                ~1 hour
Buffer for downloads + setup:         ~1 hour
─────────────────────────────────────────────
Total:                                ~17 hours

At $1.10/hr (A100 80GB on-demand):   ~$19
At $0.80/hr (spot instance):          ~$14
```

## Do NOT
- Do not use fp32. bf16 only — fp32 will OOM on 6.9B.
- Do not skip gradient checkpointing if memory is tight during training.
- Do not forget to git push before shutting down the pod. Results that exist only on a terminated pod are GONE.
- Do not use LR 2e-5 on 6.9B without testing — start at 1e-5 and increase only if divergence is too slow.
- Do not run all phases as one long script without intermediate commits. Each specialist checkpoint must be committed.
- Do not leave the pod running after experiments finish. Shut it down immediately.
- Do not skip the divergence check. If 6.9B specialists don't diverge, the result is negative and that's important to know.
