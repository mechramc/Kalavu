# KALAVAI — கலவை

20 people. 20 GPUs. 1 model none of them could build alone.

```
pip install transformers datasets torch
python kalavai_pythia_experiment.py
```

That's it. 30 minutes on one GPU. You now have a fused model that beats any individual specialist by 14.2% and beats equal-compute monolithic training by 14.5%.

## What this is

KALAVAI is a cooperative LLM training protocol. Everyone starts from the same checkpoint. Each person trains their copy on a different domain. Nobody talks to each other during training. When everyone's done, a lightweight router learns who's good at what. The fused model outperforms every individual.

The whole algorithm:

```python
# 1. Everyone starts from the same model
base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", revision="step10000")

# 2. Each person trains on their domain (independently, no communication)
specialist_code = train(copy(base), code_data, steps=2000)
specialist_science = train(copy(base), science_data, steps=2000)
specialist_fiction = train(copy(base), fiction_data, steps=2000)

# 3. A router learns who's good at what (500 steps, one linear layer)
router = nn.Linear(hidden_size, 3, bias=False)
fused = MoE(specialist_code, specialist_science, specialist_fiction, router)
train_router(fused, mixed_data, steps=500)

# 4. The fused model is better than any individual
# +14.2% over best specialist, +14.5% over monolithic training
```

No custom CUDA kernels. No distributed training framework. No LoRA. No adapters. Standard PyTorch, standard HuggingFace Transformers, standard training loop. The mechanism is the protocol, not the infrastructure.

## Results

Ran every experiment on one RTX 5090. The 6.9B scale validation on a rented A100 for $25.

| Experiment | Result |
|---|---|
| **MoE vs best specialist** (410M, 3 seeds) | **+14.2% ± 0.016%** |
| **MoE vs best specialist** (1B, 3 seeds) | **+14.8% ± 0.00%** |
| **MoE vs equal-compute monolithic** | **+14.5%** |
| MoE vs best specialist (6.9B, 3 seeds) | +2.43% ± 0.00% |
| Single-specialist hard dispatch | **-21.1%** (worse than base) |
| Wider model, 3.5× parameters | +5.9% (MoE still 2.4× better) |
| Weight averaging | +4.0% (MoE 3.5× better) |
| Qwen-1.5B fully trained | -1.0% (boundary condition) |

322 automated audit checks passed. Zero fabricated numbers. Every result reproducible from committed scripts with fixed seeds.

## The three findings that matter

**1. Frozen layers are optional insurance that becomes essential.**

We swept freeze depth from 0 (no frozen layers) to 12 (half the model). At short training horizons (< 5,000 steps), freezing costs you ~1 percentage point. At long horizons (> 10,000 steps), NOT freezing costs you — specialists drift apart and fusion degrades. The crossover is at exactly 10,000 steps.

```
Steps    freeze=0   freeze=4   Winner
500      +9.9%      +8.9%      freeze=0
1000     +12.5%     +11.3%     freeze=0
2000     +15.1%     +13.9%     freeze=0
5000     +16.4%     +15.8%     freeze=0
10000    +15.4%     +15.6%     freeze=4  ← crossover
20000    +13.6%     +14.8%     freeze=4
```

Practical guideline: if your cooperative trains specialists for under 5,000 steps, skip freezing. Over 10,000 steps, freeze.

**2. You must run all specialists. Single-expert dispatch fails catastrophically.**

A perfect domain classifier (99.3% accuracy) that routes to one specialist gets -21.1%. The MoE that runs all three specialists and selects the best gets +14.1%. Same routing accuracy, opposite results. Specialists forget out-of-domain knowledge. Running all of them and selecting per-token is what works.

```
Method                     Specialists run   Improvement
MoE (run all, select best) 3 of 3            +14.1%
Classifier (pick one)      1 of 3            -21.1%
```

The 35 percentage point gap is the difference between a system that works and one that's worse than the base model.

**3. The improvement isn't from extra parameters.**

A single model with 3.5× the parameters (+5.9%) gets less than half the improvement of the MoE (+14.2%). A multi-head baseline with identical parameter count but hard routing gets -21.1%. The mechanism is cooperative specialization plus selection, not raw capacity.

## Reproduce it

### The 30-minute version (410M, one GPU)

```bash
git clone https://github.com/mechramc/Kalavai.git
cd Kalavai
pip install transformers datasets torch accelerate
python experiments/kalavai_pythia_experiment.py
```

Requires: any GPU with 24GB+ VRAM (RTX 3090, 4090, 5090, A100, etc.)  
Produces: trained specialists, fused MoE, all evaluation numbers, figures.  
Expected output: `+14.2% ± 0.016%` on held-out mixed evaluation.

### The 1B version (same GPU, ~2 hours)

```bash
python experiments/kalavai_pythia_1b_experiment.py
```

Expected output: `+14.8% ± 0.00%`

### The 6.9B version (A100 80GB, ~8 hours)

```bash
python experiments/kalavai_pythia_6b_experiment.py
```

Expected output: `+2.43% ± 0.00%`

### All ablations

```bash
# Freeze depth sweep (0-12 frozen layers)
python experiments/kalavai_freeze_sweep.py

# Router ablation (uniform vs linear vs 2-layer MLP)
python experiments/kalavai_router_ablation.py

# Training duration crossover (500-20,000 steps, freeze=0 vs freeze=4)
python experiments/kalavai_training_duration_crossover.py

# Monolithic baseline (equal-compute comparison)
python experiments/kalavai_monolithic_baseline.py

# Domain classifier baseline (the -21.1% result)
python experiments/kalavai_domain_classifier_baseline.py

# 5-domain scaling (2→5 specialists)
python experiments/kalavai_5domain_experiment.py
```

## How it works

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED CHECKPOINT                         │
│              (e.g., Pythia-1B at step 10000)                │
└─────────┬──────────────┬──────────────┬─────────────────────┘
          │              │              │
     ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
     │  Code   │    │ Science │    │ Fiction │
     │Specialist│    │Specialist│    │Specialist│
     │(2000 steps)│ │(2000 steps)│ │(2000 steps)│
     │ on code │    │on science│    │on fiction│
     │  data   │    │  data   │    │  data   │
     └────┬────┘    └────┬────┘    └────┬────┘
          │              │              │
          │   NO COMMUNICATION DURING   │
          │        TRAINING             │
          │              │              │
     ┌────▼──────────────▼──────────────▼────┐
     │          LEARNED ROUTER               │
     │     nn.Linear(hidden_size, 3)         │
     │     trained 500 steps on mixed data   │
     │     learns: code→expert0              │
     │             science→expert1           │
     │             fiction→expert2            │
     └───────────────────┬───────────────────┘
                         │
                    FUSED MODEL
              +14.2% over best specialist
              +14.5% over monolithic training
```

The frozen layers (if used) stay identical across all specialists. They produce a shared representation that the router reads. The unfrozen layers diverge — each specialist learns its domain. The router learns to dispatch.

At inference, all specialists run. The router produces weights. The output is a weighted sum. In practice, the router hard-switches (>99.7% weight on one expert), but the small residual from other experts is present. Hard routing (argmax) and soft routing (softmax) produce identical results — the mechanism is running all experts and selecting, not the weighting scheme.

## Repository structure

```
Kalavai/
├── experiments/
│   ├── kalavai_pythia_experiment.py       # 410M main experiment (start here)
│   ├── kalavai_pythia_1b_experiment.py    # 1B scale validation
│   ├── kalavai_pythia_6b_experiment.py    # 6.9B scale validation (needs A100)
│   ├── kalavai_freeze_sweep.py            # Freeze depth ablation
│   ├── kalavai_router_ablation.py         # Router architecture comparison
│   ├── kalavai_monolithic_baseline.py     # Equal-compute monolithic comparison
│   ├── kalavai_training_duration_crossover.py  # When frozen layers matter
│   ├── kalavai_domain_classifier_baseline.py   # Why single-expert fails
│   ├── kalavai_5domain_experiment.py      # 2→5 specialist scaling
│   └── kalavai_1b_benchmarks.py           # Downstream task evaluation
├── results/
│   ├── pythia/                            # 410M + 1B results (JSON)
│   └── pythia_6b/                         # 6.9B results (JSON)
├── figures/
│   ├── pythia/                            # 410M + 1B figures
│   └── pythia_6b/                         # 6.9B figures
├── paper/
│   └── kalavai_neurips2026.pdf            # The paper
└── README.md                              # You are here
```

Every experiment is a self-contained Python file. No config files. No YAML. No framework. Read the script, understand the experiment, run it. Each one produces its own results JSON and figures.

## What KALAVAI means

கலவை (kalavai) is Tamil for mixing, fusion, blending. The protocol mixes independently trained specialists into something greater than the parts. The name reflects the method.

## Why this matters

Right now, training a competitive LLM requires millions of dollars of centralized compute. A university in Lagos, a research lab in Chennai, a hobbyist in São Paulo — none of them can build a model that covers their needs.

With KALAVAI, each of them trains one specialist on one GPU on the data they care about. The fused model handles all their domains. Cost per contributor: ~$5-10 in electricity. Cost of the equivalent model trained centrally: $5,000-10,000+.

The code is here. The results are reproducible. The mechanism is validated. What's missing is the community — people willing to contribute a GPU and a domain.

## Pending experiments

The following experiments are written and queued. Results will update the paper before NeurIPS submission.

| Experiment | Script | What it answers | Status |
|---|---|---|---|
| **A1: 1B equal-compute monolithic** | `kalavai_1b_monolithic_baseline.py` | Does KALAVAI beat monolithic at 1B? (NeurIPS gate) | Queued |
| **A2: Inference cost benchmark** | `kalavai_inference_benchmark.py` | Dense vs sparse top-1 MoE latency; routing agreement % at 410M and 1B | Queued |
| **A3: Shared init necessity** | `kalavai_shared_init_ablation.py` | Does fusion break if specialists start from different checkpoints? (NeurIPS gate) | Queued |
| B1: 6.9B step-budget sweep | `kalavai_6b_step_sweep.py` | Does more training fix the weak 6.9B result? (+2.4% → ?) | Pending A1+A3 |
| C1: Heterogeneous cooperative | `kalavai_heterogeneous_cooperative.py` | Robust to different batch sizes, LRs, training durations per contributor? | Pending Phase A |

**NeurIPS gate:** If A1 shows KALAVAI beats 1B monolithic by >5% AND A3 shows clear degradation with mismatched checkpoints, proceed to B1. Otherwise evaluate COLM/TMLR.

## Prior art

KALAVAI builds on ideas from several lines of work:

- **BTX** (Meta, COLM 2024): Branch-train-mix for expert training. KALAVAI adds the freeze analysis and cooperative framing.
- **PHATGOOSE** (ICLR 2024): Decentralized routing. KALAVAI adds monolithic baselines and training duration analysis.
- **Pari** (MIT 2025): CKA analysis of why weight averaging fails. KALAVAI shows when MoE routing succeeds.
- **STAR** (2025): Freeze-then-stack for multimodal. Same underlying principle, different application.

## Citation

```bibtex
@inproceedings{kumaresan2026kalavai,
  title     = {{KALAVAI}: When Does Independent Specialist Fusion Work?
               Conditions for Post-Hoc Cooperative {LLM} Training},
  author    = {Kumaresan, Ramchand},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://github.com/mechramc/Kalavai}
}
```

## License

MIT. Use it. Build on it. Run a cooperative.

---

*Murai Labs — முறை — method, order, disciplined process.*
