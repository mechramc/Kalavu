# KALAVAI — கலவை

20 people. 20 GPUs. 1 model none of them could build alone.

**Fusion gain ≈ 0.81 × divergence − 2.52** (R² = 0.850). Before you train a single specialist, you can predict whether the cooperative is worth it.

```
pip install transformers datasets torch
python experiments/kalavai_pythia_experiment.py
```

30 minutes on one GPU. The fused model beats any individual specialist by +7.72% and achieves oracle-optimal routing — matching the best specialist on every domain simultaneously.

## What this is

KALAVAI is a zero-communication cooperative LLM training protocol. Everyone starts from the same checkpoint. Each person trains their copy on a different domain — their language, their field, their data. Nobody talks to each other during training. When everyone's done, a lightweight router learns who's good at what. The fused model outperforms every individual.

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
# +7.72% over best specialist (corrected per-domain equal-weight eval)
```

No custom CUDA kernels. No distributed training framework. No LoRA. No adapters. Standard PyTorch, standard HuggingFace Transformers, standard training loop. The mechanism is the protocol, not the infrastructure.

## Results

All Phase 1 experiments ran on one RTX 5090. Phase 2 cross-lingual and 20-contributor experiments on rented H100s. All results use the corrected per-domain equal-weight evaluation protocol.

### Phase 1: English Domains (code / science / fiction)

| Scale | vs. Best Specialist | vs. Base | Seeds |
|---|---|---|---|
| Pythia-410M | **+7.72% ± 0.02%** | +16.3% | 3 |
| Pythia-1B | **+7.49% ± 0.01%** | +15.5% | 3 |
| Pythia-6.9B | **+6.53% ± 0.024%** | +8.6% | 3 |
| Qwen-1.5B | **+1.06% ± 0.01%** | — | 3 |

### Phase 2: High-Divergence Domains

| Experiment | Domains | vs. Best Specialist | Mean Divergence |
|---|---|---|---|
| Private-domain (410M) | Medical / Legal / Patent | **+10.17% ± 0.15pp** | 18.52% |
| Cross-lingual (410M) | Tamil / Yoruba / Welsh / Code | **+21.76% ± 0.005pp** | 25.65% |
| 20-contributor (1B) | 10 languages + 10 domains | **+16.79%** | 15.71% |

Cross-lingual highlights: Yoruba perplexity 41.9 → 7.7 (5.4×). Welsh 102.7 → 22.1 (4.6×). Contributors speaking different languages collectively built a model none could train alone.

### The Predictive Model

Across all experimental conditions, fusion gain scales linearly with specialist divergence:

```
gain ≈ 0.81 × divergence − 2.52    (R² = 0.850, n = 6)
```

Before committing to a cooperative, measure how much your specialists diverge from the base model. If divergence is 15%, expect ~+10% gain. If divergence is 25% (cross-lingual), expect ~+18% — and likely more, since high-divergence settings exceed the linear prediction. Below ~3.1% divergence, expect no gain.

| Condition | Mean Div. | Gain | Predicted | Residual |
|---|---|---|---|---|
| Qwen-1.5B | 3.16% | +1.06% | ≈0% | — |
| Pythia-6.9B | 8.29% | +6.53% | +4.17% | +2.36pp |
| Pythia-1B | 15.28% | +7.49% | +9.81% | −2.32pp |
| Pythia-410M | 15.65% | +7.72% | +10.11% | −2.39pp |
| Private-domain | 18.52% | +10.17% | +12.43% | −2.26pp |
| Cross-lingual | 25.65% | +21.76% | +18.18% | +3.58pp |
| 20-contributor | 15.71% | +16.79% | +10.16% | +6.63pp |

### Key Controls

| Experiment | Result |
|---|---|
| Single-specialist dispatch (99.3% accurate classifier) | **−21.1%** (catastrophic) |
| Wider model, 3.5× parameters | +5.9% (MoE still better) |
| Weight averaging | −3.4% vs best specialist |
| Equal-compute monolithic (410M) | MoE beats by +0.47% on aggregate; MoE wins per-domain |
| Oracle routing gap (410M) | 3 × 10⁻⁶ nats — routing is saturated |

322 automated audit checks passed. Every result reproducible from committed scripts with fixed seeds.

## The Three Findings That Matter

**1. Frozen layers are optional insurance that becomes essential.**

At short training horizons (≤2,000 steps), freezing costs ~0.5pp. The crossover is at ~5,000 steps — beyond that, unfrozen specialists over-specialise and fusion degrades. freeze=0 peaks at 2,000 steps (+8.12%); freeze=4 overtakes it at 5,000 steps.

```
Steps    freeze=0   freeze=4   Winner
500      +5.88%     +5.31%     freeze=0
1000     +5.94%     +6.48%     freeze=4 (marginal)
2000     +8.12%     +7.56%     freeze=0  ← freeze=0 peak
5000     +7.79%     +8.07%     freeze=4  ← crossover
10000    +5.83%     +7.33%     freeze=4
20000    +3.38%     +6.30%     freeze=4
```

**2. You must run all specialists. Single-expert dispatch fails catastrophically.**

A near-perfect domain classifier (99.3%) routing to one specialist: −21.1%. The MoE running all specialists with learned routing: +7.72% vs best specialist. Specialists forget out-of-domain knowledge. Running all of them and combining per-token is what works.

**3. The improvement isn't from extra parameters.**

A single model with 3.5× the parameters gets +5.9%. A multi-head baseline with identical parameter count gets −21.1%. The MoE gets +7.72%. The mechanism is cooperative specialisation plus joint inference, not raw capacity.

## Reproduce It

### The 30-minute version (410M, one GPU)

```bash
git clone https://github.com/mechramc/Kalavai.git
cd Kalavai
pip install transformers datasets torch accelerate
python experiments/kalavai_pythia_experiment.py
```

Requires: any GPU with 24GB+ VRAM (RTX 3090, 4090, 5090, A100, etc.)
Expected output: `+7.72% ± 0.02%` on corrected per-domain equal-weight evaluation.

### The 1B version (~2 hours)

```bash
python experiments/kalavai_pythia_1b_experiment.py
```

### The 6.9B version (A100 80GB, ~8 hours)

```bash
python experiments/kalavai_pythia_6b_experiment.py
```

### Phase 2 experiments

```bash
# Private-domain fusion (medical / legal / patent)
python experiments/kalavai_private_domain_experiment.py

# Cross-lingual fusion (Tamil / Yoruba / Welsh / Code)
python experiments/kalavai_crosslingual_experiment.py

# 20-contributor federation (10 languages + 10 domains, needs H100)
python experiments/kalavai_20contributor_experiment.py
```

### All ablations

```bash
python experiments/kalavai_freeze_sweep.py                    # Freeze depth (0-12 layers)
python experiments/kalavai_router_ablation.py                 # Router architecture comparison
python experiments/kalavai_training_duration_crossover.py     # When frozen layers matter
python experiments/kalavai_monolithic_baseline.py             # Equal-compute comparison
python experiments/kalavai_domain_classifier_baseline.py      # Why single-expert fails
python experiments/kalavai_5domain_experiment.py              # 2→5 specialist scaling
python experiments/kalavai_shared_init_ablation.py            # Checkpoint mismatch effects
python experiments/kalavai_heterogeneous_cooperative.py       # Robustness to contributor variation
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED CHECKPOINT                         │
│              (e.g., Pythia-1B at step 10000)                │
└─────────┬──────────────┬──────────────┬─────────────────────┘
          │              │              │
     ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
     │  Code   │    │ Science │    │ Fiction │
     │Specialist│   │Specialist│   │Specialist│
     │(2k steps)│   │(2k steps)│   │(2k steps)│
     └────┬────┘    └────┬────┘    └────┬────┘
          │              │              │
          │   NO COMMUNICATION DURING   │
          │        TRAINING             │
          │              │              │
     ┌────▼──────────────▼──────────────▼────┐
     │          LEARNED ROUTER               │
     │     nn.Linear(hidden_size, N)         │
     │     500 steps on mixed data           │
     │     Routing: near-deterministic       │
     │     (>99.7% weight on best expert)    │
     └───────────────────┬───────────────────┘
                         │
                    FUSED MODEL
              Oracle-optimal routing
              Per-domain specialist quality
              on every domain simultaneously
```

At inference, all specialists run in parallel. The router produces per-token weights. In practice, routing is near-deterministic (>99.7% weight on one expert), and at 410M the learned router matches the domain-level oracle with gap < 10⁻⁵ nats. A linear router is sufficient — a 2-layer MLP router produces identical results.

## Repository Structure

```
Kalavai/
├── experiments/
│   ├── kalavai_pythia_experiment.py          # 410M main (start here)
│   ├── kalavai_pythia_1b_experiment.py       # 1B scale
│   ├── kalavai_pythia_6b_experiment.py       # 6.9B scale (needs A100)
│   ├── kalavai_private_domain_experiment.py  # Phase 2: medical/legal/patent
│   ├── kalavai_crosslingual_experiment.py    # Phase 2: Tamil/Yoruba/Welsh/Code
│   ├── kalavai_20contributor_experiment.py   # Phase 2: 20 specialists (needs H100)
│   ├── kalavai_eval_utils.py                 # Corrected evaluation protocol
│   ├── kalavai_freeze_sweep.py               # Freeze depth ablation
│   ├── kalavai_router_ablation.py            # Router architecture comparison
│   ├── kalavai_monolithic_baseline.py        # Equal-compute comparison
│   ├── kalavai_training_duration_crossover.py
│   ├── kalavai_domain_classifier_baseline.py
│   ├── kalavai_5domain_experiment.py
│   ├── kalavai_shared_init_ablation.py
│   ├── kalavai_heterogeneous_cooperative.py
│   ├── kalavai_1b_benchmarks.py
│   └── kalavai_inference_benchmark.py
├── results/
│   ├── pythia/                               # Phase 1 results (JSON)
│   ├── pythia_6b/                            # 6.9B results
│   └── phase2/                               # Phase 2 results
│       ├── private_domain/
│       ├── crosslingual/
│       └── twenty_contributor/
├── figures/
├── paper/
│   └── kalavai_neurips2026.pdf
└── README.md
```

Every experiment is a self-contained Python file. No config files. No YAML. No framework. Read the script, understand the experiment, run it.

## Evaluation Note

Initial experiments produced +14.2% at 410M. Code review identified two evaluation inconsistencies — asymmetric batch sizes between the MoE and baselines, and a concatenated mixed evaluation that systematically underrepresented the fiction domain. The corrected per-domain equal-weight protocol (`kalavai_eval_utils.py`) yields +7.72%. All results in this repository and the paper use the corrected protocol. The inconsistencies and fix are documented in Appendix R of the paper.

## Camera-Ready Roadmap

These experiments are planned for the camera-ready version if the paper is accepted at NeurIPS 2026.

| Experiment | Purpose | Status |
|---|---|---|
| LoRA ablation (r=8, r=64) at 410M | Preempt reviewer objection: does LoRA produce sufficient divergence? | Done — LoRA r=64 produces *negative* divergence (−20% div, −13.9% gain); full FT is necessary |
| Base-PPL as conversion rate predictor | Explain why cross-lingual exceeds the linear prediction | Done — r=+0.613 (n=6, suggestive); integrated into §4.10 |
| Low-divergence ablation (50-100 training steps) | Find the divergence floor where gains go to zero | Planned |
| 20-contributor with robust data (replace thin domains) | Clean Exp3 without data-insufficient specialists | Planned |
| Multi-round contributors (thicker specialists) | Realistic cooperative: 3 rounds per contributor, fewer but deeper specialists | Planned |
| Continual cooperative (add specialist post-hoc) | Can a 4th specialist join without retraining the first 3? | Planned |

## Prior Art

- **BTX** (Meta, COLM 2024): Branch-train-mix for expert training. KALAVAI adds the predictive divergence-gain model, freeze crossover analysis, and cooperative framing.
- **PHATGOOSE** (ICLR 2024): Decentralised routing among fine-tuned models. KALAVAI adds monolithic baselines, oracle routing analysis, and training duration analysis.
- **Pari** (MIT 2025): CKA analysis of why weight averaging fails. KALAVAI shows when MoE routing succeeds and provides the empirical complement.
- **STAR** (2025): Freeze-then-stack for multimodal. Same underlying principle, different application.

## Why This Matters

Training a competitive LLM requires millions of dollars of centralised compute. A university in Lagos, a research lab in Chennai, a hobbyist in São Paulo — none of them can build a model that covers their needs.

With KALAVAI, each trains one specialist on one GPU on the data they care about. The Yoruba contributor's specialist cuts perplexity from 41.9 to 7.7. The legal contributor's specialist diverges 34% from base. The fused model handles all their domains. Cost per contributor: ~$5-10 in electricity.

The code is here. The results are reproducible. The predictive model tells you whether a cooperative is worth it before you start.

## What KALAVAI Means

கலவை (kalavai) is Tamil for mixing, fusion, blending. The protocol mixes independently trained specialists into something greater than the parts.

## Citation

```bibtex
@article{kumaresan2026kalavai,
  title     = {{KALAVAI}: Predicting When Independent Specialist Fusion Works
               --- A Quantitative Model for Post-Hoc Cooperative {LLM} Training},
  author    = {Kumaresan, Ramchand},
  journal   = {arXiv preprint},
  year      = {2026},
  url       = {https://github.com/mechramc/Kalavai}
}
```

## License

MIT. Use it. Build on it. Run a cooperative.

---

*Murai Labs — முறை — method, order, disciplined process.*
