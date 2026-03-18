# KALAVAI Phase 2: High-Divergence Experiments

**Premise:** The corrected eval data shows KALAVAI's gains scale with specialist divergence at a stable ~0.5× conversion rate:

| Scale | Mean Divergence | Fusion Gain (vs spec) | Conversion Rate |
|-------|----------------|----------------------|-----------------|
| 410M | 15.65% | +7.72% | 0.49× |
| 1B | 15.28% | +7.49% | 0.49× |
| 6.9B | 8.29% | +5.81% | 0.70× |

The conversion rate is stable at 410M/1B and actually improves at 6.9B — the lower absolute gain at 6.9B is entirely explained by lower divergence, not scale-dependent efficiency loss. This means gains are predictable: if specialists diverge X% from base, expect roughly 0.5×X% fusion gain over the best specialist. These three experiments target domains where divergence should be 25-50%+, predicting fusion gains of 12-25%.

**Prerequisites:** All corrected evals complete. Paper accurate with corrected numbers. Eval infrastructure uses per-domain evaluation with equal-weight averaging at consistent batch sizes.

**CRITICAL — Freeze=0 for all Phase 2 experiments.** The crossover finding (Section 4.4) shows freezing hurts below 5,000 steps. All Phase 2 experiments use 2,000 steps. Freeze=0 maximizes specialist divergence, which is the entire point of these experiments. Do NOT let any experiment script default to K=4 from the original templates. Every Phase 2 script must have `FREEZE_LAYERS = 0` hardcoded. Verify this before running.

---

## Experiment 1: Cross-Lingual Fusion

**The pitch:** An English base model that speaks zero Tamil, zero Yoruba, and zero Welsh. Four contributors each teach it one language. The fused model handles all four plus English.

### Setup

**Base model:** Pythia-410M step10000 (English-only pre-training, near-zero competence in other languages)

**Specialists (4 contributors):**

| Specialist | Dataset | Source | Size | Notes |
|-----------|---------|--------|------|-------|
| Tamil | cc100 Tamil subset | `cc100` (`lang="ta"`, `text_key="text"`) | ~180MB text | Classical + modern Tamil. You know this corpus. |
| Yoruba | cc100 Yoruba subset | `cc100` (`lang="yo"`, `text_key="text"`) | ~15MB text | Small but representative. Tests low-resource regime. |
| Welsh | cc100 Welsh subset | `cc100` (`lang="cy"`, `text_key="text"`) | ~50MB text | Celtic language, very different morphology from English. |
| Code | CodeSearchNet Python | `code_search_net` (python) on HuggingFace | ~200MB text | Kept from original experiments as English-adjacent control. |

**Why these languages:** Tamil (Dravidian, your personal connection, culturally meaningful for the Murai Labs story), Yoruba (Niger-Congo, extremely low-resource in LLM training), Welsh (Celtic, has a small but digitized corpus, different from Indo-European mainstream). Code stays as an anchor domain — a known quantity from the original experiments.

**Training config:**
- Freeze: K=0 (below 5k steps, your crossover finding says no freezing)
- Steps: 2,000 per specialist
- Everything else matches the corrected 410M config (AdamW, lr=2e-5, batch size 8, seq length 512)
- Seeds: 3 (42, 137, 2026)

**Evaluation:**
- Per-domain held-out eval for each language + code
- Equal-weight average across all 4 domains
- Also evaluate: base model on each domain (expected: near-random on Tamil/Yoruba/Welsh)
- Also evaluate: each specialist on every other domain (expected: catastrophic forgetting)
- Router gate analysis: 4×4 gate weight matrix per domain

**Expected divergence:** Base model should produce near-maximum loss on Tamil/Yoruba/Welsh text (essentially guessing). Specialists should diverge 40-60% from base on their assigned language. Applying the 0.5× conversion rate, predicted fusion gain is +20-30% over the best single specialist on equal-weight mixed eval.

**Additional eval — raw perplexity reporting:** For the cross-lingual experiment specifically, also report raw perplexity (exp(loss)) for the base model and each specialist on each language. If Pythia-410M produces perplexity 500+ on Tamil while the Tamil specialist produces perplexity 50, that single comparison is more compelling than any percentage. The "near-zero competence → functional competence" story is immediately graspable. Include a table like:

```
Language | Base perplexity | Specialist perplexity | MoE perplexity
Tamil    | ~500+           | ~50                   | ~50
Yoruba   | ~500+           | ~80                   | ~80
Welsh    | ~400+           | ~60                   | ~60
Code     | ~8              | ~6.5                  | ~6.5
```

(Numbers are illustrative — report actuals.)

**What this proves if it works:** KALAVAI enables a genuinely multilingual model from monolingual contributors who never share data. The "20 people, 20 GPUs" vision becomes concrete: a university in Chennai, a lab in Lagos, a group in Cardiff, and a developer anywhere collectively build something none of them could alone.

**Tokenizer caveat:** Pythia uses the GPT-NeoX tokenizer which is English-optimized. Tamil, Yoruba, and Welsh text will tokenize into many more tokens per word than English (potentially 3-5× more), increasing sequence length requirements and potentially degrading perplexity comparisons. This is a known limitation — report tokens-per-word ratio for each language alongside perplexity. If the tokenizer is severely inefficient on a language (>5 tokens per word), note this in the results. The tokenizer limitation does not invalidate the experiment — it reflects real-world conditions where contributors use a shared English-centric base model — but it should be reported honestly.

### Compute

**Hardware:** RTX 5090 (local)
**Training time:** 4 specialists × 2,000 steps × ~0.5s/step = ~70 minutes. Plus router training (10 min). Plus eval (~20 min). Total ~2 hours per seed, ~6 hours for 3 seeds.
**Cost:** Electricity only (~$2-3)

---

## Experiment 2: Private-Domain Fusion (Simulated Institutional Contributors)

**The pitch:** Three institutions with domain expertise in medicine, law, and technical engineering. None can share their training data. Each trains a specialist. The fused model handles cross-domain queries that no single institution could answer.

### Setup

**Base model:** Pythia-410M step10000

**Specialists (3 contributors):**

| Specialist | Dataset | Source | Size | Notes |
|-----------|---------|--------|------|-------|
| Medical | PubMed abstracts | `pubmed` subset of `scientific_papers` on HuggingFace, OR `ccdv/pubmed-summarization` | ~1.5GB text | Dense medical terminology, abbreviations, drug names |
| Legal | EU legislation (eurlex) | `lex_glue`/`eurlex` on HuggingFace (`text_key="text"`) | ~1GB text | Legal reasoning, EU regulatory language. `pile-of-law` is broken at datasets≥3.0. |
| Technical | USPTO patent descriptions | `big_patent` on HuggingFace | ~3GB+ text | Technical writing, specifications, engineering terminology |

**Why these domains:** They map directly to the use cases in the paper. Each domain has specialized vocabulary and reasoning patterns that are underrepresented in Pythia's Pile pre-training relative to their complexity. Unlike code/science/fiction, these domains have minimal overlap — medical, legal, and patent text are genuinely disjoint in terminology.

**Fallback datasets (if primary datasets have loading issues):**

| Specialist | Fallback | Source |
|-----------|----------|--------|
| Medical | Medical dialogue | `medical_dialog` on HuggingFace |
| Legal | EU legislation | `lex_glue`/`eurlex` on HuggingFace (verified working) |
| Technical | ArXiv CS papers | `scientific_papers` (arxiv) on HuggingFace |

**Training config:**
- Freeze: K=0
- Steps: 2,000 per specialist
- Seeds: 3 (42, 137, 2026)
- Same optimizer settings as corrected 410M

**Evaluation:**
- Per-domain held-out eval for each domain
- Equal-weight average across 3 domains
- Monolithic baseline: 6,000 steps on mixed medical+legal+technical data
- Weight averaging baseline
- Compare to corrected code/science/fiction results directly

**Expected divergence:** Medical, legal, and patent text should produce higher divergence than code/science/fiction because Pythia's Pile training has less dedicated coverage of these domains (Pile includes some PubMed and legal text, but patent text is sparse). Expected divergence: 15-25% per domain.

**What this proves if it works:** The gap between divergence and fusion gain that looked problematic at 6.9B on easy domains closes when you use domains where the base model has genuine gaps. The paper's claim that "KALAVAI is most impactful where it's most needed" is validated with data.

### Compute

**Hardware:** RTX 5090 (local)
**Training time:** 3 specialists × 2,000 steps × ~0.5s/step = ~55 minutes. Plus monolithic (6k steps, ~50 min). Plus router + eval (~30 min). Total ~2.5 hours per seed, ~7.5 hours for 3 seeds.
**Cost:** Electricity only (~$2-3)

---

## Experiment 3: 20-Contributor Federation at 1B Scale

**The pitch:** 20 people. 20 GPUs. 1 model. Tested for real (simulated).

### Setup

**Base model:** Pythia-1B step10000

**Specialists (20 contributors across 5 language families + 5 English domains):**

| # | Specialist | Dataset | Source |
|---|-----------|---------|--------|
| 1 | Tamil | cc100 ta | `cc100` (`lang="ta"`, `text_key="text"`) |
| 2 | Yoruba | cc100 yo | `cc100` (`lang="yo"`, `text_key="text"`) |
| 3 | Welsh | cc100 cy | `cc100` (`lang="cy"`, `text_key="text"`) |
| 4 | Spanish | cc100 es (subsample 50MB) | `cc100` (`lang="es"`, `text_key="text"`) |
| 5 | Hindi | cc100 hi | `cc100` (`lang="hi"`, `text_key="text"`) |
| 6 | Swahili | cc100 sw | `cc100` (`lang="sw"`, `text_key="text"`) |
| 7 | Vietnamese | cc100 vi | `cc100` (`lang="vi"`, `text_key="text"`) |
| 8 | Arabic | cc100 ar (subsample 50MB) | `cc100` (`lang="ar"`, `text_key="text"`) |
| 9 | Indonesian | cc100 id | `cc100` (`lang="id"`, `text_key="text"`) |
| 10 | Thai | cc100 th | `cc100` (`lang="th"`, `text_key="text"`) |
| 11 | Code (Python) | CodeSearchNet | `code_search_net` |
| 12 | Medical | PubMed abstracts | `ccdv/pubmed-summarization` |
| 13 | Legal | EU legislation | `lex_glue`/`eurlex` (`text_key="text"`, `trust_remote_code=True`) |
| 14 | Patents | Patent descriptions | `big_patent` |
| 15 | Math | GSM8K + MATH | `gsm8k` + `hendrycks/math` |
| 16 | Finance | SEC filings | `JanosAudworking/sec-filings-10k` or `edgartools` |
| 17 | Chemistry | Chemistry papers | `chemrxiv` subset or `scientific_papers` |
| 18 | Fiction | PG-19 books | `pg19` |
| 19 | Dialogue | DailyDialog + PersonaChat | `daily_dialog` + `bavard/personachat_truecased` |
| 20 | Instructions | Dolma/FLAN subset | `Open-Orca/FLAN` (subsample) |

**Why 20:** The paper's tagline is "20 people, 20 GPUs, 1 model." This experiment tests it literally. The mix of 10 languages + 10 English domains maximizes diversity. Languages span Dravidian, Niger-Congo, Celtic, Romance, Indo-Aryan, Bantu, Vietic, Semitic, Austronesian, and Tai — near-maximal linguistic diversity.

**Training config:**
- Freeze: K=0
- Steps: 2,000 per specialist (same compute per contributor)
- Seeds: 1 (42) for the full 20-specialist run. If results are strong, seed 137 for confirmation.
- Router: nn.Linear(hidden_size, 20) — same linear router, just wider

**Evaluation:**
- Per-domain held-out eval for all 20 domains
- Equal-weight average across all 20
- Also test: 5-specialist subset (just languages), 10-specialist subset (just languages + code/medical/legal/patent/math), full 20
- Router gate analysis: 20×20 gate weight matrix (do languages cluster? do English domains cluster?)
- Compare to: base model, best single specialist on mixed eval

**Expected result:** Base model is near-random on most of the 10 non-English languages. Each language specialist should diverge 40-60% from base. The fused model covering all 20 domains should massively outperform any single specialist on the equal-weight mixed eval. This is the experiment where the protocol's value is undeniable.

**What this proves if it works:** The "20 people, 20 GPUs" tagline is real. Cooperative training at 20 contributors produces a model that no single contributor could build, covering 10 languages and 10 domains simultaneously. The gains should be the largest in the paper because the domains are maximally diverse.

### Compute

**Hardware:** RunPod A100 80GB for the full run (training + eval).

**Why RunPod for everything, not just eval:**
- Training 20 specialists sequentially on 5090 takes ~11 hours. A100 cuts this to ~5-6 hours.
- MoE eval with 20 specialists loaded (20 × ~2GB = ~40GB) doesn't fit on 5090 (24GB VRAM). A100 80GB handles it natively.
- Running everything on one machine avoids transferring 20 specialist checkpoints (~40GB total) between machines.
- Total A100 time: ~7 hours. Cost: ~$12.

**RunPod setup checklist (before deploying):**
1. Experiment script must import from `kalavai_eval_utils.py` — no inline eval code
2. `FREEZE_LAYERS = 0` hardcoded — verify before running
3. All 20 datasets verified loadable (run the dataset check from pre-Phase-2 checklist locally first)
4. Git auth configured for result pushes after each specialist completes
5. Disk volume: 150GB (20 specialist checkpoints at ~4GB each in bf16 + datasets + HF cache)
6. Network volume: 20GB (for result JSONs and figures — survives pod preemption)

**Execution on RunPod:**
```bash
# Clone, setup
git clone https://github.com/mechramc/Kalavai.git && cd Kalavai
pip install transformers datasets torch accelerate matplotlib
pip install datasets==2.19.0  # pin for trust_remote_code compat

# Run
python -u experiments/kalavai_20contributor_experiment.py 2>&1 | tee exp3_log.txt

# Push results after completion
git add results/ figures/ && git commit -m "Exp3: 20-contributor 1B results" && git push
```

**Cost estimate:**
- A100 80GB: ~7 hours × $1.64/hr = **~$12**

**Parallel execution:** Run Experiment 3 on RunPod simultaneously with Experiment 1 on local RTX 5090. Both start after Experiment 2's stop/go decision.

---

## Summary

| Experiment | Hardware | Time (1 seed) | Total Time | Cost |
|-----------|----------|--------------|-----------|------|
| 2: Private-domain (3 specialists, 410M) | RTX 5090 | ~2.5 hrs | ~7.5 hrs (3 seeds) | ~$2-3 |
| 1: Cross-lingual (4 specialists, 410M) | RTX 5090 | ~2 hrs | ~6 hrs (3 seeds) | ~$2-3 |
| 3: 20-contributor (20 specialists, 1B) | RunPod A100 | ~7 hrs | ~7 hrs (1 seed) | ~$12 |

**Total: ~20.5 hours wall-clock (Exp 1 and 3 run in parallel), ~$16-18**

## Execution Order

1. **Experiment 2 first** (private-domain, RTX 5090). Fastest canary. If medical/legal/patent specialists diverge >15% and fusion gain is >7%, the high-divergence thesis is confirmed. Takes ~7.5 hours for 3 seeds.

2. **After Experiment 2 stop/go clears — run Experiments 1 and 3 in parallel:**
   - **Experiment 1** (cross-lingual) on **RTX 5090**. Tamil, Yoruba, Welsh. ~6 hours for 3 seeds.
   - **Experiment 3** (20-contributor) on **RunPod A100**. Spin up pod, run script, push results, terminate. ~7 hours.
   - Both run simultaneously. Wall-clock: ~7 hours for both.

3. **Total wall-clock from start to all results:** ~14.5 hours (7.5 for Exp 2, then 7 for Exp 1+3 in parallel).

## Stop/Go Criteria

**After Experiment 2:**
- **Go** (proceed to Exp 1+3): Specialist divergence >15% mean AND fusion gain >7% vs best specialist (consistent with 0.5× conversion rate)
- **Strong go**: Divergence >25% AND gain >12% — confirms high-divergence regime produces proportionally larger gains
- **Pivot** (rethink): Divergence is >20% but fusion gain is <5% — conversion rate breaks down on new domains
- **Stop**: Divergence is low (<10%) even on medical/legal/patent — Pythia's Pile training covers these domains better than expected

**After Experiment 1:**
- **Go** (proceed to Exp 3): Cross-lingual divergence >40% AND fusion gain >15% — the democratic multilingual story is real
- **Cautious go**: Divergence >40% but gain only 10-15% — conversion rate degrades on extreme divergence (interesting finding in itself)
- **Stop**: Router fails to distinguish languages — fundamental routing problem with non-English tokens (check if tokenizer is the bottleneck — Pythia's tokenizer is English-optimized)

## Dataset Verification Checklist

Before running, verify each dataset loads correctly:

```python
from datasets import load_dataset

# Experiment 1 — cc100 language subsets (text_key="text")
cc100_ta = load_dataset("cc100", lang="ta", split="train", streaming=True, trust_remote_code=True)
cc100_yo = load_dataset("cc100", lang="yo", split="train", streaming=True, trust_remote_code=True)
cc100_cy = load_dataset("cc100", lang="cy", split="train", streaming=True, trust_remote_code=True)
print(next(iter(cc100_ta))["text"][:100])   # verify text key

# Experiment 2 — private domains
pubmed  = load_dataset("ccdv/pubmed-summarization", split="train", streaming=True, trust_remote_code=True)
legal   = load_dataset("lex_glue", "eurlex", split="train", streaming=True, trust_remote_code=True)  # pile-of-law broken
patents = load_dataset("big_patent", "a", split="train", streaming=True, trust_remote_code=True)

# Experiment 3 — all 10 cc100 languages + domain datasets
cc100_es = load_dataset("cc100", lang="es", split="train", streaming=True, trust_remote_code=True)
cc100_hi = load_dataset("cc100", lang="hi", split="train", streaming=True, trust_remote_code=True)
cc100_sw = load_dataset("cc100", lang="sw", split="train", streaming=True, trust_remote_code=True)
cc100_vi = load_dataset("cc100", lang="vi", split="train", streaming=True, trust_remote_code=True)
cc100_ar = load_dataset("cc100", lang="ar", split="train", streaming=True, trust_remote_code=True)
cc100_id = load_dataset("cc100", lang="id", split="train", streaming=True, trust_remote_code=True)
cc100_th = load_dataset("cc100", lang="th", split="train", streaming=True, trust_remote_code=True)
```

**Notes:**
- All `load_dataset` calls use `trust_remote_code=True` — required for cc100 and some other HF datasets
- `pile-of-law/pile-of-law` is broken (dataset scripts no longer supported at datasets≥3.0). Use `lex_glue`/`eurlex` as the legal dataset — verified working
- cc100 text key is `"text"` for all language subsets
- Run this verification before starting any training. Better to discover loading issues before burning compute.
