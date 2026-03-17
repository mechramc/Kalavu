# KALAVAI: Strategic Summary for Claude Code
# Where We Are, What Reviewers Said, What To Do Next

---

## CURRENT STATE OF RESULTS (as of March 14, 2026)

### Completed Experiments (all committed to git)

| Experiment | Result | Seeds | Status |
|---|---|---|---|
| Synthetic 25M (held-out) | +60.7% ± 0.7% | 3 | Done |
| Pythia-410M 3-domain (held-out) | +14.2% ± 0.016% | 3 | Done |
| Pythia-1B 3-domain (held-out) | +14.8% ± 0.00% | 3 | Done |
| Qwen-1.5B code+fiction (held-out) | -1.0% ± 0.0% | 3 | Done |
| Router ablation (uniform/linear/2-layer) | linear=2-layer=+14.2%, uniform=+6.7% | 1 | Done |
| Freeze depth sweep (0-12 layers) | +14.85% to +12.36%, 2.5pp spread | 1+3 | Done |
| Maturity sweep 410M (6 checkpoints) | +14.97% (step5k) to +14.65% (step143k) | mixed | Done |
| Maturity sweep 1B (5 checkpoints) | +15.85% (step5k) to +14.75% (step143k) | 1 | Done |
| 5-domain scaling (2→5 specialists) | +17.7% (2 spec) to +14.1% (5 spec) | 3 | Done |
| Monolithic baseline (equal compute) | Mono=+6.7% vs base. MoE beats mono by +14.5% | 3 | Done |
| Downstream benchmarks (410M) | Near-random, expected at step10000 | 1 | Done |
| Downstream benchmarks (1B) | MoE leads HellaSwag (35.0%), mono worst (49.3%) | 1 | Done |
| Loss curves (training dynamics) | 3 figures: own-domain, cross-domain, fusion trajectory | 1 | Done |
| Results audit | 322/322 checks passed, 0 issues | — | Done |

### In Progress

| Experiment | Status |
|---|---|
| Pythia-6.9B 3-domain | Script crashed on disk space saving checkpoints. Fix ready (remove torch.save, keep models in GPU memory). Needs re-run on RunPod A100 80GB. |

### Key Figures Generated

- fig_paper_hero.png (4-panel: maturity curves, specialist scaling, freeze depth, router ablation)
- fig_maturity_curve_combined.png (410M + 1B + Qwen on one plot)
- fig_fusion_comparison.png (bar chart all variants)
- fig_divergence_heatmap.png (cross-domain loss matrix)
- fig_router_distribution.png (hard-switching gate weights)
- fig_specialist_own_domain.png (training dynamics — eval loss over steps)
- fig_specialist_cross_domain.png (3-panel: each specialist's cross-domain eval)
- fig_fusion_trajectory.png (fusion benefit growing over training)
- fig_monolithic_comparison.png (base vs mono vs fused)
- fig_monolithic_trajectory.png (mono loss curve vs fused horizontal line)
- Plus all 1B equivalents

---

## REVIEWER FEEDBACK AND REQUIRED RESPONSES

An external technical review raised 10 concerns. Here are the concerns, our honest assessment, and what experiments address each one.

### Concern 1: Frozen Layers Aren't The Core Mechanism

**The issue:** Freeze=0 works (+14.85%). Frozen layers cost 0.7pp for no clear benefit. Is KALAVAI really about frozen layers?

**Our answer:** No. The core mechanism is shared initialization. Frozen layers are optional structural insurance. The paper should reframe: "shared initialization enables post-hoc MoE fusion; frozen layers provide optional verifiable guarantees at minimal cost."

**Experiment needed:** Training duration crossover — run specialists for 500, 1000, 2000, 5000, 10000, 20000 steps at both freeze=0 and freeze=4. Find the point where freeze=0 degrades and freeze=4 holds. This proves frozen layers become necessary at longer horizons.

**Priority:** HIGH. ~6 hours on RTX 5090.

### Concern 2: Divergence ≠ Complementarity

**The issue:** Specialists diverge from base model, but are their errors complementary (when one is wrong, the other is right) or just randomly different?

**Our answer:** The +7.5pp gap between uniform routing (+6.7%) and learned routing (+14.2%) is evidence of complementarity — the router gains value from selecting the right expert per token. Pure ensemble diversity would make uniform routing match learned routing.

**Experiment needed:** Mutual information analysis. For each held-out token, compute log-likelihood under each specialist. Measure error correlation between specialists. Uncorrelated errors = complementary. Correlated errors = just diverse.

**Priority:** MEDIUM. ~30 min on RTX 5090, reuse existing checkpoints.

### Concern 3: Router Is Just A Domain Classifier (MOST DANGEROUS)

**The issue:** The router hard-switches with >99.7% accuracy. A simple domain classifier would do the same thing. The MoE architecture may be unnecessary.

**Our answer:** At the domain granularity tested (3-5 domains), yes, the router is effectively a domain classifier. This is honest and should be reported.

**Experiments needed (two):**

(a) Domain classifier baseline — train a logistic regression on mean-pooled frozen backbone output. Hard-route to matching specialist. Compare against MoE router. If identical (likely), report honestly and argue MoE is the correct generalization framework.

(b) Hybrid-domain evaluation — create prompts mixing domains (e.g., "Write Python code implementing the plot of Romeo and Juliet"). Visualize token-level routing. Does the router switch experts mid-sequence? If yes, it's doing more than document-level classification.

**Priority:** HIGH. ~2.5 hours total on RTX 5090.

### Concern 4: Compute Fairness

**The issue:** Specialists train on single-domain data (easier optimization) while monolithic trains on mixed data (harder, gradient interference). The gain might be from easier training, not from fusion.

**Our answer:** Both effects are real. Decompose: best individual specialist (+7.1% over base) barely beats monolithic (+6.7% over base). The fusion step (+14.2% over base) is what drives the gap. Specialization contributes ~0.4pp, fusion contributes ~7.1pp.

**Experiment needed:** None new — just explicitly decompose and report the numbers.

**Priority:** MEDIUM. Analysis only, no compute.

### Concern 5: Evaluation Scope

**The issue:** Main metric is eval loss. Reviewers want downstream task accuracy.

**Our answer:** 1B benchmarks completed. Near-parity on tasks, MoE leads HellaSwag, mono worst overall. Expected: task differentiation emerges at larger scales.

**Experiment needed:** 6.9B benchmarks (included in the RunPod experiment). At 6.9B, benchmark scores should be high enough to show meaningful differentiation.

**Priority:** HIGH. Part of the 6.9B RunPod run.

### Concern 6: Real Model Limitation

**The issue:** Qwen shows -1.0%. Does KALAVAI only work on undertrained models?

**Our answer:** The maturity sweeps show fusion works across the full Pythia training trajectory at both 410M and 1B (+13-15% at all checkpoints). The Qwen result is model-family-specific (different architecture, much larger pre-training corpus), not a universal maturity effect.

**Experiment needed:** The 6.9B result is the critical data point. If 6.9B at step10000 and step143000 both show +14%, the mechanism works at useful scale regardless of maturity.

**Priority:** CRITICAL. The 6.9B RunPod experiment.

### Concern 7: Hard Routing vs Soft Routing

**The issue:** Router uses softmax but converges to hard switching. Has explicit hard routing been tested?

**Experiment needed:** Replace softmax with argmax, run only the selected expert, compare results. Expected: identical. Takes 15 minutes, reuse existing checkpoints.

**Priority:** LOW. Expected to match, but easy to run.

### Concern 8: Parameter Capacity Confound (SECOND MOST DANGEROUS)

**The issue:** The fused model has 3× the unfrozen parameters of any individual specialist. Improvement could come from more parameters, not from routing.

**Experiments needed (two):**

(a) Wider single model — train one model with 3× FFN width on mixed data for 6000 steps. Same total parameter count as fused model. If MoE beats this, improvement is from routing, not capacity.

(b) Multi-head baseline — 3 independent LM heads on the same frozen trunk, each trained on one domain, classifier selects which head. Same parameter count as MoE. If MoE beats this, routing matters.

**Priority:** HIGH. ~4 hours on RTX 5090.

### Concern 9: How Decentralized Is "Cooperative"?

**The issue:** Contributors must agree on checkpoint, architecture, tokenizer, freeze depth, domains. That's substantial upfront coordination.

**Our answer:** The setup phase requires coordination (like Git requires agreeing on a repo). The training phase requires zero coordination (like coding independently after clone). This should be framed clearly.

**Experiment needed:** Real cooperative simulation — train specialists on 3 different machines (5090, Mac Studio M4, cheap cloud GPU). If heterogeneous hardware produces fusible specialists, the protocol works in practice.

**Priority:** MEDIUM. ~3 hours wall time.

### Concern 10: Long-Horizon Fusibility

**The issue:** Does shared initialization guarantee fusibility at 50,000 training steps?

**Our answer:** Unknown. Not tested beyond 2000 steps. This is explicitly the training duration crossover experiment from Concern 1.

**Priority:** HIGH. Same experiment as Concern 1.

---

## COMPLETE EXPERIMENT QUEUE — PRIORITIZED

### Tier 1: Must-have (addresses critical reviewer objections)

| # | Experiment | Machine | Time | Addresses |
|---|---|---|---|---|
| 1 | Fix + rerun 6.9B experiment | RunPod A100 80GB | ~15 hrs | Concerns 5, 6 |
| 2 | Domain classifier baseline | RTX 5090 | 30 min | Concern 3 |
| 3 | Hybrid-domain routing visualization | RTX 5090 | 2 hrs | Concern 3 |
| 4 | Parameter-matched wider model | RTX 5090 | 2 hrs | Concern 8 |
| 5 | Parameter-matched multi-head baseline | RTX 5090 | 2 hrs | Concern 8 |
| 6 | Training duration crossover (500-20000 steps, freeze=0 vs freeze=4) | RTX 5090 | 6 hrs | Concerns 1, 10 |

### Tier 2: Should-have (strengthens paper significantly)

| # | Experiment | Machine | Time | Addresses |
|---|---|---|---|---|
| 7 | Scale ladder (70M → 160M → 410M → 1B → 2.8B → 6.9B) | RunPod pods | 10 hrs | Scale validation |
| 8 | Hard routing verification (argmax vs softmax) | RTX 5090 | 15 min | Concern 7 |
| 9 | Mutual information / error correlation analysis | RTX 5090 | 30 min | Concern 2 |
| 10 | Improvement decomposition (specialization vs fusion) | Analysis only | 30 min | Concern 4 |
| 11 | CKA similarity tracking during training | RTX 5090 | 2 hrs | Theoretical grounding |

### Tier 3: Nice-to-have (makes paper bulletproof)

| # | Experiment | Machine | Time | Addresses |
|---|---|---|---|---|
| 12 | Maturity × scale 2D matrix (7 sizes × 3 maturities) | RunPod pods | 20 hrs | Comprehensive map |
| 13 | 12B scale point | RunPod A100 80GB | 5 hrs | Maximum scale |
| 14 | Specialist overlap experiment (3 programming languages) | RTX 5090 | 3 hrs | Domain distance |
| 15 | N-specialist scaling to 16-20 | RTX 5090 | 6 hrs | Product thesis |
| 16 | Cross-architecture (Llama, Gemma) | RTX 5090 | 8 hrs | Architecture-agnostic |
| 17 | Adversarial robustness (bad contributor) | RTX 5090 | 1 hr | Cooperative trust |
| 18 | Incremental fusion (add specialist to existing fused model) | RTX 5090 | 1 hr | Growing cooperative |
| 19 | Real cooperative simulation (3 different machines) | Multi-machine | 3 hrs | Protocol realism |

---

## EXECUTION PLAN

### Phase 1: Critical fixes (tonight)
- Fix `kalavai_pythia_6b_experiment.py` using the disk space fix prompt
- Commit fixed script to GitHub
- Spin up RunPod, clone, run 6.9B experiment overnight

### Phase 2: Local experiments on RTX 5090 (while 6.9B runs on RunPod)
Run Tier 1 experiments 2-6 in parallel on local hardware:
1. Domain classifier baseline (30 min)
2. Hybrid-domain routing visualization (2 hrs)
3. Wider model capacity control (2 hrs)
4. Multi-head capacity control (2 hrs)
5. Training duration crossover (6 hrs — run overnight)

### Phase 3: Scale ladder (after 6.9B completes)
Spin up 2-3 RunPod pods:
- Pod 1 (cheap GPU): 70M → 160M → 410M → 1B → 2.8B
- Pod 2 (A100 80GB): 6.9B maturity sweep if not done in Phase 1
- Pod 3 (A100 80GB): 12B experiment if budget allows

### Phase 4: Remaining local experiments
- Hard routing verification
- Mutual information analysis
- CKA tracking
- N-specialist scaling
- Any Tier 3 experiments time allows

### Phase 5: Paper writing
After all experiments complete, write the paper:
- Main body: 9 pages (NeurIPS format)
- Core story: mechanism + 6.9B scale validation + monolithic baseline + maturity curve
- Appendix: all ablations, all figures, full tables
- Upload to arxiv + submit to NeurIPS 2026

---

## KEY FINDINGS TO CARRY FORWARD

These findings are established and should not be re-tested:

1. **Shared initialization is the core mechanism**, not frozen layers (freeze sweep: 2.5pp spread across 0-50%)
2. **Router architecture doesn't matter** — simple linear matches 2-layer MLP (ablation: identical +14.2%)
3. **Router converges to hard switching** — >99.7% gate weight for correct domain expert
4. **Specialist-then-fuse beats equal-compute monolithic by +14.5%**
5. **Mechanism is consistent across 410M and 1B** (+14.2% and +14.8%)
6. **Improvement stable across training maturity** for Pythia models (~13-15% from step5k to step143k)
7. **Qwen at full training shows -1.0%** — model-family-specific boundary condition
8. **Near-zero variance across seeds** (±0.0-0.06% on all experiments)
9. **Adding specialists beyond 3 doesn't dilute improvement** (flat at ~14.1% from 3 to 5)
10. **Uniform routing gets +6.7%** — frozen layers + shared init provide fusibility even without learned routing

## GIT HYGIENE RULES

- One commit per completed experiment step
- Descriptive commit messages: `[kalavai] experiment_name: key_result`
- Never commit model weight files (.pt, .bin, .safetensors) to GitHub — too large
- Always commit results JSONs and figures
- Push after every commit — don't accumulate unpushed work
