# KALAVAI — Project Status

**Last Updated**: 2026-04-10
**Branch**: main
**Tests**: 149 passing (src/kalavai lib) + 13/13 passing (experiments/tests/test_moe_gpu_offload.py)
**Source Lines**: ~2,350 (src) + ~2,400 (tests)

---

## NeurIPS 2026 Paper Sprint

**Submission deadline**: May 4, 2026 (abstract) / May 6, 2026 (full paper)
**Experiment freeze**: April 28, 2026

### Paper status

| Item | Status |
|------|--------|
| Paper file | `paper/kalavai_neurips2026_submit.tex` |
| Main body pages | 9 (bibliography page 10, appendix 11–35) |
| Compilation | 0 fatal errors, 4 Overfull \hbox <20pt |
| NeurIPS reviewer rating | 7.5/10, ~40–50% acceptance probability |
| Abstract | Intuition-first, n=6 caveat, regression formula, key results |
| Contributions | 5 items with Intuitively/Surprisingly/In practice signals |
| Discussion | Cooperative sparsity paragraph (G=2, Kimi K2/DeepSeek-V3), future work |

### Experiment results

| Experiment | Result | Seeds |
|-----------|--------|-------|
| Phase 1: Pythia-410M 3-domain | +7.70% ±0.02pp vs best spec | 3 |
| Phase 1: Pythia-1B 3-domain | +7.49% ±0.01pp vs best spec | 3 |
| Phase 1: Pythia-6.9B 3-domain | +6.53% ±0.024pp vs best spec | 3 |
| Exp1: Cross-lingual (curriculum) | +21.87% ±0.12pp vs best spec | 3 |
| Exp2: Private-domain | +10.17% ±0.15pp vs best spec | 3 |
| Exp3: 20-contributor federation | +16.71% ±0.07pp vs best spec | 3 |
| FE-03: 18-contributor ablation | +21.13% ±0.01pp vs best spec | 3 |

### Regression extension — next GPU run

Designed to push the divergence-gain regression from n=6 to n=8.

| Script | Domains | Expected div | Status |
|--------|---------|-------------|--------|
| `experiments/kalavai_regression_p1_2domain.py` | code + science | ~11% | **Ready to run** |
| `experiments/kalavai_regression_p2_4domain.py` | code+science+fiction+legal | ~19–22% | **Ready to run** |

Run both in parallel on two A10G pods (~$7 total, ~4h wallclock).

---

## Overall Library Progress

| Phase | Status | Tasks | Description |
|-------|--------|-------|-------------|
| 1. Foundation | **Done** | 7/7 | Config, exceptions, checkpoint, CKA, hardware, test infra |
| 2. Cooperative Manager | **Done** | 8/8 | Tokenizer, seed, calibration, reference, manifest, coop CLI |
| 3. Module Trainer | Not started | 0/8 | Training loop, CKA anchor loss, submission |
| 4. Alignment Monitor | Not started | 0/2 | Alignment diagnostics and reporting |
| 5. Fusion Pipeline | Not started | 0/8 | Clustering, MoE fusion, post-training |
| 6. Eval & Publishing | Not started | 0/5 | CORE evaluation, model card, HF Hub |
| 7. Alpha Validation | Not started | 0/2 | 2-module E2E proof-of-concept |
| 8. GitHub Coordination | Not started | 0/4 | Remote repos, Discussions, checkpoint upload |
| 9. v1.0 Polish | Not started | 0/4 | Agent mode, state machine, incremental re-fusion |

**Total**: 15/48 tasks complete (31%)

---

## What Works Today

### CLI Commands (functional)

```bash
kalavai --help
kalavai --version
kalavai coop create --name my-coop --modules 5
kalavai coop join ./my-coop --claim-module 1
kalavai coop status ./my-coop
kalavai coop status ./my-coop --json
```

### CLI Commands (stubbed — not yet wired)

```bash
kalavai coop publish <cooperative>
kalavai train start --module <id>
kalavai train submit --module <id>
kalavai check post
kalavai fuse cluster <cooperative>
kalavai fuse build <cooperative>
kalavai fuse train <cooperative>
```

---

## Module Inventory

### Core (`src/kalavai/core/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `config.py` | 304 | kalavai.yaml schema, typed dataclasses, validation |
| `checkpoint.py` | 307 | Checkpoint save/load, SHA-256 hash validation |
| `model.py` | 221 | GPT-style transformer (RMSNorm, SwiGLU, causal attention) |
| `cka.py` | 77 | Linear CKA computation + differentiable loss |
| `exceptions.py` | ~40 | Exception hierarchy (7 classes) |

### Cooperative Manager (`src/kalavai/coop/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `create.py` | 196 | Orchestrates all 6 cooperative creation steps |
| `tokenizer.py` | 217 | BPE tokenizer (train, encode, decode, save, load) |
| `manifest.py` | 161 | Domain manifest generation + slot management |
| `join.py` | 138 | Artifact download, slot claiming, hash verification |
| `status.py` | 188 | Rich table display + JSON output |
| `calibration.py` | 97 | Calibration batch generation from tokenized corpus |
| `reference.py` | 75 | CKA reference representation computation |
| `seed.py` | 60 | Deterministic seed checkpoint generation |

### Module Trainer (`src/kalavai/train/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `hardware.py` | 87 | CUDA GPU auto-detection |

### CLI

| Module | Lines | Purpose |
|--------|-------|---------|
| `cli.py` | 178 | Click CLI with all command groups wired |

---

## Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_config.py` | 21 | Config parsing, validation, round-trip, defaults |
| `test_fixtures.py` | 15 | Shared fixture validity |
| `test_exceptions.py` | 12 | Exception hierarchy and catch behavior |
| `test_manifest.py` | 10 | Domain manifest CRUD |
| `test_tokenizer.py` | 10 | BPE train, encode/decode, determinism, save/load |
| `test_checkpoint.py` | 9 | Checkpoint save/load, hash validation |
| `test_coop_create.py` | 9 | E2E cooperative creation |
| `test_coop_status.py` | 9 | Status display, JSON output |
| `test_cka.py` | 8 | CKA computation, scale invariance, gradient flow |
| `test_hardware.py` | 8 | CUDA detection (mocked) |
| `test_coop_join.py` | 7 | Slot claiming, artifact copying |
| `test_calibration.py` | 7 | Batch generation, edge cases |
| `test_seed.py` | 13 | Model forward pass, probe extraction, reproducibility |
| `test_reference.py` | 5 | CKA reference computation |
| `test_install.py` | 4 | CLI smoke tests |
| `conftest.py` | — | Shared fixtures |

**Total: 149 tests, 0 failures**
