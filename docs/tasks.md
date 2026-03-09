# Task Plan: KALAVU

**Generated**: 2026-03-09
**Source**: KALAVU_v3_Product_Specification.docx, CLAUDE.md
**Total Tasks**: 48
**Scope**: Alpha → Beta → v1.0 (Weeks 1–12)

---

## Phase 1: Foundation (Week 1)

> Infrastructure that every subsystem depends on. Must complete first.

### Epic: Core Config & Data Models
> Priority: P0 | Effort: M | Dependencies: None

- [ ] **TASK-001**: Implement `kalavu.yaml` schema and config parser
  - **Effort**: M
  - **Dependencies**: None
  - **Component**: `src/kalavu/core/config.py`
  - **Acceptance**: Parse the full cooperative config schema from spec section 6.1 (cooperative name, modules, target_params, architecture block, alignment block, fusion block, domains list). Validate required fields, raise `ConfigError` on invalid YAML. Round-trip: load → validate → access via typed dataclass.
  - **Notes**: This is the single most depended-on task. Every subsystem reads this config.

- [ ] **TASK-002**: Define custom exception hierarchy
  - **Effort**: XS
  - **Dependencies**: None
  - **Component**: `src/kalavu/core/exceptions.py`
  - **Acceptance**: Create `KalavuError` base, plus `ConfigError`, `AlignmentError`, `CheckpointValidationError`, `FusionError`, `CooperativeError`. All inherit from `KalavuError`.

- [ ] **TASK-003**: Implement checkpoint format handler
  - **Effort**: S
  - **Dependencies**: TASK-001
  - **Component**: `src/kalavu/core/checkpoint.py`
  - **Acceptance**: Define checkpoint directory structure (model weights, alignment probes, alignment report JSON, training metadata JSON, artifact hashes). Implement `save_checkpoint()` and `load_checkpoint()`. Validate artifact hashes (tokenizer, seed) against cooperative config on load.

- [ ] **TASK-004**: Implement CKA (Centered Kernel Alignment) computation
  - **Effort**: M
  - **Dependencies**: None
  - **Component**: `src/kalavu/core/cka.py`
  - **Acceptance**: Implement linear CKA between two sets of hidden-state representations (tensors of shape `[N, D]`). Must handle batched computation. Return float in [0, 1]. Include `cka_loss(h_module, h_reference) -> tensor` that returns `1 - CKA` for use as a loss term. Unit test: CKA of identical representations = 1.0, orthogonal ≈ 0.0.

- [ ] **TASK-005**: Implement hardware auto-detection
  - **Effort**: S
  - **Dependencies**: None
  - **Component**: `src/kalavu/train/hardware.py`
  - **Acceptance**: Detect CUDA GPU (device name, VRAM), report available hardware as structured dict. For MVP, only CUDA support required. Return `{"device": "cuda", "name": "RTX 5090", "vram_gb": 32}` or raise error if no CUDA GPU found. Print human-readable summary via `rich`.

### Epic: Dev Tooling
> Priority: P0 | Effort: S | Dependencies: None

- [ ] **TASK-006**: Set up pytest fixtures and test infrastructure
  - **Effort**: S
  - **Dependencies**: None
  - **Component**: `tests/conftest.py`
  - **Acceptance**: Create fixtures for: temporary cooperative directory, sample `kalavu.yaml`, mock seed checkpoint (small random tensor), mock calibration batch. All tests should run without GPU (mock torch.cuda).

- [ ] **TASK-007**: Add `pip install -e ".[dev]"` smoke test in CI-compatible script
  - **Effort**: XS
  - **Dependencies**: TASK-001
  - **Component**: `tests/test_install.py`
  - **Acceptance**: Test that `kalavu --help` runs and prints usage. Test that `kalavu --version` prints `0.1.0`.

---

## Phase 2: Cooperative Manager — `kalavu coop` (Weeks 1–2)

> Create and manage cooperatives. This is Phase 0 in the product workflow.

### Epic: Cooperative Creation
> Priority: P0 | Effort: L | Dependencies: TASK-001, TASK-004

- [ ] **TASK-008**: Implement BPE tokenizer training via minbpe
  - **Effort**: M
  - **Dependencies**: TASK-001
  - **Component**: `src/kalavu/coop/create.py`
  - **Acceptance**: Given a corpus path (or default multi-domain corpus config), train a BPE tokenizer using minbpe and save as `tokenizer.model`. Configurable vocab size from `kalavu.yaml`. Deterministic: same corpus → same tokenizer.

- [ ] **TASK-009**: Implement canonical seed checkpoint generation
  - **Effort**: M
  - **Dependencies**: TASK-001, TASK-003
  - **Component**: `src/kalavu/coop/create.py`
  - **Acceptance**: Initialize a nanochat-compatible model at the architecture config (depth, d_model, n_heads, ffn_ratio, norm) with a fixed random seed. Save as `seed_checkpoint.pt`. Hash the checkpoint for integrity verification. Must be reproducible: same config + same seed = identical checkpoint.

- [ ] **TASK-010**: Compute CKA reference representations
  - **Effort**: S
  - **Dependencies**: TASK-004, TASK-009
  - **Component**: `src/kalavu/coop/create.py`
  - **Acceptance**: Run the seed model on the calibration batch, extract hidden states at probe layers (25%, 50%, 75% depth). Save as `cka_reference.pt`. These are the alignment targets for all modules.

- [ ] **TASK-011**: Generate calibration batch
  - **Effort**: S
  - **Dependencies**: TASK-008
  - **Component**: `src/kalavu/coop/create.py`
  - **Acceptance**: Tokenize ~1024 sequences from a configurable corpus using the trained tokenizer. Save as `calibration_batch.pt`. Fixed once per cooperative.

- [ ] **TASK-012**: Generate domain manifest
  - **Effort**: S
  - **Dependencies**: TASK-001
  - **Component**: `src/kalavu/coop/create.py`
  - **Acceptance**: Generate `domain_manifest.json` with N slots (from config), each with id, name, data_hint, and status ("open"). Default domains from spec (Code, Math, Bio, Legal, History, etc.). Allow custom domains via `kalavu.yaml`.

- [ ] **TASK-013**: Wire `kalavu coop create` end-to-end
  - **Effort**: M
  - **Dependencies**: TASK-008, TASK-009, TASK-010, TASK-011, TASK-012
  - **Component**: `src/kalavu/cli.py`, `src/kalavu/coop/create.py`
  - **Acceptance**: `kalavu coop create --name test-coop --modules 5` produces a directory with: `kalavu.yaml`, `tokenizer.model`, `seed_checkpoint.pt`, `calibration_batch.pt`, `cka_reference.pt`, `domain_manifest.json`. All files valid and loadable.

### Epic: Cooperative Membership
> Priority: P0 | Effort: M | Dependencies: TASK-013

- [ ] **TASK-014**: Implement `kalavu coop join` — download artifacts and claim slot
  - **Effort**: M
  - **Dependencies**: TASK-013
  - **Component**: `src/kalavu/coop/join.py`
  - **Acceptance**: Given a cooperative path (local directory for Alpha; GitHub repo URL for Beta), download all shared artifacts (tokenizer, seed, calibration batch, CKA reference). Claim a domain slot by ID. Update `domain_manifest.json` to mark slot as "claimed". Validate artifact hashes.

- [ ] **TASK-015**: Implement `kalavu coop status` — display cooperative health
  - **Effort**: S
  - **Dependencies**: TASK-013
  - **Component**: `src/kalavu/coop/status.py`
  - **Acceptance**: Read all module statuses and display a `rich` table showing: module ID, domain name, contributor, status (open/claimed/training/submitted), CKA scores at each probe layer, training progress %. Show summary line (e.g., "3/5 modules submitted, 2/5 aligned").

---

## Phase 3: Module Trainer — `kalavu train` (Weeks 2–3)

> The core training loop with CKA alignment constraints.

### Epic: Training Loop with CKA Anchor Loss
> Priority: P0 | Effort: XL | Dependencies: TASK-004, TASK-009, TASK-014

- [ ] **TASK-016**: Implement nanochat model wrapper
  - **Effort**: M
  - **Dependencies**: TASK-001
  - **Component**: `src/kalavu/train/model.py`
  - **Acceptance**: Wrap nanochat's GPT model to expose: `forward()` with optional hidden-state extraction at probe layers, `get_probe_representations(batch)` → dict of layer_idx → tensor. Model config loaded from `kalavu.yaml` architecture block. Initialize from seed checkpoint.

- [ ] **TASK-017**: Implement CKA anchor loss integration into training loop
  - **Effort**: L
  - **Dependencies**: TASK-004, TASK-016
  - **Component**: `src/kalavu/train/start.py`
  - **Acceptance**: Training loop computes `L_total = L_lm + λ * (1 - CKA(h_module, h_reference))` every K steps (configurable, default 500). λ follows cosine annealing schedule from `kalavu.yaml` (default: 0.05 → 0.01 over final 30% of training). Calibration batch and CKA reference loaded once at startup.

- [ ] **TASK-018**: Implement training data loading with domain assignment
  - **Effort**: M
  - **Dependencies**: TASK-008
  - **Component**: `src/kalavu/train/data.py`
  - **Acceptance**: Load training data for the assigned domain. Accept a data directory path. Tokenize with the cooperative's frozen tokenizer. Return a PyTorch DataLoader. Support streaming for large datasets.

- [ ] **TASK-019**: Implement alignment pause/warning system
  - **Effort**: S
  - **Dependencies**: TASK-017
  - **Component**: `src/kalavu/train/start.py`
  - **Acceptance**: If CKA at any probe layer drops below the cooperative's threshold during training, pause and display warning with remediation suggestions (reduce LR, increase λ, rollback to last aligned checkpoint). Log the event. Allow `--force` flag to continue despite warning.

- [ ] **TASK-020**: Implement training telemetry and progress logging
  - **Effort**: S
  - **Dependencies**: TASK-017
  - **Component**: `src/kalavu/train/start.py`
  - **Acceptance**: Log at configurable intervals: step, loss (LM + CKA components), CKA scores at probe layers, val_bpb on calibration batch, tokens/sec, VRAM usage, ETA. Output as structured JSON lines to a log file. Display live progress via `rich` progress bar.

- [ ] **TASK-021**: Wire `kalavu train start` end-to-end
  - **Effort**: M
  - **Dependencies**: TASK-016, TASK-017, TASK-018, TASK-019, TASK-020, TASK-005
  - **Component**: `src/kalavu/cli.py`, `src/kalavu/train/start.py`
  - **Acceptance**: `kalavu train start --module 1` loads cooperative config, initializes model from seed, starts training with CKA anchor loss, logs telemetry, saves checkpoints at intervals. Can be interrupted and resumed.

### Epic: Module Submission
> Priority: P0 | Effort: M | Dependencies: TASK-021

- [ ] **TASK-022**: Implement submission validation
  - **Effort**: M
  - **Dependencies**: TASK-004, TASK-003
  - **Component**: `src/kalavu/train/submit.py`
  - **Acceptance**: Run final alignment validation: CKA at all probe layers against thresholds, val_bpb on calibration data. Verify artifact hashes (tokenizer, seed match cooperative). Generate alignment report JSON. Reject with specific metrics if any threshold fails.

- [ ] **TASK-023**: Wire `kalavu train submit` end-to-end
  - **Effort**: S
  - **Dependencies**: TASK-022
  - **Component**: `src/kalavu/cli.py`, `src/kalavu/train/submit.py`
  - **Acceptance**: `kalavu train submit --module 1` runs validation, packages checkpoint directory (weights, probes, report, metadata), copies to cooperative's checkpoint store (local directory for Alpha). Updates module status to "submitted".

---

## Phase 4: Alignment Monitor — `kalavu check` (Week 3)

> Continuous alignment monitoring and reporting.

### Epic: Alignment Diagnostics
> Priority: P0 | Effort: M | Dependencies: TASK-004, TASK-016

- [ ] **TASK-024**: Implement alignment report generation
  - **Effort**: M
  - **Dependencies**: TASK-004, TASK-016
  - **Component**: `src/kalavu/check/alignment.py`
  - **Acceptance**: Compute comprehensive alignment report: CKA at each probe layer, val_bpb on calibration batch, domain-specific val_bpb, VRAM usage, throughput, ETA. Output as structured JSON (agent-parseable). Include pass/fail status per threshold.

- [ ] **TASK-025**: Wire `kalavu check post` end-to-end
  - **Effort**: S
  - **Dependencies**: TASK-024
  - **Component**: `src/kalavu/cli.py`, `src/kalavu/check/alignment.py`
  - **Acceptance**: `kalavu check post` computes alignment report from current training state, saves to cooperative directory as JSON. For Alpha: writes to local file. For Beta: posts to GitHub Discussions. Prints summary via `rich`.

---

## Phase 5: Fusion Pipeline — `kalavu fuse` (Weeks 3–4)

> The core value proposition: turning independently trained modules into one model.

### Epic: Module Clustering
> Priority: P0 | Effort: M | Dependencies: TASK-004, TASK-003

- [ ] **TASK-026**: Implement pairwise CKA similarity matrix computation
  - **Effort**: M
  - **Dependencies**: TASK-004, TASK-003
  - **Component**: `src/kalavu/fuse/cluster.py`
  - **Acceptance**: Load N submitted module checkpoints, compute pairwise CKA similarity at each probe layer using the calibration batch. Output N×N similarity matrix. Cache results to avoid recomputation.

- [ ] **TASK-027**: Implement automatic module clustering
  - **Effort**: S
  - **Dependencies**: TASK-026
  - **Component**: `src/kalavu/fuse/cluster.py`
  - **Acceptance**: Given similarity matrix and target cluster count (from config, default 4), compute optimal clustering (hierarchical or spectral). Output cluster assignments as JSON. Visualize clustering via `rich` tree display.

- [ ] **TASK-028**: Wire `kalavu fuse cluster` end-to-end
  - **Effort**: S
  - **Dependencies**: TASK-027
  - **Component**: `src/kalavu/cli.py`
  - **Acceptance**: `kalavu fuse cluster my-coop` loads submitted checkpoints, computes similarity, outputs cluster assignments. Saves `cluster_assignments.json` to cooperative directory.

### Epic: MoE Fusion (Backend A)
> Priority: P0 | Effort: XL | Dependencies: TASK-028

- [ ] **TASK-029**: Implement MoE expert conversion from module FFN layers
  - **Effort**: L
  - **Dependencies**: TASK-028, TASK-016
  - **Component**: `src/kalavu/fuse/build.py`
  - **Acceptance**: For each submitted module, extract FFN layers and wrap as MoE experts. Average attention layers across modules (BTX-style). Initialize a learned router per layer. Output: a single model with MoE layers replacing FFN layers.

- [ ] **TASK-030**: Implement MoE router initialization
  - **Effort**: M
  - **Dependencies**: TASK-029
  - **Component**: `src/kalavu/fuse/build.py`
  - **Acceptance**: Initialize router weights using cluster assignments (modules in same cluster get similar initial routing weights). Router should use top-k routing (k=2 default). Load balancing auxiliary loss included.

- [ ] **TASK-031**: Wire `kalavu fuse build` end-to-end
  - **Effort**: S
  - **Dependencies**: TASK-029, TASK-030
  - **Component**: `src/kalavu/cli.py`, `src/kalavu/fuse/build.py`
  - **Acceptance**: `kalavu fuse build my-coop` loads cluster assignments, builds MoE model, saves as a single checkpoint. Model is loadable and can run forward pass. Print architecture summary (total params, experts per layer).

### Epic: Post-Training (Coherence Annealing)
> Priority: P0 | Effort: L | Dependencies: TASK-031

- [ ] **TASK-032**: Implement progressive coherence annealing curriculum
  - **Effort**: L
  - **Dependencies**: TASK-031, TASK-018
  - **Component**: `src/kalavu/fuse/train.py`
  - **Acceptance**: Fine-tune the fused MoE model on a mixed-domain curriculum. Progressive: start with individual domain data (easy), gradually mix domains (hard). Training budget: ~8-12% of total pre-training compute. Log fusion-specific metrics (router entropy, expert utilization, cross-domain perplexity).

- [ ] **TASK-033**: Wire `kalavu fuse train` end-to-end
  - **Effort**: S
  - **Dependencies**: TASK-032
  - **Component**: `src/kalavu/cli.py`, `src/kalavu/fuse/train.py`
  - **Acceptance**: `kalavu fuse train my-coop` loads fused model, runs post-training curriculum, saves final fused checkpoint. Print training summary and final metrics.

---

## Phase 6: Evaluation & Publishing (Week 4)

> Prove the thesis and ship the result.

### Epic: Model Evaluation
> Priority: P0 | Effort: M | Dependencies: TASK-033

- [ ] **TASK-034**: Implement nanochat CORE score evaluation
  - **Effort**: M
  - **Dependencies**: TASK-016
  - **Component**: `src/kalavu/core/eval.py`
  - **Acceptance**: Evaluate a model checkpoint on nanochat's CORE benchmark. Return structured score dict. Must be runnable on individual modules AND fused model for comparison.

- [ ] **TASK-035**: Implement comparative evaluation (individual vs fused)
  - **Effort**: S
  - **Dependencies**: TASK-034
  - **Component**: `src/kalavu/core/eval.py`
  - **Acceptance**: Run CORE evaluation on all individual module checkpoints and the fused model. Output comparison table showing per-module scores vs fused score. This validates the core thesis: fused > best individual.

### Epic: Model Publishing
> Priority: P1 | Effort: M | Dependencies: TASK-033

- [ ] **TASK-036**: Implement model card auto-generation
  - **Effort**: S
  - **Dependencies**: TASK-033
  - **Component**: `src/kalavu/coop/publish.py`
  - **Acceptance**: Generate a Hugging Face model card (README.md) documenting: all contributors, their domains, alignment scores, training hardware, fusion method, evaluation results. Template-based with data populated from checkpoint metadata.

- [ ] **TASK-037**: Implement Hugging Face Hub upload
  - **Effort**: M
  - **Dependencies**: TASK-036
  - **Component**: `src/kalavu/coop/publish.py`
  - **Acceptance**: Upload fused model checkpoint + model card to a Hugging Face Hub repository. Use `huggingface_hub` library. Require HF token via env var. Create repo if it doesn't exist.

- [ ] **TASK-038**: Wire `kalavu coop publish` end-to-end
  - **Effort**: S
  - **Dependencies**: TASK-036, TASK-037
  - **Component**: `src/kalavu/cli.py`, `src/kalavu/coop/publish.py`
  - **Acceptance**: `kalavu coop publish my-coop` generates model card, uploads to HF Hub, prints URL.

---

## Phase 7: Alpha Validation (Week 4)

> End-to-end test: 2-module fusion proof-of-concept.

### Epic: 2-Module Proof of Concept
> Priority: P0 | Effort: L | Dependencies: TASK-033, TASK-034

- [ ] **TASK-039**: Create Alpha test script — full pipeline on tiny models
  - **Effort**: L
  - **Dependencies**: TASK-013, TASK-021, TASK-023, TASK-025, TASK-031, TASK-033, TASK-034
  - **Component**: `tests/test_alpha_e2e.py`
  - **Acceptance**: End-to-end test on small models (e.g., 2-layer, 128-dim): (1) `coop create` with 2 modules, (2) `train start` module 1 on domain A for N steps, (3) `train start` module 2 on domain B for N steps, (4) `train submit` both, (5) `fuse cluster`, (6) `fuse build`, (7) `fuse train`, (8) evaluate all three (module 1, module 2, fused). Fused model CORE score > best individual module score.
  - **Notes**: This is the Alpha success criterion from the spec.

- [ ] **TASK-040**: Create sample training data for 2 domains
  - **Effort**: S
  - **Dependencies**: None
  - **Component**: `tests/fixtures/`
  - **Acceptance**: Provide small but distinct domain datasets for testing (e.g., code snippets + math text). Enough data for meaningful training on tiny models (~10K tokens each).

---

## Phase 8: Beta — GitHub Coordination (Weeks 5–7)

> Move from local-only to distributed via GitHub.

### Epic: GitHub Integration
> Priority: P1 | Effort: L | Dependencies: TASK-013

- [ ] **TASK-041**: Implement GitHub repo creation for cooperatives
  - **Effort**: M
  - **Dependencies**: TASK-013
  - **Component**: `src/kalavu/coop/create.py`
  - **Acceptance**: `kalavu coop create` with `--backend github` creates a GitHub repo, uploads all shared artifacts as releases or LFS objects. Generate README with contributor instructions. Requires GitHub token via env var.

- [ ] **TASK-042**: Implement GitHub-based `coop join` (clone + claim)
  - **Effort**: M
  - **Dependencies**: TASK-041
  - **Component**: `src/kalavu/coop/join.py`
  - **Acceptance**: `kalavu coop join github.com/org/coop-repo --claim-module 3` clones repo, downloads artifacts, claims slot via PR or issue.

- [ ] **TASK-043**: Implement GitHub Discussions telemetry posting
  - **Effort**: M
  - **Dependencies**: TASK-025
  - **Component**: `src/kalavu/check/alignment.py`
  - **Acceptance**: `kalavu check post` creates a GitHub Discussion with structured alignment report (JSON in code block). Uses GitHub API. Other contributors' agents can parse these via `gh discussion list`.

- [ ] **TASK-044**: Implement checkpoint upload/download via GitHub Releases or S3
  - **Effort**: M
  - **Dependencies**: TASK-023
  - **Component**: `src/kalavu/train/submit.py`
  - **Acceptance**: `kalavu train submit` uploads checkpoint to configurable backend (GitHub Release for small models, S3/HF for large). `kalavu fuse` commands download all submitted checkpoints automatically.

---

## Phase 9: v1.0 Polish (Weeks 8–12)

> Production readiness, agent mode, external testers.

### Epic: Agent Mode
> Priority: P1 | Effort: M | Dependencies: TASK-021

- [ ] **TASK-045**: Implement `kalavu train agent-mode`
  - **Effort**: M
  - **Dependencies**: TASK-021
  - **Component**: `src/kalavu/train/start.py`
  - **Acceptance**: Agent mode restricts modification to data mixing ratios and learning rate schedule only (architecture and alignment constraints are locked). Accepts a `program.md` file with agent instructions. Trains for a fixed time budget, checks alignment, and iterates. All output is structured JSON.

### Epic: Cooperative State Machine
> Priority: P1 | Effort: M | Dependencies: TASK-015

- [ ] **TASK-046**: Implement cooperative state transitions
  - **Effort**: M
  - **Dependencies**: TASK-015
  - **Component**: `src/kalavu/coop/state.py`
  - **Acceptance**: Enforce the state machine from spec section 6.3: CREATED → RECRUITING → TRAINING → FUSING → PUBLISHED → GROWING. Validate transitions (e.g., can't fuse until 50% modules submitted). Persist state in cooperative directory. `kalavu coop status` displays current state.

### Epic: Incremental Re-Fusion (GROWING state)
> Priority: P2 | Effort: M | Dependencies: TASK-033, TASK-046

- [ ] **TASK-047**: Implement incremental re-fusion with new modules
  - **Effort**: M
  - **Dependencies**: TASK-033, TASK-046
  - **Component**: `src/kalavu/fuse/build.py`
  - **Acceptance**: After initial fusion + publish, new modules (21, 22, ...) can be submitted. `kalavu fuse build` detects new modules and re-fuses incrementally (add new experts to MoE) without re-training existing experts from scratch.

### Epic: CLI Polish
> Priority: P1 | Effort: S | Dependencies: All

- [ ] **TASK-048**: Add `--json` flag to all commands for machine-readable output
  - **Effort**: S
  - **Dependencies**: All previous tasks
  - **Component**: `src/kalavu/cli.py`
  - **Acceptance**: Every CLI command supports `--json` flag that outputs structured JSON instead of human-readable `rich` output. Required for agent compatibility per spec section 7.

---

## Dependency Graph

```
TASK-001 (Config)
├── TASK-003 (Checkpoint) ──────────────────────────────┐
├── TASK-008 (Tokenizer) ───┐                           │
├── TASK-009 (Seed) ────────┤                           │
├── TASK-012 (Manifest) ────┤                           │
│                           │                           │
TASK-002 (Exceptions)       │                           │
                            │                           │
TASK-004 (CKA) ─────────────┤                           │
├── TASK-010 (CKA Ref) ─────┤                           │
│                           │                           │
TASK-005 (Hardware) ────────┤                           │
                            │                           │
TASK-011 (Calibration) ─────┤                           │
                            ▼                           │
                    TASK-013 (coop create E2E)           │
                            │                           │
                    ┌───────┴───────┐                   │
                    ▼               ▼                   │
            TASK-014 (join)  TASK-015 (status)           │
                    │                                   │
    ┌───────────────┤                                   │
    ▼               ▼                                   │
TASK-016 (Model) TASK-018 (Data)                        │
    │               │                                   │
    ├───────────────┤                                   │
    ▼               │                                   │
TASK-017 (CKA Loss) │                                   │
    │               │                                   │
    ├── TASK-019 (Warnings)                             │
    ├── TASK-020 (Telemetry)                            │
    │               │                                   │
    ▼               ▼                                   │
TASK-021 (train start E2E) ─────────────────────────────┤
    │                                                   │
    ▼                                                   │
TASK-022 (Submit Validation) ◄──────────────────────────┘
    │
    ▼
TASK-023 (train submit E2E)
    │
    ├── TASK-024 (Alignment Report) → TASK-025 (check post E2E)
    │
    ▼
TASK-026 (CKA Matrix) → TASK-027 (Clustering) → TASK-028 (fuse cluster E2E)
                                                        │
                                                        ▼
                                    TASK-029 (MoE Experts) → TASK-030 (Router)
                                                                    │
                                                                    ▼
                                                        TASK-031 (fuse build E2E)
                                                                    │
                                                                    ▼
                                                        TASK-032 (Post-Training)
                                                                    │
                                                                    ▼
                                                        TASK-033 (fuse train E2E)
                                                                    │
                                                        ┌───────────┤
                                                        ▼           ▼
                                                TASK-034 (Eval) TASK-036 (Model Card)
                                                        │           │
                                                        ▼           ▼
                                                TASK-035 (Compare) TASK-037 (HF Upload)
                                                        │           │
                                                        ▼           ▼
                                                TASK-039 (Alpha E2E) TASK-038 (publish E2E)
```

## Critical Path

```
TASK-001 → TASK-009 → TASK-013 → TASK-014 → TASK-016 → TASK-017 → TASK-021 →
TASK-022 → TASK-023 → TASK-026 → TASK-029 → TASK-031 → TASK-032 → TASK-033 →
TASK-034 → TASK-039
```

## Parallel Execution Opportunities

| Stream A | Stream B | Stream C |
|----------|----------|----------|
| TASK-001 (Config) | TASK-002 (Exceptions) | TASK-004 (CKA) |
| TASK-005 (Hardware) | TASK-006 (Test infra) | TASK-040 (Sample data) |
| TASK-008 (Tokenizer) | TASK-009 (Seed) | TASK-012 (Manifest) |
| TASK-014 (Join) | TASK-015 (Status) | — |
| TASK-016 (Model) | TASK-018 (Data) | — |
| TASK-019 (Warnings) | TASK-020 (Telemetry) | — |
| TASK-034 (Eval) | TASK-036 (Model Card) | — |
| TASK-041 (GH repo) | TASK-045 (Agent mode) | TASK-046 (State machine) |

## Effort Summary

| Size | Count | Typical Duration |
|------|-------|------------------|
| XS | 3 | < 30 min |
| S | 17 | 30 min – 2 hours |
| M | 20 | 2 – 4 hours |
| L | 6 | 4 – 8 hours |
| XL | 2 | > 8 hours (broken into subtasks above) |

## Risk Areas

- **TASK-004/017**: CKA computation correctness is critical — wrong CKA = unfusible modules
- **TASK-029/030**: MoE conversion from independently trained FFNs is novel engineering
- **TASK-032**: Post-training curriculum design affects fusion quality more than any other step
- **TASK-039**: Alpha validation is pass/fail for the entire product thesis

## Notes

- All tasks assume NVIDIA CUDA GPU for training. MPS/CPU support is explicitly out of MVP scope.
- "nanochat fork" means we depend on Karpathy's nanochat — need to vendor or fork it as a dependency early.
- GitHub coordination (Phase 8) is Beta scope. Alpha uses local directories only.
- The spec mentions autoresearch integration — this is deferred to v1.0 (TASK-045 agent mode).
