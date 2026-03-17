# KALAVAI — Development Checkpoints

A log of completed milestones for any agent or developer resuming work.

---

## Checkpoint 2: Phase 2 Complete — Cooperative Manager (2026-03-09)

**Commit**: `c083df6` on `main`
**Tests**: 149 passing
**What's new since Checkpoint 1**: 8 tasks (TASK-008 through TASK-015)

### Completed

- **BPE tokenizer** (`coop/tokenizer.py`) — custom implementation, deterministic, save/load via JSON
- **Seed checkpoint** (`coop/seed.py`) — reproducible model initialization via BytesIO trick
- **Transformer model** (`core/model.py`) — GPT-style with RMSNorm, SwiGLU, causal attention, probe extraction
- **CKA reference** (`coop/reference.py`) — extract hidden states at probe layers from seed model
- **Calibration batch** (`coop/calibration.py`) — tokenize corpus into fixed tensor
- **Domain manifest** (`coop/manifest.py`) — 20 default domains, slot claiming, CRUD operations
- **`kalavai coop create`** (`coop/create.py`) — full orchestration of all 6 artifact creation steps
- **`kalavai coop join`** (`coop/join.py`) — copy artifacts, claim slot, store hashes
- **`kalavai coop status`** (`coop/status.py`) — rich table + JSON output

### Architecture Presets

The `create_cooperative()` function maps target_params to transformer configs:

| Target | Depth | d_model | Heads | FFN Ratio | ~Params |
|--------|-------|---------|-------|-----------|---------|
| 14M | 6 | 384 | 6 | 2.75 | ~14M |
| 125M | 12 | 768 | 12 | 2.75 | ~125M |
| 350M | 24 | 1024 | 16 | 2.75 | ~350M |
| 1B | 24 | 2048 | 16 | 2.75 | ~1B |
| 7B | 32 | 4096 | 32 | 2.75 | ~7B |

### How to resume

```bash
cd C:/Github/Kalavai
pip install -e ".[dev]"
python -m pytest -v  # should see 149 passing
```

### What to build next

Phase 3: Module Trainer — the training loop with CKA anchor loss. Start with TASK-016 (model wrapper already exists in `core/model.py` — may just need a thin adapter). The critical path is:

```
TASK-016 (model wrapper) → TASK-017 (CKA anchor loss) → TASK-021 (train start E2E)
                                                       → TASK-022 (submit validation) → TASK-023 (train submit E2E)
```

TASK-018 (data loading) and TASK-019/020 (warnings/telemetry) can be built in parallel with 017.

---

## Checkpoint 1: Phase 1 Complete — Foundation (2026-03-09)

**Commit**: `90932e3` on `main`
**Tests**: 79 passing
**What's new**: 7 tasks (TASK-001 through TASK-007)

### Completed

- **Config parser** (`core/config.py`) — 5 typed dataclasses, from_yaml/to_yaml, full validation
- **Exception hierarchy** (`core/exceptions.py`) — KalavaiError base + 6 specific exceptions
- **Checkpoint handler** (`core/checkpoint.py`) — save/load checkpoint directories, SHA-256 validation
- **CKA computation** (`core/cka.py`) — linear CKA + differentiable cka_loss
- **Hardware detection** (`train/hardware.py`) — CUDA auto-detect with rich output
- **Test infrastructure** (`tests/conftest.py`) — fixtures for config, cooperative dir, mock CUDA
- **CLI smoke tests** (`tests/test_install.py`) — --help, --version, subcommand verification
- **pyproject.toml** — package config with deps, dev tools, CLI entry point

### Key files for any future task

| Need | File | API |
|------|------|-----|
| Load config | `core/config.py` | `CooperativeConfig.from_yaml(path)` |
| Raise errors | `core/exceptions.py` | `ConfigError`, `AlignmentError`, etc. |
| Save/load checkpoints | `core/checkpoint.py` | `save_checkpoint()`, `load_checkpoint()` |
| Compute CKA | `core/cka.py` | `linear_cka(X, Y)`, `cka_loss(h, ref)` |
| Detect GPU | `train/hardware.py` | `detect_hardware()` |
| Test fixtures | `tests/conftest.py` | `cooperative_dir`, `sample_config_yaml`, etc. |

---

## Checkpoint 0: Project Initialization (2026-03-09)

**Commit**: `a1b22cc` on `main`

- Repo initialized with CLAUDE.md, pyproject.toml, .gitignore
- CLI skeleton with all command stubs (coop, train, check, fuse)
- Source layout: `src/kalavai/` with subsystem packages
- Spec files moved to `docs/spec/`
- Task plan created at `docs/tasks.md` (48 tasks across 9 phases)
