# KALAVAI — கலவை

## Project Overview

**KALAVAI** is a decentralized LLM training protocol and CLI toolchain by Murai Labs.
It lets a cooperative of GPU owners each train one piece of a large language model independently, then fuse the pieces into a unified model none of them could afford to build alone.

**Tagline**: "Git for model training."

## Tech Stack

- **Language**: Python 3.11+
- **Package Manager**: pip (published as `pip install kalavai`)
- **CLI Framework**: TBD (click or typer)
- **ML Core**: PyTorch, nanochat fork (CKA anchor loss), minbpe (tokenizer)
- **Coordination**: GitHub API / git, GitHub Discussions (async telemetry)
- **Model Publishing**: Hugging Face Hub
- **Testing**: pytest
- **Linting**: ruff

## Architecture — Four Subsystems

| Subsystem | CLI Command | Purpose |
|-----------|-------------|---------|
| Cooperative Manager | `kalavai coop` | Create cooperatives, invite members, assign domains, distribute seed + tokenizer |
| Module Trainer | `kalavai train` | Train assigned module with CKA alignment constraints, auto-detect hardware |
| Alignment Monitor | `kalavai check` | Run alignment diagnostics, post to cooperative, flag divergence |
| Fusion Pipeline | `kalavai fuse` | Collect checkpoints, cluster modules, build fusion layers, post-training |

## Key Concepts

- **Cooperative**: Group of N participants (default 20) training one module each
- **Canonical Seed θ₀**: Identical parameter initialization shared by all modules
- **CKA Alignment**: Centered Kernel Alignment — measures representational similarity between modules and reference
- **Probe Layers**: Layers at 25%, 50%, 75% depth where CKA is measured
- **Fusion Backends**: MoE Routing (v1 default) or Hierarchical Cross-Attention (advanced)
- **Domain Manifest**: Assignment of knowledge domains to module slots (Code, Math, Bio, Legal, etc.)

## Commands (Build / Test / Run)

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .

# Run formatter
ruff format .

# Run CLI
kalavai --help
```

## Project Structure

```
kalavai/
├── CLAUDE.md
├── pyproject.toml
│
├── experiments/            # All runnable Python scripts
│   ├── kalavai_pythia_experiment.py      # Main 410M experiment (template)
│   ├── kalavai_pythia_1b_experiment.py
│   ├── kalavai_pythia_6b_experiment.py
│   ├── kalavai_pythia_maturity_sweep.py
│   ├── kalavai_pythia_1b_maturity_sweep.py
│   ├── kalavai_pythia_5domain_experiment.py
│   ├── kalavai_pythia_ablation_freeze.py
│   ├── kalavai_pythia_ablation_router.py
│   ├── kalavai_pythia_monolithic_baseline.py
│   ├── kalavai_pythia_benchmarks.py
│   ├── kalavai_pythia_1b_benchmarks.py
│   ├── kalavai_results_audit.py
│   ├── kalavai_run_all.py               # Master orchestrator
│   └── ...
│
├── results/                # JSON result artifacts
│   └── pythia/             # 410M, 1B, Qwen results
│       ├── *.json
│       ├── five_domain/
│       ├── maturity_sweep_410m/
│       └── pythia_1b/
│
├── figures/                # Generated figures
│   └── pythia/
│       └── *.png
│
├── paper/                  # Writeup and paper scaffold
│   ├── KALAVAI_Results_Writeup.pdf
│   └── scaffold/
│
├── docs/                   # Technical docs + planning
│   ├── planning/           # Claude Code planning docs
│   │   ├── claude_code_*.md
│   │   └── KALAVAI_Strategic_Summary*.md
│   └── spec/               # Product specification
│
├── src/kalavai/             # Library source code
│   ├── coop/
│   ├── train/
│   ├── check/
│   ├── fuse/
│   └── core/
│
├── tests/
├── checkpoints/            # Model weights (gitignored)
└── logs/                   # Experiment logs
```

## Code Conventions

- Use type hints on all function signatures
- Docstrings: Google style, only on public APIs
- Errors: raise typed exceptions (e.g., `AlignmentError`, `CheckpointValidationError`)
- Config: all cooperative config via `kalavai.yaml` — never hardcode thresholds
- CLI: all commands must be non-interactive (flags/YAML only) for agent compatibility
- Checkpoints: always validate shared artifact hashes before any operation

## Do's

- Keep CLI commands idempotent where possible
- Use structured JSON output for all `check` commands (agent-parseable)
- Validate CKA reference hash before training or submission
- Log training telemetry at configurable intervals
- Support partial fusion (min 2 modules)

## Don'ts

- Don't modify tokenizer after cooperative creation
- Don't allow architecture deviation from cooperative config
- Don't require synchronous coordination between contributors
- Don't hardcode hardware assumptions — always auto-detect
- Don't skip alignment validation at submission time

## Key Files

- `docs/spec/KALAVAI_v3_Product_Specification.docx` — Full product specification
- `docs/spec/kalavai-v3-product-flow.jsx` — Visual product flow diagram (React/SVG)
- `kalavai.yaml` — Cooperative configuration schema (see spec section 6.1)

## MVP Scope

The MVP validates: can independently trained modules be fused into a model that outperforms any individual module?

**In MVP**: CLI skeleton (coop create, train start, check post, train submit, fuse build, fuse train, coop publish), 2-5 modules, MoE routing fusion, GitHub coordination, NVIDIA GPU support.

**Not in MVP**: Web dashboard, 20-module cooperatives, cross-attention fusion, MPS/CPU support, automated domain suggestion.
