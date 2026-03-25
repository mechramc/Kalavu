"""
Upload KALAVAI specialist checkpoints to Hugging Face Hub.

Usage:
    python experiments/upload_to_hf.py --set cross_lingual --org YOUR_HF_USERNAME
    python experiments/upload_to_hf.py --set private_domain --org YOUR_HF_USERNAME
    python experiments/upload_to_hf.py --set phase1_410m --org YOUR_HF_USERNAME
    python experiments/upload_to_hf.py --set phase1_1b --org YOUR_HF_USERNAME
    python experiments/upload_to_hf.py --set qwen --org YOUR_HF_USERNAME

Run `huggingface-cli login` first.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_ROOT = Path(__file__).parent.parent / "checkpoints"

# Each set: (checkpoint_dir, base_model_id, list of (ckpt_stem, domain_label, seed))
SETS = {
    "cross_lingual": {
        "ckpt_dir": CHECKPOINT_ROOT / "phase2" / "cross_lingual",
        "base_model": "EleutherAI/pythia-410m",
        "paper_results": "Yoruba PPL 41.9→7.7 (5.4×), Welsh 102.7→22.1 (4.6×), Tamil 4.2→3.0. "
                         "MoE fusion of 4 specialists: +21.76% over best specialist (seeds 137+2026).",
        "models": [
            ("yoruba_specialist_seed137", "Yoruba", 137),
            ("yoruba_specialist_seed2026", "Yoruba", 2026),
            ("welsh_specialist_seed137",  "Welsh",  137),
            ("welsh_specialist_seed2026", "Welsh",  2026),
            ("tamil_specialist_seed137",  "Tamil",  137),
            ("tamil_specialist_seed2026", "Tamil",  2026),
            ("code_specialist_seed137",   "Code",   137),
            ("code_specialist_seed2026",  "Code",   2026),
        ],
    },
    "private_domain": {
        "ckpt_dir": CHECKPOINT_ROOT / "phase2" / "private_domain",
        "base_model": "EleutherAI/pythia-410m",
        "paper_results": "Medical/Legal/Patent fusion: +10.17% ±0.15pp over best specialist (3 seeds). "
                         "Mean divergence 18.52%.",
        "models": [
            ("medical_specialist_seed42",  "Medical", 42),
            ("medical_specialist_seed137", "Medical", 137),
            ("medical_specialist_seed2026","Medical", 2026),
            ("legal_specialist_seed42",    "Legal",   42),
            ("legal_specialist_seed137",   "Legal",   137),
            ("legal_specialist_seed2026",  "Legal",   2026),
            ("patent_specialist_seed42",   "Patent",  42),
            ("patent_specialist_seed137",  "Patent",  137),
            ("patent_specialist_seed2026", "Patent",  2026),
        ],
    },
    "phase1_410m": {
        "ckpt_dir": CHECKPOINT_ROOT / "pythia",
        "base_model": "EleutherAI/pythia-410m",
        "paper_results": "Phase 1 English domains. MoE fusion: +7.72% ±0.02pp over best specialist (3 seeds). "
                         "Mean divergence 15.65%.",
        "models": [
            ("code_specialist_seed42",    "Code",    42),
            ("code_specialist_seed137",   "Code",    137),
            ("code_specialist_seed2026",  "Code",    2026),
            ("science_specialist_seed42", "Science", 42),
            ("science_specialist_seed137","Science", 137),
            ("science_specialist_seed2026","Science",2026),
            ("fiction_specialist_seed42", "Fiction", 42),
            ("fiction_specialist_seed137","Fiction", 137),
            ("fiction_specialist_seed2026","Fiction",2026),
        ],
    },
    "phase1_1b": {
        "ckpt_dir": CHECKPOINT_ROOT / "pythia" / "pythia_1b",
        "base_model": "EleutherAI/pythia-1b",
        "paper_results": "Phase 1 English domains at 1B scale. MoE fusion: +7.49% ±0.01pp over best specialist (3 seeds). "
                         "Mean divergence 15.28%.",
        "models": [
            ("code_specialist_seed42",    "Code",    42),
            ("code_specialist_seed137",   "Code",    137),
            ("code_specialist_seed2026",  "Code",    2026),
            ("science_specialist_seed42", "Science", 42),
            ("science_specialist_seed137","Science", 137),
            ("science_specialist_seed2026","Science",2026),
            ("fiction_specialist_seed42", "Fiction", 42),
            ("fiction_specialist_seed137","Fiction", 137),
            ("fiction_specialist_seed2026","Fiction",2026),
        ],
    },
    "qwen": {
        "ckpt_dir": CHECKPOINT_ROOT / "qwen",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "paper_results": "Qwen-2.5-1.5B specialists. MoE fusion: +1.06% ±0.01pp over best specialist (3 seeds). "
                         "Mean divergence 3.16% — near floor of gain-divergence relationship.",
        "models": [
            ("code_specialist_seed42",   "Code",    42),
            ("code_specialist_seed137",  "Code",    137),
            ("code_specialist_seed2026", "Code",    2026),
            ("fiction_specialist_seed42",   "Fiction", 42),
            ("fiction_specialist_seed137",  "Fiction", 137),
            ("fiction_specialist_seed2026", "Fiction", 2026),
        ],
    },
}

ARXIV_ID = "2603.22755"

MODEL_CARD_TEMPLATE = """\
---
base_model: {base_model}
tags:
  - kalavai
  - specialist
  - mixture-of-experts
  - decentralized-training
  - {domain_tag}
license: apache-2.0
---

# KALAVAI — {domain} Specialist ({base_model_short}, seed {seed})

Fine-tuned {base_model} on **{domain}** data as part of the
[KALAVAI](https://arxiv.org/abs/{arxiv_id}) decentralized cooperative training protocol.

## Paper results

{paper_results}

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

This model is one specialist in a KALAVAI cooperative.
To reproduce the MoE fusion results from the paper, load multiple domain specialists
and combine them with a trained MoE router (see the [paper](https://arxiv.org/abs/{arxiv_id})
and [GitHub](https://github.com/mechramc/Kalavai) for details).

## Citation

```bibtex
@article{{kalavai2026,
  title={{KALAVAI: Cooperative Decentralized LLM Training via MoE Fusion}},
  author={{[Authors]}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{2026}}
}}
```
"""


def make_repo_name(set_name: str, stem: str) -> str:
    # e.g. kalavai-cross-lingual-yoruba-specialist-seed137
    return f"kalavai-{set_name.replace('_', '-')}-{stem.replace('_', '-')}"


def upload_model(
    ckpt_path: Path,
    base_model: str,
    repo_id: str,
    domain: str,
    seed: int,
    set_cfg: dict,
    dry_run: bool,
):
    print(f"\n{'='*60}")
    print(f"  Uploading: {repo_id}")
    print(f"  From:      {ckpt_path}")
    print(f"  Base:      {base_model}")

    if not ckpt_path.exists():
        print(f"  SKIP — file not found: {ckpt_path}")
        return

    if dry_run:
        print("  DRY RUN — skipping actual upload")
        return

    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)

    print("  Loading checkpoint weights...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    domain_tag = domain.lower().replace(" ", "-")
    base_model_short = base_model.split("/")[-1]

    model_card = MODEL_CARD_TEMPLATE.format(
        base_model=base_model,
        base_model_short=base_model_short,
        domain=domain,
        domain_tag=domain_tag,
        seed=seed,
        paper_results=set_cfg["paper_results"],
        repo_id=repo_id,
        arxiv_id=ARXIV_ID,
    )

    print("  Pushing to hub (this will upload ~safetensors)...")
    model.push_to_hub(
        repo_id,
        commit_message=f"Add KALAVAI {domain} specialist (seed {seed})",
        private=False,
    )

    # Push model card
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )

    print(f"  Done: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, choices=list(SETS.keys()),
                        help="Which checkpoint set to upload")
    parser.add_argument("--org", required=True,
                        help="Your HF username or org name")
    parser.add_argument("--arxiv-id", default=None,
                        help="arXiv ID to embed in model cards (e.g. 2503.12345)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without uploading")
    parser.add_argument("--only", default=None,
                        help="Upload only this stem (e.g. yoruba_specialist_seed137)")
    args = parser.parse_args()

    global ARXIV_ID
    if args.arxiv_id:
        ARXIV_ID = args.arxiv_id

    cfg = SETS[args.set]

    for stem, domain, seed in cfg["models"]:
        if args.only and stem != args.only:
            continue

        ckpt_path = cfg["ckpt_dir"] / f"{stem}.pt"
        repo_id = f"{args.org}/{make_repo_name(args.set, stem)}"

        upload_model(
            ckpt_path=ckpt_path,
            base_model=cfg["base_model"],
            repo_id=repo_id,
            domain=domain,
            seed=seed,
            set_cfg=cfg,
            dry_run=args.dry_run,
        )

    print("\nAll done.")
    print(f"View your models at: https://huggingface.co/{args.org}")


if __name__ == "__main__":
    main()
