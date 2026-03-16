#!/usr/bin/env python3
"""
Fuse and evaluate the freeze-depth ablation sweep.

Iterates over FREEZE_DEPTHS × SEEDS, loads pre-trained LoRA specialists,
builds a TwoExpertMoE, trains its router, evaluates, and saves results.

Results land in results/real/freeze_ablation/freeze_{N}_seed_{S}.json
These files feed directly into generate_figures.py Figure 2.

Usage:
    python scripts/fuse_and_eval_ablation.py              # full run
    python scripts/fuse_and_eval_ablation.py --dry-run    # check paths only
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FREEZE_DEPTHS = [0, 1, 2, 3, 4, 6, 8]
SEEDS = [42, 137, 2026]
ROUTER_STEPS = 150
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DOMAINS = ["math", "science"]
RESULTS_DIR = Path("results/real/freeze_ablation")


# ============================================================================
# Checkpoint path resolution
# ============================================================================

def checkpoint_path(freeze: int, seed: int, domain: str) -> Path:
    """Reconstruct the checkpoint path using the same convention as train_hf.py."""
    name = f"freeze_ablation_f{freeze}"
    run_name = f"{name}_{domain}_seed{seed}"
    return Path(f"results/real/{name}/{run_name}/checkpoint")


def result_path(freeze: int, seed: int) -> Path:
    return RESULTS_DIR / f"freeze_{freeze}_seed_{seed}.json"


def check_all_paths(dry_run: bool = False) -> list[tuple[int, int]]:
    """Return list of (freeze, seed) pairs where all checkpoints exist."""
    ready = []
    print("Checking checkpoint paths...")
    for freeze in FREEZE_DEPTHS:
        for seed in SEEDS:
            missing = []
            for domain in DOMAINS:
                p = checkpoint_path(freeze, seed, domain)
                if not p.exists():
                    missing.append(str(p))
            if missing:
                print(f"  freeze={freeze} seed={seed}: MISSING {len(missing)} checkpoint(s)")
                for m in missing:
                    print(f"    - {m}")
            else:
                print(f"  freeze={freeze} seed={seed}: OK")
                ready.append((freeze, seed))

    print(f"\n{len(ready)}/{len(FREEZE_DEPTHS) * len(SEEDS)} pairs ready.")
    return ready


# ============================================================================
# Data loading
# ============================================================================

class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids_list, max_length=512):
        self.samples = [ids[:max_length] for ids in input_ids_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


def _collate(batch):
    max_len = max(len(x) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        input_ids[i, :len(x)] = x
        attention_mask[i, :len(x)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _collate_with_labels(batch):
    batch_dict = _collate(batch)
    # labels: shift right, mask padding with -100
    input_ids = batch_dict["input_ids"]
    attention_mask = batch_dict["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    batch_dict["labels"] = labels
    return batch_dict


def load_domain_texts(domain: str, split: str, n_samples: int = 1000) -> list[str]:
    from datasets import load_dataset
    if domain == "math":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in ds]
    elif domain == "science":
        ds = load_dataset("allenai/sciq", split=split)
        texts = [f"Question: {ex['question']}\nAnswer: {ex['correct_answer']}" for ex in ds]
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return texts[:n_samples]


def make_dataset(texts: list[str], tokenizer, max_length: int = 512) -> _TokenDataset:
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return _TokenDataset(enc["input_ids"], max_length)


# ============================================================================
# TwoExpertMoE
# ============================================================================

class TwoExpertMoE(nn.Module):
    """
    Sequence-level MoE over two specialist models.
    Router: mean-pooled last hidden state (averaged across both experts) →
            small MLP → 2 gates (softmax) → weighted logit sum.
    """
    def __init__(self, spec_a, spec_b, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        # Freeze specialist params — only router is trainable
        for p in self.spec_a.parameters():
            p.requires_grad_(False)
        for p in self.spec_b.parameters():
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 2),
        )

    def _run_specialist(self, model, input_ids, attention_mask):
        """Run specialist with no_grad; return (logits, mean-pooled hidden state)."""
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        logits = out.logits.detach()  # (B, T, V)
        last_h = out.hidden_states[-1].detach()  # (B, T, H)
        # Mean-pool over non-padding positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            h_pooled = (last_h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h_pooled = last_h.mean(1)
        return logits, h_pooled  # (B, T, V), (B, H)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids, attention_mask)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids, attention_mask)

        # Route on average of both experts' representations
        h_avg = (h_a + h_b) / 2.0  # (B, H) — not detached so grad flows to router
        gates = torch.softmax(self.router(h_avg), dim=-1)  # (B, 2)

        # Weighted sum of logits
        fused = gates[:, 0:1, None] * logits_a + gates[:, 1:2, None] * logits_b  # (B, T, V)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                fused.view(-1, fused.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return loss, fused, gates


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def eval_model_loss(model, loader, device, max_batches: int = 30) -> float:
    """Compute mean cross-entropy loss on a dataloader."""
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        if hasattr(model, "spec_a"):
            # TwoExpertMoE
            loss, _, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
        else:
            # Regular HF model
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

        if loss is not None:
            losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float("inf")


# ============================================================================
# Main sweep
# ============================================================================

def run_pair(freeze: int, seed: int, device: str, tokenizer):
    """Load, fuse, train router, evaluate one (freeze, seed) pair."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    result_file = result_path(freeze, seed)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"freeze_layers={freeze}  seed={seed}")
    print(f"{'='*60}")

    # Load domain specialists
    print("  Loading specialists...")
    specialists = {}
    for domain in DOMAINS:
        ckpt = checkpoint_path(freeze, seed, domain)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        specialist = PeftModel.from_pretrained(base, str(ckpt))
        specialist = specialist.merge_and_unload()
        specialist = specialist.to(device)
        specialist.eval()
        specialists[domain] = specialist
        print(f"    Loaded {domain} specialist from {ckpt}")

    hidden_size = specialists[DOMAINS[0]].config.hidden_size

    # Load data
    print("  Loading eval data...")
    eval_texts = {d: load_domain_texts(d, split="test", n_samples=500) for d in DOMAINS}
    train_texts = {d: load_domain_texts(d, split="train", n_samples=500) for d in DOMAINS}

    eval_datasets = {d: make_dataset(eval_texts[d], tokenizer) for d in DOMAINS}
    train_datasets = {d: make_dataset(train_texts[d], tokenizer) for d in DOMAINS}

    mixed_eval_texts = eval_texts[DOMAINS[0]] + eval_texts[DOMAINS[1]]
    mixed_train_texts = train_texts[DOMAINS[0]] + train_texts[DOMAINS[1]]
    mixed_eval_ds = make_dataset(mixed_eval_texts, tokenizer)
    mixed_train_ds = make_dataset(mixed_train_texts, tokenizer)

    def make_loader(ds, shuffle=False):
        return torch.utils.data.DataLoader(
            ds, batch_size=4, shuffle=shuffle,
            collate_fn=_collate_with_labels, num_workers=0,
        )

    # Evaluate individual specialists on mixed eval
    print("  Evaluating individual specialists on mixed eval...")
    individual_losses = {}
    for domain, spec in specialists.items():
        loss = eval_model_loss(spec, make_loader(mixed_eval_ds), device)
        individual_losses[domain] = loss
        print(f"    {domain}: {loss:.4f}")

    best_individual = min(individual_losses.values())

    # Build MoE
    print("  Building TwoExpertMoE and training router...")
    moe = TwoExpertMoE(
        specialists[DOMAINS[0]],
        specialists[DOMAINS[1]],
        hidden_size=hidden_size,
    ).to(device)

    optimizer = torch.optim.Adam(moe.router.parameters(), lr=1e-3)
    router_loader = make_loader(mixed_train_ds, shuffle=True)
    router_iter = iter(router_loader)

    moe.train()
    for step in range(1, ROUTER_STEPS + 1):
        try:
            batch = next(router_iter)
        except StopIteration:
            router_iter = iter(router_loader)
            batch = next(router_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, _, _ = moe(input_ids, attention_mask=attention_mask, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")

    # Evaluate fused model
    print("  Evaluating fused model on mixed eval...")
    moe.eval()
    fused_loss = eval_model_loss(moe, make_loader(mixed_eval_ds), device)
    print(f"    fused: {fused_loss:.4f}")

    improvement_pct = (best_individual - fused_loss) / best_individual * 100

    result = {
        "freeze_layers": freeze,
        "seed": seed,
        "domains": DOMAINS,
        "eval_loss": {
            **{f"specialist_{d}": round(individual_losses[d], 6) for d in DOMAINS},
            "moe_fused": round(fused_loss, 6),
        },
        "best_individual": round(best_individual, 6),
        "improvement_pct": round(improvement_pct, 4),
        "thesis_holds": improvement_pct > 0,
    }

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {result_file}")
    print(f"  Improvement: {improvement_pct:+.2f}%")

    return result


def main():
    parser = argparse.ArgumentParser(description="Fuse + eval freeze ablation sweep")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check checkpoint paths without loading models",
    )
    args = parser.parse_args()

    ready_pairs = check_all_paths(dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    if not ready_pairs:
        print("No complete checkpoint pairs found. Run run_freeze_ablation.sh first.")
        sys.exit(1)

    # Imports that require GPU only in full run
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed. Run: pip install transformers peft datasets")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    skipped = 0
    completed = 0
    failed = 0

    for freeze, seed in ready_pairs:
        rfile = result_path(freeze, seed)
        if rfile.exists():
            print(f"Skipping freeze={freeze} seed={seed} (already done: {rfile})")
            skipped += 1
            continue

        try:
            run_pair(freeze, seed, device, tokenizer)
            completed += 1
        except Exception as e:
            print(f"ERROR on freeze={freeze} seed={seed}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done. completed={completed}, skipped={skipped}, failed={failed}")
    print(f"Results in {RESULTS_DIR}/")
    print("Run python scripts/generate_figures.py to produce Figure 2.")


if __name__ == "__main__":
    main()
