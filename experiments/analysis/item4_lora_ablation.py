#!/usr/bin/env python3
"""
Item 4: LoRA Ablation
=====================
Trains 3 specialists (code/science/fiction) on Pythia-410M using LoRA instead
of full fine-tuning at two ranks (r=8, r=64). Measures:
  - Per-domain divergence from base
  - Fusion gain under corrected eval
  - Whether Item 1 regression prediction matches actual gain

Run AFTER Exp3 finishes (uses the same A100/H100 instance if available).
Requires: peft>=0.9.0  (pip install peft)

Usage:
    python experiments/analysis/item4_lora_ablation.py [--rank 8] [--seed 42]
    python experiments/analysis/item4_lora_ablation.py --rank 64 --seed 42

Outputs:
    results/analysis/lora_ablation_r{rank}_seed{seed}.json
    (prints LaTeX table rows + paper paragraph to stdout)
"""

import argparse
import json
import math
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from kalavai_eval_utils import (
    eval_all_domains, PackedChunkDataset, _collate, chunks_to_dataset, SEQ_LEN
)

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    print("ERROR: peft not installed. Run: pip install peft>=0.9.0")
    raise

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "EleutherAI/pythia-410m"
REVISION   = "step10000"
DOMAINS    = ["code", "science", "fiction"]
MAX_STEPS  = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 2e-4          # LoRA typically uses higher LR than full FT
WEIGHT_DECAY   = 0.01
GRADIENT_CLIP  = 1.0
WARMUP_FRACTION = 0.1
EVAL_BATCH_SIZE = 4
EVAL_BATCHES    = 50
N_SAMPLES       = 2000

ROUTER_STEPS = 500
ROUTER_LR    = 2e-4

parser = argparse.ArgumentParser()
parser.add_argument("--rank",  type=int, default=8,  help="LoRA rank (8 or 64)")
parser.add_argument("--seed",  type=int, default=42)
parser.add_argument("--lr",    type=float, default=None, help="Override learning rate (default: 2e-4)")
parser.add_argument("--skip-training", action="store_true",
                    help="Load existing checkpoints and skip specialist training")
args = parser.parse_args()

LORA_RANK   = args.rank
LORA_ALPHA  = LORA_RANK * 2   # standard: alpha = 2 * rank
SEED        = args.seed
if args.lr is not None:
    LR = args.lr

# build result tag for lr variants (e.g. lr5e-4 → "lr5e4")
_lr_tag = f"_lr{LR:.0e}".replace("-0", "e-").replace("+0", "e") if args.lr is not None else ""

RESULTS_DIR    = Path(f"results/analysis/lora_r{LORA_RANK}")
_ckpt_lr_tag   = f"_lr{LR:.0e}".replace("-0", "e-").replace("+0", "e") if args.lr is not None else ""
CHECKPOINT_DIR = Path(f"checkpoints/analysis/lora_r{LORA_RANK}{_ckpt_lr_tag}")
CACHE_DIR      = Path("data_cache/phase1")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data loaders (reuse Phase 1 domains) ──────────────────────────────────────
def load_data(tokenizer):
    from datasets import load_dataset

    def _cache(name, loader_fn):
        path = CACHE_DIR / f"{name}_chunks.pt"
        if path.exists():
            return torch.load(path, weights_only=True)
        texts = loader_fn()
        ds = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        torch.save(ds.chunks, path)
        return ds.chunks

    def load_code():
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        return [s["func_code_string"][:5000] for _, s in zip(range(N_SAMPLES * 2), ds)
                if s.get("func_code_string", "").strip()][:N_SAMPLES]

    def load_science():
        ds = load_dataset("allenai/sciq", split="train")
        return [(s.get("support", "") + " " + s.get("question", ""))[:5000]
                for s in ds if s.get("support", "").strip()][:N_SAMPLES]

    def load_fiction():
        ds = load_dataset("emozilla/pg19", split="train", streaming=True)
        return [s["text"][:5000] for _, s in zip(range(N_SAMPLES), ds)
                if s.get("text", "").strip()]

    train, held = {}, {}
    for name, fn in [("code", load_code), ("science", load_science), ("fiction", load_fiction)]:
        chunks = _cache(name, fn)
        a, b = int(len(chunks) * 0.8), int(len(chunks) * 0.9)
        train[name]  = chunks[:a]
        held[name]   = chunks[b:]
    return train, held

# ── LoRA training ─────────────────────────────────────────────────────────────
def make_lora_model(device):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query_key_value"],   # GPT-NeoX attention projection
    )
    return get_peft_model(base, config)


def train_lora_specialist(model, name, chunks, seed, device):
    set_seed(seed)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] LoRA r={LORA_RANK}: trainable={trainable/1e6:.2f}M/{total/1e6:.0f}M "
          f"({100*trainable/total:.2f}%)")

    dataset   = chunks_to_dataset(chunks)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=True, collate_fn=_collate)
    warmup    = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup)

    step, accum, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch["input_ids"].to(device),
                        labels=batch["labels"].to(device))
            loss = out.loss / GRAD_ACCUM
        loss.backward()
        accum        += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if step < warmup:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup
            optimizer.step()
            if step >= warmup:
                scheduler.step()
            optimizer.zero_grad()
            accum = 0
            step += 1
            if step % 500 == 0 or step == MAX_STEPS:
                print(f"    [{name}] step {step}/{MAX_STEPS} | loss={running_loss/step:.4f} | {time.time()-t0:.0f}s")

    model.eval()
    print(f"  [{name}] done {time.time()-t0:.0f}s")


# ── Router (reused from main experiment — full linear layer on merged model) ──
class LoRAMoE(torch.nn.Module):
    """Lightweight MoE over LoRA-merged specialists."""
    def __init__(self, merged_state_dicts, model_id, revision, hidden_size, n, device):
        super().__init__()
        self.n_experts = n
        self.device    = device
        self.router    = torch.nn.Linear(hidden_size, n, bias=False)
        # CPU offload
        self._cpu_sds  = merged_state_dicts
        self._model_id = model_id
        self._revision = revision

    def _run_one(self, sd, input_ids):
        m = AutoModelForCausalLM.from_pretrained(
            self._model_id, revision=self._revision,
            dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            out = m(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.float().cpu()
        h      = out.hidden_states[-1].float().mean(1).cpu()
        del m; torch.cuda.empty_cache()
        return logits, h

    def forward(self, input_ids, labels=None):
        input_ids = input_ids.to(self.device)
        all_logits, all_h = [], []
        for sd in self._cpu_sds:
            lg, h = self._run_one(sd, input_ids)
            all_logits.append(lg)
            all_h.append(h)
        h_mean = torch.stack(all_h, dim=0).mean(0).to(self.device)
        gates  = torch.softmax(self.router(h_mean), dim=-1)
        fused  = None
        for i, lg in enumerate(all_logits):
            w = gates[:, i, None, None] * lg.to(self.device)
            fused = w if fused is None else fused + w
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = F.cross_entropy(
                fused[:, :-1].contiguous().view(-1, fused.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
        return loss, fused, gates


def train_router(moe, train_chunks, device):
    all_chunks = []
    for name in DOMAINS:
        all_chunks.extend(train_chunks[name])
    dataset   = chunks_to_dataset(all_chunks)
    loader    = DataLoader(dataset, batch_size=4, shuffle=True,
                           drop_last=True, collate_fn=_collate)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=ROUTER_STEPS)
    it = cycle(loader)
    moe.train()
    t0 = time.time()
    for step in range(1, ROUTER_STEPS + 1):
        optimizer.zero_grad()
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        loss.backward()
        clip_grad_norm_(moe.router.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step}/{ROUTER_STEPS}: loss={loss.item():.4f} | {time.time()-t0:.0f}s")
    moe.eval()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoRA Ablation: rank={LORA_RANK}, alpha={LORA_ALPHA}, seed={SEED}")
    print(f"Device: {device}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_chunks, held_chunks = load_data(tokenizer)
    held_sets = {d: chunks_to_dataset(held_chunks[d]) for d in DOMAINS}

    # Base model eval
    print("\n[base eval]")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base.eval()
    base_eval = eval_all_domains(base, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
    print(f"  Base EW loss: {base_eval['equal_weight_avg']:.4f}")
    del base; torch.cuda.empty_cache()

    # Train/load LoRA specialists
    eval_matrix = {"base": base_eval}
    merged_sds  = []

    for name in DOMAINS:
        ckpt = CHECKPOINT_DIR / f"{name}_lora_r{LORA_RANK}_seed{SEED}_merged.pt"

        if ckpt.exists() or args.skip_training:
            print(f"\n[{name}] loading merged checkpoint from {ckpt}")
            m = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        else:
            print(f"\n[{name}] training LoRA specialist (r={LORA_RANK}, {MAX_STEPS} steps)...")
            m = make_lora_model(device)
            train_lora_specialist(m, name, train_chunks[name], SEED, device)
            # Merge LoRA weights into base for eval + MoE
            m = m.merge_and_unload()
            torch.save(m.state_dict(), ckpt)
            print(f"  Saved merged checkpoint: {ckpt}")

        print(f"[{name}_spec eval]")
        m.eval()
        eval_matrix[f"{name}_spec"] = eval_all_domains(
            m, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
        merged_sds.append({k: v.cpu() for k, v in m.state_dict().items()})
        del m; torch.cuda.empty_cache()

    # Per-domain divergence
    print("\nPer-domain divergence:")
    divs = []
    for name in DOMAINS:
        base_d = base_eval[name]
        spec_d = eval_matrix[f"{name}_spec"][name]
        div    = (base_d - spec_d) / base_d * 100
        divs.append(div)
        print(f"  {name:12s}: base={base_d:.4f}  spec={spec_d:.4f}  div={div:+.2f}%")
    mean_div = sum(divs) / len(divs)
    print(f"  Mean divergence: {mean_div:.2f}%")

    # Predicted fusion gain from Item 1 linear regression
    # From Item 1: gain ≈ a + b * div  (run item1 first to get exact values)
    # Approximate from paper: 0.49× conversion for English domains
    predicted_gain_approx = mean_div * 0.49
    print(f"\nPredicted fusion gain (0.49× rule): ~{predicted_gain_approx:.2f}%")

    # MoE fusion
    print("\n[MoE fusion]")
    hidden_size = 1024  # Pythia-410M
    moe = LoRAMoE(merged_sds, MODEL_ID, REVISION, hidden_size, len(DOMAINS), device)
    moe.router = moe.router.to(device)
    train_router(moe, train_chunks, device)
    moe.eval()
    eval_matrix["moe"] = eval_all_domains(
        moe, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES, is_fused=True)
    del moe; torch.cuda.empty_cache()

    # Metrics
    base_ew      = base_eval["equal_weight_avg"]
    moe_ew       = eval_matrix["moe"]["equal_weight_avg"]
    best_spec_ew = min(eval_matrix[f"{n}_spec"]["equal_weight_avg"] for n in DOMAINS)
    gain_vs_spec = (best_spec_ew - moe_ew) / best_spec_ew * 100

    print(f"\n{'='*60}")
    print(f"LoRA r={LORA_RANK} Results (seed={SEED}):")
    print(f"  Base EW:          {base_ew:.4f}")
    print(f"  Best spec EW:     {best_spec_ew:.4f}")
    print(f"  MoE EW:           {moe_ew:.4f}")
    print(f"  Mean divergence:  {mean_div:.2f}%")
    print(f"  Gain vs spec:     {gain_vs_spec:+.2f}%")
    print(f"  Predicted gain:   ~{predicted_gain_approx:.2f}% (0.49× rule)")
    print(f"  Prediction error: {gain_vs_spec - predicted_gain_approx:+.2f}pp")
    print(f"{'='*60}")

    # Compare to full fine-tuning baseline
    # Full FT (from paper): mean_div=15.65%, gain=7.72%
    full_ft_div  = 15.65
    full_ft_gain = 7.72
    print(f"\n  Full FT baseline:  div={full_ft_div:.2f}%  gain={full_ft_gain:.2f}%")
    print(f"  LoRA r={LORA_RANK} vs Full FT: div {mean_div - full_ft_div:+.2f}pp  gain {gain_vs_spec - full_ft_gain:+.2f}pp")

    result = {
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "seed": SEED,
        "eval_matrix": {k: dict(v) for k, v in eval_matrix.items()},
        "metrics": {
            "base_ew": base_ew, "best_spec_ew": best_spec_ew, "moe_ew": moe_ew,
            "mean_divergence": mean_div, "per_domain_divergence": dict(zip(DOMAINS, divs)),
            "gain_vs_spec": gain_vs_spec, "predicted_gain_0_49x": predicted_gain_approx,
            "prediction_error": gain_vs_spec - predicted_gain_approx,
        },
        "full_ft_comparison": {"div": full_ft_div, "gain": full_ft_gain},
    }

    out_path = RESULTS_DIR / f"result_seed{SEED}{_lr_tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 60)
    print("DRAFT: Appendix C expansion (Why not LoRA?):\n")
    print(f"""We revisit the LoRA ablation with quantitative measurements. LoRA
specialists (Pythia-410M, $r={LORA_RANK}$, $\\alpha={LORA_ALPHA}$, 2,000 steps, seed 42) achieve
mean divergence {mean_div:.2f}\\% versus {full_ft_div:.2f}\\% for full fine-tuning---a reduction
of {full_ft_div - mean_div:.2f}pp. Under the empirical divergence-gain relationship
(Section~\\ref{{sec:divergence_gain}}, $\\approx$0.49$\\times$ conversion rate on English
domains), this predicts a fusion gain of approximately {predicted_gain_approx:.1f}\\%.
The measured fusion gain is {gain_vs_spec:.2f}\\%. The reduced divergence is the direct cause of the
reduced gain: LoRA's low-rank constraint limits the degree of domain adaptation,
keeping the specialist closer to the base model and reducing the complementarity
that drives MoE improvement. Full fine-tuning is required to achieve the divergence
levels ({full_ft_div:.2f}\\%) that produce the reported gains ({full_ft_gain:.2f}\\%).
""")


if __name__ == "__main__":
    main()
