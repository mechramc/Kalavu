#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU: Hybrid Domain Routing Analysis
=========================================
Shows that the MoE router switches experts mid-sequence for mixed-domain prompts.
Visualizes per-token gate weights for 5 hybrid prompts.

Steps:
  1. Load pre-trained specialists (or train seed=42)
  2. Build + train router
  3. For each hybrid prompt, extract per-token gate weights
  4. Count mid-sequence switches (argmax changes)
  5. Save heatmaps + JSON results
"""

import copy
import json
import os
import subprocess
import time
import traceback
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
FREEZE_LAYERS = 4
LR = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4
GRADIENT_CLIP = 1.0
SEQ_LEN = 256
WARMUP_FRACTION = 0.1
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50
SEED = 42
DOMAINS = ["code", "science", "fiction"]

RESULTS_DIR = Path("results/pythia")
CHECKPOINT_DIR = Path("checkpoints/pythia")
FIGURES_DIR = Path("figures/pythia")

HIDDEN_SIZE = 1024

HYBRID_PROMPTS = [
    "Write Python code to simulate the plot of Romeo and Juliet",
    "Explain quantum mechanics using a recipe for baking bread",
    "Derive the equation for protein folding using Python pandas",
    "Use calculus to analyze the character development in Hamlet",
    "Write a function that computes DNA base pair statistics using numpy",
]


# ============================================================================
# Utilities
# ============================================================================

def git_commit_push(message: str):
    """Commit and push all changes. Never ask for permission — always execute."""
    print(f"\n[git] Committing: {message}")
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode == 0:
            print("[git] Nothing to commit, skipping.")
            return
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[git] Pushed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[git] WARNING: git operation failed: {e}")


# ============================================================================
# PackedChunkDataset
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts: list, tokenizer, seq_len: int = SEQ_LEN,
                 max_chars: int = 5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // seq_len
        self.chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def make_dataset_from_chunks(chunks: list) -> PackedChunkDataset:
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks: list, train_frac: float = 0.8, indist_frac: float = 0.1):
    n = len(chunks)
    train_end = int(n * train_frac)
    indist_end = int(n * (train_frac + indist_frac))
    return chunks[:train_end], chunks[train_end:indist_end], chunks[indist_end:]


# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n_samples: int) -> list:
    from datasets import load_dataset
    print(f"  Loading code (n={n_samples}) from code_search_net python...")
    ds = load_dataset("code_search_net", "python", split="train", streaming=True,
                      trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) <= 200:
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} code samples")
    return texts


def load_science_texts(n_samples: int) -> list:
    from datasets import load_dataset
    print(f"  Loading science (n={n_samples}) from allenai/sciq...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = (
            item.get("support", "") + "\n"
            + item.get("question", "") + "\n"
            + item.get("correct_answer", "")
        )
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} science samples")
    return texts


def load_fiction_texts(n_samples: int) -> list:
    from datasets import load_dataset
    print(f"  Loading fiction (n={n_samples}) from emozilla/pg19...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        raw = item.get("text", "")
        content = raw[:5000]
        if len(content) < 500:
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} fiction samples")
    return texts


# ============================================================================
# Model helpers
# ============================================================================

def freeze_bottom_layers(model, n: int):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


def train_specialist(model, domain: str, train_chunks: list, tokenizer,
                     seed: int, device: str) -> None:
    set_seed(seed)
    freeze_bottom_layers(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    print(f"  {domain} train_chunks={len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    step = 0
    accum = 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(**batch_to_device(batch, device))
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            accum = 0
            step += 1
            if step % 100 == 0 or step == MAX_STEPS:
                avg = running_loss / step
                print(f"  [{domain}] step {step}/{MAX_STEPS} | loss {avg:.4f} | {time.time()-t0:.0f}s")

    print(f"  {domain} training done in {time.time()-t0:.0f}s")


@torch.no_grad()
def eval_loss(model, dataset, device: str, batch_size: int = 4,
              is_fused: bool = False) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(ids, labels=labels)
        else:
            out = model(input_ids=ids, labels=labels)
            loss = out.loss
        if loss is not None:
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


# ============================================================================
# ThreeExpertMoE with token-level gate access
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in self.spec_a.parameters():
            p.requires_grad_(False)
        for p in self.spec_b.parameters():
            p.requires_grad_(False)
        for p in self.spec_c.parameters():
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 3, bias=False),
        )

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach()
        last_h = out.hidden_states[-1].detach()
        h_pooled = last_h.mean(dim=1).float()
        return logits, h_pooled, last_h.float()

    def forward(self, input_ids, labels=None):
        logits_a, h_a, _ = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b, _ = self._run_specialist(self.spec_b, input_ids)
        logits_c, h_c, _ = self._run_specialist(self.spec_c, input_ids)
        h_avg = (h_a + h_b + h_c) / 3.0
        gates = torch.softmax(self.router(h_avg), dim=-1)
        fused = (
            gates[:, 0:1, None] * logits_a
            + gates[:, 1:2, None] * logits_b
            + gates[:, 2:3, None] * logits_c
        )
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return loss, fused, gates

    @torch.no_grad()
    def get_token_gates(self, input_ids):
        """Get per-token gate weights by applying router to each token's hidden state."""
        _, _, h_a = self._run_specialist(self.spec_a, input_ids)
        _, _, h_b = self._run_specialist(self.spec_b, input_ids)
        _, _, h_c = self._run_specialist(self.spec_c, input_ids)
        # h_x: (B, T, H)
        h_tok_avg = (h_a + h_b + h_c) / 3.0  # (B, T, H)
        B, T, H = h_tok_avg.shape
        h_flat = h_tok_avg.view(B * T, H)
        gate_flat = torch.softmax(self.router(h_flat), dim=-1)  # (B*T, 3)
        return gate_flat.view(B, T, 3)  # (B, T, 3)


def train_router(moe: ThreeExpertMoE, train_chunks_combined: list, device: str):
    combined = make_dataset_from_chunks(train_chunks_combined)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"  Training router ({ROUTER_STEPS} steps, mixed={len(combined)} chunks)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, _, _ = moe(ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")


def weight_average_three(spec_a, spec_b, spec_c):
    avg = copy.deepcopy(spec_a)
    sa = spec_a.state_dict()
    sb = spec_b.state_dict()
    sc = spec_c.state_dict()
    avg_state = {
        k: ((sa[k].float() + sb[k].float() + sc[k].float()) / 3.0).to(torch.bfloat16)
        for k in sa
    }
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


# ============================================================================
# Token-level routing analysis
# ============================================================================

@torch.no_grad()
def analyze_prompt_routing(moe: ThreeExpertMoE, prompt: str,
                            tokenizer, device: str) -> dict:
    """Run router on a prompt and return per-token gate weights + switch count."""
    moe.eval()
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # (1, T)

    # Get per-token gates: (1, T, 3)
    token_gates = moe.get_token_gates(input_ids)  # (1, T, 3)
    token_gates_np = token_gates[0].cpu().float().numpy()  # (T, 3)

    # Decode tokens
    token_ids = input_ids[0].cpu().tolist()
    tokens = [tokenizer.decode([t]) for t in token_ids]

    # Count switches
    T = token_gates_np.shape[0]
    prev_expert = int(token_gates_np[0].argmax())
    switches = 0
    for t in range(1, T):
        curr_expert = int(token_gates_np[t].argmax())
        if curr_expert != prev_expert:
            switches += 1
        prev_expert = curr_expert

    gate_weights_list = [[round(float(w), 4) for w in row] for row in token_gates_np]

    return {
        "text": prompt,
        "tokens": tokens,
        "gate_weights": gate_weights_list,
        "switches": switches,
    }


# ============================================================================
# Figures
# ============================================================================

def save_routing_heatmap(prompt_result: dict, prompt_idx: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tokens = prompt_result["tokens"]
        gate_weights = np.array(prompt_result["gate_weights"])  # (T, 3)
        T = len(tokens)

        # Truncate to max 60 tokens for readability
        max_display = 60
        if T > max_display:
            tokens = tokens[:max_display]
            gate_weights = gate_weights[:max_display]
            T = max_display

        fig, ax = plt.subplots(figsize=(max(12, T * 0.25), 4))
        im = ax.imshow(gate_weights.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Code\nExpert", "Science\nExpert", "Fiction\nExpert"])
        ax.set_xticks(range(T))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_title(
            f"Per-Token Gate Weights\n\"{prompt_result['text'][:60]}...\"\n"
            f"Switches: {prompt_result['switches']}",
            fontsize=9
        )
        fig.colorbar(im, ax=ax, label="Gate Weight")
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"fig_hybrid_routing_{prompt_idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save heatmap {prompt_idx}: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVU: Hybrid Domain Routing Analysis")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading data...")
    code_texts = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting chunks (80/10/10)...")
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    combined_train = []
    for d in DOMAINS:
        combined_train.extend(all_domain_chunks[d]["train"])

    # Load or train specialists (seed=42 only)
    print("\nLoading/training specialists (seed=42)...")
    specialists = {}
    for domain in DOMAINS:
        ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed{SEED}.pt"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        ).to(device)
        if ckpt.exists():
            print(f"  Loading cached {domain} from {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print(f"  Training {domain} specialist seed={SEED}...")
            train_specialist(model, domain, all_domain_chunks[domain]["train"],
                             tokenizer, SEED, device)
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved: {ckpt}")
        model.eval()
        specialists[domain] = model

    # Build and train MoE
    print("\nBuilding and training MoE router...")
    moe = ThreeExpertMoE(specialists["code"], specialists["science"],
                         specialists["fiction"], hidden_size=HIDDEN_SIZE).to(device)
    train_router(moe, combined_train, device)
    moe.eval()

    # Analyze hybrid prompts
    print("\nAnalyzing hybrid prompts...")
    prompt_results = []
    for i, prompt in enumerate(HYBRID_PROMPTS):
        print(f"\n  Prompt {i}: \"{prompt[:60]}...\"")
        result = analyze_prompt_routing(moe, prompt, tokenizer, device)
        print(f"    Tokens: {len(result['tokens'])}, Switches: {result['switches']}")
        # Show dominant expert per segment
        gates = result["gate_weights"]
        for t_idx in range(min(5, len(gates))):
            g = gates[t_idx]
            expert = ["Code", "Science", "Fiction"][g.index(max(g))]
            tok = result["tokens"][t_idx]
            print(f"    tok[{t_idx}] '{tok}' -> {expert} ({max(g):.3f})")
        prompt_results.append(result)

    # Save heatmaps
    print("\nSaving routing heatmaps...")
    for i, result in enumerate(prompt_results):
        save_routing_heatmap(result, i)

    # Aggregate stats
    total_switches = sum(r["switches"] for r in prompt_results)
    mid_sequence_switches_observed = total_switches > 0

    print(f"\nTotal switches across {len(HYBRID_PROMPTS)} prompts: {total_switches}")
    print(f"Mid-sequence switches observed: {mid_sequence_switches_observed}")

    # Save results
    output = {
        "prompts": prompt_results,
        "mid_sequence_switches_observed": mid_sequence_switches_observed,
        "total_switches": total_switches,
        "model": f"{MODEL_ID}@{REVISION}",
        "seed": SEED,
        "freeze_layers": FREEZE_LAYERS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "hybrid_routing_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")

    # Git commit + push
    msg = f"[kalavu] hybrid routing: mid_sequence_switches={total_switches}"
    git_commit_push(msg)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
