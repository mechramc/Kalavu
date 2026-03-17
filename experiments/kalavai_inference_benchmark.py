#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Inference Cost Benchmarking
=====================================
Loads already-trained 410M and 1B checkpoints and measures:
  - Peak GPU VRAM (torch.cuda.max_memory_allocated)
  - Tokens per second (throughput) — 512 tokens, 10 runs, 3 warmup
  - Per-token latency
  - Total parameter count

Configurations at 410M:
  1. Base model (single forward pass)
  2. Single specialist (single forward pass)
  3. Monolithic model (single forward pass)
  4. KALAVAI MoE — all 3 specialists, softmax routing (production)
  5. KALAVAI MoE — top-1 sparse (argmax, only top-1 expert's unfrozen layers run)
  6. KALAVAI MoE — top-2 (only top-2 experts, weighted combination)

Repeat at 1B for configs 1, 4, 5, 6.

Sparse top-1 detail: frozen-layer forward pass runs once (shared), router
predicts which expert, ONLY that expert's unfrozen layers run. Reports:
  - routing agreement % vs full joint-inference top-1
  - loss under sparse output

No training — load-and-measure only.
"""

import json
import statistics
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Config
# ============================================================================

WARMUP_RUNS  = 3
MEASURE_RUNS = 10
GEN_TOKENS   = 512        # tokens generated per throughput measurement
EVAL_BATCHES = 50
BATCH_SIZE   = 1          # single-sequence latency measurement
SEQ_LEN      = 512

CHECKPOINT_DIR_410M = Path("checkpoints/pythia")
CHECKPOINT_DIR_1B   = Path("checkpoints/pythia/pythia_1b")

RESULTS_DIR = Path("results/pythia")
FIGURES_DIR = Path("figures/pythia")

# 410M config
MODEL_410M    = "EleutherAI/pythia-410m"
HIDDEN_410M   = 1024
FREEZE_410M   = 4
LAYERS_410M   = 24

# 1B config
MODEL_1B      = "EleutherAI/pythia-1b"
HIDDEN_1B     = 2048
FREEZE_1B     = 4
LAYERS_1B     = 16

REVISION = "step10000"
DOMAINS  = ["code", "science", "fiction"]


# ============================================================================
# Dataset (same as all other experiments)
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
    }


def load_eval_data(model_id, revision, n_samples=500):
    """Load small mixed eval set for loss measurement."""
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_texts = []

    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    for item in ds:
        c = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(c) > 200: all_texts.append(c[:5000])
        if len(all_texts) >= n_samples // 3: break

    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    for item in ds:
        c = (item.get("support", "") + "\n" + item.get("question", "") + "\n"
             + item.get("correct_answer", ""))
        if len(c) > 100: all_texts.append(c)
        if len(all_texts) >= 2 * n_samples // 3: break

    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    for item in ds:
        c = item.get("text", "")[:5000]
        if len(c) >= 500: all_texts.append(c)
        if len(all_texts) >= n_samples: break

    full = tokenizer(
        "\n\n".join(all_texts),
        return_tensors="pt", truncation=False,
    )["input_ids"][0]
    n_chunks = len(full) // SEQ_LEN
    chunks = [full[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(n_chunks)]
    return chunks, tokenizer


# ============================================================================
# ThreeExpertMoE — production (all experts, softmax)
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for sp in [self.spec_a, self.spec_b, self.spec_c]:
            for p in sp.parameters():
                p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, 3, bias=False)

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits  = out.logits.detach()
        h_pooled = out.hidden_states[-1].detach().mean(dim=1).float()
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids)
        logits_c, h_c = self._run_specialist(self.spec_c, input_ids)
        h_avg  = (h_a + h_b + h_c) / 3.0
        gates  = torch.softmax(self.router(h_avg), dim=-1)
        fused  = (gates[:, 0:1, None] * logits_a
                + gates[:, 1:2, None] * logits_b
                + gates[:, 2:3, None] * logits_c)
        loss = None
        if labels is not None:
            sl = fused[:, :-1, :].contiguous()
            ll = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1))
        return loss, fused, gates


# ============================================================================
# TopK sparse MoE — only runs selected expert(s)' unfrozen layers
# ============================================================================

class SparseTopKMoE(nn.Module):
    """
    Sparse inference variant:
      1. Run frozen layers once (shared across all specialists — same weights).
      2. Pool hidden state from frozen layers only.
      3. Router picks top-k experts.
      4. Run ONLY those experts' unfrozen layers on the frozen-layer output.

    This is possible because frozen layers are identical across all specialists
    (shared initialization + frozen = identical weights throughout training).
    So we only need ONE frozen-layer pass, not three.
    """
    def __init__(self, spec_a, spec_b, spec_c, hidden_size, freeze_n, top_k=1):
        super().__init__()
        self.specs      = [spec_a, spec_b, spec_c]
        self.freeze_n   = freeze_n
        self.top_k      = top_k
        self.router     = nn.Linear(hidden_size, 3, bias=False)
        self.hidden_size = hidden_size
        for sp in self.specs:
            for p in sp.parameters():
                p.requires_grad_(False)

    def _frozen_forward(self, input_ids):
        """Run embedding + first freeze_n layers (identical across specialists)."""
        model = self.specs[0]   # all frozen layers are identical
        with torch.no_grad():
            x = model.gpt_neox.embed_in(input_ids)
            for i in range(self.freeze_n):
                x = model.gpt_neox.layers[i](x)[0]
        return x  # (B, T, H)

    def _unfrozen_forward(self, spec_idx, frozen_hidden):
        """Continue forward pass from layer freeze_n onwards for one specialist."""
        model = self.specs[spec_idx]
        with torch.no_grad():
            x = frozen_hidden
            for i in range(self.freeze_n, len(model.gpt_neox.layers)):
                x = model.gpt_neox.layers[i](x)[0]
            x = model.gpt_neox.final_layer_norm(x)
            logits = model.embed_out(x)
        return logits  # (B, T, V)

    def forward(self, input_ids, labels=None):
        # 1. Shared frozen-layer pass
        frozen_h = self._frozen_forward(input_ids)         # (B, T, H)
        h_pooled = frozen_h.mean(dim=1).float()            # (B, H)

        # 2. Route
        gates_full = torch.softmax(self.router(h_pooled), dim=-1)  # (B, 3)
        top_vals, top_idx = torch.topk(gates_full, self.top_k, dim=-1)  # (B, k)

        # 3. Run only selected experts (one pass each)
        # For simplicity: process batch element by element for top-k selection
        # In practice with batch_size=1 this is trivial
        B = input_ids.size(0)
        # Collect logits for selected experts
        selected_logits = []
        for b in range(B):
            weighted = None
            gate_sum  = top_vals[b].sum()
            for rank in range(self.top_k):
                exp_idx = top_idx[b, rank].item()
                w       = top_vals[b, rank] / gate_sum
                logits_exp = self._unfrozen_forward(exp_idx, frozen_h[b:b+1])
                if weighted is None:
                    weighted = w * logits_exp
                else:
                    weighted = weighted + w * logits_exp
            selected_logits.append(weighted)
        fused = torch.cat(selected_logits, dim=0)  # (B, T, V)

        loss = None
        if labels is not None:
            sl = fused[:, :-1, :].contiguous()
            ll = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1))
        return loss, fused, gates_full, top_idx


# ============================================================================
# Measurement helpers
# ============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_params_total(*models):
    """Count unique parameters (for MoE, sum all specialists + router)."""
    return sum(count_params(m) for m in models)


@torch.no_grad()
def measure_throughput(model, input_ids, device, warmup=WARMUP_RUNS, runs=MEASURE_RUNS,
                        is_moe=False, is_sparse=False):
    """Returns tokens/sec averaged over `runs` forward passes."""
    input_ids = input_ids.to(device)
    labels    = input_ids.clone()

    def one_pass():
        if is_sparse:
            model(input_ids, labels=labels)
        elif is_moe:
            model(input_ids, labels=labels)
        else:
            model(input_ids=input_ids, labels=labels)

    # Warmup
    for _ in range(warmup):
        one_pass()
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_pass()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    tokens    = input_ids.numel()
    mean_sec  = statistics.mean(times)
    tok_per_s = tokens / mean_sec
    return tok_per_s, mean_sec


@torch.no_grad()
def measure_vram(model, input_ids, device, is_moe=False, is_sparse=False):
    """Returns peak VRAM in GB after one forward pass."""
    input_ids = input_ids.to(device)
    labels    = input_ids.clone()
    torch.cuda.reset_peak_memory_stats(device)
    if is_sparse:
        model(input_ids, labels=labels)
    elif is_moe:
        model(input_ids, labels=labels)
    else:
        model(input_ids=input_ids, labels=labels)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    return round(peak, 3)


@torch.no_grad()
def eval_loss_model(model, chunks, device, batch_size=4, is_moe=False, is_sparse=False,
                    n_batches=EVAL_BATCHES):
    ds = PackedChunkDataset(chunks)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    total, count = 0.0, 0
    for batch in loader:
        if count >= n_batches: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        if is_sparse:
            loss, _, _, _ = model(ids, labels=lbl)
        elif is_moe:
            loss, _, _ = model(ids, labels=lbl)
        else:
            out  = model(input_ids=ids, labels=lbl)
            loss = out.loss
        if loss is not None:
            total += loss.item()
            count += 1
    return round(total / count, 6) if count > 0 else float("inf")


@torch.no_grad()
def routing_agreement(dense_moe: ThreeExpertMoE, sparse_moe: SparseTopKMoE,
                       chunks, device, n_batches=20):
    """% of sequences where sparse top-1 matches dense top-1."""
    ds     = PackedChunkDataset(chunks)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    agree, total = 0, 0
    for batch in loader:
        if total >= n_batches: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        _, _, gates_dense    = dense_moe(ids, labels=lbl)
        _, _, gates_sparse, top_idx = sparse_moe(ids, labels=lbl)
        dense_top1  = gates_dense.argmax(dim=-1)   # (B,)
        sparse_top1 = top_idx[:, 0]                # (B,)
        agree += (dense_top1 == sparse_top1).sum().item()
        total += ids.size(0)
    return round(100.0 * agree / total, 2) if total > 0 else 0.0


# ============================================================================
# Load specialist models
# ============================================================================

def load_specialist(model_id, revision, ckpt_path, device, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype,
    ).to(device)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"    Loaded checkpoint: {ckpt_path}")
    else:
        print(f"    WARNING: checkpoint not found: {ckpt_path} — using base weights")
    model.eval()
    return model


def load_router_weights(moe, router_ckpt_path):
    """Load router weights if saved separately, else leave randomly initialised."""
    if router_ckpt_path and router_ckpt_path.exists():
        state = torch.load(router_ckpt_path, map_location="cpu", weights_only=True)
        router_state = {k.replace("router.", ""): v for k, v in state.items()
                        if k.startswith("router.")}
        if router_state:
            moe.router.load_state_dict(router_state)
            print(f"    Loaded router weights from {router_ckpt_path}")
        else:
            print(f"    WARNING: no router keys in {router_ckpt_path}")
    else:
        print(f"    NOTE: no router checkpoint found — router uses random init for throughput measurement")


# ============================================================================
# Benchmark one model size
# ============================================================================

def benchmark_size(model_id, revision, ckpt_dir, hidden_size, freeze_n,
                    eval_chunks, device, size_label):
    results = []

    print(f"\n{'='*60}")
    print(f"Benchmarking: {size_label}")
    print(f"{'='*60}")

    input_ids = torch.randint(0, 50277, (1, SEQ_LEN))  # random prompt for throughput

    # ---- Config 1: Base model ------------------------------------------------
    print("\n[Config 1] Base model...")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
    ).to(device)
    base.eval()

    n_params = count_params(base)
    vram     = measure_vram(base, input_ids, device)
    tok_s, lat = measure_throughput(base, input_ids, device)
    loss_base  = eval_loss_model(base, eval_chunks, device)

    results.append({
        "config":           "Base model",
        "params_M":         round(n_params / 1e6, 1),
        "peak_vram_GB":     vram,
        "tokens_per_sec":   round(tok_s, 1),
        "latency_ms":       round(lat * 1000, 2),
        "relative_latency": 1.0,
        "eval_loss":        loss_base,
        "notes":            "",
    })
    base_lat = lat
    print(f"  params={n_params/1e6:.0f}M  vram={vram}GB  tok/s={tok_s:.1f}  loss={loss_base:.4f}")

    # Keep base for re-use
    del base
    torch.cuda.empty_cache()

    # ---- Config 2 (410M only): Single specialist ----------------------------
    if size_label == "410M":
        print("\n[Config 2] Single specialist (seed=42, code domain)...")
        spec = load_specialist(model_id, revision,
                               ckpt_dir / "code_specialist_seed42.pt", device)
        n_params = count_params(spec)
        vram     = measure_vram(spec, input_ids, device)
        tok_s, lat = measure_throughput(spec, input_ids, device)
        loss_spec  = eval_loss_model(spec, eval_chunks, device)

        results.append({
            "config":           "Single specialist",
            "params_M":         round(n_params / 1e6, 1),
            "peak_vram_GB":     vram,
            "tokens_per_sec":   round(tok_s, 1),
            "latency_ms":       round(lat * 1000, 2),
            "relative_latency": round(lat / base_lat, 3),
            "eval_loss":        loss_spec,
            "notes":            "code specialist, seed=42",
        })
        print(f"  params={n_params/1e6:.0f}M  vram={vram}GB  tok/s={tok_s:.1f}  loss={loss_spec:.4f}")
        del spec
        torch.cuda.empty_cache()

    # ---- Config 3 (410M only): Monolithic -----------------------------------
    if size_label == "410M":
        print("\n[Config 3] Monolithic model (seed=42)...")
        mono = load_specialist(model_id, revision,
                               ckpt_dir / "monolithic_seed42.pt", device)
        n_params  = count_params(mono)
        vram      = measure_vram(mono, input_ids, device)
        tok_s, lat = measure_throughput(mono, input_ids, device)
        loss_mono  = eval_loss_model(mono, eval_chunks, device)

        results.append({
            "config":           "Monolithic",
            "params_M":         round(n_params / 1e6, 1),
            "peak_vram_GB":     vram,
            "tokens_per_sec":   round(tok_s, 1),
            "latency_ms":       round(lat * 1000, 2),
            "relative_latency": round(lat / base_lat, 3),
            "eval_loss":        loss_mono,
            "notes":            "6000-step mixed training, seed=42",
        })
        print(f"  params={n_params/1e6:.0f}M  vram={vram}GB  tok/s={tok_s:.1f}  loss={loss_mono:.4f}")
        del mono
        torch.cuda.empty_cache()

    # ---- Load three specialists for MoE configs -----------------------------
    print("\nLoading 3 specialists for MoE configs...")
    spec_a = load_specialist(model_id, revision, ckpt_dir / "code_specialist_seed42.pt",    device)
    spec_b = load_specialist(model_id, revision, ckpt_dir / "science_specialist_seed42.pt", device)
    spec_c = load_specialist(model_id, revision, ckpt_dir / "fiction_specialist_seed42.pt", device)

    # ---- Config 4: Dense MoE (all 3 specialists, softmax routing) -----------
    print(f"\n[Config {'4' if size_label=='410M' else '2'}] KALAVAI MoE — dense (all 3 specialists)...")
    dense_moe = ThreeExpertMoE(spec_a, spec_b, spec_c, hidden_size).to(device)
    # Note: router uses random init for throughput measurement — weights don't
    # affect timing since all code paths are identical regardless of gate values
    dense_moe.eval()

    n_params_moe = count_params_total(spec_a, spec_b, spec_c) + count_params(dense_moe.router)
    vram         = measure_vram(dense_moe, input_ids, device, is_moe=True)
    tok_s, lat   = measure_throughput(dense_moe, input_ids, device, is_moe=True)
    loss_dense   = eval_loss_model(dense_moe, eval_chunks, device, is_moe=True)

    cfg_num = "4" if size_label == "410M" else "2"
    results.append({
        "config":           "KALAVAI MoE — dense (top-3)",
        "params_M":         round(n_params_moe / 1e6, 1),
        "peak_vram_GB":     vram,
        "tokens_per_sec":   round(tok_s, 1),
        "latency_ms":       round(lat * 1000, 2),
        "relative_latency": round(lat / base_lat, 3),
        "eval_loss":        loss_dense,
        "notes":            "3 full specialist passes + router",
    })
    print(f"  params={n_params_moe/1e6:.0f}M  vram={vram}GB  tok/s={tok_s:.1f}  loss={loss_dense:.4f}")
    dense_moe_lat = lat

    # ---- Config 5: Sparse top-1 MoE -----------------------------------------
    print(f"\n[Config {'5' if size_label=='410M' else '3'}] KALAVAI MoE — sparse top-1...")
    sparse_top1 = SparseTopKMoE(spec_a, spec_b, spec_c, hidden_size,
                                  freeze_n=freeze_n, top_k=1).to(device)
    # Copy router weights from dense MoE for fair routing comparison
    sparse_top1.router.load_state_dict(dense_moe.router.state_dict())
    sparse_top1.eval()

    vram       = measure_vram(sparse_top1, input_ids, device, is_sparse=True)
    tok_s, lat = measure_throughput(sparse_top1, input_ids, device, is_sparse=True)
    loss_s1    = eval_loss_model(sparse_top1, eval_chunks, device, is_sparse=True)
    agree_pct  = routing_agreement(dense_moe, sparse_top1, eval_chunks, device)

    cfg_num = "5" if size_label == "410M" else "3"
    results.append({
        "config":               "KALAVAI MoE — sparse top-1",
        "params_M":             round(n_params_moe / 1e6, 1),  # loaded params same, but only 1 runs
        "peak_vram_GB":         vram,
        "tokens_per_sec":       round(tok_s, 1),
        "latency_ms":           round(lat * 1000, 2),
        "relative_latency":     round(lat / base_lat, 3),
        "eval_loss":            loss_s1,
        "routing_agreement_pct": agree_pct,
        "notes":                "1 frozen-layer pass + 1 expert unfrozen pass",
    })
    print(f"  params={n_params_moe/1e6:.0f}M  vram={vram}GB  tok/s={tok_s:.1f}  "
          f"loss={loss_s1:.4f}  routing_agreement={agree_pct}%")

    # ---- Config 6: Sparse top-2 MoE -----------------------------------------
    print(f"\n[Config {'6' if size_label=='410M' else '4'}] KALAVAI MoE — sparse top-2...")
    sparse_top2 = SparseTopKMoE(spec_a, spec_b, spec_c, hidden_size,
                                  freeze_n=freeze_n, top_k=2).to(device)
    sparse_top2.router.load_state_dict(dense_moe.router.state_dict())
    sparse_top2.eval()

    vram       = measure_vram(sparse_top2, input_ids, device, is_sparse=True)
    tok_s, lat = measure_throughput(sparse_top2, input_ids, device, is_sparse=True)
    loss_s2    = eval_loss_model(sparse_top2, eval_chunks, device, is_sparse=True)

    cfg_num = "6" if size_label == "410M" else "4"
    results.append({
        "config":           "KALAVAI MoE — sparse top-2",
        "params_M":         round(n_params_moe / 1e6, 1),
        "peak_vram_GB":     vram,
        "tokens_per_sec":   round(tok_s, 1),
        "latency_ms":       round(lat * 1000, 2),
        "relative_latency": round(lat / base_lat, 3),
        "eval_loss":        loss_s2,
        "notes":            "1 frozen-layer pass + 2 expert unfrozen passes",
    })
    print(f"  params={n_params_moe/1e6:.0f}M  vram={vram}GB  tok/s={tok_s:.1f}  loss={loss_s2:.4f}")

    del dense_moe, sparse_top1, sparse_top2, spec_a, spec_b, spec_c
    torch.cuda.empty_cache()

    return results, base_lat


# ============================================================================
# Print table
# ============================================================================

def print_table(results, size_label, base_lat_s):
    print(f"\n{'='*90}")
    print(f"INFERENCE BENCHMARK RESULTS — {size_label}")
    print(f"{'='*90}")
    hdr = (f"{'Config':<36} {'Params':>8} {'VRAM':>8} {'Tok/s':>8} "
           f"{'Lat(ms)':>9} {'Rel.Lat':>8} {'Loss':>8}")
    print(hdr)
    print("-" * 90)
    for r in results:
        agree = f"  (agree={r['routing_agreement_pct']}%)" if "routing_agreement_pct" in r else ""
        print(f"{r['config']:<36} {r['params_M']:>7.0f}M {r['peak_vram_GB']:>7.2f}G "
              f"{r['tokens_per_sec']:>8.1f} {r['latency_ms']:>9.1f} "
              f"{r['relative_latency']:>7.2f}x {r['eval_loss']:>8.4f}{agree}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Inference Cost Benchmark")
    print("=" * 70)
    print(f"Warmup runs: {WARMUP_RUNS}  |  Measure runs: {MEASURE_RUNS}")
    print(f"Sequence length: {SEQ_LEN}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Load eval data -------------------------------------------------------
    print("\nLoading eval data for 410M...")
    eval_chunks_410m, _ = load_eval_data(MODEL_410M, REVISION, n_samples=300)
    print(f"  eval chunks: {len(eval_chunks_410m)}")

    # ---- 410M benchmark -------------------------------------------------------
    results_410m, base_lat_410m = benchmark_size(
        model_id    = MODEL_410M,
        revision    = REVISION,
        ckpt_dir    = CHECKPOINT_DIR_410M,
        hidden_size = HIDDEN_410M,
        freeze_n    = FREEZE_410M,
        eval_chunks = eval_chunks_410m,
        device      = device,
        size_label  = "410M",
    )
    print_table(results_410m, "410M", base_lat_410m)

    # ---- 1B benchmark --------------------------------------------------------
    print("\nLoading eval data for 1B...")
    eval_chunks_1b, _ = load_eval_data(MODEL_1B, REVISION, n_samples=300)
    print(f"  eval chunks: {len(eval_chunks_1b)}")

    results_1b, base_lat_1b = benchmark_size(
        model_id    = MODEL_1B,
        revision    = REVISION,
        ckpt_dir    = CHECKPOINT_DIR_1B,
        hidden_size = HIDDEN_1B,
        freeze_n    = FREEZE_1B,
        eval_chunks = eval_chunks_1b,
        device      = device,
        size_label  = "1B",
    )
    print_table(results_1b, "1B", base_lat_1b)

    # ---- Save JSON -----------------------------------------------------------
    output = {
        "experiment":    "inference_benchmark",
        "device":        device,
        "gpu_name":      torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "warmup_runs":   WARMUP_RUNS,
        "measure_runs":  MEASURE_RUNS,
        "seq_len":       SEQ_LEN,
        "results_410m":  results_410m,
        "results_1b":    results_1b,
        "summary": {
            "410m_base_ms":          round(base_lat_410m * 1000, 2),
            "410m_dense_moe_ms":     next((r["latency_ms"] for r in results_410m
                                           if "dense" in r["config"]), None),
            "410m_sparse_top1_ms":   next((r["latency_ms"] for r in results_410m
                                           if "top-1" in r["config"]), None),
            "410m_routing_agreement_pct": next((r.get("routing_agreement_pct")
                                                for r in results_410m if "top-1" in r["config"]), None),
            "1b_base_ms":            round(base_lat_1b * 1000, 2),
            "1b_dense_moe_ms":       next((r["latency_ms"] for r in results_1b
                                           if "dense" in r["config"]), None),
            "1b_sparse_top1_ms":     next((r["latency_ms"] for r in results_1b
                                           if "top-1" in r["config"]), None),
            "1b_routing_agreement_pct": next((r.get("routing_agreement_pct")
                                              for r in results_1b if "top-1" in r["config"]), None),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "inference_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Figure --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, results, title in [
            (axes[0], results_410m, "Pythia-410M"),
            (axes[1], results_1b,   "Pythia-1B"),
        ]:
            configs = [r["config"].replace("KALAVAI MoE — ", "MoE\n") for r in results]
            lats    = [r["latency_ms"] for r in results]
            colors  = ["#95a5a6", "#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"][:len(results)]
            ax.bar(range(len(configs)), lats, color=colors[:len(lats)], alpha=0.85, width=0.6)
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, fontsize=7, rotation=15, ha="right")
            ax.set_ylabel("Latency (ms per sequence)")
            ax.set_title(f"Inference Latency — {title}")
            ax.grid(True, axis="y", alpha=0.3)
            for i, (lat, r) in enumerate(zip(lats, results)):
                ax.text(i, lat + max(lats) * 0.01, f"{r['relative_latency']:.1f}x",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        fig.suptitle("KALAVAI Inference Cost: Dense vs Sparse MoE", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = FIGURES_DIR / "fig_inference_benchmark.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {path}")
    except Exception as e:
        print(f"WARNING: figure failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
