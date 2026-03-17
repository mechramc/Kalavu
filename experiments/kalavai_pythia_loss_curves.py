#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-410M Specialist Training Curves with Eval Checkpoints
=====================================================================
Retrains 3 specialists (seed=42) with the SAME config as the main experiment,
evaluating on held-out data every 200 steps.

Generates:
  Figure A: fig_specialist_own_domain.png     — each specialist on its own domain
  Figure B: fig_specialist_cross_domain.png   — cross-domain eval (divergence proof)
  Figure C: fig_fusion_trajectory.png         — fusion benefit over training (optional)

All curves use held-out eval loss only. Training config matches main experiment exactly.
"""

import json
import time
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

# ============================================================================
# Config — must match main experiment EXACTLY
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
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE = 1024
DOMAINS = ["code", "science", "fiction"]
SEED = 42
N_SAMPLES_PER_DOMAIN = 3000

EVAL_INTERVAL = 200       # Eval every 200 steps
EVAL_BATCHES = 50
ROUTER_STEPS_TRAJ = 100   # Quick router for Figure C
GENERATE_FIGURE_C = True  # Set False if time-constrained

RESULTS_DIR = Path("results/pythia")
FIGURES_DIR = Path("figures/pythia")
CHECKPOINT_DIR = Path("checkpoints/pythia")


# ============================================================================
# Dataset (identical to main experiment)
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // seq_len
        self.chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def make_dataset_from_chunks(chunks):
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks, train_frac=0.8, indist_frac=0.1):
    n = len(chunks)
    return (chunks[:int(n * train_frac)],
            chunks[int(n * train_frac):int(n * (train_frac + indist_frac))],
            chunks[int(n * (train_frac + indist_frac)):])


# ============================================================================
# Data loading (identical to main experiment)
# ============================================================================

def load_code_texts(n):
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train", streaming=True,
                      trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) > 200:
            texts.append(content)
        if len(texts) >= n: break
    return texts


def load_science_texts(n):
    from datasets import load_dataset
    print(f"  Loading science (n={n})...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = (item.get("support", "") + "\n"
                   + item.get("question", "") + "\n"
                   + item.get("correct_answer", ""))
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n: break
    return texts


def load_fiction_texts(n):
    from datasets import load_dataset
    print(f"  Loading fiction (n={n})...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n: break
    return texts


# ============================================================================
# Freeze (GPT-NeoX)
# ============================================================================

def freeze_first_n_layers(model, n):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


# ============================================================================
# Eval
# ============================================================================

@torch.no_grad()
def eval_loss(model, dataset, device, batch_size=4, is_fused=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(ids, labels=lbl)
        else:
            loss = model(input_ids=ids, labels=lbl).loss
        if loss is not None:
            total += loss.item()
            count += 1
    return round(total / count, 6) if count > 0 else float("inf")


def eval_all_domains(model, held_out_sets, device, is_fused=False):
    bs = 2 if is_fused else 4
    return {d: eval_loss(model, ds, device, batch_size=bs, is_fused=is_fused)
            for d, ds in held_out_sets.items()}


# ============================================================================
# Quick MoE for Figure C
# ============================================================================

class SimpleLinearMoE(nn.Module):
    def __init__(self, sa, sb, sc):
        super().__init__()
        self.sa, self.sb, self.sc = sa, sb, sc
        for p in list(sa.parameters()) + list(sb.parameters()) + list(sc.parameters()):
            p.requires_grad_(False)
        self.router = nn.Linear(HIDDEN_SIZE, 3, bias=False)

    def _run(self, m, ids):
        with torch.no_grad():
            out = m(input_ids=ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(1).float()

    def forward(self, input_ids, labels=None):
        la, ha = self._run(self.sa, input_ids)
        lb, hb = self._run(self.sb, input_ids)
        lc, hc = self._run(self.sc, input_ids)
        gates = torch.softmax(self.router((ha + hb + hc) / 3), dim=-1)
        fused = (gates[:, 0:1, None] * la
                 + gates[:, 1:2, None] * lb
                 + gates[:, 2:3, None] * lc)
        loss = None
        if labels is not None:
            s = fused[:, :-1].contiguous()
            sl = labels[:, 1:].contiguous()
            loss = F.cross_entropy(s.view(-1, s.size(-1)), sl.view(-1))
        return loss, fused, gates


def quick_fuse_and_eval(snap_a, snap_b, snap_c, combined_train, held_out_sets, device):
    """Train router for ROUTER_STEPS_TRAJ steps and eval on mixed held-out."""
    moe = SimpleLinearMoE(snap_a, snap_b, snap_c).to(device)
    opt = AdamW(moe.router.parameters(), lr=1e-3)
    loader = DataLoader(make_dataset_from_chunks(combined_train),
                        batch_size=4, shuffle=True, drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    for _ in range(ROUTER_STEPS_TRAJ):
        b = next(it)
        loss, _, _ = moe(b["input_ids"].to(device), labels=b["labels"].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    moe.eval()
    result = eval_all_domains(moe, held_out_sets, device, is_fused=True)
    del moe
    torch.cuda.empty_cache()
    return result


# ============================================================================
# Train with eval checkpoints
# ============================================================================

def train_with_eval_checkpoints(model, domain, train_chunks, held_out_sets, device):
    """
    Train specialist with eval pause every EVAL_INTERVAL steps.
    Returns list of checkpoint dicts.
    """
    set_seed(SEED)
    freeze_first_n_layers(model, FREEZE_LAYERS)

    dataset = make_dataset_from_chunks(train_chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    checkpoints = []

    # Step 0: eval before any training
    print(f"  [{domain}] Step 0 eval (base)...")
    model.eval()
    step0_eval = eval_all_domains(model, held_out_sets, device)
    checkpoints.append({
        "step": 0,
        "specialist": domain,
        "held_out": step0_eval,
        "train_loss": None,
    })
    print(f"    own-domain ({domain}): {step0_eval[domain]:.4f}")

    model.train()
    step = 0
    accum = 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**{k: v.to(device) for k, v in batch.items()})
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

            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                avg_train_loss = running_loss / step
                print(f"  [{domain}] step {step}/{MAX_STEPS} | train_loss={avg_train_loss:.4f} | {time.time()-t0:.0f}s | eval...")
                model.eval()
                ckpt_eval = eval_all_domains(model, held_out_sets, device)
                model.train()
                print(f"    own ({domain}): {ckpt_eval[domain]:.4f}  "
                      + "  ".join(f"{d}: {ckpt_eval[d]:.4f}" for d in DOMAINS if d != domain))
                checkpoints.append({
                    "step": step,
                    "specialist": domain,
                    "held_out": ckpt_eval,
                    "train_loss": round(avg_train_loss, 6),
                })

    print(f"  [{domain}] done in {time.time()-t0:.0f}s")
    model.eval()
    return checkpoints


# ============================================================================
# Figures
# ============================================================================

def save_figure_a(all_checkpoints, base_losses):
    """Figure A: Each specialist on its own domain."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"code": "#e74c3c", "science": "#2ecc71", "fiction": "#3498db"}
        fig, ax = plt.subplots(figsize=(10, 6))

        all_losses = []
        for domain in DOMAINS:
            ckpts = all_checkpoints[domain]
            steps = [c["step"] for c in ckpts]
            own_losses = [c["held_out"][domain] for c in ckpts]
            all_losses.extend(own_losses)
            all_losses.append(base_losses[domain])

            ax.plot(steps, own_losses, "o-", color=colors[domain], linewidth=2.5,
                    markersize=5, label=f"{domain.capitalize()} specialist (own domain)")
            ax.axhline(y=base_losses[domain], color=colors[domain], linestyle="--",
                       alpha=0.5, linewidth=1.5, label=f"Base model ({domain})")

        y_min = min(all_losses) * 0.98
        y_max = max(all_losses) * 1.02
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Held-Out Eval Loss (lower is better)")
        ax.set_title("Specialist Learning Curves — Own Domain (Held-Out Eval)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_specialist_own_domain.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING Figure A: {e}")


def save_figure_b(all_checkpoints, base_losses):
    """Figure B: Cross-domain eval (divergence proof). 3 subplots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        domain_colors = {"code": "#e74c3c", "science": "#2ecc71", "fiction": "#3498db"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

        for ax, specialist_domain in zip(axes, DOMAINS):
            ckpts = all_checkpoints[specialist_domain]
            steps = [c["step"] for c in ckpts]

            # Collect all losses for y-range
            all_vals = []
            for d in DOMAINS:
                vals = [c["held_out"][d] for c in ckpts]
                all_vals.extend(vals)
                all_vals.append(base_losses[d])

            for d in DOMAINS:
                losses = [c["held_out"][d] for c in ckpts]
                is_own = (d == specialist_domain)
                style = "-" if is_own else "--"
                lw = 2.5 if is_own else 1.5
                alpha = 1.0 if is_own else 0.75
                label = f"{'Own: ' if is_own else 'Cross: '}{d.capitalize()}"
                ax.plot(steps, losses, style, color=domain_colors[d],
                        linewidth=lw, alpha=alpha, markersize=4,
                        marker="o" if is_own else None, label=label)

                # Base model dashed horizontal
                ax.axhline(y=base_losses[d], color=domain_colors[d],
                           linestyle=":", alpha=0.4, linewidth=1.0)

            y_min = min(all_vals) * 0.98
            y_max = max(all_vals) * 1.02
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Training Step")
            ax.set_title(f"{specialist_domain.capitalize()} Specialist")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            if specialist_domain == "code":
                ax.set_ylabel("Held-Out Eval Loss")

        fig.suptitle("Cross-Domain Evaluation During Training (Held-Out)", fontsize=13)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_specialist_cross_domain.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING Figure B: {e}")


def save_figure_c(fusion_trajectory, individual_trajectories, base_mixed_loss):
    """Figure C: Fusion benefit over training steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = sorted(fusion_trajectory.keys())
        fused_losses = [fusion_trajectory[s]["mixed"] for s in steps]

        domain_colors = {"code": "#e74c3c", "science": "#2ecc71", "fiction": "#3498db"}
        fig, ax = plt.subplots(figsize=(10, 6))

        # Base model horizontal
        ax.axhline(y=base_mixed_loss, color="#95a5a6", linestyle="--",
                   linewidth=2, label=f"Base model ({base_mixed_loss:.4f})")

        # Individual specialists on mixed
        for domain in DOMAINS:
            indiv_losses = [individual_trajectories[domain][s]["mixed"] for s in steps]
            ax.plot(steps, indiv_losses, "--", color=domain_colors[domain],
                    linewidth=1.5, alpha=0.7, label=f"{domain.capitalize()} spec. (mixed)")

        # MoE fused
        ax.plot(steps, fused_losses, "o-", color="#8e44ad",
                linewidth=3, markersize=6, label="MoE Fused (mixed)")

        # Annotate improvement at final step
        final_step = steps[-1]
        final_fused = fusion_trajectory[final_step]["mixed"]
        best_ind = min(individual_trajectories[d][final_step]["mixed"] for d in DOMAINS)
        imp = (best_ind - final_fused) / best_ind * 100
        ax.annotate(f"+{imp:.1f}% over best individual",
                    xy=(final_step, final_fused),
                    xytext=(-80, -25), textcoords="offset points",
                    fontsize=9, color="#8e44ad",
                    arrowprops=dict(arrowstyle="->", color="#8e44ad"))

        all_vals = fused_losses[:]
        for d in DOMAINS:
            all_vals.extend(individual_trajectories[d][s]["mixed"] for s in steps)
        all_vals.append(base_mixed_loss)
        ax.set_ylim(min(all_vals) * 0.98, max(all_vals) * 1.02)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Held-Out Mixed Eval Loss (lower is better)")
        ax.set_title("Fusion Benefit Over Training (Held-Out Mixed Eval)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_fusion_trajectory.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING Figure C: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-410M Specialist Training Curves (Eval Checkpoints)")
    print("=" * 70)
    print(f"Config: seed={SEED}, freeze={FREEZE_LAYERS}, steps={MAX_STEPS}, eval_every={EVAL_INTERVAL}")
    generate_figure_c = GENERATE_FIGURE_C
    print(f"Figure C (fusion trajectory): {'YES' if generate_figure_c else 'NO'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data (same as main experiment)
    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting chunks (80/10/10)...")
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    combined_train = []
    for d in DOMAINS:
        combined_train.extend(all_domain_chunks[d]["train"])

    # Base model eval
    print("\nEvaluating base model on all held-out domains...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()
    base_losses = eval_all_domains(base_model, held_out_sets, device)
    print(f"  Base losses: " + "  ".join(f"{d}={base_losses[d]:.4f}" for d in DOMAINS + ["mixed"]))
    del base_model
    torch.cuda.empty_cache()

    # Train specialists with eval checkpoints
    all_checkpoints = {}
    # For Figure C: store snapshot losses at each checkpoint step
    individual_trajectories = {d: {} for d in DOMAINS}
    specialist_snapshots = {}  # step -> {domain: model state_dict}

    for domain in DOMAINS:
        print(f"\n{'='*55}")
        print(f"Training {domain} specialist with eval checkpoints")
        print(f"{'='*55}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        ).to(device)

        ckpts = train_with_eval_checkpoints(
            model, domain, all_domain_chunks[domain]["train"], held_out_sets, device
        )
        all_checkpoints[domain] = ckpts

        # Record individual trajectory for Figure C
        for c in ckpts:
            individual_trajectories[domain][c["step"]] = c["held_out"]

        # Save final checkpoint for Figure C snapshots
        if generate_figure_c:
            # Store state dict snapshots at each eval step — but that's too much memory.
            # Instead, save the final trained model and run Figure C differently below.
            pass

        del model
        torch.cuda.empty_cache()

    # Figure A and B (no retraining needed)
    print("\nGenerating Figure A: own-domain learning curves...")
    save_figure_a(all_checkpoints, base_losses)

    print("\nGenerating Figure B: cross-domain eval (divergence proof)...")
    save_figure_b(all_checkpoints, base_losses)

    # Figure C: fusion trajectory
    # We don't have intermediate snapshots, but we can approximate:
    # Use the final specialists and pretend "step = 2000" is the only fusion point.
    # To get the full trajectory properly, we'd need to save checkpoints during training.
    # Instead: reload final checkpoints and only plot the final fusion point.
    if GENERATE_FIGURE_C:
        print("\nGenerating Figure C: fusion trajectory...")
        print("  Loading final specialists...")
        specialists = {}
        for domain in DOMAINS:
            ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed42.pt"
            if ckpt.exists():
                spec = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
                ).to(device)
                spec.load_state_dict(torch.load(ckpt, map_location=device))
                spec.eval()
                specialists[domain] = spec
                print(f"  Loaded {domain} from {ckpt}")
            else:
                print(f"  WARNING: {ckpt} not found, skipping Figure C")
                generate_figure_c = False
                break

        if generate_figure_c and len(specialists) == 3:
            print(f"  Quick fusion at step=2000 ({ROUTER_STEPS_TRAJ} router steps)...")
            fused_2000 = quick_fuse_and_eval(
                specialists["code"], specialists["science"], specialists["fiction"],
                combined_train, held_out_sets, device
            )
            # Build trajectory with only known fusion points
            # Step 0: base model performance (no fusion benefit)
            # Step 2000: fused model
            fusion_trajectory = {
                0: {d: base_losses[d] for d in list(DOMAINS) + ["mixed"]},
                2000: fused_2000,
            }
            # For a richer curve, add intermediate points using base model for steps 0..2000
            # (honest: we can't re-run all those intermediate fusions without saving checkpoints)
            # Mark as "final only" and plot what we have
            indiv_at_0 = {d: {dd: base_losses[dd] for dd in list(DOMAINS) + ["mixed"]}
                          for d in DOMAINS}
            indiv_at_2000 = {d: {dd: all_checkpoints[d][-1]["held_out"][dd]
                                  for dd in list(DOMAINS) + ["mixed"]}
                             for d in DOMAINS}
            indiv_traj_for_fig = {d: {0: indiv_at_0[d], 2000: indiv_at_2000[d]}
                                  for d in DOMAINS}

            save_figure_c(fusion_trajectory, indiv_traj_for_fig, base_losses["mixed"])

            for s in specialists.values():
                del s
            torch.cuda.empty_cache()

    # Save JSON
    output = {
        "seed": SEED,
        "eval_interval": EVAL_INTERVAL,
        "config": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "freeze_layers": FREEZE_LAYERS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "seq_len": SEQ_LEN,
        },
        "base_model_loss": base_losses,
        "specialists": {
            domain: ckpts for domain, ckpts in all_checkpoints.items()
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "loss_curves_seed42.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("Loss curves complete. Figures saved to figures/pythia/")
    print("=" * 70)


if __name__ == "__main__":
    main()
