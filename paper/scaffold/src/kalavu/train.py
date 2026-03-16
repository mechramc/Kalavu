"""
Synthetic experiment training loop for KALAVU proof-of-concept.

Runs the full experiment for each seed in the config:
  1. Initialize canonical seed model (θ₀)
  2. Train one MiniGPT specialist per domain (frozen shared backbone)
  3. Fuse by weight averaging and by MoE routing
  4. Evaluate all models on all domains + mixed eval
  5. Save results JSON

Usage:
    python -m kalavu.train --config configs/synthetic/2mod_3M.yaml
    python -m kalavu.train --config configs/synthetic/2mod_3M.yaml --seed 42
"""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kalavu.config import ExperimentConfig
from kalavu.data import build_datasets, make_mixed_dataset, TextDataset
from kalavu.model import MiniGPT, ModelConfig, SimpleMoEFusion


# ============================================================================
# Config bridge: ExperimentConfig → ModelConfig
# ============================================================================

def _make_model_config(cfg: ExperimentConfig) -> ModelConfig:
    arch = cfg.architecture
    return ModelConfig(
        n_layers=arch.n_layers,
        d_model=arch.d_model,
        n_heads=arch.n_heads,
        d_ff=arch.d_ff,
        context_length=arch.context_length,
        vocab_size=arch.vocab_size,
        dropout=arch.dropout,
        freeze_layers=cfg.alignment.freeze_layers,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_steps=cfg.training.max_steps,
        eval_interval=cfg.training.eval_interval,
        warmup_steps=cfg.training.warmup_steps,
    )


# ============================================================================
# Training
# ============================================================================

def train_module(
    model: MiniGPT,
    train_dataset: TextDataset,
    eval_datasets: dict[str, TextDataset],
    config: ModelConfig,
    name: str,
    device: str,
) -> dict:
    """Train a single module and return training history."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    train_iter = iter(train_loader)

    history = {
        "train_loss": [],
        "eval_losses": {n: [] for n in eval_datasets},
        "steps": [],
    }

    total = model.count_params()
    trainable = model.count_params(trainable_only=True)
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"  Total:     {total:,}  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"{'='*60}")

    t0 = time.time()
    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Cosine LR with linear warmup
        if step < config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
            lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config.eval_interval == 0 or step == 1:
            model.eval()
            history["steps"].append(step)
            history["train_loss"].append(round(loss.item(), 6))

            with torch.no_grad():
                for eval_name, eval_ds in eval_datasets.items():
                    loader = DataLoader(eval_ds, batch_size=config.batch_size, num_workers=0)
                    eval_losses = []
                    for ex, ey in loader:
                        ex, ey = ex.to(device), ey.to(device)
                        _, el = model(ex, ey)
                        eval_losses.append(el.item())
                        if len(eval_losses) >= 10:
                            break
                    history["eval_losses"][eval_name].append(
                        round(sum(eval_losses) / len(eval_losses), 6)
                    )

            elapsed = time.time() - t0
            evals_str = " | ".join(
                f"{k}: {v[-1]:.4f}" for k, v in history["eval_losses"].items()
            )
            print(f"  [{name}] step {step:5d} | train: {loss.item():.4f} | {evals_str} | {elapsed:.0f}s")
            model.train()

    return history


@torch.no_grad()
def evaluate_model(
    model,
    eval_datasets: dict[str, TextDataset],
    config: ModelConfig,
    device: str,
    max_batches: int = 20,
) -> dict[str, float]:
    """Evaluate a model on all eval datasets."""
    model = model.to(device)
    model.eval()
    results = {}
    for eval_name, eval_ds in eval_datasets.items():
        loader = DataLoader(eval_ds, batch_size=config.batch_size, num_workers=0)
        losses = []
        for ex, ey in loader:
            ex, ey = ex.to(device), ey.to(device)
            _, el = model(ex, ey)
            losses.append(el.item())
            if len(losses) >= max_batches:
                break
        results[eval_name] = round(sum(losses) / len(losses), 6) if losses else float("inf")
    return results


# ============================================================================
# Fusion
# ============================================================================

def fuse_by_averaging(module_a: MiniGPT, module_b: MiniGPT, config: ModelConfig) -> MiniGPT:
    """Average unfrozen layer weights between two modules."""
    fused = copy.deepcopy(module_a)
    for i in range(config.freeze_layers, config.n_layers):
        for pa, pb in zip(
            fused.blocks[i].parameters(),
            module_b.blocks[i].parameters(),
        ):
            pa.data = (pa.data + pb.data) / 2.0
    return fused


def fuse_n_by_averaging(modules: list[MiniGPT], config: ModelConfig) -> MiniGPT:
    """Average unfrozen layer weights across N modules."""
    fused = copy.deepcopy(modules[0])
    n = len(modules)
    for layer_idx in range(config.freeze_layers, config.n_layers):
        for param_name, param in fused.blocks[layer_idx].named_parameters():
            param.data.zero_()
            for mod in modules:
                src = dict(mod.blocks[layer_idx].named_parameters())[param_name]
                param.data += src.data
            param.data /= n
    return fused


def fuse_by_moe(
    module_a: MiniGPT,
    module_b: MiniGPT,
    config: ModelConfig,
    train_dataset: TextDataset,
    device: str,
    router_steps: int = 500,
) -> SimpleMoEFusion:
    """Train a SimpleMoEFusion router on mixed-domain data."""
    moe = SimpleMoEFusion(config, module_a, module_b).to(device)

    for name, param in moe.named_parameters():
        if "router" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(moe.router.parameters(), lr=1e-3)
    loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    loader_iter = iter(loader)

    print(f"\n  Training MoE router for {router_steps} steps...")
    moe.train()
    for step in range(1, router_steps + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)
        _, loss = moe(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == router_steps:
            print(f"    Router step {step:4d}: loss={loss.item():.4f}")

    return moe


# ============================================================================
# Main
# ============================================================================

def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Run synthetic KALAVU experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override seed (default: run all seeds in config)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = cfg.resolve_device()
    mcfg = _make_model_config(cfg)

    seeds = [args.seed] if args.seed is not None else cfg.seeds
    output_base = Path(cfg.output_dir) / cfg.name
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"KALAVU Synthetic Experiment: {cfg.name}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Architecture: {mcfg.n_layers}L  d={mcfg.d_model}  heads={mcfg.n_heads}")
    print(f"Domains: {cfg.data.domains}")
    print(f"Freeze: {mcfg.freeze_layers} layers")
    print(f"Seeds: {seeds}")

    # Build datasets once (same synthetic data across seeds)
    train_sets, eval_sets = build_datasets(cfg)

    for seed in seeds:
        result_file = output_base / f"seed_{seed}.json"
        if result_file.exists():
            print(f"\nSkipping seed {seed} (already done: {result_file})")
            continue

        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        _set_seed(seed)

        # Canonical seed model
        seed_model = MiniGPT(mcfg)
        seed_checkpoint = copy.deepcopy(seed_model.state_dict())
        print(f"  Created canonical seed model ({seed_model.count_params():,} params)")

        # Build mixed eval
        mixed_eval = make_mixed_dataset(eval_sets, mcfg.context_length)
        all_eval = {**eval_sets, "mixed": mixed_eval}

        # Train one specialist per domain
        modules: dict[str, MiniGPT] = {}
        histories: dict[str, dict] = {}
        for domain in cfg.data.domains:
            model = MiniGPT(mcfg)
            model.load_state_dict(seed_checkpoint)
            model.freeze_early_layers(mcfg.freeze_layers)

            history = train_module(
                model, train_sets[domain], all_eval, mcfg, domain, device,
            )
            modules[domain] = model
            histories[domain] = history

        # Evaluate individuals
        print("\n  Evaluating individual specialists...")
        individual_results: dict[str, dict[str, float]] = {}
        for domain, model in modules.items():
            individual_results[domain] = evaluate_model(model, all_eval, mcfg, device)
            avg = sum(individual_results[domain].values()) / len(individual_results[domain])
            print(f"    {domain}: mixed={individual_results[domain].get('mixed', 0):.4f}  avg={avg:.4f}")

        # Fusion
        print("\n  Fusing modules...")
        domain_list = cfg.data.domains
        fused_results: dict[str, dict[str, float]] = {}

        if len(domain_list) == 2:
            mod_a = modules[domain_list[0]]
            mod_b = modules[domain_list[1]]

            fused_avg = fuse_by_averaging(mod_a, mod_b, mcfg)
            fused_results["averaged"] = evaluate_model(fused_avg, all_eval, mcfg, device)
            avg_avg = sum(fused_results["averaged"].values()) / len(fused_results["averaged"])
            print(f"    averaged: mixed={fused_results['averaged'].get('mixed', 0):.4f}  avg={avg_avg:.4f}")

            mixed_train = make_mixed_dataset(train_sets, mcfg.context_length)
            router_steps = cfg.fusion.router_steps if cfg.fusion else 500
            fused_moe = fuse_by_moe(mod_a, mod_b, mcfg, mixed_train, device, router_steps)
            fused_results["moe"] = evaluate_model(fused_moe, all_eval, mcfg, device)
            avg_moe = sum(fused_results["moe"].values()) / len(fused_results["moe"])
            print(f"    moe:      mixed={fused_results['moe'].get('mixed', 0):.4f}  avg={avg_moe:.4f}")

        else:
            # N > 2: only averaging
            fused_avg_n = fuse_n_by_averaging(list(modules.values()), mcfg)
            fused_results["averaged"] = evaluate_model(fused_avg_n, all_eval, mcfg, device)
            avg_n = sum(fused_results["averaged"].values()) / len(fused_results["averaged"])
            print(f"    averaged ({len(domain_list)}-way): mixed={fused_results['averaged'].get('mixed', 0):.4f}  avg={avg_n:.4f}")

        # Improvement
        best_individual_mixed = min(
            r.get("mixed", float("inf")) for r in individual_results.values()
        )
        best_fused_mixed = min(
            r.get("mixed", float("inf")) for r in fused_results.values()
        )
        improvement_pct = (
            (best_individual_mixed - best_fused_mixed) / best_individual_mixed * 100
            if best_individual_mixed > 0 else 0.0
        )

        print(f"\n  Best individual (mixed): {best_individual_mixed:.4f}")
        print(f"  Best fused (mixed):      {best_fused_mixed:.4f}")
        print(f"  Improvement:             {improvement_pct:+.2f}%")

        # Save
        output = {
            "config_name": cfg.name,
            "seed": seed,
            "domains": domain_list,
            "model_params": seed_model.count_params(),
            "freeze_layers": mcfg.freeze_layers,
            "individual_eval": individual_results,
            "fused_eval": fused_results,
            "best_individual_mixed": round(best_individual_mixed, 6),
            "best_fused_mixed": round(best_fused_mixed, 6),
            "improvement_pct": round(improvement_pct, 4),
            "fusion_works": improvement_pct > 0,
            "history_a": histories[domain_list[0]],
            "history_b": histories[domain_list[1]] if len(domain_list) > 1 else {},
        }

        with open(result_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved → {result_file}")

    print("\nAll seeds done.")


if __name__ == "__main__":
    main()
