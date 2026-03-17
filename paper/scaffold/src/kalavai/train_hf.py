"""
Training loop for real HuggingFace models with LoRA fine-tuning.

Usage:
    python -m kalavai.train_hf --config configs/real/qwen_2mod_math_science.yaml --seed 42

This handles:
  - Loading a base model from HF Hub
  - Applying LoRA to unfrozen layers only (respecting freeze_layers)
  - Fine-tuning on domain-specific data
  - Saving checkpoints for later fusion
  - Logging to Weights & Biases
"""

import argparse
import json
import time
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from kalavai.config import ExperimentConfig


def get_target_modules_for_unfrozen_layers(model, freeze_layers):
    """
    Identify LoRA target modules only in layers ABOVE the freeze boundary.
    This ensures frozen layers remain exactly shared across all modules.
    """
    target_modules = []
    for name, _ in model.named_modules():
        # Match transformer layer indices (model-family specific patterns)
        # Qwen: model.layers.{idx}.self_attn.{q,k,v,o}_proj
        # Llama: model.layers.{idx}.self_attn.{q,k,v,o}_proj
        for pattern in [".self_attn.", ".mlp."]:
            if pattern in name:
                parts = name.split(".")
                # Find the layer index
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            if layer_idx >= freeze_layers:
                                # This module is in an unfrozen layer
                                # Extract the leaf module name for LoRA targeting
                                leaf = parts[-1]
                                if leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"):
                                    full_name = ".".join(parts)
                                    if full_name not in target_modules:
                                        target_modules.append(full_name)
                        except ValueError:
                            pass
    return target_modules


def prepare_domain_dataset(dataset_id, tokenizer, max_samples=None, max_length=512):
    """Load and tokenize a domain dataset from HuggingFace."""
    ds = load_dataset(dataset_id, split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def tokenize_fn(examples):
        # Handle different column names
        text_col = "text" if "text" in examples else list(examples.keys())[0]
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    return tokenized


def train_specialist(
    config: ExperimentConfig,
    domain_name: str,
    dataset_id: str,
    seed: int,
    output_base: str,
):
    """
    Fine-tune a single specialist module via LoRA on domain data.
    Only layers above freeze_layers get LoRA adapters.
    """
    device = config.resolve_device()
    model_id = config.data.model_id
    freeze_layers = config.alignment.freeze_layers
    run_name = f"{config.name}_{domain_name}_seed{seed}"
    output_dir = Path(output_base) / run_name

    print(f"\n{'='*60}")
    print(f"Training specialist: {domain_name}")
    print(f"  Base model: {model_id}")
    print(f"  Frozen layers: 0-{freeze_layers - 1}")
    print(f"  Dataset: {dataset_id}")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if config.training.precision == "bf16" else torch.float32,
        trust_remote_code=True,
    )

    # Explicitly freeze early layers (belt and suspenders with LoRA targeting)
    for i in range(freeze_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = False

    # Configure LoRA on unfrozen layers only
    target_modules = get_target_modules_for_unfrozen_layers(model, freeze_layers)
    if not target_modules:
        # Fallback: target standard module names and let layer freezing handle it
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        print(f"  Warning: using generic target modules (layer freezing still applies)")
    else:
        print(f"  LoRA targets: {len(target_modules)} modules in layers {freeze_layers}+")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load domain data
    dataset = prepare_domain_dataset(
        dataset_id, tokenizer,
        max_samples=config.data.max_train_samples,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=(config.training.precision == "bf16"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config.training.eval_interval,
        save_strategy="steps",
        save_steps=config.training.max_steps,  # save at end
        seed=seed,
        report_to="wandb",
        run_name=run_name,
        gradient_accumulation_steps=1,
        max_grad_norm=config.training.gradient_clip,
    )

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={
            "domain": domain_name,
            "dataset": dataset_id,
            "seed": seed,
            "freeze_layers": freeze_layers,
            "model_id": model_id,
        },
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save
    model.save_pretrained(output_dir / "checkpoint")
    tokenizer.save_pretrained(output_dir / "checkpoint")

    # Save metadata
    meta = {
        "domain": domain_name,
        "dataset": dataset_id,
        "seed": seed,
        "freeze_layers": freeze_layers,
        "model_id": model_id,
        "training_time_sec": elapsed,
        "max_steps": config.training.max_steps,
        "final_train_loss": trainer.state.log_history[-1].get("loss", None),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    wandb.finish()
    print(f"  Done. Saved to {output_dir}. Training took {elapsed:.1f}s.")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train a KALAVAI specialist module")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--domain", type=str, default=None, help="Train only this domain")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    output_base = Path(config.output_dir) / config.name

    domains = config.data.domains
    datasets = config.data.dataset_ids or {}

    if args.domain:
        # Train single domain
        if args.domain not in datasets:
            raise ValueError(f"Domain '{args.domain}' not in config. Available: {list(datasets.keys())}")
        train_specialist(config, args.domain, datasets[args.domain], args.seed, output_base)
    else:
        # Train all domains
        for domain_name in domains:
            dataset_id = datasets[domain_name]
            train_specialist(config, domain_name, dataset_id, args.seed, output_base)


if __name__ == "__main__":
    main()
