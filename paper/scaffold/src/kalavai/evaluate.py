"""
Evaluation module for KALAVAI experiments.

Two types of evaluation:
  1. Eval loss: cross-entropy on held-out domain data (fast, used during development)
  2. Benchmarks: lm-eval harness scores on standard tasks (slow, used for paper tables)
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================================
# Eval Loss (fast, for development iterations)
# ============================================================================

@torch.no_grad()
def eval_loss(model, eval_datasets, batch_size=16, device="cuda", max_batches=20):
    """
    Compute cross-entropy loss on multiple eval datasets.

    Args:
        model: the model to evaluate
        eval_datasets: dict of {name: Dataset}
        batch_size: eval batch size
        device: compute device
        max_batches: cap on number of batches per dataset (for speed)

    Returns:
        dict of {dataset_name: avg_loss}
    """
    model = model.to(device)
    model.eval()
    results = {}

    for name, dataset in eval_datasets.items():
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            # Handle models that return (logits, loss) or (logits, loss, extras)
            out = model(x, y)
            if isinstance(out, tuple):
                loss = out[1]
            else:
                loss = out.loss
            if loss is not None:
                losses.append(loss.item())

        results[name] = sum(losses) / len(losses) if losses else float("inf")

    return results


# ============================================================================
# Benchmark Evaluation (lm-eval harness, for paper tables)
# ============================================================================

def run_lm_eval(
    model_path: str,
    tasks: list,
    output_dir: str,
    num_fewshot: int = 0,
    batch_size: int = 8,
    device: str = "cuda",
    model_type: str = "hf",
    base_model: Optional[str] = None,
):
    """
    Run lm-eval harness on a model checkpoint.

    Args:
        model_path: path to model checkpoint or HF model ID
        tasks: list of benchmark names (e.g., ["gsm8k", "arc_challenge"])
        output_dir: where to save results
        num_fewshot: number of few-shot examples
        batch_size: eval batch size
        device: compute device
        model_type: "hf" for HuggingFace, "hf-peft" for LoRA checkpoints
        base_model: base model ID (required for PEFT models)

    Returns:
        dict of results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tasks_str = ",".join(tasks)

    cmd = [
        "lm_eval",
        "--model", model_type,
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", output_dir,
    ]

    # For LoRA models, specify base model
    if model_type == "hf-peft" and base_model:
        cmd[4] = f"pretrained={base_model},peft={model_path}"

    print(f"Running lm-eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"lm-eval failed:\n{result.stderr}")
        return None

    # Parse results from output directory
    results_files = list(Path(output_dir).glob("**/*.json"))
    if results_files:
        with open(results_files[0]) as f:
            return json.load(f)
    return None


def eval_all_variants(
    model_paths: dict,
    tasks: list,
    output_base: str,
    num_fewshot: int = 0,
    batch_size: int = 8,
):
    """
    Run benchmarks on all model variants and produce a comparison table.

    Args:
        model_paths: dict of {variant_name: model_path}
        tasks: list of benchmark names
        output_base: base directory for results
        num_fewshot: few-shot count
        batch_size: eval batch size

    Returns:
        dict of {variant_name: {task: score}}
    """
    all_results = {}
    for variant_name, model_path in model_paths.items():
        print(f"\nEvaluating: {variant_name}")
        output_dir = str(Path(output_base) / variant_name)
        results = run_lm_eval(
            model_path=model_path,
            tasks=tasks,
            output_dir=output_dir,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
        )
        if results:
            # Extract accuracy metrics
            task_scores = {}
            for task in tasks:
                if task in results.get("results", {}):
                    # lm-eval stores accuracy under different keys per task
                    task_result = results["results"][task]
                    score = (
                        task_result.get("acc,none") or
                        task_result.get("acc_norm,none") or
                        task_result.get("exact_match,none") or
                        0.0
                    )
                    task_scores[task] = score
            all_results[variant_name] = task_scores

    return all_results


def format_results_table(all_results: dict, tasks: list) -> str:
    """Format results as a markdown table for the paper."""
    header = "| Model | " + " | ".join(tasks) + " | Avg |"
    separator = "|---|" + "|".join(["---"] * len(tasks)) + "|---|"

    rows = [header, separator]
    for variant, scores in all_results.items():
        values = [scores.get(t, 0.0) for t in tasks]
        avg = sum(values) / len(values) if values else 0.0
        row = f"| {variant} | " + " | ".join(f"{v:.2%}" for v in values) + f" | {avg:.2%} |"
        rows.append(row)

    return "\n".join(rows)
