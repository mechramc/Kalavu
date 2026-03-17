"""
Experiment configuration.
Every experiment is fully specified by a YAML config file.
"""

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ArchitectureConfig:
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    context_length: int = 256
    vocab_size: int = 512
    dropout: float = 0.0


@dataclass
class AlignmentConfig:
    freeze_layers: int = 2
    seed: int = 42


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    eval_interval: int = 200
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    precision: str = "bf16"  # bf16, fp16, fp32


@dataclass
class FusionConfig:
    method: str = "moe"  # moe, averaging, btx
    router_steps: int = 500
    router_lr: float = 1e-3
    # BTX-specific: average attention, MoE-route FFN only
    btx_average_attention: bool = False


@dataclass
class DataConfig:
    mode: str = "synthetic"  # synthetic, huggingface
    domains: list = field(default_factory=lambda: ["code", "stories"])
    # HuggingFace-specific
    model_id: Optional[str] = None  # e.g. "Qwen/Qwen2.5-1.5B"
    dataset_ids: Optional[dict] = None  # domain -> HF dataset path
    max_train_samples: Optional[int] = None
    max_eval_samples: int = 500
    # Synthetic-specific
    data_tokens: int = 2_000_000


@dataclass
class EvalConfig:
    benchmarks: list = field(default_factory=lambda: [])  # e.g. ["gsm8k", "arc_challenge"]
    eval_batch_size: int = 8
    num_fewshot: int = 0


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    seeds: list = field(default_factory=lambda: [42, 137, 2026])
    output_dir: str = "results"
    wandb_project: str = "kalavai-paper"
    device: str = "auto"  # auto, cuda, mps, cpu

    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load experiment config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()
        config.name = raw.get("name", config.name)
        config.description = raw.get("description", config.description)
        config.seeds = raw.get("seeds", config.seeds)
        config.output_dir = raw.get("output_dir", config.output_dir)
        config.wandb_project = raw.get("wandb_project", config.wandb_project)
        config.device = raw.get("device", config.device)

        # Nested configs
        if "architecture" in raw:
            config.architecture = ArchitectureConfig(**raw["architecture"])
        if "alignment" in raw:
            config.alignment = AlignmentConfig(**raw["alignment"])
        if "training" in raw:
            config.training = TrainingConfig(**raw["training"])
        if "fusion" in raw:
            config.fusion = FusionConfig(**raw["fusion"])
        if "data" in raw:
            config.data = DataConfig(**raw["data"])
        if "eval" in raw:
            config.eval = EvalConfig(**raw["eval"])

        return config

    def to_yaml(self, path: str):
        """Save config to YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def resolve_device(self):
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
