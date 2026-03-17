"""End-to-end cooperative creation for Kalavai.

Orchestrates all setup steps: config generation, tokenizer training,
seed checkpoint, calibration batch, CKA reference, and domain manifest.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from rich.console import Console

from kalavai.core.config import (
    AlignmentConfig,
    ArchitectureConfig,
    CooperativeConfig,
    FusionConfig,
)
from kalavai.coop.calibration import generate_calibration_batch
from kalavai.coop.manifest import generate_manifest
from kalavai.coop.reference import compute_cka_reference
from kalavai.coop.seed import generate_seed_checkpoint
from kalavai.coop.tokenizer import save_tokenizer, train_tokenizer

console = Console()

# ---------------------------------------------------------------------------
# Architecture presets keyed by target_params string
# ---------------------------------------------------------------------------

_ARCH_PRESETS: dict[str, ArchitectureConfig] = {
    "14M": ArchitectureConfig(depth=4, d_model=128, n_heads=4, ffn_ratio=2.75),
    "125M": ArchitectureConfig(depth=12, d_model=768, n_heads=12, ffn_ratio=2.75),
    "350M": ArchitectureConfig(depth=24, d_model=1024, n_heads=16, ffn_ratio=2.75),
    "1B": ArchitectureConfig(depth=24, d_model=2048, n_heads=16, ffn_ratio=2.75),
    "7B": ArchitectureConfig(depth=32, d_model=4096, n_heads=32, ffn_ratio=2.75),
}


def _default_arch(target_params: str) -> ArchitectureConfig:
    """Return a sensible architecture config for the given target size."""
    key = target_params.upper()
    if key in _ARCH_PRESETS:
        return _ARCH_PRESETS[key]
    # Fallback to smallest preset for unknown sizes
    return _ARCH_PRESETS["14M"]


def _generate_synthetic_corpus(output_path: Path) -> Path:
    """Write a small synthetic corpus for testing and return its path."""
    corpus_path = output_path / "_synthetic_corpus.txt"
    # Generate enough text so the tokenizer can learn merges and the
    # calibration batch has enough tokens (need at least 128 tokens).
    text = textwrap.dedent("""\
        The quick brown fox jumps over the lazy dog.
        A stitch in time saves nine.
        To be or not to be, that is the question.
        All that glitters is not gold.
        Knowledge is power, and power corrupts.
        The pen is mightier than the sword.
        Actions speak louder than words.
        Fortune favours the bold.
        Where there is a will, there is a way.
        Practice makes perfect, but nobody is perfect.
    """)
    # Repeat to ensure enough tokens for calibration
    corpus_path.write_text(text * 50, encoding="utf-8")
    return corpus_path


def create_cooperative(
    name: str,
    modules: int,
    target_params: str = "1B",
    output_dir: Path | None = None,
    corpus_path: Path | None = None,
    vocab_size: int = 4096,
    seed: int = 42,
) -> Path:
    """Create a complete cooperative directory with all required artifacts.

    Steps:
        1. Save ``kalavai.yaml`` configuration
        2. Train and save tokenizer
        3. Generate seed checkpoint
        4. Generate calibration batch
        5. Compute CKA reference representations
        6. Generate domain manifest

    Args:
        name: Cooperative name (also used as directory name).
        modules: Number of module slots in the cooperative.
        target_params: Target parameter count per module (e.g. "1B", "125M").
        output_dir: Directory to create. Defaults to ``./name``.
        corpus_path: Path to a UTF-8 text corpus for tokenizer training.
            If ``None``, a synthetic corpus is generated.
        vocab_size: Tokenizer vocabulary size.
        seed: Random seed for reproducible initialization.

    Returns:
        Path to the created cooperative directory.
    """
    if output_dir is None:
        output_dir = Path.cwd() / name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arch = _default_arch(target_params)
    probe_layers = _compute_probe_layers(arch.depth)

    config = CooperativeConfig(
        name=name,
        modules=modules,
        target_params_per_module=target_params,
        architecture=arch,
        alignment=AlignmentConfig(
            lambda_max=0.05,
            lambda_min=0.01,
            anneal_start=0.7,
            probe_layers=probe_layers,
            calibration_interval=500,
            thresholds={f"layer_{i}": round(0.7 - 0.05 * idx, 2) for idx, i in enumerate(probe_layers)},
        ),
        fusion=FusionConfig(backend="moe_routing", n_clusters=min(4, modules)),
    )

    # If no corpus provided, generate a synthetic one
    synthetic = False
    if corpus_path is None:
        synthetic = True
        corpus_path = _generate_synthetic_corpus(output_dir)

    # ── Step 1: Save config ────────────────────────────────────────────
    config_path = output_dir / "kalavai.yaml"
    console.print("[bold cyan][1/6][/] Saving kalavai.yaml...")
    config.to_yaml(config_path)

    # ── Step 2: Train & save tokenizer ────────────────────────────────
    tokenizer_path = output_dir / "tokenizer.model"
    console.print("[bold cyan][2/6][/] Training tokenizer...")
    tok = train_tokenizer(corpus_path, vocab_size=vocab_size)
    save_tokenizer(tok, tokenizer_path)

    # ── Step 3: Generate seed checkpoint ──────────────────────────────
    seed_path = output_dir / "seed_checkpoint.pt"
    console.print("[bold cyan][3/6][/] Generating seed checkpoint...")
    generate_seed_checkpoint(
        arch_config=arch,
        output_path=seed_path,
        vocab_size=tok.vocab_size,
        seed=seed,
    )

    # ── Step 4: Generate calibration batch ────────────────────────────
    calib_path = output_dir / "calibration_batch.pt"
    console.print("[bold cyan][4/6][/] Generating calibration batch...")
    generate_calibration_batch(
        tokenizer_path=tokenizer_path,
        corpus_path=corpus_path,
        output_path=calib_path,
        n_sequences=64,
        seq_length=128,
    )

    # ── Step 5: Compute CKA reference ─────────────────────────────────
    cka_path = output_dir / "cka_reference.pt"
    console.print("[bold cyan][5/6][/] Computing CKA reference...")
    compute_cka_reference(
        seed_checkpoint_path=seed_path,
        calibration_batch_path=calib_path,
        arch_config=arch,
        probe_layers=probe_layers,
        output_path=cka_path,
        vocab_size=tok.vocab_size,
    )

    # ── Step 6: Generate domain manifest ──────────────────────────────
    manifest_path = output_dir / "domain_manifest.json"
    console.print("[bold cyan][6/6][/] Generating domain manifest...")
    generate_manifest(config, manifest_path)

    # Clean up synthetic corpus if we created one
    if synthetic:
        corpus_path.unlink(missing_ok=True)

    console.print(f"\n[bold green]Cooperative '{name}' created at {output_dir}[/]")
    return output_dir


def _compute_probe_layers(depth: int) -> list[int]:
    """Pick 3 evenly-spaced probe layers for CKA alignment."""
    if depth <= 3:
        return list(range(depth))
    quarter = depth // 4
    return [quarter, 2 * quarter, 3 * quarter]
