"""CKA reference representation computation for Kalavu cooperatives.

Runs the canonical seed model on a calibration batch and extracts hidden
states at the configured probe layers. The resulting tensors serve as
alignment targets for all cooperative modules during training.
"""

from __future__ import annotations

from pathlib import Path

import torch

from kalavu.core.config import ArchitectureConfig
from kalavu.core.model import create_model_from_config


def compute_cka_reference(
    seed_checkpoint_path: Path,
    calibration_batch_path: Path,
    arch_config: ArchitectureConfig,
    probe_layers: list[int],
    output_path: Path,
    vocab_size: int = 4096,
) -> None:
    """Compute and save CKA reference representations from the seed model.

    Loads the seed checkpoint, runs it on the calibration batch in eval mode
    with no gradient tracking, and extracts hidden states at the specified
    probe layers. The result is saved as a ``{layer_idx: tensor}`` dict.

    Args:
        seed_checkpoint_path: Path to the seed checkpoint (``.pt`` state_dict).
        calibration_batch_path: Path to the calibration batch (``.pt`` tensor
            of shape ``[N, seq_len]``).
        arch_config: Architecture configuration matching the seed checkpoint.
        probe_layers: Layer indices (0-based) at which to extract hidden states.
        output_path: Destination path for the saved reference file.
        vocab_size: Vocabulary size for the model (must match the seed checkpoint).
    """
    # Load seed model
    state_dict = torch.load(
        seed_checkpoint_path, map_location="cpu", weights_only=True
    )
    model = create_model_from_config(arch_config, vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model.eval()

    # Load calibration batch
    calibration_batch = torch.load(
        calibration_batch_path, map_location="cpu", weights_only=True
    )

    # Extract probe representations
    with torch.no_grad():
        representations = model.get_probe_representations(
            calibration_batch, probe_layers
        )

    # Save reference
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(representations, output_path)


def load_cka_reference(path: Path) -> dict[int, torch.Tensor]:
    """Load a previously saved CKA reference file.

    Args:
        path: Path to the ``cka_reference.pt`` file.

    Returns:
        Dictionary mapping layer index to reference hidden-state tensor.
    """
    return torch.load(path, map_location="cpu", weights_only=True)
