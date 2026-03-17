"""Hardware auto-detection for Kalavai training.

Detects CUDA GPU availability and reports device name, VRAM, and CUDA version
as a structured dataclass. Used by `kalavai train start` to auto-configure
training parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kalavai.core.exceptions import HardwareError


@dataclass
class HardwareInfo:
    """Detected hardware information.

    Attributes:
        device: Torch device string (e.g. "cuda").
        name: GPU device name (e.g. "NVIDIA GeForce RTX 5090").
        vram_gb: Total GPU VRAM in gigabytes, rounded down to nearest integer.
        cuda_version: CUDA runtime version string (e.g. "12.4").
    """

    device: str
    name: str
    vram_gb: int
    cuda_version: str


def detect_hardware() -> HardwareInfo:
    """Detect CUDA GPU and return hardware info.

    Returns:
        A HardwareInfo dataclass with device details.

    Raises:
        HardwareError: If no CUDA-capable GPU is available.
    """
    if not torch.cuda.is_available():
        raise HardwareError(
            "No CUDA GPU detected. Kalavai requires a CUDA-capable GPU for training. "
            "Please ensure you have an NVIDIA GPU with the correct drivers installed."
        )

    device_index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_index)

    total_bytes = torch.cuda.get_device_properties(device_index).total_mem
    vram_gb = math.floor(total_bytes / (1024**3))

    cuda_version = torch.version.cuda or "unknown"

    return HardwareInfo(
        device="cuda",
        name=name,
        vram_gb=vram_gb,
        cuda_version=cuda_version,
    )


def print_hardware_summary(info: HardwareInfo) -> None:
    """Print a rich-formatted hardware summary to the console.

    Args:
        info: The detected hardware information to display.
    """
    console = Console()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Device", info.device)
    table.add_row("GPU", info.name)
    table.add_row("VRAM", f"{info.vram_gb} GB")
    table.add_row("CUDA", info.cuda_version)

    panel = Panel(table, title="[bold green]Hardware Detected[/bold green]", expand=False)
    console.print(panel)
