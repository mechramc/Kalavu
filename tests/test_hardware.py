"""Tests for kalavai.train.hardware — GPU auto-detection (all mocked, no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kalavai.train.hardware import HardwareError, HardwareInfo, detect_hardware, print_hardware_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeDeviceProperties:
    """Mimics torch.cuda.get_device_properties return value."""

    def __init__(self, total_mem: int) -> None:
        self.total_mem = total_mem


# ---------------------------------------------------------------------------
# detect_hardware — CUDA available
# ---------------------------------------------------------------------------


class TestDetectHardwareCudaAvailable:
    """When CUDA is available, detect_hardware returns correct HardwareInfo."""

    @patch("kalavai.train.hardware.torch")
    def test_returns_hardware_info(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 5090"
        mock_torch.cuda.get_device_properties.return_value = _FakeDeviceProperties(
            total_mem=32 * (1024**3),  # 32 GB
        )
        mock_torch.version.cuda = "12.4"

        info = detect_hardware()

        assert isinstance(info, HardwareInfo)
        assert info.device == "cuda"
        assert info.name == "NVIDIA GeForce RTX 5090"
        assert info.vram_gb == 32
        assert info.cuda_version == "12.4"

    @patch("kalavai.train.hardware.torch")
    def test_vram_rounds_down(self, mock_torch: MagicMock) -> None:
        """VRAM should be floored, not rounded — 23.7 GB → 23."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        # 23.7 GB in bytes
        mock_torch.cuda.get_device_properties.return_value = _FakeDeviceProperties(
            total_mem=int(23.7 * (1024**3)),
        )
        mock_torch.version.cuda = "12.1"

        info = detect_hardware()
        assert info.vram_gb == 23

    @patch("kalavai.train.hardware.torch")
    def test_calls_correct_device_index(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 2
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value = _FakeDeviceProperties(
            total_mem=8 * (1024**3),
        )
        mock_torch.version.cuda = "11.8"

        detect_hardware()

        mock_torch.cuda.get_device_name.assert_called_once_with(2)
        mock_torch.cuda.get_device_properties.assert_called_once_with(2)

    @patch("kalavai.train.hardware.torch")
    def test_cuda_version_none_becomes_unknown(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value = _FakeDeviceProperties(
            total_mem=16 * (1024**3),
        )
        mock_torch.version.cuda = None

        info = detect_hardware()
        assert info.cuda_version == "unknown"


# ---------------------------------------------------------------------------
# detect_hardware — no CUDA
# ---------------------------------------------------------------------------


class TestDetectHardwareNoCuda:
    """When CUDA is not available, detect_hardware raises HardwareError."""

    @patch("kalavai.train.hardware.torch")
    def test_raises_hardware_error(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = False

        with pytest.raises(HardwareError, match="No CUDA GPU detected"):
            detect_hardware()

    @patch("kalavai.train.hardware.torch")
    def test_error_is_kalavai_error_subclass(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = False

        from kalavai.core.exceptions import KalavaiError

        with pytest.raises(KalavaiError):
            detect_hardware()


# ---------------------------------------------------------------------------
# print_hardware_summary
# ---------------------------------------------------------------------------


class TestPrintHardwareSummary:
    """print_hardware_summary renders without errors."""

    def test_prints_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        info = HardwareInfo(
            device="cuda",
            name="NVIDIA GeForce RTX 5090",
            vram_gb=32,
            cuda_version="12.4",
        )
        # Should not raise
        print_hardware_summary(info)

    def test_output_contains_device_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        info = HardwareInfo(
            device="cuda",
            name="NVIDIA RTX 4090",
            vram_gb=24,
            cuda_version="12.1",
        )
        print_hardware_summary(info)
        captured = capsys.readouterr().out
        assert "NVIDIA RTX 4090" in captured
        assert "24 GB" in captured
        assert "12.1" in captured
