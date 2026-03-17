"""Smoke tests to verify kalavai package installs and CLI runs."""

import subprocess
import sys

from kalavai import __version__


def test_version_string():
    assert __version__ == "0.1.0"


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "kalavai.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "KALAVAI" in result.stdout


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "kalavai.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_subcommands_exist():
    for cmd in ["coop", "train", "check", "fuse"]:
        result = subprocess.run(
            [sys.executable, "-m", "kalavai.cli", cmd, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"'{cmd}' subcommand failed"
