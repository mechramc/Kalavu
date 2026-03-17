"""Tests for CKA computation."""

from __future__ import annotations

import torch
import pytest

from kalavai.core.cka import linear_cka, cka_loss


class TestLinearCKA:
    """Tests for ``linear_cka``."""

    def test_identical_representations(self) -> None:
        X = torch.randn(50, 16)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_representations(self) -> None:
        # Build centered orthogonal matrices: center first, then QR.
        A = torch.randn(128, 32)
        A = A - A.mean(dim=0, keepdim=True)  # center before QR
        Q, _ = torch.linalg.qr(A)
        X = Q[:, :16]
        Y = Q[:, 16:32]
        assert linear_cka(X, Y) == pytest.approx(0.0, abs=1e-5)

    def test_scale_invariance(self) -> None:
        X = torch.randn(50, 16)
        assert linear_cka(X, X * 42.0) == pytest.approx(1.0, abs=1e-5)

    def test_different_feature_dims(self) -> None:
        X = torch.randn(50, 8)
        Y = torch.randn(50, 32)
        score = linear_cka(X, Y)
        assert 0.0 <= score <= 1.0

    def test_mismatched_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="Sample counts"):
            linear_cka(torch.randn(10, 4), torch.randn(20, 4))


class TestCKALoss:
    """Tests for ``cka_loss``."""

    def test_identical_gives_zero_loss(self) -> None:
        X = torch.randn(50, 16)
        loss = cka_loss(X, X)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flows(self) -> None:
        h = torch.randn(50, 16, requires_grad=True)
        ref = torch.randn(50, 16)
        loss = cka_loss(h, ref)
        loss.backward()
        assert h.grad is not None
        assert h.grad.shape == h.shape

    def test_loss_range(self) -> None:
        loss = cka_loss(torch.randn(50, 16), torch.randn(50, 16))
        assert 0.0 <= loss.item() <= 1.0
