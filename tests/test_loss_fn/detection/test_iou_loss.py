"""Tests for IoU / GIoU / DIoU / CIoU loss."""

from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn


class TestIoULoss:
    """Test IoULoss in all four modes."""

    @pytest.fixture(params=["iou", "giou", "diou", "ciou"])
    def fn(self, request):
        return build_loss_fn(
            "iou_loss", category="detection", mode=request.param
        )

    def test_perfect_overlap(self, fn):
        """Perfectly overlapping boxes → loss near 0."""
        p = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        t = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        loss = fn(p, t)
        assert loss.item() < 0.1, f"Expected near-zero loss, got {loss.item()}"

    def test_no_overlap(self, fn):
        """Non-overlapping boxes → loss > 0.5."""
        p = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        t = torch.tensor([[10.0, 10.0, 11.0, 11.0]])
        loss = fn(p, t)
        assert loss.item() > 0.5, f"Expected large loss, got {loss.item()}"

    def test_batched(self, fn):
        """Batched inputs produce scalar output."""
        loss = fn(torch.rand(4, 4), torch.rand(4, 4))
        assert loss.shape == ()

    def test_cxcywh_format(self):
        """cxcywh box format works correctly."""
        fn = build_loss_fn(
            "iou_loss", category="detection", mode="iou", box_format="cxcywh"
        )
        # Center (0.5,0.5), w=1, h=1  →  xyxy: (0,0,1,1)
        p = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        t = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        loss = fn(p, t)
        assert loss.item() < 0.1

    def test_different_modes_different_values(self):
        """Different modes give different loss values for the same boxes."""
        # Use boxes with different aspect ratios so CIoU != DIoU
        p = torch.tensor([[0.0, 0.0, 3.0, 1.0]])   # w=3, h=1
        t = torch.tensor([[1.0, 0.0, 4.0, 2.0]])   # w=3, h=2
        iou_fn = build_loss_fn("iou_loss", category="detection", mode="iou")
        giou_fn = build_loss_fn("iou_loss", category="detection", mode="giou")
        diou_fn = build_loss_fn("iou_loss", category="detection", mode="diou")
        ciou_fn = build_loss_fn("iou_loss", category="detection", mode="ciou")
        losses = [fn(p, t).item() for fn in (iou_fn, giou_fn, diou_fn, ciou_fn)]
        # All should be distinct for this partial overlap case
        assert len(set(round(l, 4) for l in losses)) == 4, (
            f"Expected distinct losses, got {losses}"
        )

    def test_invalid_mode(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError):
            build_loss_fn(
                "iou_loss", category="detection", mode="invalid"
            )

    def test_gradient_flow(self):
        """Loss is differentiable w.r.t. prediction."""
        fn = build_loss_fn("iou_loss", category="detection", mode="iou")
        p = torch.randn(4, 4, requires_grad=True)
        t = torch.rand(4, 4)
        loss = fn(p, t)
        loss.backward()
        assert p.grad is not None

    def test_reduction_none(self):
        """reduction='none' returns per-sample losses."""
        fn = build_loss_fn(
            "iou_loss", category="detection", mode="iou", reduction="none"
        )
        p = torch.rand(4, 4)
        t = torch.rand(4, 4)
        loss = fn(p, t)
        assert loss.shape == (4,)
