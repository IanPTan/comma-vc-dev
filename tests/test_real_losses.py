"""Sanity checks on the real challenge-network losses.

Runs slow (~seconds) because it loads segnet+posenet weights. Skipped
automatically if the safetensors aren't present.
"""
from __future__ import annotations

from pathlib import Path

import torch

from gsplat_video.constants import CAMERA_H, CAMERA_W


_HERE = Path(__file__).resolve().parents[1]
_SEG_W = _HERE / "challenge_deps/models/segnet.safetensors"
_POS_W = _HERE / "challenge_deps/models/posenet.safetensors"

_WEIGHTS_AVAILABLE = _SEG_W.exists() and _POS_W.exists()


def _dummy_frame_pair(seed: int = 0) -> torch.Tensor:
    """(B=1, seq_len=2, H, W, 3) in [0, 255] float, shaped like real video."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 2, CAMERA_H, CAMERA_W, 3, generator=g) * 255).float()


def test_segnet_loss_forward_and_eval():
    if not _WEIGHTS_AVAILABLE:
        print("[skip] segnet weights not present")
        return
    from gsplat_video.losses import SegnetLoss

    device = "cpu"
    seg = SegnetLoss(weights_path=_SEG_W, device=device)
    rendered = _dummy_frame_pair(0).to(device)
    target = _dummy_frame_pair(1).to(device)
    rendered.requires_grad_(True)

    ce = seg(rendered, target)
    assert ce.dim() == 0 and torch.isfinite(ce), f"segnet CE not scalar/finite: {ce}"
    ce.backward()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all(), \
        "segnet CE gradient not flowing back into rendered frames"

    d = seg.eval_distortion(rendered.detach(), target)
    assert torch.all((d >= 0) & (d <= 1))
    print(f"[ok] segnet: CE={ce.item():.4f}  eval_distortion={d.mean().item():.4f}")


def test_posenet_loss_forward_and_eval():
    if not _WEIGHTS_AVAILABLE:
        print("[skip] posenet weights not present")
        return
    from gsplat_video.losses import PosenetLoss

    device = "cpu"
    pose = PosenetLoss(weights_path=_POS_W, device=device)
    rendered = _dummy_frame_pair(0).to(device)
    target = _dummy_frame_pair(1).to(device)
    rendered.requires_grad_(True)

    mse = pose(rendered, target)
    assert mse.dim() == 0 and torch.isfinite(mse), f"posenet MSE not scalar/finite: {mse}"
    mse.backward()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()

    d = pose.eval_distortion(rendered.detach(), target)
    assert torch.all(d >= 0)
    print(f"[ok] posenet: MSE={mse.item():.4f}  eval_distortion={d.mean().item():.6f}")


def test_score_shaped_with_real_losses():
    if not _WEIGHTS_AVAILABLE:
        print("[skip] weights not present")
        return
    from gsplat_video.losses import PosenetLoss, SegnetLoss, score_shaped_loss

    device = "cpu"
    seg_fn = SegnetLoss(weights_path=_SEG_W, device=device)
    pose_fn = PosenetLoss(weights_path=_POS_W, device=device)

    rendered = _dummy_frame_pair(0).to(device)
    rendered.requires_grad_(True)
    target = _dummy_frame_pair(1).to(device)

    seg = seg_fn(rendered, target)
    pose = pose_fn(rendered, target)
    rate = torch.tensor(0.005, device=device)

    loss = score_shaped_loss(seg, pose, rate, w_photo=0.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(rendered.grad).all()
    print(f"[ok] end-to-end differentiable score-shaped loss = {loss.item():.4f}")


def main() -> None:
    tests = [
        test_segnet_loss_forward_and_eval,
        test_posenet_loss_forward_and_eval,
        test_score_shaped_with_real_losses,
    ]
    for fn in tests:
        fn()
    if _WEIGHTS_AVAILABLE:
        print(f"\n{len(tests)}/{len(tests)} real-loss tests passed.")
    else:
        print(f"\n{len(tests)} tests skipped (weights missing).")


if __name__ == "__main__":
    main()
