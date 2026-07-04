"""Loss functions matching the challenge scoring formula.

    score = 100 * segnet_dist + 25 * rate + sqrt(10 * posenet_dist)

Training loss mirrors this. Segnet / posenet losses are stubbed until we wire
in the real challenge networks; the stubs use MSE on target frames so the
skeleton is runnable end-to-end for shape / gradient sanity checks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def photometric_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """MSE for pixel warmup — used before segnet/posenet losses take over."""
    return F.mse_loss(rendered, target)


class SegnetLossStub(nn.Module):
    """Placeholder until we plug in the real challenge segnet.

    Signature matches what we'll swap in: takes rendered + target frames,
    returns a scalar disagreement. For now, uses per-pixel L1 as a proxy.
    """

    def forward(self, rendered: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(rendered, target)


class PosenetLossStub(nn.Module):
    """Placeholder for posenet distortion (MSE on posenet outputs between
    original and reconstructed consecutive frames)."""

    def forward(self, rendered_pair: tuple[Tensor, Tensor],
                target_pair: tuple[Tensor, Tensor]) -> Tensor:
        # Rough proxy: per-frame temporal difference agreement.
        rd = rendered_pair[1] - rendered_pair[0]
        td = target_pair[1] - target_pair[0]
        return F.mse_loss(rd, td)


def score_shaped_loss(seg: Tensor, pose: Tensor, rate: Tensor,
                      w_photo: float = 0.0, photo: Tensor | None = None,
                      posenet_eps: float = 1e-6) -> Tensor:
    """Weighted combination mirroring the challenge score.

    seg   : scalar segnet distortion estimate (or its differentiable proxy)
    pose  : scalar posenet distortion estimate
    rate  : scalar differentiable rate estimate (bits / uncompressed size)
    photo : optional pixel MSE for optimization stability, weighted by w_photo
    """
    loss = 100.0 * seg + 25.0 * rate + torch.sqrt(10.0 * pose + posenet_eps)
    if photo is not None and w_photo > 0.0:
        loss = loss + w_photo * photo
    return loss
