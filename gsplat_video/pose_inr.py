"""Ego-pose implicit representation: t -> SE(3) camera pose.

Small fourier-feature MLP overfit to the ego trajectory. Outputs 12 numbers:
3 for translation + 6-dim continuous rotation representation (Zhou et al. 2019)
which we then Gram-Schmidt to a valid rotation matrix. 6D repr is smoother to
optimize than quaternions.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_freqs: int, sigma: float = 4.0):
        super().__init__()
        # Log-linearly spaced frequencies from 1 to 2**(n_freqs-1).
        freqs = 2.0 ** torch.arange(n_freqs) * sigma
        # (in_dim, n_freqs) -> flattened to (in_dim * n_freqs * 2).
        self.register_buffer("freqs", freqs)
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.out_dim = in_dim * n_freqs * 2

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_dim)
        proj = x[..., None] * self.freqs[None, :]  # (..., in_dim, n_freqs)
        return torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(-2)


def _six_d_to_rotmat(six: Tensor) -> Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt on first two columns."""
    a1 = six[..., 0:3]
    a2 = six[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


class EgoPoseINR(nn.Module):
    """Tiny MLP that maps normalized time t in [0, 1] to SE(3) pose.

    Default: 32 hidden units, 2 layers, 8 fourier frequencies -> ~2 KB after
    INT8 quantization.
    """

    def __init__(self, hidden: int = 32, n_freqs: int = 8):
        super().__init__()
        self.ff = FourierFeatures(in_dim=1, n_freqs=n_freqs)
        self.net = nn.Sequential(
            nn.Linear(self.ff.out_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 9),  # 3 translation + 6d rotation
        )

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """t: scalar or (B,) tensor in [0, 1].

        Returns (R, tvec) where R is (..., 3, 3) and tvec is (..., 3).
        """
        if t.dim() == 0:
            t = t[None]
        feats = self.ff(t[..., None])          # (B, ff_dim)
        raw = self.net(feats)                  # (B, 9)
        tvec = raw[..., 0:3]
        R = _six_d_to_rotmat(raw[..., 3:9])
        return R, tvec

    def viewmat(self, t: Tensor) -> Tensor:
        """Return world->camera 4x4 matrix. gsplat expects this format."""
        R, tvec = self(t)
        if R.dim() == 2:
            R = R[None]
            tvec = tvec[None]
        B = R.shape[0]
        M = torch.zeros(B, 4, 4, device=R.device, dtype=R.dtype)
        M[:, 0:3, 0:3] = R
        M[:, 0:3, 3] = tvec
        M[:, 3, 3] = 1.0
        return M[0] if B == 1 else M
