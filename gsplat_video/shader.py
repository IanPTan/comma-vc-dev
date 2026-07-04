"""View-dependent color shader — optional refinement over raw Gaussian RGB.

For each Gaussian and each view direction, the shader can output a small color
delta. Kept intentionally tiny: input = (base_rgb, view_dir), output = delta_rgb.
~3 KB post-quantization.

Initially unused; hook into rasterizer once we're past the pixel-fidelity
warmup phase and want per-Gaussian view effects (sun glare, shiny metal, etc.).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ViewDependentShader(nn.Module):
    def __init__(self, hidden: int = 24):
        super().__init__()
        # Input: base_rgb (3) + view direction (3) = 6.
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, base_rgb: Tensor, view_dir: Tensor) -> Tensor:
        """Return corrected RGB in [0, 1].

        base_rgb: (N, 3) in [0, 1]
        view_dir: (N, 3) unit vectors, camera-to-Gaussian direction
        """
        x = torch.cat([base_rgb, view_dir], dim=-1)
        delta = self.net(x)
        return torch.sigmoid(torch.logit(base_rgb.clamp(1e-4, 1 - 1e-4)) + delta)
