"""gsplat wrapper: SceneRepresentation + camera pose -> rendered frame.

Handles activation of raw Gaussian parameters (sigmoid/softplus/normalize) and
invokes gsplat's differentiable rasterizer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from gsplat import rasterization
    _HAS_GSPLAT = True
except ImportError:
    rasterization = None
    _HAS_GSPLAT = False

from .populations import SceneRepresentation


def _activate(raw: dict[str, Tensor]) -> dict[str, Tensor]:
    """Apply activations to raw Gaussian parameters."""
    return {
        "means": raw["means"],
        "scales": F.softplus(raw["scales"]),
        "quats": F.normalize(raw["quats"], dim=-1),
        "opacities": torch.sigmoid(raw["opacities"]),
        "colors": torch.sigmoid(raw["colors"]),
    }


def render(scene_or_gaussians: SceneRepresentation | dict[str, Tensor],
           t: Tensor | float | None,
           viewmat: Tensor,        # (4, 4) world -> camera
           K: Tensor,              # (3, 3) intrinsics
           width: int,
           height: int,
           near: float = 0.1,
           far: float = 1000.0,
           background: Tensor | None = None) -> Tensor:
    """Render a single frame from the given camera.

    First arg may be a SceneRepresentation (in which case `t` is queried) OR
    a raw Gaussian dict (means/scales/quats/opacities/colors) which is used
    as-is. The raw-dict path is what inflate.py uses at decode.

    Returns:
        image: (H, W, 3) in [0, 1].
    """
    if not _HAS_GSPLAT:
        raise RuntimeError("gsplat is not installed. `pip install gsplat` to render.")

    if isinstance(scene_or_gaussians, dict):
        raw = scene_or_gaussians
    else:
        raw = scene_or_gaussians.gaussians(t if isinstance(t, Tensor) else torch.tensor(t or 0.0))
    g = _activate(raw)

    # gsplat expects colors as either RGB (N,3) or SH coefficients (N,K,3).
    # We're using RGB (SH degree 0).
    colors = g["colors"]  # (N, 3)

    # gsplat rasterization returns (renders, alphas, meta).
    # viewmats: (C, 4, 4) — we render one camera at a time.
    renders, _alphas, _meta = rasterization(
        means=g["means"],
        quats=g["quats"],
        scales=g["scales"],
        opacities=g["opacities"],
        colors=colors,
        viewmats=viewmat[None, ...],
        Ks=K[None, ...],
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        render_mode="RGB",
    )
    image = renders[0]  # (H, W, 3)

    if background is not None:
        # gsplat renders on black by default; blend with background if given.
        alpha = _alphas[0]  # (H, W, 1)
        image = image + (1.0 - alpha) * background

    return image
