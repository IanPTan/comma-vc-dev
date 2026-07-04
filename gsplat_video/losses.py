"""Loss functions matching the challenge scoring formula.

    score = 100 * segnet_dist + 25 * rate + sqrt(10 * posenet_dist)

We provide two loss classes wrapping the *real* segnet + posenet from
challenge_deps/modules.py:

  - SegnetLoss.forward() : DIFFERENTIABLE proxy for CE (target argmax vs.
    rendered logits). Used during training.
  - SegnetLoss.eval_distortion() : the exact challenge metric (argmax
    disagreement fraction). Used for validation only.

  - PosenetLoss.forward() : MSE on first 6 pose dims (matches challenge).
  - PosenetLoss.eval_distortion() : same, no argmax involved -> no gap.

Real weights are loaded from challenge_deps/models/{segnet,posenet}.safetensors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# challenge_deps/ is where the challenge repo's modules.py + safetensors live.
_CHALLENGE_DEPS = Path(__file__).resolve().parent.parent / "challenge_deps"
if str(_CHALLENGE_DEPS) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE_DEPS))


def _load_challenge_modules():
    """Lazy import: challenge modules pull in heavy deps (timm, smp)."""
    import modules as _m
    return _m


def _differentiable_rgb_to_yuv6(rgb_chw: Tensor) -> Tensor:
    """Differentiable port of frame_utils.rgb_to_yuv6 (BT.601 limited range).

    Input:  (..., 3, H, W) with values in [0, 255].
    Output: (..., 6, H/2, W/2) matching the challenge preprocessor.
    """
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2 * H2, :2 * W2]

    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]

    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)

    U_sub = 0.25 * (U[..., 0::2, 0::2] + U[..., 1::2, 0::2]
                    + U[..., 0::2, 1::2] + U[..., 1::2, 1::2])
    V_sub = 0.25 * (V[..., 0::2, 0::2] + V[..., 1::2, 0::2]
                    + V[..., 0::2, 1::2] + V[..., 1::2, 1::2])

    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


# ---------------------------------------------------------------------------
# Photometric warmup loss (unchanged).
# ---------------------------------------------------------------------------
def photometric_loss(rendered: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(rendered, target)


# ---------------------------------------------------------------------------
# Real SegNet loss.
# ---------------------------------------------------------------------------
class SegnetLoss(nn.Module):
    """Wraps the challenge SegNet. Training uses CE (differentiable);
    eval uses the exact challenge metric (per-pixel argmax disagreement)."""

    def __init__(self, weights_path: Path | None = None, device: str = "cuda"):
        super().__init__()
        m = _load_challenge_modules()
        self.model = m.SegNet().eval()

        wp = Path(weights_path or m.segnet_sd_path)
        from safetensors.torch import load_file
        sd = load_file(str(wp), device=str(device))
        self.model.load_state_dict(sd)
        self.model.to(device)

        # Segnet operates on (B, seq_len, C, H, W) via last-frame slice.
        # We freeze all its weights: no gradient into segnet's parameters.
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _preprocess(self, rgb_bthwc_uint8_range: Tensor) -> Tensor:
        """Take (B, T, H, W, 3) in [0, 255] float, return (B, C, H_seg, W_seg)
        ready for segnet.forward()."""
        import einops
        x = einops.rearrange(rgb_bthwc_uint8_range,
                             'b t h w c -> b t c h w').float()
        return self.model.preprocess_input(x)

    def _logits(self, rgb_bthwc: Tensor) -> Tensor:
        x = self._preprocess(rgb_bthwc)
        return self.model(x)          # (B, num_classes, H_seg, W_seg)

    def forward(self, rendered_bthwc: Tensor, target_bthwc: Tensor) -> Tensor:
        """Differentiable proxy loss.

        rendered/target: (B, T, H, W, 3), values in [0, 255] float. Segnet
        uses only the LAST frame; consistent with challenge behavior.
        """
        with torch.no_grad():
            target_logits = self._logits(target_bthwc)
            target_labels = target_logits.argmax(dim=1)   # (B, H_seg, W_seg)
        pred_logits = self._logits(rendered_bthwc)        # requires_grad via rendered
        return F.cross_entropy(pred_logits, target_labels)

    @torch.inference_mode()
    def eval_distortion(self, rendered_bthwc: Tensor, target_bthwc: Tensor) -> Tensor:
        pred_logits = self._logits(rendered_bthwc)
        target_logits = self._logits(target_bthwc)
        diff = (pred_logits.argmax(dim=1) != target_logits.argmax(dim=1)).float()
        return diff.mean(dim=tuple(range(1, diff.ndim)))


# ---------------------------------------------------------------------------
# Real PoseNet loss.
# ---------------------------------------------------------------------------
class PosenetLoss(nn.Module):
    """Wraps the challenge PoseNet. Distortion = MSE on first N/2 pose dims
    (matches challenge scoring). Fully differentiable — no argmax anywhere."""

    def __init__(self, weights_path: Path | None = None, device: str = "cuda"):
        super().__init__()
        m = _load_challenge_modules()
        self.model = m.PoseNet().eval()

        wp = Path(weights_path or m.posenet_sd_path)
        from safetensors.torch import load_file
        sd = load_file(str(wp), device=str(device))
        self.model.load_state_dict(sd)
        self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self._pose_head_out = 12  # from modules.HEADS

    def _preprocess(self, rgb_bthwc: Tensor) -> Tensor:
        """(B, T, H, W, 3) [0,255] float -> segnet-format posenet input.

        Reimplements challenge's preprocess without `@torch.no_grad()` so
        gradients flow back to the rendered frames.
        """
        import einops
        from gsplat_video.constants import SEGNET_INPUT_HW
        seg_h, seg_w = SEGNET_INPUT_HW
        b, t = rgb_bthwc.shape[:2]
        x = einops.rearrange(rgb_bthwc, 'b t h w c -> (b t) c h w').float()
        x = F.interpolate(x, size=(seg_h, seg_w), mode="bilinear", align_corners=False)
        yuv6 = _differentiable_rgb_to_yuv6(x)                       # (B*T, 6, H2, W2)
        return einops.rearrange(yuv6, '(b t) c h w -> b (t c) h w', b=b, t=t)

    def _pose_out(self, rgb_bthwc: Tensor) -> Tensor:
        x = self._preprocess(rgb_bthwc)
        return self.model(x)["pose"]         # (B, 12)

    def forward(self, rendered_bthwc: Tensor, target_bthwc: Tensor) -> Tensor:
        with torch.no_grad():
            target = self._pose_out(target_bthwc)[..., : self._pose_head_out // 2]
        pred = self._pose_out(rendered_bthwc)[..., : self._pose_head_out // 2]
        return F.mse_loss(pred, target)

    @torch.inference_mode()
    def eval_distortion(self, rendered_bthwc: Tensor, target_bthwc: Tensor) -> Tensor:
        rendered_out = self._pose_out(rendered_bthwc)
        target_out = self._pose_out(target_bthwc)
        half = self._pose_head_out // 2
        return (rendered_out[..., :half] - target_out[..., :half]).pow(2).mean(
            dim=tuple(range(1, rendered_out.ndim - 1))
        )


# ---------------------------------------------------------------------------
# Score-shaped combined loss.
# ---------------------------------------------------------------------------
def score_shaped_loss(seg: Tensor, pose: Tensor, rate: Tensor | float,
                      w_photo: float = 0.0, photo: Tensor | None = None,
                      posenet_eps: float = 1e-6) -> Tensor:
    """Weighted combination mirroring the challenge score.

        loss = 100 * seg + 25 * rate + sqrt(10 * pose + eps) + w_photo * photo

    - `seg` and `pose` come from Segnet/PosenetLoss.forward() (differentiable).
    - `rate` is the differentiable rate estimate from codec.GaussianCodec.rate().
    - `photo` is optional pixel MSE for optimization stability early on.
    """
    if not isinstance(rate, torch.Tensor):
        rate = torch.tensor(float(rate), device=seg.device, dtype=seg.dtype)
    loss = 100.0 * seg + 25.0 * rate + torch.sqrt(10.0 * pose + posenet_eps)
    if photo is not None and w_photo > 0.0:
        loss = loss + w_photo * photo
    return loss


# ---------------------------------------------------------------------------
# Backward-compat stubs kept for the existing smoke tests.
# ---------------------------------------------------------------------------
class SegnetLossStub(nn.Module):
    """L1 proxy for tests that don't want to load the real weights."""
    def forward(self, rendered: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(rendered, target)


class PosenetLossStub(nn.Module):
    def forward(self, rendered_pair, target_pair) -> Tensor:
        rd = rendered_pair[1] - rendered_pair[0]
        td = target_pair[1] - target_pair[0]
        return F.mse_loss(rd, td)
