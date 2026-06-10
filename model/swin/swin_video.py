"""
Video Swin Transformer encoder.

Adapted from "Swin MAE: Masked Autoencoders for Small Datasets"
(https://arxiv.org/abs/2212.13805), with the masked / MAE objective deliberately
left out. This module only implements the Swin encoder backbone applied to a
video clip.

Spec from the request:
    - Each cell (patch) covers 16x16 pixels.
    - Each attention window covers 4x4 cells.

So a single spatial window spans 64x64 pixels. With the default input of
256x256 frames, the patch grid is 16x16 and there are 4x4 = 16 windows per
frame, which lets the shifted-window attention actually shift.

Input tensor shape: (B, C, T, H, W) — matches what `DaliDataLoader` yields.
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """(B, T, H, W, C) -> (B*nW, wT*wH*wW, C)."""
    B, T, H, W, C = x.shape
    wT, wH, wW = window_size
    x = x.view(B, T // wT, wT, H // wH, wH, W // wW, wW, C)
    # B, nT, nH, nW, wT, wH, wW, C
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wT * wH * wW, C)


def _window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int],
                    B: int, T: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, wT*wH*wW, C) -> (B, T, H, W, C)."""
    wT, wH, wW = window_size
    nT, nH, nW = T // wT, H // wH, W // wW
    x = windows.view(B, nT, nH, nW, wT, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, T, H, W, -1)


def _compute_shift_mask(T: int, H: int, W: int,
                        window_size: Tuple[int, int, int],
                        shift_size: Tuple[int, int, int],
                        device) -> torch.Tensor:
    """SW-MSA attention mask so tokens from different shifted regions don't mix."""
    img_mask = torch.zeros(1, T, H, W, 1, device=device)
    cnt = 0
    t_slices = (slice(0, -window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None)) if shift_size[0] > 0 else (slice(None),)
    h_slices = (slice(0, -window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None)) if shift_size[1] > 0 else (slice(None),)
    w_slices = (slice(0, -window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None)) if shift_size[2] > 0 else (slice(None),)
    for t in t_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, t, h, w, :] = cnt
                cnt += 1

    mask_windows = _window_partition(img_mask, window_size).squeeze(-1)  # (nW, wT*wH*wW)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, 0.0
    )
    return attn_mask  # (nW, N, N)


# --------------------------------------------------------------------------- #
# Patch embedding & merging
# --------------------------------------------------------------------------- #
class PatchEmbed3D(nn.Module):
    """Split (B, C, T, H, W) video into non-overlapping 3D patches via a Conv3d."""

    def __init__(self, patch_size: Tuple[int, int, int] = (2, 16, 16),
                 in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) -> (B, D, T', H', W')
        x = self.proj(x)
        B, D, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, D)
        return self.norm(x)


class PatchMerging(nn.Module):
    """Halve H and W, concatenate the 4 sub-tokens, project 4D -> 2D."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"H,W must be even, got {H}x{W}"
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, T, H/2, W/2, 4C)
        x = self.norm(x)
        return self.reduction(x)


# --------------------------------------------------------------------------- #
# Window attention + MLP + block
# --------------------------------------------------------------------------- #
class WindowAttention3D(nn.Module):
    """3D windowed multi-head self attention with a learned relative position bias."""

    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wT, wH, wW)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        wT, wH, wW = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wT - 1) * (2 * wH - 1) * (2 * wW - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_t = torch.arange(wT)
        coords_h = torch.arange(wH)
        coords_w = torch.arange(wW)
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)  # (3, wT*wH*wW)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
        rel = rel.permute(1, 2, 0).contiguous()  # (N, N, 3)
        rel[..., 0] += wT - 1
        rel[..., 1] += wH - 1
        rel[..., 2] += wW - 1
        rel[..., 0] *= (2 * wH - 1) * (2 * wW - 1)
        rel[..., 1] *= (2 * wW - 1)
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B_, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B_, h, N, N)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()  # (h, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock3D(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 window_size: Tuple[int, int, int],
                 shift_size: Tuple[int, int, int],
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        wT, wH, wW = self.window_size
        sT, sH, sW = self.shift_size
        assert T % wT == 0 and H % wH == 0 and W % wW == 0, (
            f"Feature map {T}x{H}x{W} not divisible by window {self.window_size}"
        )

        shortcut = x
        x = self.norm1(x)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-sT, -sH, -sW), dims=(1, 2, 3))
            attn_mask = _compute_shift_mask(T, H, W, self.window_size, self.shift_size, x.device)
        else:
            attn_mask = None

        windows = _window_partition(x, self.window_size)  # (B*nW, N, C)
        attended = self.attn(windows, mask=attn_mask)
        x = _window_reverse(attended, self.window_size, B, T, H, W)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(sT, sH, sW), dims=(1, 2, 3))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Stage + full encoder
# --------------------------------------------------------------------------- #
class SwinStage3D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: Tuple[int, int, int],
                 feat_size: Tuple[int, int, int],
                 downsample: bool, mlp_ratio: float = 4.0):
        super().__init__()
        # Clamp window to feature size (Swin handles "feature smaller than window"
        # by treating the whole feature map as a single window with no shifting).
        eff_window = tuple(min(w, f) for w, f in zip(window_size, feat_size))
        shift = tuple((w // 2) if w == f else (w // 2)
                      for w, f in zip(eff_window, feat_size))
        # If feature == window in a dim, shifting is a no-op cycle, so zero it.
        shift = tuple(0 if w == f else s
                      for s, w, f in zip(shift, eff_window, feat_size))

        self.blocks = nn.ModuleList([
            SwinBlock3D(
                dim, num_heads, eff_window,
                shift_size=(0, 0, 0) if (i % 2 == 0) else shift,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinVideoEncoder(nn.Module):
    """
    Video Swin encoder (no masking / no MAE decoder).

    Args:
        input_size: (T, H, W) of the input clip *after* any preprocessing.
            Must satisfy: T % patch_t == 0, H % patch_s == 0, W % patch_s == 0,
            and the resulting feature-map dims must be divisible by the window
            sizes at every stage.
        patch_size: (t, h, w) — defaults to (2, 16, 16) per the request.
        window_size: (t, h, w) cells per attention window. Defaults to
            (8, 4, 4) — i.e. 4x4 spatial cells per the request, 8 temporal cells.
        embed_dim: channels after patch embedding.
        depths: number of Swin blocks per stage.
        num_heads: heads per stage (channel dim doubles each stage, so heads do too).
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (200, 256, 256),
        in_channels: int = 3,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        window_size: Tuple[int, int, int] = (8, 4, 4),
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert len(depths) == len(num_heads)
        self.input_size = input_size
        self.patch_size = patch_size
        self.window_size = window_size

        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        # Feature map size after patch embedding.
        T = input_size[0] // patch_size[0]
        H = input_size[1] // patch_size[1]
        W = input_size[2] // patch_size[2]
        self._validate_dims(T, H, W, len(depths))

        self.stages = nn.ModuleList()
        dim = embed_dim
        cur_T, cur_H, cur_W = T, H, W
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            is_last = i == len(depths) - 1
            self.stages.append(SwinStage3D(
                dim=dim, depth=d, num_heads=h,
                window_size=window_size,
                feat_size=(cur_T, cur_H, cur_W),
                downsample=not is_last,
                mlp_ratio=mlp_ratio,
            ))
            if not is_last:
                dim *= 2
                cur_H //= 2
                cur_W //= 2

        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def _validate_dims(self, T: int, H: int, W: int, n_stages: int):
        wT, wH, wW = self.window_size
        # Effective window is min(window, feat) per dim, so divisibility only
        # needs to hold when the window is strictly smaller than the feat map.
        def check(name: str, feat: int, win: int, stage: int):
            if feat < win:
                return  # whole feature map becomes one window
            if feat % win != 0:
                raise ValueError(
                    f"Stage {stage}: {name}={feat} not divisible by window={win}"
                )

        for stage in range(n_stages):
            check("T", T, wT, stage)
            check("H", H, wH, stage)
            check("W", W, wW, stage)
            if stage < n_stages - 1:
                if H % 2 != 0 or W % 2 != 0:
                    raise ValueError(
                        f"Stage {stage}: cannot patch-merge odd dims H={H} W={W}"
                    )
                H //= 2
                W //= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) in float, already normalized to roughly [0, 1] or
               standardized — this module does no normalization.
        Returns:
            features: (B, T', H', W', C_out) — the final stage's tokens.
        """
        x = self.patch_embed(x)  # (B, T, H, W, D)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Decoder: mirrors the encoder so we can train end-to-end with pixel MSE.
# --------------------------------------------------------------------------- #
class PatchExpand(nn.Module):
    """Inverse of PatchMerging — double H and W, halve channels."""

    def __init__(self, dim: int):
        super().__init__()
        # dim -> 2*dim, then pixel-shuffle into 2x2 spatial blocks -> dim/2 per token.
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = self.expand(x)                               # (B, T, H, W, 2C)
        x = x.view(B, T, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # (B, T, H, 2, W, 2, C/2)
        x = x.view(B, T, H * 2, W * 2, C // 2)
        return self.norm(x)


class SwinVideoDecoder(nn.Module):
    """
    Symmetric Swin decoder. Takes encoder feature tokens and reconstructs the
    input clip in pixel space. Architecture mirror of `SwinVideoEncoder`:
        - PatchExpand to undo PatchMerging at each stage.
        - SwinStage3D (no downsample) to refine after each expand.
        - ConvTranspose3d to undo the initial 3D patch embedding.
    """

    def __init__(
        self,
        encoder_feat_size: Tuple[int, int, int],  # (T', H', W') after all encoder stages
        encoder_out_dim: int,                     # C_out of encoder
        out_channels: int,
        patch_size: Tuple[int, int, int],
        window_size: Tuple[int, int, int],
        embed_dim: int,                           # encoder's embed_dim
        depths: Tuple[int, ...],                  # encoder depths (used reversed)
        num_heads: Tuple[int, ...],               # encoder num_heads (used reversed)
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        n_stages = len(depths)
        self.window_size = window_size

        # Build the reverse chain: start at deepest feat size & dim, expand spatial
        # back up at each stage, halve channels, run a Swin block.
        cur_T, cur_H, cur_W = encoder_feat_size
        dim = encoder_out_dim

        self.stages = nn.ModuleList()
        self.expands = nn.ModuleList()

        # We have (n_stages - 1) patch-merges in the encoder -> same number of expands here.
        # Order from deep -> shallow.
        rev_depths = list(reversed(depths))
        rev_heads = list(reversed(num_heads))

        for i in range(n_stages):
            # Refine current resolution with a Swin stage (no downsample).
            self.stages.append(SwinStage3D(
                dim=dim,
                depth=rev_depths[i],
                num_heads=rev_heads[i],
                window_size=window_size,
                feat_size=(cur_T, cur_H, cur_W),
                downsample=False,
                mlp_ratio=mlp_ratio,
            ))
            # Then expand (unless this is the last stage, which equals embed_dim).
            if i < n_stages - 1:
                self.expands.append(PatchExpand(dim))
                dim //= 2
                cur_H *= 2
                cur_W *= 2

        assert dim == embed_dim, f"decoder dim ended at {dim}, expected {embed_dim}"
        self.final_norm = nn.LayerNorm(dim)

        # Undo the 3D patch embedding with a transposed conv: (T, H, W) tokens of
        # dim `embed_dim` -> (T*pT, H*pH, W*pW) pixels of `out_channels`.
        self.unpatch = nn.ConvTranspose3d(
            embed_dim, out_channels,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T', H', W', C_out) from encoder
        x = feats
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.expands):
                x = self.expands[i](x)
        x = self.final_norm(x)                       # (B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()    # (B, C, T, H, W)
        return self.unpatch(x)


# --------------------------------------------------------------------------- #
# Full autoencoder wrapper (Swin encoder + symmetric Swin decoder).
# This is the standalone trainable model — no masking, no MAE, no codebook.
# --------------------------------------------------------------------------- #
class SwinVideoAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (16, 256, 256),
        in_channels: int = 3,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        window_size: Tuple[int, int, int] = (8, 4, 4),
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.encoder = SwinVideoEncoder(
            input_size=input_size, in_channels=in_channels,
            patch_size=patch_size, window_size=window_size,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Feature-map size at encoder output (used to size the decoder).
        T = input_size[0] // patch_size[0]
        H = input_size[1] // patch_size[1]
        W = input_size[2] // patch_size[2]
        for _ in range(len(depths) - 1):
            H //= 2
            W //= 2

        self.decoder = SwinVideoDecoder(
            encoder_feat_size=(T, H, W),
            encoder_out_dim=self.encoder.out_dim,
            out_channels=in_channels,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon: (B, C, T, H, W) reconstruction
            loss:  scalar MSE between input and reconstruction
        """
        feats = self.encoder(x)
        recon = self.decoder(feats)
        loss = F.mse_loss(recon, x)
        return recon, loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# --------------------------------------------------------------------------- #
# Tiny smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = SwinVideoAutoencoder(input_size=(16, 256, 256))
    x = torch.randn(1, 3, 16, 256, 256)
    with torch.no_grad():
        recon, loss = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"params:  {n_params/1e6:.2f}M  (enc {n_enc/1e6:.2f}M, dec {n_dec/1e6:.2f}M)")
    print(f"input:   {tuple(x.shape)}")
    print(f"recon:   {tuple(recon.shape)}")
    print(f"loss:    {loss.item():.4f}")
