#!/usr/bin/env python
import torch

def rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    """
    Converts RGB tensor (..., 3, H, W) to YUV6 tensor (..., 6, H/2, W/2).
    Matches the rgb_to_yuv6 transformation in frame_utils.py.
    """
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]

    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]

    kYR, kYG, kYB = 0.299, 0.587, 0.114
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)

    U_sub = (
        U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
        U[..., 0::2, 1::2] + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
        V[..., 0::2, 1::2] + V[..., 1::2, 1::2]
    ) * 0.25

    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)
