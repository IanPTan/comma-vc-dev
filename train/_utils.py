import torch

def _flatten_time(x):
    """(B, C, T, H, W) -> (B*T, C, H, W). Returns reshaped tensor plus (B, T)."""
    B, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
    return x, B, T


def _unflatten_time(x, B, T):
    """(B*T, ...) -> (B, T, ...)"""
    return x.view(B, T, *x.shape[1:])
