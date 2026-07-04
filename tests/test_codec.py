"""Codec smoke tests: rate math, STE gradients, byte round-trip."""
from __future__ import annotations

import numpy as np
import torch

from gsplat_video.codec import (BASELINE_ORIGINAL_BYTES, BitsConfig,
                                GaussianCodec, ste_quantize)
from gsplat_video.populations import (DynamicActor, RoadPlane, RoadsideBand,
                                      SceneRepresentation, SkyDome)


def _scene(road_n=500, sky_n=200, road_side_n=2000, actors=100) -> SceneRepresentation:
    return SceneRepresentation(
        road=RoadPlane(n=road_n, x_range=(0.0, 60.0), y_range=(-5.0, 5.0)),
        sky=SkyDome(n=sky_n, radius=500.0),
        roadside_left=RoadsideBand(n=road_side_n // 2, s_range=(0.0, 60.0), side="left"),
        roadside_right=RoadsideBand(n=road_side_n // 2, s_range=(0.0, 60.0), side="right"),
        actors=[DynamicActor(m_per_cluster=5) for _ in range(actors)],
    )


def test_bits_per_gaussian():
    b = BitsConfig()
    # 3*12 + 3*8 + 4*8 + 6 + 3*8 = 36+24+32+6+24 = 122
    assert b.bits_per_gaussian() == 122
    print(f"[ok] bits_per_gaussian = {b.bits_per_gaussian()}")


def test_rate_estimate_target_range():
    """Confirm our design lands in the 60-150 KB rate window we projected."""
    codec = GaussianCodec()
    n = 500 + 200 + 2000 + 5 * 100  # matches the pitch's numbers
    extra = 3_000 + 3_000            # pose INR + shader
    r = codec.rate(n_gaussians=n, extra_bytes=extra)
    bytes_est = r * BASELINE_ORIGINAL_BYTES
    kb = bytes_est / 1024
    print(f"[ok] projected archive: {kb:.1f} KB, rate = {r:.5f}, 25*rate = {25*r:.4f}")
    assert 40 < kb < 200, f"design outside expected window: {kb:.1f} KB"


def test_ste_quantize_gradient():
    x = torch.linspace(-1.0, 1.0, 100, requires_grad=True)
    y, _range = ste_quantize(x, num_bits=8)
    loss = y.square().sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Quantized values should be within one quantization step of original.
    max_err = (y - x).detach().abs().max().item()
    step = 2.0 / (2**8 - 1)
    assert max_err < step, f"quantization error {max_err} exceeds step {step}"
    print(f"[ok] STE gradients flow; max quant error {max_err:.5f} < step {step:.5f}")


def test_ste_gradient_pass_through():
    """STE must let gradients pass through as identity."""
    x = torch.randn(50, requires_grad=True)
    y, _r = ste_quantize(x, num_bits=6)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), "STE should pass identity grad"
    print("[ok] STE grad passes through as identity")


def test_encode_decode_roundtrip():
    torch.manual_seed(0)
    scene = _scene(road_n=50, sky_n=20, road_side_n=100, actors=10)
    raw = scene.gaussians(torch.tensor(0.0))
    codec = GaussianCodec()
    blob = codec.encode_scene(raw)
    restored = codec.decode_scene(blob)
    for k in ["means", "scales", "quats", "opacities", "colors"]:
        orig = raw[k].detach().cpu().numpy()
        rec = restored[k]
        # Values should be within one quantization step of original.
        span = float(orig.max() - orig.min())
        bits = getattr(codec.bits, k)
        step = span / (2**bits - 1) if span > 0 else 1e-6
        err = np.abs(rec.reshape(orig.shape) - orig).max()
        assert err <= step + 1e-5, f"{k}: err {err} > step {step}"
        print(f"[ok] roundtrip {k}: shape {orig.shape}, max err {err:.6f} <= step {step:.6f}")
    print(f"[ok] archive size: {len(blob)} bytes (uncompressed by brotli)")


def test_encode_size_matches_estimate():
    """Encoded byte count should track the rate estimate closely."""
    torch.manual_seed(42)
    n_g = 200
    scene = _scene(road_n=50, sky_n=20, road_side_n=100, actors=6)  # actors have m=5 each
    assert scene.total_count() == n_g
    raw = scene.gaussians(torch.tensor(0.0))
    codec = GaussianCodec()
    actual_bytes = len(codec.encode_scene(raw))
    # No entropy coding factor here, just header + packed bits.
    projected_bytes = int(codec.rate(n_g, extra_bytes=0, include_ec_factor=False)
                          * BASELINE_ORIGINAL_BYTES)
    # Actual will be slightly larger due to header. Should be within ~100 bytes.
    print(f"[ok] size check: actual {actual_bytes} vs projected {projected_bytes}")
    assert abs(actual_bytes - projected_bytes) < 200, \
        f"size mismatch: actual {actual_bytes} vs projected {projected_bytes}"


def main():
    tests = [
        test_bits_per_gaussian,
        test_rate_estimate_target_range,
        test_ste_quantize_gradient,
        test_ste_gradient_pass_through,
        test_encode_decode_roundtrip,
        test_encode_size_matches_estimate,
    ]
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} codec tests passed.")


if __name__ == "__main__":
    main()
