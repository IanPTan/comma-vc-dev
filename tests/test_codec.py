"""Codec smoke tests: rate math, STE gradients, module + scene roundtrip."""
from __future__ import annotations

import numpy as np
import torch

from gsplat_video.codec import (BASELINE_ORIGINAL_BYTES, BitsConfig,
                                GaussianCodec, pack_module, pack_scene,
                                ste_quantize, unpack_module, unpack_scene)
from gsplat_video.populations import (DynamicActor, RoadPlane, RoadsideBand,
                                      SceneRepresentation, SkyDome)
from gsplat_video.pose_inr import EgoPoseINR


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
    assert b.bits_per_gaussian() == 122        # 3*12 + 3*8 + 4*8 + 6 + 3*8
    print(f"[ok] bits_per_gaussian = {b.bits_per_gaussian()}")


def test_rate_estimate_target_range():
    codec = GaussianCodec()
    n = 500 + 200 + 2000 + 5 * 100
    extra = 6_000                              # pose INR + shader
    r = codec.rate(n_gaussians=n, extra_bytes=extra)
    kb = r * BASELINE_ORIGINAL_BYTES / 1024
    print(f"[ok] projected archive: {kb:.1f} KB, rate={r:.5f}, 25*rate={25*r:.4f}")
    assert 40 < kb < 200


def test_ste_quantize_gradient():
    x = torch.linspace(-1.0, 1.0, 100, requires_grad=True)
    y, _ = ste_quantize(x, num_bits=8)
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    max_err = (y - x).detach().abs().max().item()
    step = 2.0 / (2**8 - 1)
    assert max_err < step
    print(f"[ok] STE gradients flow; max quant error {max_err:.5f} < step {step:.5f}")


def test_ste_gradient_pass_through():
    x = torch.randn(50, requires_grad=True)
    y, _ = ste_quantize(x, num_bits=6)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
    print("[ok] STE grad passes through as identity")


def test_pack_module_roundtrip():
    """pack_module + unpack_module preserves an MLP within INT8 tolerance."""
    torch.manual_seed(0)
    m = EgoPoseINR(hidden=32, n_freqs=8)
    payload, header = pack_module(m, num_bits=8)

    m2 = EgoPoseINR(hidden=32, n_freqs=8)
    unpack_module(m2, payload, header)

    for (n1, p1), (n2, p2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2
        span = (p1.max() - p1.min()).item()
        step = span / 255 if span > 0 else 1e-4
        err = (p1 - p2).abs().max().item()
        assert err <= step + 1e-4, f"{n1}: err {err} > step {step}"
    print(f"[ok] pack_module roundtrip: payload {len(payload)} bytes")


def test_pack_scene_roundtrip_preserves_actor_motion():
    """pack_scene + unpack_scene preserves DynamicActor trajectory motion."""
    torch.manual_seed(42)
    actor = DynamicActor(m_per_cluster=4, traj_degree=2, t_range=(0, 60),
                         initial_center=(10.0, 0.0, 1.0))
    with torch.no_grad():
        actor.traj_coeffs[1, 0] = 5.0

    scene = SceneRepresentation(
        road=RoadPlane(n=5, x_range=(0, 60), y_range=(-5, 5)),
        sky=SkyDome(n=5, radius=500.0),
        roadside_left=RoadsideBand(n=5, s_range=(0, 60), side="left"),
        roadside_right=RoadsideBand(n=5, s_range=(0, 60), side="right"),
        actors=[actor],
    )
    payload, header = pack_scene(scene, num_bits=8)
    scene2 = unpack_scene(payload, header)

    dx_orig = (scene.gaussians(torch.tensor(60.0))["means"][-4:].mean(0)
               - scene.gaussians(torch.tensor(0.0))["means"][-4:].mean(0))[0].item()
    dx_r = (scene2.gaussians(torch.tensor(60.0))["means"][-4:].mean(0)
            - scene2.gaussians(torch.tensor(0.0))["means"][-4:].mean(0))[0].item()
    print(f"[ok] pack_scene actor motion: orig dx={dx_orig:.3f}, roundtrip dx={dx_r:.3f}")
    assert abs(dx_orig - dx_r) < 0.2


def main():
    tests = [
        test_bits_per_gaussian,
        test_rate_estimate_target_range,
        test_ste_quantize_gradient,
        test_ste_gradient_pass_through,
        test_pack_module_roundtrip,
        test_pack_scene_roundtrip_preserves_actor_motion,
    ]
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} codec tests passed.")


if __name__ == "__main__":
    main()
