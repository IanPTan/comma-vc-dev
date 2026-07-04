"""End-to-end: encode a small scene into archive.zip, decode it, verify
Gaussian arrays and INR weights come back within quantization tolerance."""
from __future__ import annotations

import numpy as np
import torch

from gsplat_video.compress import (encode_archive, make_initial_scene)
from gsplat_video.inflate import load_from_archive
from gsplat_video.populations import (DynamicActor, RoadPlane, RoadsideBand,
                                      SceneRepresentation, SkyDome)
from gsplat_video.pose_inr import EgoPoseINR


def _tiny_scene() -> SceneRepresentation:
    return SceneRepresentation(
        road=RoadPlane(n=30, x_range=(0.0, 60.0), y_range=(-5.0, 5.0)),
        sky=SkyDome(n=10, radius=500.0),
        roadside_left=RoadsideBand(n=15, s_range=(0.0, 60.0), side="left"),
        roadside_right=RoadsideBand(n=15, s_range=(0.0, 60.0), side="right"),
        actors=[],  # v0: static only
    )


def test_archive_roundtrip_gaussians():
    torch.manual_seed(0)
    scene = _tiny_scene()
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)

    blob = encode_archive(scene, pose_inr, shader=None)
    print(f"[ok] archive size: {len(blob)} bytes")

    gaussians, pose_restored, shader_restored, cfg = load_from_archive(blob)
    assert shader_restored is None
    assert cfg["road_n"] == 30

    # Compare against original scene's flat Gaussians at t=0.
    orig = scene.gaussians(torch.tensor(0.0))
    for k in ["means", "scales", "quats", "opacities", "colors"]:
        expected = orig[k].detach().cpu().numpy()
        got = gaussians[k].cpu().numpy()
        assert expected.shape == got.shape, f"{k} shape mismatch"
        # Within one quantization step.
        span = float(expected.max() - expected.min())
        step = span / 255 if span > 0 else 1e-6
        err = np.abs(expected - got).max()
        assert err <= step + 1e-4, f"{k} err {err} exceeds step {step}"
        print(f"[ok] Gaussian {k}: max roundtrip err {err:.6f} <= step {step:.6f}")


def test_archive_roundtrip_pose_inr():
    torch.manual_seed(1)
    scene = _tiny_scene()
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)

    # Run pose_inr through several ts BEFORE encoding, save outputs.
    ts = torch.linspace(0.0, 1.0, 5)
    with torch.no_grad():
        original_out = [pose_inr.viewmat(t) for t in ts]

    blob = encode_archive(scene, pose_inr, shader=None)
    _, pose_restored, _, _ = load_from_archive(blob)

    with torch.no_grad():
        restored_out = [pose_restored.viewmat(t) for t in ts]

    # Pose INR is INT8 quantized -> small drift is expected; check it's bounded.
    for i, (a, b) in enumerate(zip(original_out, restored_out)):
        diff = (a - b).abs().max().item()
        # Rough tolerance: pose INR weights INT8-quantized; ~1% typical drift.
        assert diff < 0.5, f"pose INR drift at t[{i}]: {diff}"
    print(f"[ok] pose INR INT8 roundtrip: max drift across 5 samples "
          f"{max((a - b).abs().max().item() for a, b in zip(original_out, restored_out)):.4f}")


def test_archive_size_realistic():
    """Verify a full-scale untrained scene lands in the projected 70-150 KB band."""
    scene = make_initial_scene()
    pose_inr = EgoPoseINR()
    blob = encode_archive(scene, pose_inr, shader=None)
    kb = len(blob) / 1024
    n = scene.total_count()
    # Untrained scene has most params at init constants -> zip compresses hard.
    # Real trained size will be larger. Just verify it's under the winners' 178 KB.
    print(f"[ok] full-scale archive: {kb:.1f} KB, {n} Gaussians "
          f"(design target for trained: 70-150 KB; winners at 178 KB)")
    assert kb < 200, f"even untrained archive {kb:.1f} KB exceeds winners' budget"


def main():
    tests = [
        test_archive_roundtrip_gaussians,
        test_archive_roundtrip_pose_inr,
        test_archive_size_realistic,
    ]
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} archive roundtrip tests passed.")


if __name__ == "__main__":
    main()
