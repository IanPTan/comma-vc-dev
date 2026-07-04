"""End-to-end: encode -> decode via per-population encoding. Verify each
population's native parameters survive AND that dynamic actor trajectories
still produce time-varying positions after roundtrip."""
from __future__ import annotations

import numpy as np
import torch

from gsplat_video.compress import (encode_archive, make_initial_scene)
from gsplat_video.inflate import load_from_archive
from gsplat_video.populations import (DynamicActor, RoadPlane, RoadsideBand,
                                      SceneRepresentation, SkyDome)
from gsplat_video.pose_inr import EgoPoseINR


def _small_scene_with_actors() -> SceneRepresentation:
    return SceneRepresentation(
        road=RoadPlane(n=30, x_range=(0.0, 60.0), y_range=(-5.0, 5.0)),
        sky=SkyDome(n=10, radius=500.0),
        roadside_left=RoadsideBand(n=15, s_range=(0.0, 60.0), side="left"),
        roadside_right=RoadsideBand(n=15, s_range=(0.0, 60.0), side="right"),
        actors=[DynamicActor(m_per_cluster=4, traj_degree=2,
                             t_range=(0.0, 60.0), initial_center=(10.0, 0.0, 1.0))
                for _ in range(3)],
    )


def test_archive_static_populations_roundtrip():
    torch.manual_seed(0)
    scene = _small_scene_with_actors()
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)

    blob = encode_archive(scene, pose_inr, shader=None)
    print(f"[ok] archive size: {len(blob)} bytes")

    scene_r, pose_r, shader_r, cfg = load_from_archive(blob)
    assert shader_r is None

    # Compare the concatenated gaussians at t=0. Because we quantize each
    # population's NATIVE parameters (not the projected means directly), the
    # effective drift on means/scales depends on how sensitive each
    # population's projection is to input quantization.
    # Sky (radius=500) is especially sensitive to angular INT8 -> ~6 m position
    # step is expected. Sky bit budget will need tuning in Phase 5.
    orig = scene.gaussians(torch.tensor(0.0))
    got = scene_r.gaussians(torch.tensor(0.0))
    tolerance = {"means": 10.0, "scales": 0.05, "quats": 0.02,
                 "opacities": 0.02, "colors": 0.02}
    for k in ["means", "scales", "quats", "opacities", "colors"]:
        e = orig[k].detach().cpu().numpy()
        g = got[k].detach().cpu().numpy()
        assert e.shape == g.shape, f"{k}: shape mismatch {e.shape} vs {g.shape}"
        err = np.abs(e - g).max()
        assert err <= tolerance[k], f"{k} err {err} > tol {tolerance[k]}"
        print(f"[ok] {k} @ t=0: max err {err:.6f} (tol {tolerance[k]})")


def test_archive_dynamic_actor_moves():
    """Dynamic actors should have DIFFERENT positions at different ts, both
    before AND after archive roundtrip."""
    torch.manual_seed(42)
    actor = DynamicActor(m_per_cluster=4, traj_degree=2, t_range=(0.0, 60.0),
                         initial_center=(10.0, 0.0, 1.0))
    with torch.no_grad():
        actor.traj_coeffs[1, 0] = 5.0  # linear +x term
        actor.traj_coeffs[2, 1] = 2.0  # quadratic y term

    scene = SceneRepresentation(
        road=RoadPlane(n=5, x_range=(0.0, 60.0), y_range=(-5.0, 5.0)),
        sky=SkyDome(n=5, radius=500.0),
        roadside_left=RoadsideBand(n=5, s_range=(0.0, 60.0), side="left"),
        roadside_right=RoadsideBand(n=5, s_range=(0.0, 60.0), side="right"),
        actors=[actor],
    )
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)

    # Actor positions at t=0 and t=60 in the original scene.
    means_at = {t: scene.gaussians(torch.tensor(float(t)))["means"][-4:].detach().clone()
                for t in (0, 60)}
    dx_orig = (means_at[60] - means_at[0]).mean(0)
    print(f"[ok] original actor motion over t=[0,60]: dx={dx_orig[0]:.3f} "
          f"dy={dx_orig[1]:.3f} dz={dx_orig[2]:.3f}")
    assert dx_orig[0] > 2.0, "linear x motion should be present in original"

    # Round trip and check motion still works after decode.
    blob = encode_archive(scene, pose_inr)
    scene_r, _, _, _ = load_from_archive(blob)
    means_r = {t: scene_r.gaussians(torch.tensor(float(t)))["means"][-4:].detach().clone()
               for t in (0, 60)}
    dx_r = (means_r[60] - means_r[0]).mean(0)
    print(f"[ok] roundtrip actor motion over t=[0,60]: dx={dx_r[0]:.3f} "
          f"dy={dx_r[1]:.3f} dz={dx_r[2]:.3f}")
    # Motion should agree with original within quantization tolerance.
    diff = (dx_orig - dx_r).abs().max().item()
    assert diff < 0.5, f"actor motion changed too much after roundtrip: {diff}"


def test_archive_pose_inr_roundtrip():
    torch.manual_seed(1)
    scene = _small_scene_with_actors()
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)

    ts = torch.linspace(0.0, 1.0, 5)
    with torch.no_grad():
        orig_out = [pose_inr.viewmat(t) for t in ts]

    blob = encode_archive(scene, pose_inr)
    _, pose_r, _, _ = load_from_archive(blob)
    with torch.no_grad():
        r_out = [pose_r.viewmat(t) for t in ts]

    max_drift = max((a - b).abs().max().item() for a, b in zip(orig_out, r_out))
    assert max_drift < 0.5, f"pose INR drift {max_drift} too large"
    print(f"[ok] pose INR INT8 roundtrip: max drift {max_drift:.4f}")


def test_archive_size_realistic():
    scene = make_initial_scene()
    pose_inr = EgoPoseINR()
    blob = encode_archive(scene, pose_inr, shader=None)
    kb = len(blob) / 1024
    n = scene.total_count()
    print(f"[ok] full-scale archive (untrained): {kb:.1f} KB, {n} Gaussians "
          f"(design target trained: 70-150 KB; winners at 178 KB)")
    assert kb < 200, f"even untrained archive {kb:.1f} KB exceeds winners' budget"


def main():
    tests = [
        test_archive_static_populations_roundtrip,
        test_archive_dynamic_actor_moves,
        test_archive_pose_inr_roundtrip,
        test_archive_size_realistic,
    ]
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} archive roundtrip tests passed.")


if __name__ == "__main__":
    main()
