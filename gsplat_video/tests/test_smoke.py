"""Smoke tests: shapes, activations, gradient flow. No gsplat required."""
from __future__ import annotations

import math

import torch

from gsplat_video.losses import (PosenetLossStub, SegnetLossStub,
                                 score_shaped_loss)
from gsplat_video.populations import (DynamicActor, RoadPlane, RoadsideBand,
                                      SceneRepresentation, SkyDome)
from gsplat_video.pose_inr import EgoPoseINR
from gsplat_video.shader import ViewDependentShader


def _tiny_scene() -> SceneRepresentation:
    return SceneRepresentation(
        road=RoadPlane(n=20, x_range=(0.0, 60.0), y_range=(-5.0, 5.0)),
        sky=SkyDome(n=10, radius=500.0),
        roadside_left=RoadsideBand(n=15, s_range=(0.0, 60.0), side="left"),
        roadside_right=RoadsideBand(n=15, s_range=(0.0, 60.0), side="right"),
        actors=[DynamicActor(m_per_cluster=4), DynamicActor(m_per_cluster=4)],
    )


def test_populations_shapes():
    scene = _tiny_scene()
    t = torch.tensor(0.5)
    g = scene.gaussians(t)
    expected = 20 + 10 + 15 + 15 + 4 + 4
    assert scene.total_count() == expected
    assert g["means"].shape == (expected, 3)
    assert g["scales"].shape == (expected, 3)
    assert g["quats"].shape == (expected, 4)
    assert g["opacities"].shape == (expected,)
    assert g["colors"].shape == (expected, 3)
    print(f"[ok] populations shapes: total={expected} Gaussians")


def test_road_z_zero():
    road = RoadPlane(n=50, x_range=(0.0, 60.0), y_range=(-5.0, 5.0))
    g = road.gaussians(torch.tensor(0.0))
    assert torch.all(g["means"][:, 2] == 0.0), "road Gaussians must be at z=0"
    print("[ok] road Gaussians pinned to z=0")


def test_sky_on_sphere():
    sky = SkyDome(n=30, radius=500.0)
    g = sky.gaussians(torch.tensor(0.0))
    r = g["means"].norm(dim=-1)
    assert torch.allclose(r, torch.full_like(r, 500.0), atol=1e-3), \
        f"sky must lie on sphere of radius 500, got {r.min():.2f}..{r.max():.2f}"
    print("[ok] sky Gaussians on sphere of radius 500")


def test_roadside_side_signs():
    left = RoadsideBand(n=20, s_range=(0.0, 60.0), side="left")
    right = RoadsideBand(n=20, s_range=(0.0, 60.0), side="right")
    gl = left.gaussians(torch.tensor(0.0))
    gr = right.gaussians(torch.tensor(0.0))
    assert torch.all(gl["means"][:, 1] > 0), "left band should have y > 0"
    assert torch.all(gr["means"][:, 1] < 0), "right band should have y < 0"
    print("[ok] roadside bands on correct sides")


def test_dynamic_trajectory():
    actor = DynamicActor(m_per_cluster=4, traj_degree=2,
                         t_range=(0.0, 60.0), initial_center=(10.0, 0.0, 1.0))
    # Give it a nontrivial trajectory: linear motion along +x with 5 unit velocity
    with torch.no_grad():
        actor.traj_coeffs[1, 0] = 5.0   # linear x term
    g0 = actor.gaussians(torch.tensor(0.0))
    g1 = actor.gaussians(torch.tensor(60.0))
    center0 = g0["means"].mean(dim=0)   # cluster center at t=0
    center1 = g1["means"].mean(dim=0)   # cluster center at t=60
    dx = (center1 - center0)[0]
    assert torch.allclose(dx, torch.tensor(5.0), atol=1e-4), \
        f"actor should have moved +5 in x, got {dx.item():.3f}"
    print(f"[ok] dynamic actor trajectory: dx over full window = {dx.item():.3f}")


def test_pose_inr_rotation_orthogonal():
    pose = EgoPoseINR(hidden=32, n_freqs=8)
    ts = torch.linspace(0.0, 1.0, 5)
    for t in ts:
        viewmat = pose.viewmat(t)
        R = viewmat[0:3, 0:3]
        should_be_I = R @ R.transpose(-1, -2)
        assert torch.allclose(should_be_I, torch.eye(3), atol=1e-4), \
            f"rotation not orthogonal at t={t.item():.2f}: R R^T = {should_be_I}"
    print("[ok] pose INR outputs orthogonal rotation matrices across time")


def test_pose_inr_gradient_flows():
    pose = EgoPoseINR(hidden=16, n_freqs=4)
    t = torch.tensor(0.3)
    R, tvec = pose(t)
    (R.sum() + tvec.sum()).backward()
    for name, p in pose.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
    print("[ok] pose INR gradients flow and are finite")


def test_scene_gradient_flows():
    scene = _tiny_scene()
    t = torch.tensor(0.5)
    g = scene.gaussians(t)
    # Dummy scalar loss combining all outputs
    loss = (g["means"].square().mean() + g["scales"].square().mean()
            + g["quats"].square().mean() + g["opacities"].square().mean()
            + g["colors"].square().mean())
    loss.backward()
    for name, p in scene.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
    print("[ok] scene gradients flow through all populations")


def test_score_shaped_loss():
    seg = torch.tensor(0.001, requires_grad=True)
    pose = torch.tensor(0.0001, requires_grad=True)
    rate = torch.tensor(0.005, requires_grad=True)
    photo = torch.tensor(0.05, requires_grad=True)

    loss = score_shaped_loss(seg, pose, rate, w_photo=0.1, photo=photo)
    expected = 100 * 0.001 + 25 * 0.005 + math.sqrt(10 * 0.0001 + 1e-6) + 0.1 * 0.05
    assert abs(loss.item() - expected) < 1e-4, f"expected {expected}, got {loss.item()}"

    loss.backward()
    assert seg.grad is not None and pose.grad is not None and rate.grad is not None
    print(f"[ok] score-shaped loss = {loss.item():.6f} (matches formula)")


def test_shader_shapes():
    shader = ViewDependentShader(hidden=24)
    base = torch.rand(50, 3)
    view = torch.nn.functional.normalize(torch.randn(50, 3), dim=-1)
    out = shader(base, view)
    assert out.shape == (50, 3)
    assert (out >= 0).all() and (out <= 1).all()
    print("[ok] shader produces valid RGB in [0,1]")


def test_loss_stubs_run():
    seg_fn = SegnetLossStub()
    pose_fn = PosenetLossStub()
    r = torch.rand(3, 32, 32)
    t = torch.rand(3, 32, 32)
    r2 = torch.rand(3, 32, 32)
    t2 = torch.rand(3, 32, 32)
    s = seg_fn(r, t)
    p = pose_fn((r, r2), (t, t2))
    assert s.dim() == 0 and p.dim() == 0
    print(f"[ok] stubs: seg={s.item():.4f} pose={p.item():.4f}")


def main():
    tests = [
        test_populations_shapes,
        test_road_z_zero,
        test_sky_on_sphere,
        test_roadside_side_signs,
        test_dynamic_trajectory,
        test_pose_inr_rotation_orthogonal,
        test_pose_inr_gradient_flows,
        test_scene_gradient_flows,
        test_score_shaped_loss,
        test_shader_shapes,
        test_loss_stubs_run,
    ]
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} smoke tests passed.")


if __name__ == "__main__":
    main()
