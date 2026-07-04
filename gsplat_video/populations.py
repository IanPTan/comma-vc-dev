"""Four physically-structured Gaussian populations for dashcam scenes.

Coordinate frame: right-handed, +x forward (driving direction), +y left, +z up.
Ego camera is somewhere near the origin, driving along +x.

Each population exposes `.gaussians(t)` -> dict of tensors with keys:
    means:    (N, 3)   world-space centers
    scales:   (N, 3)   log-space scales (softplus at render time)
    quats:    (N, 4)   unnormalized quaternions (normalized at render time)
    opacities:(N,)     pre-sigmoid logits
    colors:   (N, 3)   pre-sigmoid RGB (view-dependent shading added by shader)

The rasterizer concatenates all populations' outputs before invoking gsplat.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _quat_identity(n: int, device=None) -> Tensor:
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q


class RoadPlane(nn.Module):
    """2D Gaussians constrained to the ground plane z = 0.

    Params per Gaussian: (u, v) position in plane, (log_scale_u, log_scale_v),
    yaw angle for in-plane rotation, and RGB color. Total: 8 raw floats.
    At INT8 that's 8 bytes/Gaussian.
    """

    def __init__(self, n: int, x_range: tuple[float, float], y_range: tuple[float, float]):
        super().__init__()
        self.n = n
        self.x_range = x_range
        self.y_range = y_range

        u = torch.rand(n) * (x_range[1] - x_range[0]) + x_range[0]
        v = torch.rand(n) * (y_range[1] - y_range[0]) + y_range[0]
        self.uv = nn.Parameter(torch.stack([u, v], dim=-1))
        self.log_scale = nn.Parameter(torch.full((n, 2), math.log(0.3)))
        self.yaw = nn.Parameter(torch.zeros(n))
        self.color = nn.Parameter(torch.zeros(n, 3))          # pre-sigmoid RGB
        self.opacity = nn.Parameter(torch.full((n,), 2.0))    # pre-sigmoid, biases toward opaque

    def gaussians(self, t: Tensor | None = None) -> dict[str, Tensor]:
        means = torch.zeros(self.n, 3, device=self.uv.device)
        means[:, 0] = self.uv[:, 0]
        means[:, 1] = self.uv[:, 1]
        # z = 0 always -> road plane constraint enforced structurally

        # 2D scales (u,v) extruded to 3D with tiny vertical scale so Gaussian
        # is flat against the ground.
        scales = torch.empty(self.n, 3, device=self.uv.device)
        scales[:, 0:2] = self.log_scale
        scales[:, 2] = math.log(0.01)  # thin in z

        # Yaw-only rotation as quaternion around +z axis
        half = self.yaw * 0.5
        quats = torch.stack([torch.cos(half),
                             torch.zeros_like(half),
                             torch.zeros_like(half),
                             torch.sin(half)], dim=-1)

        return dict(means=means, scales=scales, quats=quats,
                    opacities=self.opacity, colors=self.color)


class SkyDome(nn.Module):
    """2D Gaussians on a sphere at large radius. Direction-parameterized;
    infinite-depth content, no perspective foreshortening from ego motion.

    Params per Gaussian: (theta, phi) direction, (log_scale_theta, log_scale_phi),
    RGB color, opacity. 7 raw floats -> ~7 bytes/Gaussian at INT8.
    """

    def __init__(self, n: int, radius: float = 500.0):
        super().__init__()
        self.n = n
        self.radius = radius

        theta = torch.rand(n) * math.pi           # 0..pi (elevation from +z)
        phi = torch.rand(n) * 2 * math.pi         # 0..2pi (azimuth around z)
        self.dir = nn.Parameter(torch.stack([theta, phi], dim=-1))
        self.log_scale = nn.Parameter(torch.full((n, 2), math.log(0.3)))
        self.color = nn.Parameter(torch.zeros(n, 3))
        self.opacity = nn.Parameter(torch.full((n,), 2.0))

    def gaussians(self, t: Tensor | None = None) -> dict[str, Tensor]:
        theta, phi = self.dir[:, 0], self.dir[:, 1]
        st, ct, sp, cp = torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)
        means = self.radius * torch.stack([st * cp, st * sp, ct], dim=-1)

        # Angular scales become linear scales at radius R -> mean * R.
        angular_scale = self.log_scale.exp() * self.radius
        scales = torch.empty(self.n, 3, device=means.device)
        scales[:, 0:2] = angular_scale.log()
        scales[:, 2] = math.log(0.01 * self.radius)  # thin in radial direction

        # Rotation aligns the Gaussian tangent plane with the sphere tangent.
        # Simplification: identity quat for now, refine if needed.
        quats = _quat_identity(self.n, device=means.device)

        return dict(means=means, scales=scales, quats=quats,
                    opacities=self.opacity, colors=self.color)


class RoadsideBand(nn.Module):
    """3D Gaussians confined to two lateral bands parallel to the driving axis.

    Split into left (y > 0) and right (y < 0) sides. Position parameterized as
    (s along path, w lateral offset within band, h height above ground).
    Prevents Gaussians from drifting into road/sky during optimization.

    Params per Gaussian: (s, w, h), (log_sx, log_sy, log_sz), 4-vec quat,
    RGB, opacity. ~14 raw floats -> ~14 bytes/Gaussian at INT8.
    """

    def __init__(self, n: int, s_range: tuple[float, float],
                 w_range: tuple[float, float] = (2.0, 20.0),
                 h_range: tuple[float, float] = (0.0, 15.0),
                 side: str = "left"):
        super().__init__()
        self.n = n
        self.s_range, self.w_range, self.h_range = s_range, w_range, h_range
        self.side_sign = 1.0 if side == "left" else -1.0

        s = torch.rand(n) * (s_range[1] - s_range[0]) + s_range[0]
        w = torch.rand(n) * (w_range[1] - w_range[0]) + w_range[0]
        h = torch.rand(n) * (h_range[1] - h_range[0]) + h_range[0]
        self.pos_band = nn.Parameter(torch.stack([s, w, h], dim=-1))

        self.log_scale = nn.Parameter(torch.full((n, 3), math.log(0.5)))
        self.quat = nn.Parameter(_quat_identity(n))
        self.color = nn.Parameter(torch.zeros(n, 3))
        self.opacity = nn.Parameter(torch.full((n,), 2.0))

    def gaussians(self, t: Tensor | None = None) -> dict[str, Tensor]:
        s = self.pos_band[:, 0]
        w = self.pos_band[:, 1]
        h = self.pos_band[:, 2]
        means = torch.stack([s, self.side_sign * w, h], dim=-1)
        return dict(means=means, scales=self.log_scale, quats=self.quat,
                    opacities=self.opacity, colors=self.color)


class DynamicActor(nn.Module):
    """A cluster of 3D Gaussians representing one moving object (car, ped).

    Cluster center follows a polynomial trajectory in world space:
        c(t) = sum_k coeff_k * t^k     with degree D.
    Each Gaussian in the cluster has a fixed offset from the center (in the
    object's local frame) plus scale/rot/color/opacity.

    Params:
        traj_coeffs: (D+1, 3) -> position polynomial. D=2 (quadratic) is enough
                     for straight-line and mild accel over a short window.
        offsets:    (M, 3)   local frame offsets for M Gaussians in the cluster
        + per-Gaussian scale/quat/color/opacity as usual.

    For a 5-Gaussian actor with quadratic trajectory:
        traj: 3*3 = 9 floats
        offsets: 5*3 = 15 floats
        scales: 5*3 = 15
        quats:  5*4 = 20
        colors: 5*3 = 15
        opacities: 5 = 5
        = 79 floats per actor. At INT8: ~79 bytes/actor.

    100 actors -> ~8 KB.
    """

    def __init__(self, m_per_cluster: int = 5, traj_degree: int = 2,
                 t_range: tuple[float, float] = (0.0, 60.0),
                 initial_center: tuple[float, float, float] = (10.0, 0.0, 1.0)):
        super().__init__()
        self.m = m_per_cluster
        self.degree = traj_degree
        self.t_min, self.t_max = t_range

        # Trajectory coefficients c_k such that c(t_norm) = sum coeff_k * t_norm^k
        # where t_norm = (t - t_min) / (t_max - t_min) in [0, 1].
        coeffs = torch.zeros(traj_degree + 1, 3)
        coeffs[0] = torch.tensor(initial_center)  # constant term = initial pos
        self.traj_coeffs = nn.Parameter(coeffs)

        self.offsets = nn.Parameter(torch.randn(m_per_cluster, 3) * 0.5)
        self.log_scale = nn.Parameter(torch.full((m_per_cluster, 3), math.log(0.3)))
        self.quat = nn.Parameter(_quat_identity(m_per_cluster))
        self.color = nn.Parameter(torch.zeros(m_per_cluster, 3))
        self.opacity = nn.Parameter(torch.full((m_per_cluster,), 2.0))

    def _center_at(self, t: Tensor) -> Tensor:
        # t is a scalar tensor (single frame) — batching to be added later.
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        powers = torch.stack([t_norm ** k for k in range(self.degree + 1)])  # (D+1,)
        return (powers[:, None] * self.traj_coeffs).sum(dim=0)  # (3,)

    def gaussians(self, t: Tensor) -> dict[str, Tensor]:
        center = self._center_at(t)                # (3,)
        means = center[None, :] + self.offsets     # (M, 3)
        return dict(means=means, scales=self.log_scale, quats=self.quat,
                    opacities=self.opacity, colors=self.color)


class SceneRepresentation(nn.Module):
    """Container that assembles all Gaussian populations for a scene."""

    def __init__(self, road: RoadPlane, sky: SkyDome,
                 roadside_left: RoadsideBand, roadside_right: RoadsideBand,
                 actors: list[DynamicActor]):
        super().__init__()
        self.road = road
        self.sky = sky
        self.roadside_left = roadside_left
        self.roadside_right = roadside_right
        self.actors = nn.ModuleList(actors)

    def gaussians(self, t: Tensor) -> dict[str, Tensor]:
        parts = [self.road.gaussians(t), self.sky.gaussians(t),
                 self.roadside_left.gaussians(t), self.roadside_right.gaussians(t)]
        parts.extend(actor.gaussians(t) for actor in self.actors)

        return {k: torch.cat([p[k] for p in parts], dim=0)
                for k in ["means", "scales", "quats", "opacities", "colors"]}

    def total_count(self) -> int:
        return (self.road.n + self.sky.n + self.roadside_left.n + self.roadside_right.n
                + sum(a.m for a in self.actors))
