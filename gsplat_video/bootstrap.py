"""Phase 1: turn a dashcam video into an initialized SceneRepresentation.

Pipeline:
  1. estimate_ego_trajectory  - straight-line forward motion + small yaw drift.
     (Later: replace with COLMAP or a real VO frontend.)
  2. estimate_depth           - Depth-Anything-V2 monocular depth per frame.
  3. semantic_classes         - segnet per-pixel class labels.
  4. unproject_frame          - depth + K + pose -> world-space points.
  5. distribute_points        - assign each point to a population by
     geometry-plus-semantics, seed each population's parameters.

Depth-Anything and segnet are heavy models; they run on the training device
(Colab GPU). Locally we test the geometry + distribution logic against
synthetic depth/class maps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .constants import (CAMERA_CX, CAMERA_CY, CAMERA_FL, CAMERA_H, CAMERA_W,
                        FPS, N_FRAMES, intrinsics)
from .populations import (DynamicActor, RoadPlane, RoadsideBand,
                          SceneRepresentation, SkyDome)


# ---------------------------------------------------------------------------
# Ego trajectory: simple straight-line forward motion.
# ---------------------------------------------------------------------------
def estimate_ego_trajectory_simple(
    n_frames: int = N_FRAMES,
    fps: int = FPS,
    forward_velocity_mps: float = 13.0,   # ~47 km/h city driving
) -> Tensor:
    """Return (n_frames, 4, 4) world<-camera poses at each frame.

    Assumes camera moves along +x at constant velocity, y=0, z=eye_height.
    Real ego trajectory has curves; for bootstrap we lean on photometric fit
    to refine this. Poses are camera-in-world (not view matrices).
    """
    dt = 1.0 / fps
    xs = torch.arange(n_frames, dtype=torch.float32) * (forward_velocity_mps * dt)

    poses = torch.zeros(n_frames, 4, 4)
    # Camera axes: +x forward, +y left, +z up.
    R = torch.tensor([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
    for i in range(n_frames):
        poses[i, 0:3, 0:3] = R
        poses[i, 0, 3] = xs[i]
        poses[i, 1, 3] = 0.0
        poses[i, 2, 3] = 1.5    # eye height
        poses[i, 3, 3] = 1.0
    return poses


def viewmat_from_camera_pose(pose_c2w: Tensor) -> Tensor:
    """Invert a camera-to-world pose to world-to-camera (what gsplat wants)."""
    if pose_c2w.dim() == 2:
        return torch.linalg.inv(pose_c2w)
    return torch.stack([torch.linalg.inv(p) for p in pose_c2w])


# ---------------------------------------------------------------------------
# Depth: Depth-Anything-V2 wrapper.
# ---------------------------------------------------------------------------
class DepthEstimator:
    """Thin wrapper around Depth-Anything-V2.  Downloads weights on first use.

    Depth values are RELATIVE (unitless); we scale to meters via a hand-tuned
    coefficient at unprojection time, since monocular depth has no absolute
    scale. The scale can be refined during photometric warmup.
    """

    def __init__(self, model_size: str = "small", device: str = "cuda"):
        self.device = torch.device(device)
        self.model_size = model_size
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        # transformers has Depth-Anything-V2 support.
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError(
                "transformers not installed. `pip install transformers` on Colab."
            ) from e
        model_id = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }[self.model_size]
        self._model = pipeline(task="depth-estimation", model=model_id,
                               device=self.device)

    def estimate(self, frame_hwc_uint8: Tensor) -> Tensor:
        """(H, W, 3) uint8 -> (H, W) relative depth (higher = farther)."""
        self._lazy_load()
        from PIL import Image
        img = Image.fromarray(frame_hwc_uint8.cpu().numpy())
        out = self._model(img)
        # Pipeline returns {"depth": PIL, "predicted_depth": Tensor}. Prefer tensor.
        d = out["predicted_depth"]
        if d.dim() == 3:
            d = d[0]
        # Resize back to original if the model output was downscaled.
        if d.shape != frame_hwc_uint8.shape[:2]:
            d = torch.nn.functional.interpolate(
                d[None, None].float(), size=frame_hwc_uint8.shape[:2],
                mode="bilinear", align_corners=False)[0, 0]
        return d.to(self.device)


# ---------------------------------------------------------------------------
# Unprojection: depth map -> 3D points in world frame.
# ---------------------------------------------------------------------------
def unproject_frame(
    depth_hw: Tensor,
    pose_c2w: Tensor,
    K: Tensor | None = None,
    stride: int = 8,
    depth_scale: float = 30.0,
) -> Tensor:
    """Convert a depth map into world-space 3D points.

    Args:
        depth_hw    : (H, W) relative depth from Depth-Anything.
        pose_c2w    : (4, 4) camera-in-world pose.
        K           : (3, 3) intrinsics; defaults to `constants.intrinsics()`.
        stride      : subsample factor (produces ~ HxW / stride^2 points).
        depth_scale : multiplier from unitless relative depth to meters.
            Monocular depth has no absolute scale; this is a rough hand-tune.

    Returns:
        (N, 3) tensor of world-space 3D points.
    """
    if K is None:
        K = torch.tensor(intrinsics(), device=depth_hw.device, dtype=depth_hw.dtype)

    H, W = depth_hw.shape
    ys, xs = torch.meshgrid(
        torch.arange(0, H, stride, device=depth_hw.device),
        torch.arange(0, W, stride, device=depth_hw.device),
        indexing="ij",
    )
    depth_sampled = depth_hw[ys, xs]                  # (H', W')
    valid = depth_sampled > 0
    depth_m = depth_sampled[valid] * depth_scale       # (N,)
    xs_valid = xs[valid].float()
    ys_valid = ys[valid].float()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Camera frame: +x right, +y down, +z forward (standard CV).  We convert
    # to the driving frame (+x forward, +y left, +z up) at the pose step.
    x_cam = (xs_valid - cx) * depth_m / fx
    y_cam = (ys_valid - cy) * depth_m / fy
    z_cam = depth_m
    ones = torch.ones_like(z_cam)
    p_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # (N, 4)

    # Camera-to-world.  pose_c2w is written in the DRIVING frame; we need to
    # rotate the CV-camera frame to align: CV +z (forward) -> driving +x,
    # CV +x (right) -> driving -y, CV +y (down) -> driving -z.
    cv_to_driving = torch.tensor([[0., 0., 1., 0.],
                                   [-1., 0., 0., 0.],
                                   [0., -1., 0., 0.],
                                   [0., 0., 0., 1.]],
                                 device=depth_hw.device, dtype=depth_hw.dtype)
    p_driving_cam = (cv_to_driving @ p_cam.T).T           # (N, 4)
    p_world = (pose_c2w @ p_driving_cam.T).T[:, :3]       # (N, 3)
    return p_world


# ---------------------------------------------------------------------------
# Distribute points across the four populations by geometry + semantics.
# ---------------------------------------------------------------------------
@dataclass
class DistributionConfig:
    road_z_max: float = 0.5         # points below this altitude count as road
    sky_r_min: float = 100.0        # points farther than this count as sky
    roadside_y_abs_min: float = 3.0 # points beyond this lateral distance = roadside
    # Semantic overrides (integer class ids from segnet).  If a point's segnet
    # class is in one of these sets, that's the assignment regardless of
    # geometry.  Empty set = pure geometry.
    road_classes: tuple[int, ...] = ()
    sky_classes: tuple[int, ...] = ()
    roadside_classes: tuple[int, ...] = ()
    actor_classes: tuple[int, ...] = ()


def distribute_points_to_populations(
    points_world: Tensor,
    class_labels: Tensor | None = None,
    cfg: DistributionConfig | None = None,
) -> dict[str, Tensor]:
    """Split a world-space point cloud into per-population buckets.

    Returns dict with keys road/sky/roadside_left/roadside_right/actor and
    (N_i, 3) tensors as values.  Empty populations may be present with (0, 3).
    """
    cfg = cfg or DistributionConfig()
    device = points_world.device
    N = points_world.shape[0]
    assign = torch.full((N,), -1, dtype=torch.int64, device=device)

    if class_labels is not None:
        for pop_id, ids in enumerate([cfg.road_classes, cfg.sky_classes,
                                       cfg.roadside_classes, cfg.actor_classes]):
            if ids:
                mask = torch.zeros_like(class_labels, dtype=torch.bool)
                for cid in ids:
                    mask |= (class_labels == cid)
                assign[mask] = pop_id

    # Geometry-based fallback for anything still unassigned.
    r = points_world.norm(dim=-1)
    unassigned = assign == -1
    is_sky = unassigned & (r > cfg.sky_r_min)
    is_road = unassigned & (~is_sky) & (points_world[:, 2] < cfg.road_z_max)
    is_roadside = unassigned & (~is_sky) & (~is_road) & (points_world[:, 1].abs() > cfg.roadside_y_abs_min)
    assign[is_sky] = 1
    assign[is_road] = 0
    assign[is_roadside] = 2
    # Anything still unassigned defaults to "actor" (movable / small object).
    assign[assign == -1] = 3

    def _sel(pid: int) -> Tensor:
        return points_world[assign == pid]

    road_pts = _sel(0)
    sky_pts = _sel(1)
    side_pts = _sel(2)
    actor_pts = _sel(3)
    roadside_left = side_pts[side_pts[:, 1] > 0] if side_pts.numel() else side_pts
    roadside_right = side_pts[side_pts[:, 1] < 0] if side_pts.numel() else side_pts

    return {
        "road": road_pts,
        "sky": sky_pts,
        "roadside_left": roadside_left,
        "roadside_right": roadside_right,
        "actor": actor_pts,
    }


# ---------------------------------------------------------------------------
# Seed a SceneRepresentation from distributed points.
# ---------------------------------------------------------------------------
def seed_scene_from_points(
    buckets: dict[str, Tensor],
    scene_length_m: float = 800.0,
    m_per_actor: int = 5,
    n_actor_clusters: int = 100,
    sky_radius: float = 500.0,
) -> SceneRepresentation:
    """Build a SceneRepresentation with each population's parameters seeded
    from `buckets`.  Where a bucket is empty, fall back to random init.
    """
    def _uv_from_xy(pts: Tensor) -> Tensor:
        return pts[:, :2].clone()

    road = RoadPlane(
        n=max(int(buckets["road"].shape[0]), 50),
        x_range=(0.0, scene_length_m),
        y_range=(-6.0, 6.0),
    )
    if buckets["road"].numel():
        n = min(road.n, buckets["road"].shape[0])
        with torch.no_grad():
            road.uv.data[:n] = _uv_from_xy(buckets["road"][:n])

    sky = SkyDome(n=max(int(buckets["sky"].shape[0]), 30), radius=sky_radius)
    if buckets["sky"].numel():
        n = min(sky.n, buckets["sky"].shape[0])
        pts = buckets["sky"][:n]
        # Convert world XYZ to sphere (theta, phi).
        unit = torch.nn.functional.normalize(pts, dim=-1)
        theta = torch.acos(unit[:, 2].clamp(-1, 1))
        phi = torch.atan2(unit[:, 1], unit[:, 0])
        with torch.no_grad():
            sky.dir.data[:n, 0] = theta
            sky.dir.data[:n, 1] = phi

    left = RoadsideBand(n=max(int(buckets["roadside_left"].shape[0]), 50),
                        s_range=(0.0, scene_length_m), side="left")
    if buckets["roadside_left"].numel():
        n = min(left.n, buckets["roadside_left"].shape[0])
        pts = buckets["roadside_left"][:n]
        with torch.no_grad():
            left.pos_band.data[:n, 0] = pts[:, 0]
            left.pos_band.data[:n, 1] = pts[:, 1].abs()
            left.pos_band.data[:n, 2] = pts[:, 2]

    right = RoadsideBand(n=max(int(buckets["roadside_right"].shape[0]), 50),
                         s_range=(0.0, scene_length_m), side="right")
    if buckets["roadside_right"].numel():
        n = min(right.n, buckets["roadside_right"].shape[0])
        pts = buckets["roadside_right"][:n]
        with torch.no_grad():
            right.pos_band.data[:n, 0] = pts[:, 0]
            right.pos_band.data[:n, 1] = pts[:, 1].abs()
            right.pos_band.data[:n, 2] = pts[:, 2]

    # DynamicActors: for bootstrap we cluster actor points by proximity and
    # spawn N_actor_clusters clusters with initial centers from cluster means.
    actors = []
    if buckets["actor"].shape[0] >= n_actor_clusters:
        pts = buckets["actor"]
        indices = torch.linspace(0, pts.shape[0] - 1,
                                 n_actor_clusters).long()
        centers = pts[indices]
        for c in centers:
            actor = DynamicActor(m_per_cluster=m_per_actor,
                                 t_range=(0.0, N_FRAMES / FPS),
                                 initial_center=tuple(c.tolist()))
            actors.append(actor)
    else:
        for i in range(n_actor_clusters):
            actors.append(DynamicActor(
                m_per_cluster=m_per_actor,
                t_range=(0.0, N_FRAMES / FPS),
                initial_center=(20.0 + 5 * i, 0.0, 1.0)))

    return SceneRepresentation(road=road, sky=sky,
                                roadside_left=left, roadside_right=right,
                                actors=actors)
