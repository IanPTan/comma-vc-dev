"""Encoder entrypoint: video -> trained scene -> archive.zip.

Ships as `submissions/<name>/compress.sh` -> python -m gsplat_video.compress ...

Pipeline:
  1. Load frames from videos/0.mkv (uses challenge_deps.frame_utils.AVVideoDataset).
  2. Bootstrap initial Gaussian positions from monocular SfM/depth (TODO).
  3. Train SceneRepresentation + EgoPoseINR against score-shaped loss.
  4. Quantize + entropy-code trained scene.
  5. Bundle into archive.zip alongside inflate.py + gsplat wheel dep spec.

Only the model artifacts + code needed at decode ship in the archive. Segnet,
posenet, COLMAP, Depth-Anything, and the training video itself are all
ENCODE-TIME ONLY and never shipped.
"""
from __future__ import annotations

import argparse
import io
import struct
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .codec import BitsConfig, GaussianCodec
from .constants import N_FRAMES
from .populations import (DynamicActor, RoadPlane, RoadsideBand,
                          SceneRepresentation, SkyDome)
from .pose_inr import EgoPoseINR
from .shader import ViewDependentShader


# ---------------------------------------------------------------------------
# Scene initialization (Phase-1 bootstrap will replace random init with
# COLMAP + Depth-Anything V2 seeded positions).
# ---------------------------------------------------------------------------
def make_initial_scene(
    n_road: int = 500,
    n_sky: int = 200,
    n_roadside: int = 2000,
    n_actors: int = 100,
    m_per_actor: int = 5,
    scene_length_m: float = 800.0,   # ~60s at ~50 km/h
) -> SceneRepresentation:
    return SceneRepresentation(
        road=RoadPlane(n=n_road, x_range=(0.0, scene_length_m), y_range=(-6.0, 6.0)),
        sky=SkyDome(n=n_sky, radius=500.0),
        roadside_left=RoadsideBand(n=n_roadside // 2, s_range=(0.0, scene_length_m),
                                   side="left"),
        roadside_right=RoadsideBand(n=n_roadside // 2, s_range=(0.0, scene_length_m),
                                    side="right"),
        actors=[DynamicActor(m_per_cluster=m_per_actor,
                             t_range=(0.0, N_FRAMES / 20),
                             initial_center=(20.0 + 5 * i, 0.0, 1.0))
                for i in range(n_actors)],
    )


# ---------------------------------------------------------------------------
# Serialization: pack trained scene + INRs into a single archive.
# ---------------------------------------------------------------------------
def _pack_nn(module: torch.nn.Module, num_bits: int = 8) -> tuple[bytes, dict]:
    """INT-N quantize an entire nn.Module's state dict, return bytes + header.

    Header maps parameter names to (shape, min, max, dtype='intN').
    """
    sd = module.state_dict()
    header: dict[str, dict] = {}
    payload = bytearray()
    for name, tensor in sd.items():
        arr = tensor.detach().cpu().float().numpy().flatten()
        xmin, xmax = float(arr.min()), float(arr.max())
        span = max(xmax - xmin, 1e-8)
        levels = 2 ** num_bits - 1
        q = np.clip(np.round((arr - xmin) / span * levels), 0, levels).astype(np.uint8)
        header[name] = {
            "shape": list(tensor.shape),
            "min": xmin, "max": xmax,
            "bits": num_bits, "count": arr.size,
        }
        payload.extend(q.tobytes())
    return bytes(payload), header


def _archive_bundle(
    scene_bytes: bytes,
    pose_bytes: bytes, pose_header: dict,
    shader_bytes: bytes, shader_header: dict,
    scene_cfg: dict,
) -> bytes:
    """Combine into a compressed zip. The zip *is* archive.zip."""
    import json
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("scene.bin", scene_bytes)
        zf.writestr("pose_inr.bin", pose_bytes)
        zf.writestr("pose_inr.json", json.dumps(pose_header))
        zf.writestr("shader.bin", shader_bytes)
        zf.writestr("shader.json", json.dumps(shader_header))
        zf.writestr("scene_cfg.json", json.dumps(scene_cfg))
    return buf.getvalue()


def encode_archive(
    scene: SceneRepresentation,
    pose_inr: EgoPoseINR,
    shader: ViewDependentShader | None,
    codec: GaussianCodec | None = None,
    output_path: Path | None = None,
) -> bytes:
    """Serialize a fully-trained model into the submission archive.

    The archive is what comma unzips before running inflate.sh.
    """
    codec = codec or GaussianCodec()

    raw = scene.gaussians(torch.tensor(0.0))   # dynamic actors evaluated at t=0
    scene_bytes = codec.encode_scene(raw)

    pose_bytes, pose_header = _pack_nn(pose_inr, num_bits=8)
    shader_bytes, shader_header = (b"", {}) if shader is None else _pack_nn(shader, num_bits=8)

    scene_cfg = {
        "bits": asdict(codec.bits),
        "n_actors": len(scene.actors),
        "m_per_actor": scene.actors[0].m if scene.actors else 0,
        "actor_traj_degree": scene.actors[0].degree if scene.actors else 0,
        "actor_t_range": [scene.actors[0].t_min, scene.actors[0].t_max] if scene.actors else [0, 0],
        "road_n": scene.road.n, "sky_n": scene.sky.n,
        "roadside_left_n": scene.roadside_left.n,
        "roadside_right_n": scene.roadside_right.n,
        "has_shader": shader is not None,
        # Config for reconstructing the tiny NNs at decode.
        "pose_inr": {"hidden": pose_inr.net[0].out_features,
                     "n_freqs": pose_inr.ff.n_freqs},
        "shader": {"hidden": shader.net[0].out_features} if shader is not None else None,
    }

    blob = _archive_bundle(scene_bytes, pose_bytes, pose_header,
                           shader_bytes, shader_header, scene_cfg)

    if output_path is not None:
        Path(output_path).write_bytes(blob)
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("videos/0.mkv"))
    ap.add_argument("--output", type=Path, default=Path("submissions/gsplat_v0/archive.zip"))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--skip-training", action="store_true",
                    help="Encode an untrained scene (useful for size/format testing).")
    args = ap.parse_args()

    scene = make_initial_scene()
    pose_inr = EgoPoseINR()
    shader = None  # skip shader for v0 to save bytes

    if not args.skip_training:
        # TODO Phase-2/3/4: actual training loop.
        # from .train import train, TrainConfig, FrameDataset
        # train(scene, pose_inr, FrameDataset(args.video), TrainConfig(steps=args.steps))
        raise NotImplementedError("Training pipeline not wired yet — use --skip-training "
                                  "to test the encode format only.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    blob = encode_archive(scene, pose_inr, shader, output_path=args.output)
    print(f"[compress] wrote {args.output} ({len(blob):,} bytes)")


if __name__ == "__main__":
    main()
