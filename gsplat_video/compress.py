"""Encoder entrypoint: video -> trained scene -> archive.zip.

Pipeline:
  1. Load frames from videos/0.mkv.
  2. Bootstrap initial Gaussian positions (Phase 1 gap: currently random init).
  3. Train SceneRepresentation + EgoPoseINR against score-shaped loss.
  4. Quantize per-population + pack into archive.zip.

Segnet, posenet, COLMAP, Depth-Anything, and the training video itself are
all ENCODE-TIME ONLY and never shipped in the archive.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from .codec import BitsConfig, GaussianCodec, pack_module, pack_scene
from .constants import N_FRAMES
from .populations import (DynamicActor, RoadPlane, RoadsideBand,
                          SceneRepresentation, SkyDome)
from .pose_inr import EgoPoseINR
from .shader import ViewDependentShader


# ---------------------------------------------------------------------------
# Scene initialization.  Phase 1 will replace random init with COLMAP +
# Depth-Anything-V2 seeded positions.
# ---------------------------------------------------------------------------
def make_initial_scene(
    n_road: int = 500,
    n_sky: int = 200,
    n_roadside: int = 2000,
    n_actors: int = 100,
    m_per_actor: int = 5,
    scene_length_m: float = 800.0,
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
# Archive bundling.
# ---------------------------------------------------------------------------
def encode_archive(
    scene: SceneRepresentation,
    pose_inr: EgoPoseINR,
    shader: ViewDependentShader | None = None,
    codec: GaussianCodec | None = None,
    output_path: Path | None = None,
) -> bytes:
    codec = codec or GaussianCodec()

    scene_payload, scene_header = pack_scene(scene, num_bits=8)
    pose_payload, pose_header = pack_module(pose_inr, num_bits=8)
    shader_payload, shader_header = ((b"", {}) if shader is None
                                     else pack_module(shader, num_bits=8))

    scene_cfg = {
        "bits": asdict(codec.bits),
        "scene_header": scene_header,
        "pose_inr": {"hidden": pose_inr.net[0].out_features,
                     "n_freqs": pose_inr.ff.n_freqs},
        "shader": ({"hidden": shader.net[0].out_features}
                   if shader is not None else None),
        "has_shader": shader is not None,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("populations.bin", scene_payload)
        zf.writestr("pose_inr.bin", pose_payload)
        zf.writestr("pose_inr.json", json.dumps(pose_header))
        zf.writestr("shader.bin", shader_payload)
        zf.writestr("shader.json", json.dumps(shader_header))
        zf.writestr("scene_cfg.json", json.dumps(scene_cfg))
    blob = buf.getvalue()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(blob)
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("videos/0.mkv"))
    ap.add_argument("--output", type=Path,
                    default=Path("submissions/gsplat_v0/archive.zip"))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--skip-training", action="store_true",
                    help="Encode an untrained scene (useful for format testing).")
    args = ap.parse_args()

    scene = make_initial_scene()
    pose_inr = EgoPoseINR()

    if not args.skip_training:
        raise NotImplementedError(
            "Training pipeline not wired yet — use --skip-training to test the "
            "encode format only.")

    blob = encode_archive(scene, pose_inr, shader=None, output_path=args.output)
    print(f"[compress] wrote {args.output} ({len(blob):,} bytes)")


if __name__ == "__main__":
    main()
