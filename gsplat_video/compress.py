"""Encoder entrypoint: video -> trained scene -> archive.zip.

Pipeline:
  1. Load frames from videos/0.mkv.
  2. (Optional) Bootstrap initial Gaussian positions from monocular depth.
  3. Train SceneRepresentation + EgoPoseINR against score-shaped loss.
  4. Quantize per-population + pack into archive.zip.

Segnet, posenet, COLMAP, Depth-Anything, and the training video itself are
all ENCODE-TIME ONLY and never shipped in the archive.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
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


def encode_archive(
    scene: SceneRepresentation,
    pose_inr: EgoPoseINR,
    shader: ViewDependentShader | None = None,
    codec: GaussianCodec | None = None,
    output_path: Path | None = None,
    n_frames: int = N_FRAMES,
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
        "n_frames": n_frames,
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


def encode_from_checkpoint(ckpt_path: Path, output_path: Path,
                           make_scene_fn=make_initial_scene) -> bytes:
    """Rebuild scene from a trained checkpoint and pack into archive.zip."""
    scene = make_scene_fn()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    scene.load_state_dict(ckpt["scene_state_dict"])
    pose_cfg = ckpt.get("pose_inr_config", {})
    pose_inr = EgoPoseINR(**{k: v for k, v in pose_cfg.items()
                             if k in {"hidden", "n_freqs"}})
    pose_inr.load_state_dict(ckpt["pose_inr_state_dict"])
    return encode_archive(scene, pose_inr, output_path=output_path)


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("videos/0.mkv"))
    ap.add_argument("--output", type=Path,
                    default=Path("submissions/gsplat_v0/archive.zip"))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--from-checkpoint", type=Path,
                    help="Skip training and encode from an existing ckpt_XXXXXX.pt.")
    ap.add_argument("--bootstrap-from", type=Path,
                    help="Pickle produced by scripts/bootstrap_scene.py.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/gsplat_train"))
    args = ap.parse_args()

    if args.from_checkpoint is not None:
        blob = encode_from_checkpoint(args.from_checkpoint, args.output)
        print(f"[compress] wrote {args.output} ({len(blob):,} bytes) from "
              f"{args.from_checkpoint}")
        return

    scene = make_initial_scene()
    pose_inr = EgoPoseINR()

    if args.skip_training:
        blob = encode_archive(scene, pose_inr, shader=None, output_path=args.output)
        print(f"[compress] wrote {args.output} ({len(blob):,} bytes) [untrained]")
        return

    from .dataset import VideoFrameDataset
    from .train import TrainConfig, train

    ds = VideoFrameDataset(args.video)
    cfg = TrainConfig(steps=args.steps, output_dir=args.output_dir)
    stats = train(scene, pose_inr, ds, cfg, bootstrap_from=args.bootstrap_from)

    blob = encode_archive(scene, pose_inr, output_path=args.output)
    print(f"[compress] wrote {args.output} ({len(blob):,} bytes)")
    print(f"[compress] final val: {stats.get('final_val')}")


def main() -> None:
    _cli()


if __name__ == "__main__":
    main()
