"""Decoder entrypoint. THIS is the file comma runs during evaluation.

Loads per-population parameters + tiny INR weights from archive.zip,
rebuilds a SceneRepresentation with the correct time-varying trajectories,
rasterizes every frame via gsplat, writes decoded frames to `out/`.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from .codec import unpack_module, unpack_scene
from .constants import (CAMERA_H, CAMERA_W, N_FRAMES, intrinsics)
from .populations import SceneRepresentation
from .pose_inr import EgoPoseINR
from .shader import ViewDependentShader


def load_from_archive(archive_bytes: bytes) -> tuple[SceneRepresentation,
                                                     EgoPoseINR,
                                                     ViewDependentShader | None,
                                                     dict]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        pops_blob = zf.read("populations.bin")
        pose_blob = zf.read("pose_inr.bin")
        pose_header = json.loads(zf.read("pose_inr.json"))
        shader_blob = zf.read("shader.bin")
        shader_header = json.loads(zf.read("shader.json"))
        scene_cfg = json.loads(zf.read("scene_cfg.json"))

    scene = unpack_scene(pops_blob, scene_cfg["scene_header"])

    pose_cfg = scene_cfg.get("pose_inr", {})
    pose_inr = EgoPoseINR(**{k: v for k, v in pose_cfg.items()
                             if k in {"hidden", "n_freqs"}})
    unpack_module(pose_inr, pose_blob, pose_header)

    shader: ViewDependentShader | None = None
    if scene_cfg.get("has_shader", False):
        shader_cfg = scene_cfg.get("shader") or {}
        shader = ViewDependentShader(**{k: v for k, v in shader_cfg.items()
                                        if k == "hidden"})
        unpack_module(shader, shader_blob, shader_header)

    return scene, pose_inr, shader, scene_cfg


def render_all_frames(
    scene: SceneRepresentation,
    pose_inr: EgoPoseINR,
    output_dir: Path,
    n_frames: int = N_FRAMES,
    device: str = "cuda",
) -> None:
    from .rasterizer import render

    output_dir.mkdir(parents=True, exist_ok=True)
    K = torch.tensor(intrinsics(), dtype=torch.float32, device=device)
    scene.to(device)
    pose_inr.to(device)

    for i in range(n_frames):
        t_norm = torch.tensor(i / max(n_frames - 1, 1), device=device)
        viewmat = pose_inr.viewmat(t_norm)
        image = render(scene, t_norm, viewmat, K, CAMERA_W, CAMERA_H)
        arr = (image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        np.save(output_dir / f"frame_{i:05d}.npy", arr)
        if i % 100 == 0:
            print(f"[inflate] rendered {i}/{n_frames}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.archive_dir.is_file() and args.archive_dir.suffix == ".zip":
        blob = args.archive_dir.read_bytes()
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for p in args.archive_dir.iterdir():
                if p.is_file():
                    zf.writestr(p.name, p.read_bytes())
        blob = buf.getvalue()

    scene, pose_inr, _, _ = load_from_archive(blob)
    render_all_frames(scene, pose_inr, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
