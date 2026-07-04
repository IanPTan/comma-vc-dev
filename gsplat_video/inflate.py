"""Decoder entrypoint. THIS is the file comma runs during evaluation.

Usage (from evaluate.sh):
    unzip archive.zip -d archive/
    bash inflate.sh    # -> python -m gsplat_video.inflate archive/ out/

Loads Gaussian primitives + tiny INR weights from archive.zip, rasterizes every
frame via gsplat, writes decoded frames to `out/`.

We do NOT reconstruct SceneRepresentation / populations at decode: the
rasterizer accepts raw Gaussian dicts directly. Population structure is a
training-time abstraction; at inference we just need the primitives.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from .codec import BitsConfig, GaussianCodec
from .constants import (CAMERA_H, CAMERA_W, FPS, N_FRAMES, intrinsics)
from .pose_inr import EgoPoseINR
from .shader import ViewDependentShader


# ---------------------------------------------------------------------------
# Unpack routine (inverse of compress._pack_nn).
# ---------------------------------------------------------------------------
def _unpack_nn(module: torch.nn.Module, payload: bytes, header: dict) -> None:
    p = 0
    sd = module.state_dict()
    for name, meta in header.items():
        count = int(meta["count"])
        bits = int(meta["bits"])
        assert bits == 8, "only INT8 unpack implemented"
        q = np.frombuffer(payload[p:p + count], dtype=np.uint8).astype(np.float32)
        p += count
        span = max(meta["max"] - meta["min"], 1e-8)
        arr = meta["min"] + q / (2 ** bits - 1) * span
        sd[name].copy_(torch.from_numpy(arr).reshape(meta["shape"]))
    module.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Load an archive into rasterizer-ready state.
# ---------------------------------------------------------------------------
def load_from_archive(archive_bytes: bytes) -> tuple[dict[str, torch.Tensor],
                                                     EgoPoseINR,
                                                     ViewDependentShader | None,
                                                     dict]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        scene_blob = zf.read("scene.bin")
        pose_blob = zf.read("pose_inr.bin")
        pose_header = json.loads(zf.read("pose_inr.json"))
        shader_blob = zf.read("shader.bin")
        shader_header = json.loads(zf.read("shader.json"))
        scene_cfg = json.loads(zf.read("scene_cfg.json"))

    codec = GaussianCodec(BitsConfig(**scene_cfg["bits"]))
    flat = codec.decode_scene(scene_blob)

    # Static Gaussians are already activation-ready (encoder wrote raw
    # pre-activation values). Convert to tensors; rasterizer applies
    # softplus/sigmoid/normalize on the raw values.
    gaussians = {k: torch.from_numpy(v) for k, v in flat.items()}

    pose_cfg = scene_cfg.get("pose_inr", {})
    pose_inr = EgoPoseINR(**{k: v for k, v in pose_cfg.items()
                             if k in {"hidden", "n_freqs"}})
    _unpack_nn(pose_inr, pose_blob, pose_header)

    shader: ViewDependentShader | None = None
    if scene_cfg.get("has_shader", False):
        shader_cfg = scene_cfg.get("shader") or {}
        shader = ViewDependentShader(**{k: v for k, v in shader_cfg.items() if k == "hidden"})
        _unpack_nn(shader, shader_blob, shader_header)

    return gaussians, pose_inr, shader, scene_cfg


# ---------------------------------------------------------------------------
# Rendering loop.
# ---------------------------------------------------------------------------
def render_all_frames(
    gaussians: dict[str, torch.Tensor],
    pose_inr: EgoPoseINR,
    output_dir: Path,
    n_frames: int = N_FRAMES,
    device: str = "cuda",
) -> None:
    from .rasterizer import render

    output_dir.mkdir(parents=True, exist_ok=True)
    K = torch.tensor(intrinsics(), dtype=torch.float32, device=device)
    pose_inr.to(device)
    g_dev = {k: v.to(device) for k, v in gaussians.items()}

    for i in range(n_frames):
        t_norm = torch.tensor(i / (n_frames - 1), device=device)
        viewmat = pose_inr.viewmat(t_norm)
        image = render(g_dev, None, viewmat, K, CAMERA_W, CAMERA_H)
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

    gaussians, pose_inr, shader, cfg = load_from_archive(blob)
    render_all_frames(gaussians, pose_inr, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
