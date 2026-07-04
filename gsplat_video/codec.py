"""Codec: per-population quantization + differentiable rate estimation.

Each Gaussian population owns its own native parameter tensors (RoadPlane.uv,
SkyDome.dir, etc.); DynamicActor also carries trajectory coefficients so its
Gaussians move with time.

Rather than projecting everything to a single flat Gaussian block at t=0
(which would lose actor motion), the codec serializes each population's
state_dict separately and stores the constructor args to rebuild them at
decode.  Rendering at time t then calls `scene.gaussians(t)` which reproduces
the correct time-varying positions.

Archive layout:
    scene_cfg.json    : population sizes + constructor args + bits config
    populations.bin   : concatenated INT8-quantized state dicts + shapes
    pose_inr.{bin,json} : tiny MLP weights
    shader.{bin,json}   : tiny MLP weights, optional
"""
from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

# From the baseline evaluate.sh report.
BASELINE_ORIGINAL_BYTES = 37_545_489

# Empirical estimate: brotli/zstd shaves ~25% off an INT8 stream. Tune once we
# have real archive sizes.
ENTROPY_CODING_COMPRESSION_FACTOR = 0.75


@dataclass
class BitsConfig:
    means: int = 12
    scales: int = 8
    quats: int = 8
    opacities: int = 6
    colors: int = 8

    def bits_per_gaussian(self) -> int:
        return (3 * self.means + 3 * self.scales + 4 * self.quats
                + self.opacities + 3 * self.colors)


@dataclass
class TensorRange:
    min: float
    max: float


# ---------------------------------------------------------------------------
# STE quantization (used during training so the loss sees quantized values).
# ---------------------------------------------------------------------------
def ste_quantize(x: Tensor, num_bits: int,
                 x_min: float | None = None,
                 x_max: float | None = None) -> tuple[Tensor, TensorRange]:
    """Straight-through-estimator uniform quantization."""
    if x_min is None:
        x_min = x.detach().min().item()
    if x_max is None:
        x_max = x.detach().max().item()
    x_min_t = torch.tensor(x_min, dtype=x.dtype, device=x.device)
    x_max_t = torch.tensor(x_max, dtype=x.dtype, device=x.device)
    span = (x_max_t - x_min_t).clamp_min(1e-8)
    levels = 2 ** num_bits - 1

    x_norm = (x - x_min_t) / span
    q = torch.round(x_norm * levels).clamp(0, levels)
    x_hat = x_min_t + (q / levels) * span

    return x + (x_hat - x).detach(), TensorRange(min=float(x_min), max=float(x_max))


# ---------------------------------------------------------------------------
# Module-level pack / unpack (works for any nn.Module).
# ---------------------------------------------------------------------------
def pack_module(module: nn.Module, num_bits: int = 8) -> tuple[bytes, dict]:
    """INT-N quantize an entire nn.Module state_dict.

    Returns (payload_bytes, header_dict). Only num_bits == 8 is implemented
    at the packing level; other widths would need bit-packing (see the
    Gaussian-attribute path further down for that treatment).
    """
    assert num_bits == 8, "only INT8 pack_module implemented; extend if needed"
    header: dict[str, dict] = {}
    payload = bytearray()
    for name, tensor in module.state_dict().items():
        arr = tensor.detach().cpu().float().numpy().flatten()
        xmin, xmax = float(arr.min()), float(arr.max())
        span = max(xmax - xmin, 1e-8)
        levels = 2 ** num_bits - 1
        q = np.clip(np.round((arr - xmin) / span * levels), 0, levels).astype(np.uint8)
        header[name] = {"shape": list(tensor.shape), "min": xmin, "max": xmax,
                        "bits": num_bits, "count": int(arr.size)}
        payload.extend(q.tobytes())
    return bytes(payload), header


def unpack_module(module: nn.Module, payload: bytes, header: dict) -> None:
    p = 0
    sd = module.state_dict()
    for name, meta in header.items():
        count = int(meta["count"])
        bits = int(meta["bits"])
        assert bits == 8
        q = np.frombuffer(payload[p:p + count], dtype=np.uint8).astype(np.float32)
        p += count
        span = max(meta["max"] - meta["min"], 1e-8)
        arr = meta["min"] + q / (2 ** bits - 1) * span
        sd[name].copy_(torch.from_numpy(arr).reshape(meta["shape"]))
    module.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Rate estimation used inside the training loss.
# ---------------------------------------------------------------------------
class GaussianCodec:
    """Rate accountant.  Uses BitsConfig to translate Gaussian count into a
    projected archive size, so the training loss can penalize models that
    would encode to too many bytes.
    """

    def __init__(self, bits: BitsConfig | None = None,
                 original_bytes: int = BASELINE_ORIGINAL_BYTES):
        self.bits = bits or BitsConfig()
        self.original_bytes = original_bytes

    def rate(self, n_gaussians: int, extra_bytes: int = 0,
             include_ec_factor: bool = True) -> float:
        bits = n_gaussians * self.bits.bits_per_gaussian()
        raw_bytes = (bits + 7) // 8 + extra_bytes
        est_bytes = raw_bytes * ENTROPY_CODING_COMPRESSION_FACTOR if include_ec_factor else raw_bytes
        return est_bytes / self.original_bytes


# ---------------------------------------------------------------------------
# Helpers to serialize a whole SceneRepresentation as structured populations.
# ---------------------------------------------------------------------------
def pack_scene(scene, num_bits: int = 8) -> tuple[bytes, dict]:
    """Serialize each population's parameters separately.

    Returns (payload, header) where header maps population name to
    {constructor_args, module_header}.
    """
    parts: list[bytes] = []
    header: dict[str, Any] = {"num_bits": num_bits, "populations": {}}
    for name in ("road", "sky", "roadside_left", "roadside_right"):
        pop = getattr(scene, name)
        payload_i, hdr_i = pack_module(pop, num_bits)
        parts.append(payload_i)
        header["populations"][name] = {
            "ctor": _ctor_args(pop),
            "kind": type(pop).__name__,
            "module_header": hdr_i,
            "payload_len": len(payload_i),
        }
    # Actors are a list -> pack each with an index in the name.
    header["populations"]["actors"] = []
    for i, actor in enumerate(scene.actors):
        payload_i, hdr_i = pack_module(actor, num_bits)
        parts.append(payload_i)
        header["populations"]["actors"].append({
            "ctor": _ctor_args(actor),
            "kind": type(actor).__name__,
            "module_header": hdr_i,
            "payload_len": len(payload_i),
        })
    return b"".join(parts), header


def unpack_scene(payload: bytes, header: dict):
    """Rebuild a SceneRepresentation from a packed scene payload."""
    from .populations import (DynamicActor, RoadPlane, RoadsideBand,
                              SceneRepresentation, SkyDome)

    KIND_MAP = {"RoadPlane": RoadPlane, "SkyDome": SkyDome,
                "RoadsideBand": RoadsideBand, "DynamicActor": DynamicActor}
    p = 0

    def _next(pop_header: dict, cls_map=KIND_MAP):
        nonlocal p
        cls = cls_map[pop_header["kind"]]
        pop = cls(**_normalize_ctor(pop_header["ctor"]))
        length = pop_header["payload_len"]
        unpack_module(pop, payload[p:p + length], pop_header["module_header"])
        p += length
        return pop

    road = _next(header["populations"]["road"])
    sky = _next(header["populations"]["sky"])
    left = _next(header["populations"]["roadside_left"])
    right = _next(header["populations"]["roadside_right"])
    actors = [_next(a) for a in header["populations"]["actors"]]

    return SceneRepresentation(road=road, sky=sky, roadside_left=left,
                               roadside_right=right, actors=actors)


# ---------------------------------------------------------------------------
# Constructor-args extraction / normalization for each population type.
# ---------------------------------------------------------------------------
def _ctor_args(pop) -> dict:
    """Extract the constructor args we need to rebuild `pop`.

    We don't try to be fully generic — just cover the four population types.
    """
    from .populations import (DynamicActor, RoadPlane, RoadsideBand, SkyDome)
    if isinstance(pop, RoadPlane):
        return {"n": pop.n, "x_range": list(pop.x_range), "y_range": list(pop.y_range)}
    if isinstance(pop, SkyDome):
        return {"n": pop.n, "radius": pop.radius}
    if isinstance(pop, RoadsideBand):
        return {"n": pop.n, "s_range": list(pop.s_range),
                "w_range": list(pop.w_range), "h_range": list(pop.h_range),
                "side": "left" if pop.side_sign > 0 else "right"}
    if isinstance(pop, DynamicActor):
        return {"m_per_cluster": pop.m, "traj_degree": pop.degree,
                "t_range": [pop.t_min, pop.t_max]}
    raise TypeError(f"unknown population type {type(pop)!r}")


def _normalize_ctor(d: dict) -> dict:
    """JSON round-trip converts tuples to lists; convert range fields back."""
    d = dict(d)
    for k in ("x_range", "y_range", "s_range", "w_range", "h_range", "t_range"):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return d
