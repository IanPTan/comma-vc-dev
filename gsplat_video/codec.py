"""Codec: quantization, differentiable rate estimation, byte-level encode/decode.

The archive is dominated by Gaussian parameters. We quantize each parameter
group with its own bit width (chosen by sensitivity, similar to LightGaussian),
STE for gradient flow during training, and pack to bytes at export time.

Rate model for training:
    - Uniform-bits rate = N_gaussians * bits_per_gaussian.  Constant w.r.t. the
      Gaussian *values*, so this term does not directly train the Gaussian
      values — but it DOES train the number of Gaussians via opacity-based
      pruning (small opacity -> Gaussian gets dropped, reducing N).
    - Compression factor accounts for downstream entropy coding (brotli/zstd
      typically achieve ~30% reduction on quantized weight streams).

Archive layout (v0):
    header: BitsConfig + population sizes + per-tensor (min, max) ranges
    payload: bit-packed quantized values, concatenated by group

TODO: swap uniform bits for a learned entropy model once we have real weights
to profile. That gives us a differentiable rate that also trains the values.
"""
from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field, fields
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

# From the baseline evaluate.sh report: uncompressed size of videos/0.mkv
BASELINE_ORIGINAL_BYTES = 37_545_489

# Empirical assumption: brotli/zstd shave ~25% off a uniformly-quantized stream.
# Tune once we measure real archive sizes.
ENTROPY_CODING_COMPRESSION_FACTOR = 0.75


@dataclass
class BitsConfig:
    means: int = 12       # 3D positions need more precision
    scales: int = 8       # log-scales; 256 levels plenty
    quats: int = 8        # unit quaternions
    opacities: int = 6    # bimodal distribution, few bits enough
    colors: int = 8       # 8-bit RGB standard

    def bits_per_gaussian(self) -> int:
        return (3 * self.means + 3 * self.scales + 4 * self.quats
                + self.opacities + 3 * self.colors)


@dataclass
class TensorRange:
    min: float
    max: float


def ste_quantize(x: Tensor, num_bits: int,
                 x_min: Tensor | float | None = None,
                 x_max: Tensor | float | None = None) -> tuple[Tensor, TensorRange]:
    """Straight-through-estimator uniform quantization.

    Returns the STE-quantized tensor (forward: quantized value, backward:
    identity through) and the range used, so the encoder can save it.
    """
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


class GaussianCodec:
    """Per-group quantization + rate estimation + byte round-trip.

    Instances are stateless w.r.t. optimization — they read/write raw tensors.
    A codec instance IS shipped in the archive (as the BitsConfig header), so
    the decoder knows how many bits per group and what ranges to unpack.
    """

    def __init__(self, bits: BitsConfig | None = None,
                 original_bytes: int = BASELINE_ORIGINAL_BYTES):
        self.bits = bits or BitsConfig()
        self.original_bytes = original_bytes

    # ------------------------------------------------------------------
    # Rate estimation for the training loss.
    # ------------------------------------------------------------------
    def rate(self, n_gaussians: int,
             extra_bytes: int = 0,
             include_ec_factor: bool = True) -> float:
        """Rate = archive_bytes / original_bytes, as a plain float.

        n_gaussians   : total Gaussians in the scene
        extra_bytes   : bytes for pose INR + shader + codebooks + headers
        include_ec_factor: multiply by ENTROPY_CODING_COMPRESSION_FACTOR
        """
        bits = n_gaussians * self.bits.bits_per_gaussian()
        raw_bytes = (bits + 7) // 8 + extra_bytes
        est_bytes = raw_bytes * ENTROPY_CODING_COMPRESSION_FACTOR if include_ec_factor else raw_bytes
        return est_bytes / self.original_bytes

    # ------------------------------------------------------------------
    # STE quantization applied to a scene's raw Gaussian dict.
    # ------------------------------------------------------------------
    def quantize_scene(self, raw: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, TensorRange]]:
        """Return (quantized_params, per-group ranges).

        `raw` must be the dict from SceneRepresentation.gaussians(t) — raw
        pre-activation parameters. Ranges come out per-group as a single
        TensorRange each (per-tensor quantization for simplicity).
        """
        q, ranges = {}, {}
        for name, bits in [("means", self.bits.means),
                          ("scales", self.bits.scales),
                          ("quats", self.bits.quats),
                          ("opacities", self.bits.opacities),
                          ("colors", self.bits.colors)]:
            q[name], ranges[name] = ste_quantize(raw[name], bits)
        return q, ranges

    # ------------------------------------------------------------------
    # Byte packing for the actual archive.
    # ------------------------------------------------------------------
    def encode_scene(self, raw: dict[str, Tensor]) -> bytes:
        """Serialize a scene to bytes. Header + bit-packed values.

        Layout:
            uint8   version = 0
            5 x uint8 : bits per group (means, scales, quats, opacities, colors)
            5 x (float32 min, float32 max) : per-group ranges
            uint32 : n_gaussians
            uint32 : payload byte length
            bytes  : packed payload
        """
        n = raw["means"].shape[0]
        header = bytearray()
        header.append(0)  # version
        header.extend(struct.pack("BBBBB",
                                  self.bits.means, self.bits.scales,
                                  self.bits.quats, self.bits.opacities,
                                  self.bits.colors))

        payload_bits: list[int] = []
        for name, bits in [("means", self.bits.means),
                          ("scales", self.bits.scales),
                          ("quats", self.bits.quats),
                          ("opacities", self.bits.opacities),
                          ("colors", self.bits.colors)]:
            x = raw[name].detach().cpu()
            xmin, xmax = float(x.min()), float(x.max())
            header.extend(struct.pack("ff", xmin, xmax))
            span = max(xmax - xmin, 1e-8)
            levels = 2 ** bits - 1
            x_norm = ((x - xmin) / span).clamp(0, 1)
            q = torch.round(x_norm * levels).to(torch.int64).flatten().tolist()
            payload_bits.extend(_int_to_bits(v, bits) for v in q)

        header.extend(struct.pack("II", n, sum(len(b) for b in payload_bits)))
        payload_bytes = _pack_bits(_flatten(payload_bits))
        return bytes(header) + payload_bytes

    def decode_scene(self, blob: bytes) -> dict[str, np.ndarray]:
        """Inverse of encode_scene. Returns numpy arrays keyed by group."""
        p = 0
        version = blob[p]; p += 1
        assert version == 0, f"unknown codec version {version}"
        bits_means, bits_scales, bits_quats, bits_opac, bits_cols = \
            struct.unpack_from("BBBBB", blob, p)
        p += 5
        ranges: dict[str, tuple[float, float]] = {}
        for name in ("means", "scales", "quats", "opacities", "colors"):
            xmin, xmax = struct.unpack_from("ff", blob, p)
            p += 8
            ranges[name] = (xmin, xmax)
        n, _ = struct.unpack_from("II", blob, p)
        p += 8
        payload = blob[p:]

        # Read bits in the same order we wrote them.
        stream = _unpack_bits(payload)
        cursor = [0]
        out: dict[str, np.ndarray] = {}
        for name, bits, shape in [
                ("means", bits_means, (n, 3)),
                ("scales", bits_scales, (n, 3)),
                ("quats", bits_quats, (n, 4)),
                ("opacities", bits_opac, (n,)),
                ("colors", bits_cols, (n, 3)),
                ]:
            flat_count = int(np.prod(shape))
            values = np.empty(flat_count, dtype=np.float32)
            xmin, xmax = ranges[name]
            span = max(xmax - xmin, 1e-8)
            levels = 2 ** bits - 1
            for i in range(flat_count):
                v = _read_bits(stream, cursor, bits)
                values[i] = xmin + (v / levels) * span
            out[name] = values.reshape(shape)
        return out


# ---------------------------------------------------------------------------
# Bit-packing helpers.
# ---------------------------------------------------------------------------
def _int_to_bits(v: int, n: int) -> list[int]:
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def _flatten(list_of_lists: Iterable[list[int]]) -> list[int]:
    out: list[int] = []
    for xs in list_of_lists:
        out.extend(xs)
    return out


def _pack_bits(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def _unpack_bits(blob: bytes) -> bytes:
    return blob


def _read_bits(stream: bytes, cursor: list[int], n: int) -> int:
    v = 0
    for _ in range(n):
        i = cursor[0]
        bit = (stream[i >> 3] >> (7 - (i & 7))) & 1
        v = (v << 1) | bit
        cursor[0] += 1
    return v
