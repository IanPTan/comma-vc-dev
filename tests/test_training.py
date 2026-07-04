"""Training-loop unit tests.

Focus on the pieces that DON'T require gsplat (checkpointing, LR schedule,
config plumbing). Full loop test is impossible locally without a working
rasterizer.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from gsplat_video.compress import encode_from_checkpoint, make_initial_scene
from gsplat_video.pose_inr import EgoPoseINR
from gsplat_video.train import (TrainConfig, _lr_at, load_checkpoint,
                                save_checkpoint)


def test_lr_schedule_constant_during_warmup():
    lrs = [_lr_at(s, warmup=1000, total=10000, base=0.01) for s in range(0, 1000, 100)]
    assert all(abs(l - 0.01) < 1e-9 for l in lrs), \
        f"expected 0.01 throughout warmup, got {lrs}"
    print("[ok] LR constant at 0.01 during warmup")


def test_lr_schedule_cosine_after_warmup():
    end_lr = _lr_at(10000, warmup=1000, total=10000, base=0.01)
    mid_lr = _lr_at(5500, warmup=1000, total=10000, base=0.01)
    assert 0.004 < mid_lr < 0.006, f"midpoint LR {mid_lr} not near half-decay"
    assert end_lr < 0.001, f"end LR {end_lr} not near 0"
    print(f"[ok] cosine decay: mid={mid_lr:.5f}, end={end_lr:.5f}")


def test_checkpoint_roundtrip():
    torch.manual_seed(0)
    scene = make_initial_scene(n_road=20, n_sky=10, n_roadside=20, n_actors=5)
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)
    opt = torch.optim.AdamW(list(scene.parameters()) + list(pose_inr.parameters()),
                            lr=1e-3)
    cfg = TrainConfig(output_dir=Path("/tmp/gs_ckpt_test"))

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "test_ckpt.pt"
        save_checkpoint(scene, pose_inr, opt, step=1234, cfg=cfg, path=ck)

        scene2 = make_initial_scene(n_road=20, n_sky=10, n_roadside=20, n_actors=5)
        pose2 = EgoPoseINR(hidden=16, n_freqs=4)
        opt2 = torch.optim.AdamW(list(scene2.parameters()) + list(pose2.parameters()),
                                 lr=1e-3)
        step = load_checkpoint(scene2, pose2, opt2, ck)
        assert step == 1234

        # Verify weights match.
        for (n1, p1), (n2, p2) in zip(scene.state_dict().items(),
                                     scene2.state_dict().items()):
            assert torch.equal(p1, p2), f"scene param {n1} mismatch"
        for (n1, p1), (n2, p2) in zip(pose_inr.state_dict().items(),
                                     pose2.state_dict().items()):
            assert torch.equal(p1, p2), f"pose param {n1} mismatch"
    print("[ok] checkpoint save/load round-trip preserves all weights + step")


def test_encode_from_checkpoint():
    """Simulate: save ckpt during 'training', then encode from it."""
    torch.manual_seed(1)
    scene = make_initial_scene(n_road=20, n_sky=10, n_roadside=20, n_actors=5)
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)
    opt = torch.optim.AdamW(list(scene.parameters()) + list(pose_inr.parameters()),
                            lr=1e-3)

    def _small_scene():
        return make_initial_scene(n_road=20, n_sky=10, n_roadside=20, n_actors=5)

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ckpt.pt"
        arc = Path(d) / "archive.zip"
        save_checkpoint(scene, pose_inr, opt, step=100,
                        cfg=TrainConfig(), path=ck)
        blob = encode_from_checkpoint(ck, arc, make_scene_fn=_small_scene)
        assert arc.exists() and len(blob) > 0
        print(f"[ok] encode_from_checkpoint: {len(blob)} bytes archive")


def main() -> None:
    tests = [
        test_lr_schedule_constant_during_warmup,
        test_lr_schedule_cosine_after_warmup,
        test_checkpoint_roundtrip,
        test_encode_from_checkpoint,
    ]
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} training tests passed.")


if __name__ == "__main__":
    main()
