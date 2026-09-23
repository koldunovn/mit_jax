"""Plan Task 20 (tier 2, GPU node with >= 4 GPUs): the tile-sharded model on 4 GPUs equals the 1-GPU model bitwise
over a day, two 1-GPU runs are bitwise identical (the GPU floor), and the sharded gradient agrees with the 1-GPU one
within the GPU gradient repeat floor. Runs scripts/runs/gpu_sharding.py's logic.
Recorded 2026-09-23 (4x A100-80, job 27640194): floor 0, 4-GPU vs 1-GPU forward 0 on all fields after 24 steps,
gradient rel 1.5e-11 (repeat floor ~4e-11, Task 18); 0.337 s/step on 1 GPU, 0.429 s/step on 4 GPUs."""

import importlib.util
from pathlib import Path

import jax
import pytest

REPO = Path(__file__).resolve().parents[2]


def test_gpu_sharding_bitwise(capsys):
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    assert len(gpus) >= 4, f"tier-2 GPU test needs 4 GPUs, found {jax.devices()}"
    spec = importlib.util.spec_from_file_location("gs", REPO / "scripts" / "runs" / "gpu_sharding.py")
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    assert gs.main(["--nsteps", "6", "--nproc", "4"]) == 0
    out = capsys.readouterr().out
    floor = [l for l in out.splitlines() if l.startswith("GPU floor")][0]
    shard = [l for l in out.splitlines() if "sharded vs 1-GPU" in l][0]
    grad = [l for l in out.splitlines() if l.startswith("gradient")][0]
    assert all(v.endswith("0.0e+00") for v in floor.split(": ", 1)[1].split(", ")), floor
    assert all(v.endswith("0.0e+00") for v in shard.split(": ", 1)[1].split(", ")), shard
    assert float(grad.split("rel ")[1].split(";")[0]) < 1e-10, grad
