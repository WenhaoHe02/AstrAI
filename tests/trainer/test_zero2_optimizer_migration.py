import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from scripts.tools.train import MatrixAwareOptimizer


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Parameter(torch.zeros(3))
        self.expert = nn.Parameter(torch.zeros(1, 2, 2))
        self.right = nn.Parameter(torch.zeros(5))


class _ToyFusedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_right = nn.Parameter(torch.zeros(8))
        self.expert = nn.Parameter(torch.zeros(1, 2, 2))


class _ToyFusedExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Parameter(torch.zeros(4))
        self.expert_up_gate = nn.Parameter(torch.zeros(1, 4, 2))
        self.expert_down = nn.Parameter(torch.zeros(1, 2, 2))


def _migration_worker(rank: int, init_file: str, fused: bool):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{init_file}"
    )
    model = _ToyFusedModel() if fused else _ToyModel()
    optimizer = MatrixAwareOptimizer(model, enable_muon=False)
    param_groups = optimizer.adamw.state_dict()["param_groups"]
    step = torch.tensor(7.0)
    expert_value = 10.0 + rank
    if rank == 0:
        state = {
            0: {
                "step": step,
                "exp_avg": torch.tensor([0.0, 1.0, 2.0]),
                "exp_avg_sq": torch.tensor([100.0, 101.0, 102.0]),
            },
            1: {
                "step": step,
                "exp_avg": torch.full((1, 2, 2), expert_value),
                "exp_avg_sq": torch.full((1, 2, 2), expert_value + 100),
            },
            2: {
                "step": step,
                "exp_avg": torch.tensor([3.0]),
                "exp_avg_sq": torch.tensor([103.0]),
            },
        }
    else:
        state = {
            1: {
                "step": step,
                "exp_avg": torch.full((1, 2, 2), expert_value),
                "exp_avg_sq": torch.full((1, 2, 2), expert_value + 100),
            },
            2: {
                "step": step,
                "exp_avg": torch.tensor([4.0, 5.0, 6.0, 7.0]),
                "exp_avg_sq": torch.tensor([104.0, 105.0, 106.0, 107.0]),
            },
        }
    optimizer.load_state_dict({"adamw": {"state": state, "param_groups": param_groups}})

    if fused:
        assert torch.equal(
            optimizer.adamw.state[model.left_right]["exp_avg"],
            torch.arange(8, dtype=torch.float32),
        )
    else:
        assert torch.equal(
            optimizer.adamw.state[model.left]["exp_avg"],
            torch.tensor([0.0, 1.0, 2.0]),
        )
        assert torch.equal(
            optimizer.adamw.state[model.right]["exp_avg"],
            torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0]),
        )
    assert torch.equal(
        optimizer.adamw.state[model.expert]["exp_avg"],
        torch.full((1, 2, 2), expert_value),
    )
    dist.destroy_process_group()


def _run_migration(fused: bool):
    fd, init_file = tempfile.mkstemp()
    os.close(fd)
    os.unlink(init_file)
    try:
        mp.spawn(_migration_worker, args=(init_file, fused), nprocs=2, join=True)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_zero2_optimizer_state_is_reassembled_and_experts_stay_local():
    _run_migration(False)


def test_zero2_optimizer_state_survives_nonexpert_parameter_fusion():
    _run_migration(True)


def _expert_fusion_worker(rank: int, init_file: str):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{init_file}"
    )
    model = _ToyFusedExpertModel()
    optimizer = MatrixAwareOptimizer(model, enable_muon=False)
    param_groups = optimizer.adamw.state_dict()["param_groups"]
    step = torch.tensor(11.0)
    base = float(rank * 100)
    state = {
        0: {
            "step": step,
            "exp_avg": torch.arange(4, dtype=torch.float32),
            "exp_avg_sq": torch.arange(4, dtype=torch.float32) + 1000,
        },
        1: {
            "step": step,
            "exp_avg": torch.full((1, 2, 2), base + 1),
            "exp_avg_sq": torch.full((1, 2, 2), base + 11),
        },
        2: {
            "step": step,
            "exp_avg": torch.full((1, 2, 2), base + 2),
            "exp_avg_sq": torch.full((1, 2, 2), base + 12),
        },
        3: {
            "step": step,
            "exp_avg": torch.full((1, 2, 2), base + 3),
            "exp_avg_sq": torch.full((1, 2, 2), base + 13),
        },
    }
    optimizer.load_state_dict({"adamw": {"state": state, "param_groups": param_groups}})

    fused = optimizer.adamw.state[model.expert_up_gate]["exp_avg"]
    assert torch.equal(fused[:, :2], torch.full((1, 2, 2), base + 1))
    assert torch.equal(fused[:, 2:], torch.full((1, 2, 2), base + 2))
    assert torch.equal(
        optimizer.adamw.state[model.expert_down]["exp_avg"],
        torch.full((1, 2, 2), base + 3),
    )
    dist.destroy_process_group()


def test_zero2_optimizer_state_merges_legacy_expert_up_gate_moments():
    fd, init_file = tempfile.mkstemp()
    os.close(fd)
    os.unlink(init_file)
    try:
        mp.spawn(_expert_fusion_worker, args=(init_file,), nprocs=2, join=True)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
