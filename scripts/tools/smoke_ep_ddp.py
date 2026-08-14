"""Smoke-test that EP-aware DDP syncs shared weights but not local experts."""

import os

import torch
import torch.distributed as dist
from torch import nn

from astrai.parallel.executor import DDPExecutor


class _LocalExpert(nn.Module):
    def __init__(self, rank: int, world_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4, 4), float(rank + 1)))
        self._expert_parallel_local = True
        self.expert_parallel_size = world_size
        self.process_group = dist.group.WORLD

    def forward(self, x):
        return x @ self.weight


class _ToyEPModel(nn.Module):
    def __init__(self, rank: int, world_size: int):
        super().__init__()
        self.shared = nn.Linear(4, 4, bias=False)
        self.expert = _LocalExpert(rank, world_size)

    def forward(self, x):
        return self.shared(x) + self.expert(x)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    model = _ToyEPModel(rank, world_size).cuda(local_rank)
    with torch.no_grad():
        model.shared.weight.fill_(float(rank + 1))
    wrapped = DDPExecutor(
        gradient_as_bucket_view=True, broadcast_buffers=False
    )._prepare_model(model)

    # DDP broadcasts shared weights from rank 0, but ignored experts retain
    # their distinct rank-local initialization.
    assert torch.all(model.shared.weight == 1)
    assert torch.all(model.expert.weight == rank + 1)

    x = torch.full((2, 4), float(rank + 1), device=local_rank)
    wrapped(x).sum().backward()
    shared_grad = model.shared.weight.grad.clone()
    expert_grad = model.expert.weight.grad.clone()
    gathered_shared = [torch.empty_like(shared_grad) for _ in range(world_size)]
    gathered_expert = [torch.empty_like(expert_grad) for _ in range(world_size)]
    dist.all_gather(gathered_shared, shared_grad)
    dist.all_gather(gathered_expert, expert_grad)
    assert all(torch.equal(gathered_shared[0], item) for item in gathered_shared[1:])
    assert any(not torch.equal(gathered_expert[0], item) for item in gathered_expert[1:])
    if rank == 0:
        print("EP_DDP_SMOKE_OK", f"world_size={world_size}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
