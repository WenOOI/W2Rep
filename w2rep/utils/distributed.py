from __future__ import annotations

import os

import torch
import torch.distributed as dist


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Pretraining requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / dist.get_world_size()


def finish_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

