"""Fail fast if a requested multi-node CUDA collective does not work."""

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    world, rank = dist.get_world_size(), dist.get_rank()
    if world != args.world_size:
        raise ValueError("Wrong distributed world size")
    value = torch.full((1024 * 1024,), float(rank), device=device)
    dist.all_reduce(value)
    if not torch.all(value == world * (world - 1) / 2):
        raise ValueError("Incorrect all-reduce result")
    if rank == 0:
        print(
            "COLLECTIVE_OK "
            + json.dumps(
                {"world_size": world, "gpu": torch.cuda.get_device_name(device)}
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
