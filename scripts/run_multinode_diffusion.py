"""Iris gang launcher: static torchrun rendezvous, collective test, fork/resume gate."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time


def torchrun_command(nodes, gpus, rank, endpoint):
    host, port = endpoint.rsplit(":", 1)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nodes}",
        f"--nproc_per_node={gpus}",
        f"--node_rank={rank}",
        f"--master_addr={host}",
        f"--master_port={port}",
        "--max_restarts=0",
    ]


def iris_identity(environ):
    task, attempt = environ["IRIS_TASK_ID"].rsplit(":", 1)
    job, rank = task.rsplit("/", 1)
    return {
        "job": job,
        "rank": int(rank),
        "attempt": int(attempt),
        "nodes": int(environ["IRIS_NUM_TASKS"]),
        "host": environ["IRIS_ADVERTISE_HOST"],
    }


def rendezvous(info, out, stage):
    from sparsa.train import storage
    from sparsa.train_diffusion import write_json

    # Job-scoped and attempt-scoped: never join a previous gang's rank zero.
    job_key = hashlib.sha256(info["job"].encode()).hexdigest()[:16]
    uri = out + f"/rendezvous/{job_key}-attempt-{info['attempt']}-{stage}.json"
    fs, path = storage(uri)
    if info["rank"] == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        endpoint = f"{info['host']}:{port}"
        write_json(
            uri, {"endpoint": endpoint, "job": info["job"], "attempt": info["attempt"]}
        )
        return endpoint
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if fs.exists(path):
            metadata = json.loads(fs.cat_file(path))
            if (
                metadata["job"] == info["job"]
                and metadata["attempt"] == info["attempt"]
            ):
                return metadata["endpoint"]
        time.sleep(5)
    raise TimeoutError("Timed out waiting for this attempt's rank-zero rendezvous")


def main():
    from sparsa.train import storage
    from sparsa.train_diffusion import inherited_counts, load_checkpoint, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fork-from", required=True)
    parser.add_argument("--fork-step", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    info = iris_identity(os.environ)
    if info["nodes"] != args.nodes:
        raise ValueError("Launch as an Iris gang with the requested replica count")
    os.environ.setdefault("SPARSA_CHECKPOINT_CACHE", "/tmp/sparsa-checkpoint-cache")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    def run_stage(stage, arguments):
        endpoint = rendezvous(info, args.out, stage)
        print(
            "TORCH_RENDEZVOUS "
            + json.dumps(
                {
                    "stage": stage,
                    "node_rank": info["rank"],
                    "nodes": args.nodes,
                    "endpoint": endpoint,
                    "attempt": info["attempt"],
                }
            ),
            flush=True,
        )
        launch = torchrun_command(
            args.nodes, args.gpus_per_node, info["rank"], endpoint
        )
        subprocess.run(launch + arguments, check=True)

    train = [
        "-m",
        "sparsa.train_diffusion",
        "--config",
        args.config,
        "--out",
        args.out,
        "--auto-resume",
        "--fork-from",
        args.fork_from,
        "--eval-every",
        str(args.eval_every),
    ]
    fs, root = storage(args.out)
    gate_complete = fs.exists(root + "/parallelism_preflight.json")
    run_stage(
        "collective",
        [
            "scripts/check_distributed.py",
            "--world-size",
            str(args.nodes * args.gpus_per_node),
        ],
    )
    # Every node uses the same stages even if rank zero publishes latest.json
    # before another launcher gets here. Completed boundaries safely no-op.
    if not gate_complete:
        for boundary in (args.fork_step + 2, args.fork_step + 4):
            run_stage(f"train-{boundary}", train + ["--stop-after", str(boundary)])
        if info["rank"] == 0:
            latest = json.loads(fs.cat_file(root + "/latest.json"))
            state = load_checkpoint(latest["checkpoint"])
            if state["world_size"] != args.nodes * args.gpus_per_node:
                raise ValueError("Resumed world size mismatch")
            if (
                state["training_config"]["logical_batch_size"]
                != state["training_config"]["batch_size"]
            ):
                raise ValueError("This experiment must not accumulate gradients")
            if (
                state["fork_metadata"]["parent_step"] != args.fork_step
                or state["step"] < args.fork_step + 4
            ):
                raise ValueError("Fork/resume gate used an unexpected parent or step")
            counts = inherited_counts(state)
            write_json(
                args.out + "/parallelism_preflight.json",
                {
                    "resumed_through_step": state["step"],
                    "world_size": state["world_size"],
                    "global_data_counts": counts,
                    "inherited_data_counts": state["inherited_data_counts"],
                    "optimizer_parameters": len(state["optimizer"]["state"]),
                    "fork_metadata": state["fork_metadata"],
                    "validation_split": "eval-val",
                    "held_out_used": False,
                },
            )
            del state
    # Other torchrun nodes wait for rank zero during process-group init.
    if args.stop_after:
        train += ["--stop-after", str(args.stop_after)]
    run_stage("long-training", train)


if __name__ == "__main__":
    main()
