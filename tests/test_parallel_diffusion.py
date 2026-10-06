import copy
from pathlib import Path

import pytest
import torch
import yaml

from scripts.run_multinode_diffusion import iris_identity, torchrun_command
from sparsa.train import contact_loss, contact_ranking_loss
from sparsa.train_diffusion import (
    inherited_counts,
    ranking_normalization,
    relocate_sources,
    validate_parallel_fork,
)


def configs():
    original = yaml.safe_load(Path("configs/pairformer_noisy_128.yaml").read_text())
    branch = yaml.safe_load(Path("configs/pairformer_noisy_128_32gpu.yaml").read_text())
    return {"training_config": original, "world_size": 8, "rng": [{}] * 8}, branch


def test_parallel_fork_constraints():
    state, cfg = configs()
    validate_parallel_fork(state, cfg, 32)
    for key, value in [
        ("lr", 1e-3),
        ("logical_batch_size", 8),
        ("ranking_group_size", 1),
        ("stream_seed", 23),
    ]:
        changed = dict(cfg, **{key: value})
        with pytest.raises(ValueError):
            validate_parallel_fork(state, changed, 32)
    changed = copy.deepcopy(cfg)
    changed["model"]["pairformer_layers"] = 24
    with pytest.raises(ValueError):
        validate_parallel_fork(state, changed, 32)


def test_parent_exposure_and_nested_fork():
    state, _ = configs()
    state.update(
        step=3,
        rng=[{"data_counts": {"proteins": 48, "afdb": 8, "esm": 40}} for _ in range(8)],
    )
    assert inherited_counts(state) == {"proteins": 384, "afdb": 64, "esm": 320}
    state["inherited_data_counts"] = {"proteins": 128}
    with pytest.raises(ValueError, match="exposure"):
        inherited_counts(state)
    state["step"] = 4
    assert inherited_counts(state)["proteins"] == 512


def test_relocate_preserves_frozen_source_order():
    state, cfg = configs()
    old = state["training_config"]["data_root"]
    provenance = dict(
        state,
        source_files={
            "afdb": ["file://" + old + "/afdb/b.parquet", old + "/afdb/a.parquet"]
        },
    )
    mapped = relocate_sources(provenance, cfg)
    assert mapped["afdb"] == [
        cfg["data_root"] + "/afdb/b.parquet",
        cfg["data_root"] + "/afdb/a.parquet",
    ]
    provenance["source_files"]["afdb"][0] = "/unrelated/b.parquet"
    with pytest.raises(ValueError, match="outside"):
        relocate_sources(provenance, cfg)


def test_32gpu_loss_and_gradient_match_8gpu_accumulation():
    # Include zero-contact proteins and a whole empty normalization group.
    torch.manual_seed(31)
    target = (torch.rand(128, 12, 12) < 0.1).float()
    target[:19] = 0
    mask = torch.ones_like(target, dtype=torch.bool).triu(6)
    initial = torch.randn_like(target)
    values = []
    for parallel in (False, True):
        logits = initial.clone().requires_grad_()
        total = torch.zeros(())
        counts = (
            ((target > 0.5) & mask).flatten(1).any(1).reshape(32, 4).sum(1).tolist()
        )
        width, replicas = (4, 32) if parallel else (16, 8)
        for rank in range(replicas):
            sl = slice(rank * width, (rank + 1) * width)
            normalizer = ranking_normalization(counts, rank, 4, 1) if parallel else None
            loss = contact_loss(logits[sl], target[sl], mask[sl], 4.0)
            loss += 0.05 * contact_ranking_loss(
                logits[sl], target[sl], mask[sl], normalizer
            )
            total += loss / replicas
        total.backward()
        values.append((total.detach(), logits.grad))
    torch.testing.assert_close(*[v[0] for v in values])
    torch.testing.assert_close(*[v[1] for v in values])


def test_multinode_command_uses_static_rank_and_no_elastic_restart():
    command = torchrun_command(4, 8, 3, "192.0.2.1:12345")
    assert "--nnodes=4" in command and "--node_rank=3" in command
    assert "--master_addr=192.0.2.1" in command and "--master_port=12345" in command
    assert "--standalone" not in command and "--max_restarts=0" in command


def test_iris_identity_scopes_rendezvous_to_job_and_attempt():
    assert iris_identity(
        {
            "IRIS_TASK_ID": "/user/job/3:9",
            "IRIS_NUM_TASKS": "4",
            "IRIS_ADVERTISE_HOST": "192.0.2.2",
        }
    ) == {"job": "/user/job", "rank": 3, "attempt": 9, "nodes": 4, "host": "192.0.2.2"}


def test_checkpoint_cache_downloads_once_and_recovers_partial(tmp_path, monkeypatch):
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    from sparsa import train_diffusion

    source = tmp_path / "source.pt"
    torch.save({"step": 1200, "weight": torch.arange(4)}, source)
    calls = []

    class Store:
        def get_file(self, remote, local):
            calls.append(remote)
            if len(calls) == 1:
                Path(local).write_bytes(b"incomplete download")
                raise OSError("interrupted download")
            shutil.copyfile(source, local)

    monkeypatch.setenv("SPARSA_CHECKPOINT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(train_diffusion, "storage", lambda uri: (Store(), uri))
    uri = "s3://test/run/checkpoints/step-1200.pt"
    with pytest.raises(OSError, match="interrupted"):
        train_diffusion.load_checkpoint(uri)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(train_diffusion.load_checkpoint, [uri] * 4))
    assert len(calls) == 2
    for result in results:
        assert result["step"] == 1200
        torch.testing.assert_close(result["weight"], torch.arange(4))
