from sparsa.train_diffusion import validation_due


def test_validation_cadence_retries_only_incomplete_milestones():
    assert not validation_due(0, 30000, 10000)
    assert not validation_due(9999, 30000, 10000)
    assert validation_due(10000, 30000, 10000)
    assert not validation_due(10000, 30000, 10000, complete=True)
    assert validation_due(30000, 30000, 0)
    assert not validation_due(30000, 30000, 0, complete=True)


def test_local_checkpoint_write_preserves_previous_file_on_publish_failure(
    tmp_path, monkeypatch
):
    import pytest

    from sparsa import train_diffusion

    target = tmp_path / "checkpoints" / "latest.json"
    train_diffusion.write_bytes(str(target), b"previous")

    def fail_publish(source, destination):
        assert target.read_bytes() == b"previous"
        raise OSError("simulated publish failure")

    monkeypatch.setattr(train_diffusion.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        train_diffusion.write_bytes(str(target), b"new checkpoint")
    assert target.read_bytes() == b"previous"
    assert not list(target.parent.glob("*.partial"))


def test_checkpoint_pruning_preserves_validation_milestones(tmp_path):
    from sparsa.train_diffusion import prune_checkpoints

    root = tmp_path / "checkpoints"
    root.mkdir()
    for step in [9000, 10000, 11000, 12000]:
        (root / f"step-{step}.pt").write_bytes(b"checkpoint")
    prune_checkpoints(str(tmp_path), keep=2, keep_every=10000)
    assert {p.name for p in root.iterdir()} == {
        "step-10000.pt",
        "step-11000.pt",
        "step-12000.pt",
    }
    prune_checkpoints(str(tmp_path), keep=2)
    assert {p.name for p in root.iterdir()} == {"step-11000.pt", "step-12000.pt"}
