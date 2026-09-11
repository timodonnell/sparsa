from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from sparsa.data import TeacherBatches
from sparsa.grow import grow_from_ema
from sparsa.model import ALPHABET, ContactModel, ModelConfig
from sparsa.train import contact_loss


@pytest.mark.parametrize("factor,depth", [(1, 2), (2, 1), (2, 2), (4, 3)])
def test_growth_preserves_predictions_and_learns(factor, depth):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    cfg = ModelConfig(
        sequence_dim=32,
        sequence_layers=2,
        heads=4,
        pair_dim=8,
        pair_layers=4,
        triangle_every=2,
        triangle_dim=4,
    )
    old = ContactModel(cfg).eval()
    state = {"alphabet": ALPHABET, "model_config": asdict(cfg), "ema": old.state_dict()}
    new = ContactModel(
        replace(
            cfg, sequence_dim=32 * factor, heads=4 * factor, sequence_layers=2 * depth
        )
    ).eval()
    grow_from_ema(new, state)
    tokens = torch.randint(1, 22, (2, 23))
    tokens[0, 17:] = 0
    with torch.no_grad():
        expected = old(tokens)
        torch.testing.assert_close(new(tokens), expected, atol=4e-6, rtol=4e-6)
    new.train()
    target = torch.zeros_like(expected)
    target[:, 2, 12] = 1
    valid = tokens != 0
    mask = (valid[:, :, None] & valid[:, None, :]).triu(6)
    before = new.embedding.weight.detach().clone()
    opt = torch.optim.AdamW(new.parameters(), lr=1e-3)
    loss = contact_loss(new(tokens), target, mask, 4)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()
    )
    if factor > 1:
        grad = new.embedding.weight.grad.reshape(-1, factor, 32)
        assert not torch.equal(grad[:, 0], grad[:, 1])
    if depth > 1:
        assert new.sequence[1].out.weight.grad.abs().sum() > 0
    opt.step()
    assert not torch.equal(before, new.embedding.weight)


def test_growth_rejects_incompatible_source():
    cfg = ModelConfig(
        sequence_dim=32,
        sequence_layers=2,
        heads=4,
        pair_dim=8,
        pair_layers=4,
        triangle_every=2,
        triangle_dim=4,
    )
    model = ContactModel(cfg)
    state = {
        "alphabet": ALPHABET,
        "model_config": asdict(cfg),
        "ema": model.state_dict(),
    }
    with pytest.raises(ValueError, match="pair architecture"):
        grow_from_ema(ContactModel(replace(cfg, pair_dim=16)), state)
    with pytest.raises(ValueError, match="uncapped"):
        grow_from_ema(
            model, state | {"inference_config": {"relative_max_distance": 100}}
        )
    with pytest.raises(ValueError, match="vocabulary"):
        grow_from_ema(model, state | {"alphabet": ALPHABET[::-1]})


def test_teacher_mixture_and_cursor_restore(monkeypatch):
    from sparsa import data

    class Stream:
        def __init__(self, paths, seed, state=None):
            self.source = paths[0]
            self.row = state["row"] if state else 0

        def state(self):
            return {"row": self.row}

        def __next__(self):
            self.row += 1
            return {
                "sequence": "ACDEFGHIKLMN",
                "contacts": np.array([[0, 8]]),
                "entry_id": f"{self.source}:{self.row}",
            }

    monkeypatch.setattr(data, "ShardStream", Stream)
    shards = {"afdb": ["afdb"], "esm": ["esm"]}
    for probability in (0.0, 0.06, 0.5, 1.0):
        batches = list(
            TeacherBatches(
                shards, 4, 12, 17, total_batches=100, afdb_probability=probability
            )
        )
        states = {0: batches[42]["data_state"]}
        resumed = list(
            TeacherBatches(
                shards,
                4,
                12,
                17,
                consumed=43,
                total_batches=100,
                states=states,
                afdb_probability=probability,
            )
        )
        assert [b["ids"] for b in resumed] == [b["ids"] for b in batches[43:]]
        afdb = sum(s.startswith("afdb:") for b in batches for s in b["ids"])
        assert abs(afdb / 400 - probability) < 0.08
    with pytest.raises(ValueError, match="probability"):
        TeacherBatches(shards, 4, 12, 17, afdb_probability=1.01)


def test_compute_guard_bounds_missing_attempt_timing():
    from scripts.guard_compute import conservative_hours

    measured = {
        "wall_seconds": 7200,
        "tasks": [{"gpu_count": 8, "attempts": [{"running_gpu_hours": 4.0}]}],
    }
    missing = {
        "wall_seconds": 3600,
        "tasks": [
            {
                "gpu_count": 8,
                "attempts": [{"running_gpu_hours": 1.0}, {"running_gpu_hours": None}],
            }
        ],
    }
    # The missing job is charged its entire lifetime, including any queue time;
    # its known attempt is not counted twice.
    assert conservative_hours({"jobs": [measured, missing]}) == 12.0


def test_recovery_retention_preserves_validated_and_protected(tmp_path):
    from sparsa.train import prune_recovery_checkpoints

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "validation").mkdir()
    for step in (100, 200, 300, 400, 500, 600, 700):
        (tmp_path / f"checkpoints/step-{step}.pt").write_bytes(b"checkpoint")
    (tmp_path / "validation/step-200.json").write_text("{}")
    # Preserve an unusual best pointer even if its validation metadata is absent.
    protected = str(tmp_path / "checkpoints/step-100.pt")
    removed = prune_recovery_checkpoints(str(tmp_path), 2, [protected])
    assert len(removed) == 3
    assert {p.name for p in (tmp_path / "checkpoints").iterdir()} == {
        "step-100.pt",
        "step-200.pt",
        "step-600.pt",
        "step-700.pt",
    }
    with pytest.raises(ValueError, match="at least one"):
        prune_recovery_checkpoints(str(tmp_path), 0)


def test_ranking_loss_masks_shift_invariance_and_empty_positives():
    from sparsa.train import contact_ranking_loss

    logits = torch.zeros(2, 12, 12, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 8] = 1
    mask = torch.ones_like(logits, dtype=torch.bool).triu(6)
    loss = contact_ranking_loss(logits, target, mask)
    torch.testing.assert_close(loss, contact_ranking_loss(logits + 7, target, mask))
    loss.backward()
    assert logits.grad[0, 0, 8] < 0
    assert logits.grad[0, 0, 9] > 0
    assert logits.grad[1].abs().sum() == 0
    assert logits.grad[~mask].abs().sum() == 0
    better = logits.detach().clone()
    better[0, 0, 8] = 2
    assert contact_ranking_loss(better, target, mask) < loss
    # A high-scoring false contact must increase the ranking loss.
    better[0, 0, 9] = 5
    assert contact_ranking_loss(better, target, mask) > loss
    empty = contact_ranking_loss(logits, torch.zeros_like(target), mask)
    assert empty == 0 and torch.isfinite(empty)


def test_resume_architecture_ignores_only_checkpointing():
    from sparsa.train import architecture_config

    a = asdict(ModelConfig())
    assert architecture_config(a) == architecture_config(
        a | {"gradient_checkpointing": False}
    )
    assert architecture_config(a) != architecture_config(a | {"sequence_layers": 9})
    assert architecture_config(a) != architecture_config(a | {"dropout": 0.1})
