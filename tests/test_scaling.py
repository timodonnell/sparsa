from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from sparsa.data import TeacherBatches
from sparsa.grow import grow_from_ema
from sparsa.model import ALPHABET, ContactModel, ModelConfig
from sparsa.train import contact_loss


@pytest.mark.parametrize("factor,depth", [(1, 2), (2, 1), (2, 2)])
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
    if factor == 2:
        grad = new.embedding.weight.grad
        assert not torch.equal(grad[:, :32], grad[:, 32:])
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
