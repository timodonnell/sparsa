import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from sparsa.data import TeacherBatches
from sparsa.experiment import (
    advance_data_digest,
    initialize_encoder,
    learning_rate_scale,
    microbatches,
)
from sparsa.model import ALPHABET, ContactModel, ModelConfig
from sparsa.pair_trunk import TriangleAttention
from sparsa.train import contact_loss, contact_ranking_loss


def tiny(family="conv", directed=False):
    return ModelConfig(
        sequence_dim=32,
        sequence_layers=2,
        heads=4,
        pair_dim=16,
        pair_layers=2,
        triangle_dim=8,
        triangle_every=2,
        pair_family=family,
        pair_directional=directed,
        pair_outer_rank=4,
        pair_attention_heads=4,
        pair_attention_chunk=3,
        pair_residue_dim=32,
    )


@pytest.mark.parametrize(
    "family,directed",
    [
        ("conv", False),
        ("conv", True),
        ("triangle", True),
        ("attention", True),
        ("joint", True),
    ],
)
def test_pair_family_contract_and_learning(family, directed):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = ContactModel(tiny(family, directed))
    tokens = torch.randint(1, 22, (2, 19))
    tokens[0, 13:] = 0
    valid = tokens != 0
    mask = (valid[:, :, None] & valid[:, None, :]).triu(6)
    target = torch.zeros(2, 19, 19)
    target[:, 0, 10] = 1
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    for _ in range(4):
        opt.zero_grad()
        logits = model(tokens)
        contact_loss(logits, target, mask, 4).backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        opt.step()
    model.eval()
    with torch.no_grad():
        result = model(tokens)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, result.transpose(1, 2))
        torch.testing.assert_close(
            result[:1, :13, :13], model(tokens[:1, :13]), atol=1e-5, rtol=1e-5
        )
        other = ContactModel(tiny(family, directed)).eval()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(other(tokens), result, atol=0, rtol=0)
    if family in ("attention", "joint"):
        assert model.pairs[-1].attentions[0].qkv.weight.grad.abs().sum() > 0
    if family == "joint":
        assert model.pairs[-1].feedback.bias.weight.grad.abs().sum() > 0


def test_chunked_triangle_output_and_gradient_match():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    for ending in (False, True):
        a = TriangleAttention(16, 4, 3, ending)
        torch.nn.init.normal_(a.out.weight, std=0.1)
        b = copy.deepcopy(a)
        b.chunk = 20
        z = torch.randn(2, 9, 9, 16, requires_grad=True)
        z2 = z.detach().clone().requires_grad_()
        valid = torch.ones(2, 9, dtype=torch.bool)
        valid[0, 6:] = False
        mask = valid[:, :, None] & valid[:, None, :]
        x, y = a(z, mask), b(z2, mask)
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        x.square().sum().backward()
        y.square().sum().backward()
        torch.testing.assert_close(z.grad, z2.grad, atol=1e-5, rtol=1e-5)
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5)


def test_encoder_copy_resets_pair_and_rejects_shape():
    a = ContactModel(tiny())
    b = ContactModel(tiny("joint", True))
    before = {k: v.clone() for k, v in b.state_dict().items() if k.startswith("pairs.")}
    state = {
        "ema": a.state_dict(),
        "model_config": asdict(a.config),
        "alphabet": ALPHABET,
    }
    initialize_encoder(b, state)
    for k, v in before.items():
        torch.testing.assert_close(v, b.state_dict()[k])
    for p, q in zip(a.sequence.parameters(), b.sequence.parameters(), strict=True):
        torch.testing.assert_close(p, q)
    with pytest.raises(ValueError, match="architecture"):
        initialize_encoder(ContactModel(replace(tiny(), sequence_layers=3)), state)


def test_fixed_schedule_survives_stage_extension():
    a = {
        "steps": 25000,
        "schedule": "wsd",
        "schedule_steps": 100000,
        "decay_start": 80000,
        "warmup": 1000,
    }
    b = a | {"steps": 100000}
    for step in (0, 999, 1000, 24999, 25000):
        assert learning_rate_scale(a, step) == learning_rate_scale(b, step)
    assert learning_rate_scale(b, 50000) == 1
    assert learning_rate_scale(b, 90000) == pytest.approx(0.55)
    assert learning_rate_scale(b, 100000) == pytest.approx(0.1)


def test_ranking_normalization_matches_across_microbatches_with_empty_contacts():
    torch.manual_seed(7)
    logits = torch.randn(8, 12, 12, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 8] = targets[5, 0, 8] = targets[6, 0, 9] = 1
    mask = torch.ones_like(targets, dtype=torch.bool).triu(6)
    full = contact_ranking_loss(logits, targets, mask)
    full_grad = torch.autograd.grad(full, logits)[0]
    for size in (1, 2, 4, 8):
        parts = 8 // size
        loss = sum(
            contact_ranking_loss(
                logits[i : i + size],
                targets[i : i + size],
                mask[i : i + size],
                3 / parts,
            )
            / parts
            for i in range(0, 8, size)
        )
        torch.testing.assert_close(loss, full)
        torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], full_grad)


def test_logical_batch_microbatch_independence_and_replay(monkeypatch):
    from sparsa import data

    class Stream:
        def __init__(self, paths, seed, state=None):
            self.path = paths[0]
            self.i = state or 0

        def state(self):
            return self.i

        def __next__(self):
            self.i += 1
            return {
                "entry_id": f"{self.path}-{self.i}",
                "sequence": "ACDEFGHIKLMNPQRSTVWY" * 2,
                "contacts": np.array([[0, 10], [10, 30]]),
            }

    monkeypatch.setattr(data, "ShardStream", Stream)
    shards = {"afdb": ["a"], "esm": ["e"]}
    batches = list(TeacherBatches(shards, 8, 24, 17, total_batches=10))
    resumed = list(
        TeacherBatches(
            shards,
            8,
            24,
            17,
            consumed=4,
            states={0: batches[3]["data_state"]},
            total_batches=10,
        )
    )
    digest1 = digest2 = "00" * 32
    for a, b in zip(batches[4:], resumed, strict=True):
        assert a["ids"] == b["ids"] and a["crop_starts"] == b["crop_starts"]
        for size in (1, 2, 4, 8):
            parts = list(microbatches(a, size))
            torch.testing.assert_close(
                torch.cat([p["tokens"] for p in parts]), b["tokens"]
            )
            torch.testing.assert_close(
                torch.cat([p["targets"] for p in parts]), b["targets"]
            )
        digest1 = advance_data_digest(digest1, a)
        digest2 = advance_data_digest(digest2, b)
        assert digest1 == digest2
