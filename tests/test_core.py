import random

import numpy as np
import pytest
import torch

from sparsa.data import benchmark, collate, decode_document
from sparsa.model import ContactModel, ModelConfig
from sparsa.train import contact_loss
from sparsa.vendor.marinfold_metrics import metric_rows, resolved_pairs, true_matrix


def example():
    names = [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
    ]
    seq = [f"<p{(1996 + i) % 2000}> <{aa}>" for i, aa in enumerate(names)]
    random.Random(7).shuffle(seq)
    return {
        "entry_id": "example",
        "seq_len": 12,
        "n_term_index": 1996,
        "contacts_emitted": 2,
        "truncated": False,
        "document": "<contacts-v1> <begin_sequence> <n-term> <p1996> <c-term> <p7> "
        + " ".join(seq)
        + " <begin_statements> <contact> <p3> <p1996> <contact> <p1999> <p6> <end>",
    }


def test_decode_wrapped_shuffled_and_flipped():
    decoded = decode_document(example())
    assert decoded["sequence"] == "ACDEFGHIKLMN"
    np.testing.assert_array_equal(decoded["contacts"], [[0, 7], [3, 10]])


@pytest.mark.parametrize(
    "change", [{"truncated": True}, {"contacts_emitted": 3}, {"seq_len": 13}]
)
def test_reject_bad_teacher(change):
    with pytest.raises(ValueError):
        decode_document(example() | change)


def tiny():
    return ContactModel(
        ModelConfig(
            sequence_dim=32,
            sequence_layers=2,
            heads=4,
            pair_dim=8,
            pair_layers=4,
            triangle_every=2,
            triangle_dim=4,
        )
    )


def test_symmetry_padding_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    model = tiny().eval()
    tokens = torch.randint(1, 22, (1, 17))
    padded = torch.zeros(2, 25, dtype=torch.long)
    padded[0, :17] = tokens[0]
    padded[1] = torch.randint(1, 22, (25,))
    a, b = model(tokens), model(padded)[0:1, :17, :17]
    torch.testing.assert_close(a, a.transpose(-1, -2), atol=0, rtol=0)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    model.train()
    batch = collate([decode_document(example())], 24, random.Random(1))
    loss = contact_loss(model(batch["tokens"]), batch["targets"], batch["mask"])
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_loss_ignores_unresolved_and_lower_triangle():
    logits = torch.zeros(1, 12, 12, requires_grad=True)
    labels = torch.zeros_like(logits)
    labels[0, 0, 7] = 1
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[0, 0, 7] = True
    loss = contact_loss(logits, labels, mask)
    loss.backward()
    assert logits.grad[0, 0, 7] < 0
    assert torch.count_nonzero(logits.grad) == 1


def test_overfit_contact_map_and_restore(tmp_path):
    torch.manual_seed(5)
    model = tiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    batch = collate([decode_document(example())], 16, random.Random(1))
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        loss = contact_loss(model(batch["tokens"]), batch["targets"], batch["mask"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.1
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = tiny()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(batch["tokens"]), model(batch["tokens"]))


def test_frozen_benchmark():
    assert len(benchmark()) == 97
    assert len(benchmark(splits=("eval-test",))) == 217
    assert len(benchmark(splits=("eval-denovo",))) == 19
    rec = benchmark()[0]
    gt = true_matrix(rec["L"], rec["contacts"])
    pi, pj, sep = resolved_pairs(np.asarray(rec["resolved"]))
    rows = metric_rows(gt.astype(float), gt, pi, pj, sep, rec["L"], with_precision=True)
    assert (
        next(r["precision"] for r in rows if r["range"] == "all" and r["cut"] == "R")
        == 1
    )


def test_stream_resume_with_workers(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader

    from sparsa.data import TeacherBatches

    rows = [example() | {"entry_id": str(i)} for i in range(20)]
    path = tmp_path / "teacher.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    shards = {"afdb": [str(path)], "esm": [str(path)]}
    loader = DataLoader(
        TeacherBatches(shards, 2, 12, 17, total_batches=8),
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    stream = iter(loader)
    batches = list(stream)
    original = [batch["ids"] for batch in batches]
    states = {batch["worker_id"]: batch["data_state"] for batch in batches[:3]}
    del stream
    resumed = DataLoader(
        TeacherBatches(shards, 2, 12, 17, consumed=3, total_batches=8),
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    stream = iter(resumed)
    assert [batch["ids"] for batch in stream] == original[3:]
    del stream
    resumed = DataLoader(
        TeacherBatches(shards, 2, 12, 17, consumed=3, states=states, total_batches=8),
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    stream = iter(resumed)
    assert [batch["ids"] for batch in stream] == original[3:]


def test_helico_export(tmp_path):
    from sparsa.cli import export_contacts

    scores = np.zeros((12, 12))
    scores[0, 1] = 1  # invalid near-diagonal candidate must never be emitted
    scores[1, 9] = 0.8
    scores[0, 8] = 0.7
    out = tmp_path / "contacts.txt"
    export_contacts(scores, out, 2)
    assert out.read_text() == "1 9\n0 8\n"


def test_distance_readout_and_checkpoint(tmp_path):
    from dataclasses import asdict

    from sparsa.cli import load_model
    from sparsa.model import ALPHABET, tokenize

    model = tiny().eval()
    tokens = tokenize("ACDEFGHIKLMN")[None]
    with torch.inference_mode():
        original = model(tokens)
        model.relative_max_distance = 11
        torch.testing.assert_close(model(tokens), original, rtol=0, atol=0)
        model.relative_max_distance = 7
        capped = model(tokens)
        assert not torch.allclose(capped, original)
    path = tmp_path / "inference.pt"
    torch.save(
        {
            "format_version": 2,
            "alphabet": ALPHABET,
            "model_config": asdict(model.config),
            "ema": model.state_dict(),
            "inference_config": {"relative_max_distance": 7},
        },
        path,
    )
    restored, _ = load_model(str(path), torch.device("cpu"))
    with torch.inference_mode():
        torch.testing.assert_close(restored(tokens), capped)
