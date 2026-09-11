"""Strict decoding and deterministic streaming of MarinFold teacher documents."""

import hashlib
import json
import pickle
import random
import re
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from sparsa.model import tokenize

THREE_TO_ONE = dict(
    zip(
        [
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
            "PRO",
            "GLN",
            "ARG",
            "SER",
            "THR",
            "VAL",
            "TRP",
            "TYR",
            "UNK",
        ],
        "ACDEFGHIKLMNPQRSTVWYX",
        strict=True,
    )
)
RESIDUE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")
CONTACT = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
ROOT = "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/data"


def decode_document(row: dict) -> dict:
    """Undo randomized statement order, orientation, and the 2000-token ring.

    Truncated documents cannot define negative labels; callers must exclude them.
    Unlike tolerant inference parsers, this fails on corrupt authored examples.
    """
    if row.get("truncated", False):
        raise ValueError("Truncated contact targets cannot supervise negatives")
    doc, length, start = row["document"], int(row["seq_len"]), int(row["n_term_index"])
    if not 1 <= length <= 2000 or not 0 <= start < 2000:
        raise ValueError("Invalid sequence length or ring origin")
    if (
        doc.count("<n-term>") != 1
        or doc.count("<begin_statements>") != 1
        or not doc.rstrip().endswith("<end>")
    ):
        raise ValueError("Expected a complete monomer contacts-v1 document")
    sequence_section, contact_section = doc.split("<begin_statements>")
    residues = [None] * length
    for pos, aa in RESIDUE.findall(sequence_section):
        index = (int(pos) - start) % 2000
        if index >= length or residues[index] is not None or aa not in THREE_TO_ONE:
            raise ValueError("Invalid or duplicate sequence statement")
        residues[index] = THREE_TO_ONE[aa]
    if any(aa is None for aa in residues):
        raise ValueError("Incomplete sequence")
    if "<retract>" in contact_section:
        raise ValueError("Only unedited teacher documents are supported")
    pairs = set()
    for p, q in CONTACT.findall(contact_section):
        i, j = sorted(((int(p) - start) % 2000, (int(q) - start) % 2000))
        if not 0 <= i < j < length or j - i < 6 or (i, j) in pairs:
            raise ValueError("Invalid or duplicate contact")
        pairs.add((i, j))
    if len(pairs) != int(row["contacts_emitted"]):
        raise ValueError("Contact metadata mismatch")
    return {
        "sequence": "".join(residues),
        "contacts": np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2),
        "entry_id": str(row["entry_id"]),
    }


def inventory(root: str = ROOT) -> dict[str, list[str]]:
    fs, path = fsspec.core.url_to_fs(root)
    result = {}
    for source in ("afdb", "esm"):
        paths = sorted(fs.glob(f"{path}/{source}/**/*.parquet"))
        if not paths:
            raise ValueError(f"No teacher shards: {root}/{source}")
        result[source] = [fs.unstrip_protocol(p) for p in paths]
    return result


def load_shard(uri: str) -> list[dict]:
    with fsspec.open(uri, "rb") as f:
        rows = pq.read_table(f).to_pylist()
    return rows


def collate(records: list[dict], crop: int, rng: random.Random) -> dict:
    length = min(crop, max(len(r["sequence"]) for r in records))
    # Multiples of 8 aid tensor cores; validity remains explicit.
    length = (length + 7) // 8 * 8
    tokens = torch.zeros(len(records), length, dtype=torch.long)
    targets = torch.zeros(len(records), length, length)
    ids = []
    for b, rec in enumerate(records):
        seq = rec["sequence"]
        start = rng.randrange(max(1, len(seq) - crop + 1))
        seq = seq[start : start + crop]
        tokens[b, : len(seq)] = tokenize(seq)
        contacts = np.asarray(rec["contacts"], dtype=np.int64).reshape(-1, 2) - start
        contacts = contacts[(contacts[:, 0] >= 0) & (contacts[:, 1] < len(seq))]
        if len(contacts):
            i, j = contacts.T
            targets[b, i, j] = 1
            targets[b, j, i] = 1
        ids.append(rec["entry_id"])
    valid = tokens != 0
    mask = valid[:, :, None] & valid[:, None, :]
    mask &= torch.ones(length, length, dtype=torch.bool).triu(6)
    return {"tokens": tokens, "targets": targets, "mask": mask, "ids": ids}


class ShardStream:
    """A reproducible stream with a compact cursor, independent of document size."""

    def __init__(self, paths, seed, state=None):
        self.paths, self.seed = list(paths), seed
        self.epoch, self.shard, self.row = state or (0, 0, 0)
        self.rows = None

    def state(self):
        return self.epoch, self.shard, self.row

    def __next__(self):
        while True:
            if self.rows is None:
                paths = list(self.paths)
                random.Random(f"{self.seed}:{self.epoch}").shuffle(paths)
                self.rows = load_shard(paths[self.shard])
                random.Random(f"{self.seed}:{self.epoch}:{self.shard}").shuffle(
                    self.rows
                )
            while self.row < len(self.rows):
                row = self.rows[self.row]
                self.row += 1
                if row.get("truncated", False) or int(row["seq_len"]) < 12:
                    continue
                return decode_document(row)
            self.rows = None
            self.row = 0
            self.shard += 1
            if self.shard == len(self.paths):
                self.shard = 0
                self.epoch += 1


class TeacherBatches(IterableDataset):
    """Shuffle shards and rows, then sample AFDB/ESM equally at protein level.

    Each rank/worker receives a distinct deterministic shard ordering. Compact
    cursors and RNG state travel with consumed batches; resume loads only the
    current shards, preserving the exact stream. Prefetched batches replay.
    """

    def __init__(
        self,
        shards,
        batch_size,
        crop,
        seed,
        rank=0,
        world_size=1,
        consumed=0,
        limit_shards=0,
        states=None,
        total_batches=None,
    ):
        self.shards = shards
        self.batch_size, self.crop, self.seed = batch_size, crop, seed
        self.rank, self.world_size, self.consumed = rank, world_size, consumed
        self.limit_shards = limit_shards
        self.states = states or {}
        self.total_batches = total_batches

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        remaining = (
            None
            if self.total_batches is None
            else max(
                0,
                (self.total_batches - self.consumed - worker_id + workers - 1)
                // workers,
            )
        )
        worker_id = (worker_id + self.consumed) % workers
        rng = random.Random(self.seed + 100003 * self.rank + 1009 * worker_id)
        saved = (
            pickle.loads(self.states[worker_id]) if worker_id in self.states else None
        )
        streams = {}
        for source, paths in self.shards.items():
            paths = list(paths[: self.limit_shards] if self.limit_shards else paths)
            streams[source] = ShardStream(
                paths, rng.getrandbits(64), saved["streams"][source] if saved else None
            )
        if saved:
            rng.setstate(saved["rng"])
        skip = (
            0 if saved else max(0, (self.consumed - worker_id + workers - 1) // workers)
        )
        batch_index = 0
        while remaining is None or batch_index < skip + remaining:
            records = [
                next(streams["afdb" if rng.random() < 0.5 else "esm"])
                for _ in range(self.batch_size)
            ]
            batch = collate(records, self.crop, rng)
            if batch_index >= skip:
                batch["worker_id"] = worker_id
                batch["data_state"] = pickle.dumps(
                    {
                        "rng": rng.getstate(),
                        "streams": {k: s.state() for k, s in streams.items()},
                    }
                )
                yield batch
            batch_index += 1


def benchmark(root="data/benchmark", splits=("eval-val",)) -> list[dict]:
    import csv

    root = Path(root)
    pins = {
        "eval_sets.csv": "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861",
        "gt_universe_scored.jsonl": "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5",
    }
    for name, digest in pins.items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Frozen MarinFold benchmark changed: {name}")
    with (root / "eval_sets.csv").open() as f:
        manifest = {r["stem"]: r for r in csv.DictReader(f) if r["scorable"] == "1"}
    records = []
    with (root / "gt_universe_scored.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            meta = manifest[rec["stem"]]
            if meta["eval_set"] not in splits:
                continue
            rec.update(sequence=meta["sequence"], eval_set=meta["eval_set"])
            if rec["L"] != len(rec["sequence"]):
                raise ValueError("Benchmark sequence/GT length mismatch")
            records.append(rec)
    expected = {"eval-val": 97, "eval-test": 217, "eval-denovo": 19}
    for split in splits:
        if sum(r["eval_set"] == split for r in records) != expected[split]:
            raise ValueError(f"Benchmark coverage mismatch for {split}")
    return records


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
