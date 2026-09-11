"""Read only teacher eligibility columns to count the exact supervised corpus."""

import json
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pyarrow.parquet as pq

from sparsa.data import inventory
from sparsa.train import write_json


def one(uri):
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb", block_size=65536) as f:
        table = pq.read_table(
            f,
            columns=[
                "seq_len",
                "truncated",
                "contacts_emitted",
                "contacts_passing_min_degree",
            ],
        )
    d = table.to_pydict()
    kept = [
        i
        for i, (length, truncated) in enumerate(
            zip(d["seq_len"], d["truncated"], strict=True)
        )
        if length >= 12 and not truncated
    ]
    if any(
        d["contacts_emitted"][i] != d["contacts_passing_min_degree"][i] for i in kept
    ):
        raise ValueError(f"Untruncated row omitted above-threshold contacts: {uri}")
    return {
        "rows": len(d["seq_len"]),
        "truncated": sum(d["truncated"]),
        "retained": len(kept),
        "retained_residues": sum(d["seq_len"][i] for i in kept),
        "retained_contacts": sum(d["contacts_emitted"][i] for i in kept),
    }


if __name__ == "__main__":
    result = {}
    for source, paths in inventory().items():
        counts = {}
        with ThreadPoolExecutor(12) as pool:
            for i, row in enumerate(pool.map(one, paths)):
                for key, value in row.items():
                    counts[key] = counts.get(key, 0) + value
                if (i + 1) % 100 == 0:
                    print(source, i + 1, counts, flush=True)
        result[source] = counts
        write_json(
            result,
            "s3://marin-us-east-02a/marin/protein-structure/sparsa/data/teacher_audit.json",
        )
    print(json.dumps(result), flush=True)
