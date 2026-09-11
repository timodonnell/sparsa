"""Exercise a recovered checkpoint through the CLI and the actual Helico parser.

Uses a validation sequence only. This checks interoperability, not folding
quality or contact accuracy; experimental accuracy comes from final_evaluate.
Helico's Python environment must already have its dependencies installed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sparsa.data import benchmark


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--helico-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    record = benchmark(splits=("eval-val",))[0]
    sequence = record["sequence"]
    length = len(sequence)
    command = [
        sys.executable,
        "-m",
        "sparsa.cli",
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--sequence",
        sequence,
        "--out",
        str(out),
        "--device",
        args.device,
        "--top-l",
        "1",
    ]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, check=False
    )
    (out / "cli.stdout.txt").write_text(result.stdout)
    (out / "cli.stderr.txt").write_text(result.stderr)
    result.check_returncode()
    with np.load(out / "contacts.npz", allow_pickle=False) as matrix:
        scores, valid = matrix["score"], matrix["valid"]
        assert str(matrix["sequence"]) == sequence
    assert scores.shape == valid.shape == (length, length)
    assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
    assert np.allclose(scores, scores.T, atol=1e-6, rtol=0)
    expected_valid = abs(np.arange(length)[:, None] - np.arange(length)) >= 6
    assert valid.dtype == np.bool_ and np.array_equal(valid, expected_valid)
    pairs = np.loadtxt(out / "helico_contacts.txt", dtype=int, ndmin=2)
    i, j = np.triu_indices(length, k=6)
    ranked = np.argsort(-scores[i, j], kind="stable")[:length]
    assert np.array_equal(pairs, np.column_stack((i[ranked], j[ranked])))

    # Run in Helico's real environment, without replacing imported modules.
    helico_code = """
import hashlib, inspect, json, pathlib, subprocess, sys
import torch
import helico.contacts
import helico.data
from helico.inference import contacts_from_pairs
from helico.data import CONTACT_PRESENT, CONTACT_UNKNOWN
pairs = [tuple(map(int, line.split())) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
length = int(sys.argv[2])
contacts = contacts_from_pairs(pairs, seq_len=length, one_indexed=False, strict=True)
expected = torch.full((length, length), CONTACT_UNKNOWN, dtype=torch.uint8)
for i, j in pairs:
    expected[i, j] = expected[j, i] = CONTACT_PRESENT
assert torch.equal(contacts, expected)
source = pathlib.Path(inspect.getfile(contacts_from_pairs)).resolve()
revision = subprocess.check_output(['git', '-C', str(source.parent), 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', '-C', str(source.parent), 'status', '--porcelain'], text=True)
print(json.dumps(dict(parser=str(source), parser_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    dependency_sha256={inspect.getfile(m): hashlib.sha256(pathlib.Path(inspect.getfile(m)).read_bytes()).hexdigest()
        for m in (helico.contacts, helico.data)},
    repository_revision=revision, repository_dirty=bool(dirty), positive_unordered_pairs=len(pairs),
    unlisted_pairs_remain_unknown=True, strict_parser_passed=True)))
"""
    parsed = subprocess.run(
        [
            str(args.helico_python.absolute()),
            "-c",
            helico_code,
            str(out / "helico_contacts.txt"),
            str(length),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = {
        "observed_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "prediction": json.loads((out / "prediction.json").read_text()),
        "validation_protein": record["stem"],
        "sequence_length": length,
        "finite_symmetric_probability_matrix": True,
        "separation_mask_verified": True,
        "stable_top_l_export_verified": True,
        "helico": json.loads(parsed.stdout),
        "artifact_sha256": {
            name: sha256(out / name)
            for name in ("contacts.npz", "helico_contacts.txt", "prediction.json")
        },
        "verification_source_sha256": {
            str(path): sha256(path)
            for path in (Path(__file__), Path("sparsa/cli.py"), Path("sparsa/model.py"))
        },
        "scope": "CLI and contact-parser integration; not a Helico structure prediction or an accuracy evaluation",
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
