"""Seed ideas followed by structured code proposals from the installed Codex CLI."""

import json
import os
import signal
import subprocess
from pathlib import Path

from research.core import write_json

SCHEMA = {
    "type": "object",
    "properties": {
        k: {"type": "string"}
        for k in ("hypothesis", "model_source", "model_config_json")
    },
    "required": ["hypothesis", "model_source", "model_config_json"],
    "additionalProperties": False,
}


class ProposerUnavailable(RuntimeError):
    """Authentication, CLI, or service failure; stop instead of burning trial ideas."""


def seeded_proposal(index, source, config):
    cfg = dict(config)
    if index == 1:
        cfg.update(sequence_layers=16, pair_layers=8)
        hypothesis = "Move depth toward global sequence context: 16 sequence blocks and 8 pair blocks."
    elif index == 2:
        cfg.update(sequence_layers=8, pair_layers=24)
        hypothesis = "Move depth toward explicit pair reasoning: 8 sequence blocks and 24 pair blocks."
    elif index == 3:
        old = "self.relative = nn.Embedding(130, config.pair_dim)"
        new = (
            old
            + "\n        self.difference = nn.Linear(config.sequence_dim, config.pair_dim, bias=False)"
        )
        anchor = "a[:, :, None]\n            + a[:, None, :]"
        if old not in source or anchor not in source:
            raise ValueError(
                "Seeded code proposal needs the original model; use Codex for this parent"
            )
        source = source.replace(old, new).replace(
            anchor,
            anchor
            + "\n            + self.difference((x[:, :, None] - x[:, None, :]).abs())",
        )
        hypothesis = "Add symmetric absolute sequence-feature differences to pair initialization, complementing sums and products."
    else:
        raise ValueError("Seed idea queue exhausted")
    return {
        "hypothesis": hypothesis,
        "model_source": source,
        "model_config_json": json.dumps(cfg),
    }


def codex_proposal(directory, source, config, history, protocol, settings, guidance=""):
    """No benchmark files or credentials are copied into the proposal directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "schema.json", SCHEMA)
    prompt = (
        Path(__file__).with_name("program.md").read_text()
        + "\n\nPROTOCOL\n"
        + json.dumps(protocol, indent=2)
        + "\n\nPARENT MODEL CONFIGURATION\n"
        + json.dumps(config)
        + "\n\nPARENT MODEL SOURCE\n"
        + source
        + "\n\nEXPERIMENT HISTORY (validation only)\n"
        + json.dumps(history, indent=2)
        + "\n\nRESEARCH STEERING\n"
        + guidance
    )
    (directory / "prompt.txt").write_text(prompt)
    output = directory / "proposal.json"
    if output.exists():
        return json.loads(output.read_text())
    argv = [
        settings.get("codex", "codex"),
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        str(directory / "schema.json"),
        "--output-last-message",
        str(output),
        "-",
    ]
    if settings.get("codex_model"):
        argv[2:2] = ["--model", settings["codex_model"]]
    write_json(
        directory / "invocation.json",
        {"argv": argv, "timeout_seconds": protocol["proposer_timeout_seconds"]},
    )
    with (directory / "events.jsonl").open("w") as log:
        process = subprocess.Popen(
            argv,
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(prompt, timeout=protocol["proposer_timeout_seconds"])
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if process.returncode:
        raise ProposerUnavailable(
            f"Codex proposal failed ({process.returncode}); see {directory / 'events.jsonl'}"
        )
    return json.loads(output.read_text())
