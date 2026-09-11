import copy
from dataclasses import asdict

import pytest
import torch

from scripts.average_checkpoints import average_ema
from sparsa.cli import load_model
from sparsa.model import ALPHABET, ContactModel, ModelConfig, tokenize


def checkpoints(tmp_path):
    config = ModelConfig(
        sequence_dim=16,
        sequence_layers=1,
        heads=2,
        pair_dim=8,
        pair_layers=1,
        triangle_every=0,
        triangle_dim=4,
    )
    model = ContactModel(config).eval()
    first = {
        "format_version": 2,
        "alphabet": ALPHABET,
        "model_config": asdict(config),
        "training_config": {"crop": 12, "pos_weight": 4},
        "step": 100,
        "ema": copy.deepcopy(model.state_dict()),
    }
    second = copy.deepcopy(first)
    second["step"] = 200
    second["ema"]["head.1.bias"].add_(2)
    paths = [str(tmp_path / "first.pt"), str(tmp_path / "second.pt")]
    torch.save(first, paths[0])
    torch.save(second, paths[1])
    return model, second, paths


def test_average_checkpoint_reloads_expected_predictions(tmp_path):
    expected, _, paths = checkpoints(tmp_path)
    tokens = tokenize("ACDEFGHIKLMN")[None]
    with torch.inference_mode():
        original_logits = expected(tokens)
        expected.head[-1].bias.add_(1)
        expected_logits = expected(tokens)
    average = average_ema(paths)
    path = tmp_path / "average.pt"
    torch.save(average, path)
    loaded, state = load_model(str(path), torch.device("cpu"))
    with torch.inference_mode():
        actual_logits = loaded(tokens)
    assert not torch.equal(original_logits, actual_logits)
    torch.testing.assert_close(actual_logits, expected_logits)
    assert state["step"] == 200
    assert [r["weight"] for r in state["averaged_checkpoints"]] == [0.5, 0.5]
    assert [r["checkpoint"] for r in state["averaged_checkpoints"]] == paths


def test_average_rejects_mismatched_amino_acid_semantics(tmp_path):
    _, second, paths = checkpoints(tmp_path)
    second["alphabet"] = ALPHABET[::-1]
    torch.save(second, paths[1])
    with pytest.raises(ValueError, match="alphabet"):
        average_ema(paths)


def test_average_accepts_execution_change_but_rejects_architecture(tmp_path):
    _, second, paths = checkpoints(tmp_path)
    second["model_config"]["gradient_checkpointing"] = False
    torch.save(second, paths[1])
    assert average_ema(paths)["step"] == 200
    second["model_config"]["dropout"] = 0.1
    torch.save(second, paths[1])
    with pytest.raises(ValueError, match="model_config"):
        average_ema(paths)
