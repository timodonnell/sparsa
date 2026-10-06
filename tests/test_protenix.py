"""Transfer-specific invariants that can be checked without pretrained assets."""

import copy
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("optree")
pytest.importorskip("rdkit")

from scripts.run_protenix_pilot import stage_plan
from sparsa.diffusion import DiffusionModelConfig, build_diffusion_model
from sparsa.model import TOKEN_IDS
from sparsa.protenix import sequence_features
from sparsa.train_diffusion import validate_transfer_branch


def config():
    return yaml.safe_load(Path("configs/protenix_v1_frozen.yaml").read_text())


def test_sequence_features_are_query_only_and_chemical():
    tokens = [TOKEN_IDS[x] for x in "AGX"]
    f = sequence_features(tokens, "cpu")
    assert torch.equal(f["restype"], f["profile"])
    assert not f["deletion_mean"].any()
    assert not f["token_bonds"].any()
    assert not {"msa", "template", "coordinates", "esm_token_embedding"} & f.keys()
    assert f["restype"].argmax(-1).tolist() == [0, 7, 20]
    names = [
        "".join(chr(c + 32) for c in row).strip()
        for row in f["ref_atom_name_chars"].argmax(-1).tolist()
    ]
    assert names.count("OXT") == 1
    assert names[-1] == "OXT"
    for i in range(3):
        xyz = f["ref_pos"][f["atom_to_token_idx"] == i]
        torch.testing.assert_close(xyz.mean(0), torch.zeros(3), atol=1e-6, rtol=0)
    # C, N, O only: one-hot element indices are atomic number minus one.
    assert set(f["ref_element"].argmax(-1).tolist()) == {5, 6, 7}
    with pytest.raises(ValueError):
        sequence_features([0], "cpu")


def test_transfer_branch_allows_only_unfreezing():
    cfg = config()
    state = {"training_config": cfg, "world_size": 16, "rng": [{}] * 16}
    validate_transfer_branch(state, cfg, 16)
    changed = dict(cfg, train_backbone=True)
    validate_transfer_branch(state, changed, 16)
    for key, value in [
        ("lr", 1e-3),
        ("seed", 5),
        ("steps", 9000),
        ("logical_batch_size", 16),
    ]:
        with pytest.raises(ValueError):
            validate_transfer_branch(state, dict(changed, **{key: value}), 16)
    with pytest.raises(ValueError):
        validate_transfer_branch(state, changed, 8)
    assert cfg == config()


def test_pilot_branches_share_parent_and_global_batch():
    frozen = config()
    full = yaml.safe_load(Path("configs/protenix_v1_finetune.yaml").read_text())
    assert {k: v for k, v in frozen.items() if k != "train_backbone"} == {
        k: v for k, v in full.items() if k != "train_backbone"
    }
    assert frozen["logical_batch_size"] * 16 == 128
    assert frozen["logical_batch_size"] * frozen["ranking_group_size"] == 16
    stages = dict(stage_plan("s3://pilot", "frozen.yaml", "full.yaml"))
    for arm in ["frozen", "finetune"]:
        command = stages[arm]
        assert (
            command[command.index("--branch-from") + 1]
            == "s3://pilot/warmup/checkpoints/step-1000.pt"
        )
        assert command[command.index("--stop-after") + 1] == "6000"
    evaluation = stages["warmup-validation"]
    assert evaluation[evaluation.index("--n-rollouts") + 1] == "100"


def test_native_backbone_coverage_and_optimizer_unfreeze():
    torch.set_num_threads(2)
    model = build_diffusion_model(DiffusionModelConfig(**config()["model"]))
    assert sum(p.numel() for p in model.backbone.parameters()) == 148895680
    model.configure_trainability(False)
    groups = model.optimizer_groups(2e-4, 1e-5)
    optimizer = torch.optim.AdamW(groups)
    model.adapter_gate.square().backward()
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    model.configure_trainability(True)
    restored = torch.optim.AdamW(model.optimizer_groups(2e-4, 1e-5))
    restored.load_state_dict(saved)
    assert [g["base_lr"] for g in restored.param_groups] == [2e-4, 1e-5]
    assert len(restored.state) == 1
    # Unfreezing preserves adapter moments and lazily creates backbone moments.
    weight = model.backbone.linear_no_bias_z_cycle.weight
    model.zero_grad(set_to_none=True)
    weight.square().mean().backward()
    restored.step()
    assert weight in restored.state
