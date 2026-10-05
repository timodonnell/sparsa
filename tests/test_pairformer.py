from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from scripts.profile_pairformer_overhead import original_attention_forward
from sparsa.diffusion import (
    BinaryDiffusion,
    DiffusionModelConfig,
    build_diffusion_model,
    sample_contact_maps,
)
from sparsa.evaluate_diffusion import evaluate_records
from sparsa.pair_trunk import TriangleAttention


def config(mode, checkpointing=False, width=8):
    return DiffusionModelConfig(
        sequence_dim=16,
        sequence_layers=1,
        heads=2,
        pair_dim=width,
        triangle_dim=width,
        pair_attention_heads=2,
        pair_attention_chunk=4,
        pair_outer_rank=2,
        transition_expansion=2,
        diffusion_steps=2,
        gradient_checkpointing=checkpointing,
        pairformer_layers=3,
        pairformer_mode=mode,
        denoiser_layers=2,
    )


@pytest.mark.parametrize("mode", ["cached", "noisy"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_pairformer_loss_reaches_all_blocks_and_both_streams(mode, checkpointing):
    torch.manual_seed(31)
    model = build_diffusion_model(config(mode, checkpointing))
    tokens = torch.randint(1, 22, (2, 9))
    tokens[0, 7:] = 0
    logits = model(tokens, torch.zeros(2, 9, 9, dtype=torch.bool), torch.tensor([1, 2]))
    torch.testing.assert_close(logits, logits.transpose(1, 2))
    logits.square().mean().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )
    assert model.single_left.weight.grad.abs().sum() > 0
    assert model.sequence[0].qkv.weight.grad.abs().sum() > 0
    for block in model.trunk:
        assert block.output.weight.grad.abs().sum() > 0
        assert block.pair.transition.output.weight.grad.abs().sum() > 0


def test_factorial_arms_have_identical_parameters_and_only_move_noise_entry():
    torch.manual_seed(32)
    cached = build_diffusion_model(config("cached")).eval()
    noisy = build_diffusion_model(config("noisy")).eval()
    noisy.load_state_dict(cached.state_dict(), strict=True)
    assert sum(p.numel() for p in cached.parameters()) == sum(
        p.numel() for p in noisy.parameters()
    )
    tokens = torch.randint(1, 22, (1, 9))
    zero = torch.zeros(1, 9, 9, dtype=torch.bool)
    one = zero.clone()
    one[:, 0, 8] = one[:, 8, 0] = True
    t = torch.tensor([1])
    for model in (cached, noisy):
        inputs = []
        hook = model.trunk[0].register_forward_pre_hook(
            lambda module, args, inputs=inputs: inputs.append(args[0].detach().clone())
        )
        a = model(tokens, zero, t)
        b = model(tokens, one, t)
        assert not torch.allclose(a, b)
        if model.config.pairformer_mode == "cached":
            torch.testing.assert_close(inputs[0], inputs[1])
        else:
            assert not torch.equal(inputs[0], inputs[1])
        hook.remove()


@pytest.mark.parametrize("mode", ["cached", "noisy"])
def test_cached_features_preserve_seeded_sampling_and_are_not_mutated(mode):
    torch.manual_seed(33)
    model = build_diffusion_model(config(mode)).eval()
    tokens = torch.randint(1, 22, (1, 9))
    schedule = BinaryDiffusion(2)
    with torch.inference_mode():
        encoded = model.encode(tokens)
        before = tuple(x.clone() for x in encoded)
    args = (model, schedule, tokens, 2)
    a = sample_contact_maps(*args, torch.Generator().manual_seed(34))
    b = sample_contact_maps(*args, torch.Generator().manual_seed(34), encoded=encoded)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for x, y in zip(before, encoded, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert a[1].isfinite().all()


@pytest.mark.parametrize("mode", ["cached", "noisy"])
def test_evaluation_caches_once_per_protein_but_noisy_trunk_runs_every_step(mode):
    model = build_diffusion_model(config(mode)).eval()
    calls = {"sequence": 0, "trunk": 0}

    def count(name):
        def callback(module, args):
            calls[name] += 1

        return callback

    hooks = [
        model.sequence[0].register_forward_pre_hook(count("sequence")),
        model.trunk[0].register_forward_pre_hook(count("trunk")),
    ]
    rec = {
        "dataset": "test",
        "stem": "x",
        "eval_set": "eval-val",
        "L": 9,
        "sequence": "ACDEFGHIK",
        "resolved": list(range(9)),
        "contacts": [(0, 8, 1.0)],
    }
    proteins, _ = evaluate_records(
        model,
        BinaryDiffusion(2),
        [rec],
        torch.device("cpu"),
        n_rollouts=3,
        rollout_batch=1,
    )
    assert calls == {"sequence": 1, "trunk": 1 if mode == "cached" else 6}
    assert proteins[0]["n_rollouts"] == 3
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("mode", ["cached", "noisy"])
def test_pairformer_valid_logits_ignore_padding_and_checkpointing_matches(mode):
    torch.manual_seed(35)
    plain = build_diffusion_model(config(mode))
    checked = build_diffusion_model(replace(config(mode), gradient_checkpointing=True))
    checked.load_state_dict(plain.state_dict())
    tokens = torch.randint(1, 22, (1, 8))
    noisy = torch.zeros(1, 8, 8, dtype=torch.bool)
    t = torch.tensor([1])
    a, b = plain(tokens, noisy, t), checked(tokens, noisy, t)
    torch.testing.assert_close(a, b)
    a.sum().backward()
    b.sum().backward()
    for x, y in zip(plain.parameters(), checked.parameters(), strict=True):
        torch.testing.assert_close(x.grad, y.grad)
    padded = torch.nn.functional.pad(tokens, (0, 3))
    out = plain(padded, torch.zeros(1, 11, 11, dtype=torch.bool), t)
    torch.testing.assert_close(out[:, :8, :8], a, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", ["cached", "noisy"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_attention_bookkeeping_preserves_logits_gradients_and_updates(
    mode, checkpointing
):
    torch.manual_seed(36)
    original = build_diffusion_model(config(mode, checkpointing))
    # Nonzero residual projections exercise every path, unlike fresh zero init.
    with torch.no_grad():
        for module in original.modules():
            if isinstance(module, torch.nn.Linear) and not module.weight.any():
                torch.nn.init.normal_(module.weight, std=0.02)
    optimized = build_diffusion_model(config(mode, checkpointing))
    optimized.load_state_dict(original.state_dict())
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (original, optimized)
    ]
    tokens = torch.randint(1, 22, (2, 9))
    tokens[0, 7:] = 0  # Partial final chunk and padding in both attention axes.
    noisy = torch.rand(2, 9, 9) > 0.5
    timesteps = torch.tensor([1, 2])
    for _ in range(2):
        outputs = []
        for model, optimizer, forward in zip(
            (original, optimized),
            optimizers,
            (original_attention_forward, TriangleAttention.forward),
            strict=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            with patch.object(TriangleAttention, "forward", forward):
                logits = model(tokens, noisy, timesteps)
                logits.square().mean().backward()
            outputs.append(logits.detach())
            optimizer.step()
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for a, b in zip(original.parameters(), optimized.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
