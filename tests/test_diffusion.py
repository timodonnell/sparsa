import torch

from sparsa.diffusion import (
    BinaryDiffusion,
    DiffusionModelConfig,
    TriangleDiffusionModel,
    sample_contact_maps,
)


def tiny(inner_loops=1, untied_cells=1):
    return DiffusionModelConfig(
        sequence_dim=32,
        sequence_layers=1,
        heads=4,
        pair_dim=8,
        triangle_dim=4,
        pair_attention_heads=2,
        pair_attention_chunk=3,
        pair_outer_rank=4,
        transition_expansion=2,
        diffusion_steps=4,
        inner_loops=inner_loops,
        untied_cells=untied_cells,
        gradient_checkpointing=False,
    )


def example_tokens():
    tokens = torch.randint(1, 22, (2, 13))
    tokens[0, 11:] = 0
    return tokens


def test_binary_schedule_terminal_prior_and_exact_posteriors():
    schedule = BinaryDiffusion(4, (0.2, 0.1, 0.05))
    torch.testing.assert_close(
        schedule.qbar[-1, :, 0, 1], schedule.priors, atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        schedule.qbar[-1, :, 1, 1], schedule.priors, atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        schedule.posterior[1:].sum(-1),
        torch.ones_like(schedule.posterior[1:].sum(-1)),
        atol=1e-6,
        rtol=0,
    )


def test_forward_noise_is_symmetric_and_respects_pair_mask():
    schedule = BinaryDiffusion(4, (0.2, 0.1, 0.05))
    tokens = example_tokens()
    clean = torch.zeros(2, 13, 13)
    clean[:, 0, 8] = clean[:, 8, 0] = 1
    noisy = schedule.sample_forward(
        clean,
        tokens,
        torch.tensor([1, 4]),
        torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(noisy, noisy.transpose(1, 2))
    assert not noisy[:, torch.arange(13), torch.arange(13)].any()
    assert not noisy[0, 11:].any() and not noisy[0, :, 11:].any()
    i, j = torch.nonzero(noisy[1], as_tuple=True)
    assert ((i - j).abs() >= 6).all()


def test_all_denoisers_backpropagate_through_every_parameter():
    torch.manual_seed(8)
    tokens = example_tokens()
    noisy = torch.zeros_like(tokens[:, :, None] * tokens[:, None, :], dtype=torch.bool)
    timestep = torch.tensor([2, 3])
    for config in (tiny(1, 1), tiny(2, 1), tiny(1, 2)):
        model = TriangleDiffusionModel(config)
        output = model(tokens, noisy, timestep)
        output.square().mean().backward()
        assert output.shape == (2, 13, 13)
        torch.testing.assert_close(output, output.transpose(1, 2))
        assert all(parameter.numel() for parameter in model.parameters())
        assert all(parameter.grad is not None for parameter in model.parameters())
        for cell in model.cells:
            assert cell.transition[-1].weight.grad is not None


def test_sampling_runs_shared_cell_over_reverse_steps():
    torch.manual_seed(9)
    model = TriangleDiffusionModel(tiny()).eval()
    schedule = BinaryDiffusion(4, (0.2, 0.1, 0.05))
    tokens = torch.randint(1, 22, (1, 12))
    state, probability = sample_contact_maps(
        model,
        schedule,
        tokens,
        3,
        torch.Generator().manual_seed(11),
    )
    assert state.shape == probability.shape == (3, 12, 12)
    torch.testing.assert_close(state, state.transpose(1, 2))
    assert torch.isfinite(probability).all()
