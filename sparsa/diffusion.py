"""Discrete contact-map diffusion with a convolution-free triangle denoiser.

The forward process independently refreshes every eligible binary edge toward
an empirical, sequence-separation-conditioned contact prior.  The denoiser is
global: outgoing/incoming triangle multiplication and starting/ending triangle
attention are followed by a pair transition.  G1 shares one cell over reverse
diffusion time; L2 additionally reuses that cell twice inside each denoising
call, while U2 uses two untied cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from sparsa.model import ALPHABET, SequenceBlock
from sparsa.pair_trunk import ResidueToPair, TriangleAttention, TriangleUpdate, zero


@dataclass
class DiffusionModelConfig:
    sequence_dim: int = 384
    sequence_layers: int = 6
    heads: int = 6
    pair_dim: int = 64
    triangle_dim: int = 64
    pair_attention_heads: int = 4
    pair_attention_chunk: int = 32
    pair_outer_rank: int = 32
    transition_expansion: int = 4
    diffusion_steps: int = 8
    inner_loops: int = 1
    untied_cells: int = 1
    dropout: float = 0.0
    gradient_checkpointing: bool = True


class TriangleDenoiserCell(nn.Module):
    """One convolution-free Evoformer-style pair update."""

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        c = config.pair_dim
        self.outgoing = TriangleUpdate(c, config.triangle_dim)
        self.incoming = TriangleUpdate(c, config.triangle_dim, incoming=True)
        self.starting = TriangleAttention(
            c,
            config.pair_attention_heads,
            config.pair_attention_chunk,
            ending=False,
        )
        self.ending = TriangleAttention(
            c,
            config.pair_attention_heads,
            config.pair_attention_chunk,
            ending=True,
        )
        hidden = c * config.transition_expansion
        self.transition = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, hidden),
            nn.GELU(),
            zero(nn.Linear(hidden, c)),
        )

    def forward(self, z, mask):
        z = self.incoming(self.outgoing(z, mask), mask)
        z = self.starting(z, mask)
        z = self.ending(z, mask)
        return (z + self.transition(z)) * mask[..., None]


class TriangleDiffusionModel(nn.Module):
    """Sequence-conditioned x0 predictor used at every reverse timestep."""

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        if config.sequence_dim % config.heads:
            raise ValueError("Sequence dimension must divide into heads")
        if (config.sequence_dim // config.heads) % 2:
            raise ValueError("Rotary sequence head dimension must be even")
        if config.pair_dim % config.pair_attention_heads:
            raise ValueError("Pair dimension must divide into attention heads")
        if config.inner_loops < 1 or config.untied_cells < 1:
            raise ValueError("Denoiser depth must be positive")
        if config.inner_loops > 1 and config.untied_cells > 1:
            raise ValueError("Choose tied inner loops or untied cells, not both")
        self.config = config
        self.embedding = nn.Embedding(
            len(ALPHABET) + 1, config.sequence_dim, padding_idx=0
        )
        self.sequence = nn.ModuleList(
            [SequenceBlock(config) for _ in range(config.sequence_layers)]
        )
        self.seqnorm = nn.LayerNorm(config.sequence_dim)
        self.pair_init = ResidueToPair(
            config.sequence_dim,
            config.pair_dim,
            config.pair_outer_rank,
            zero_output=False,
        )
        self.relative = nn.Embedding(257, config.pair_dim)
        self.noisy_state = nn.Embedding(2, config.pair_dim)
        self.time = nn.Embedding(config.diffusion_steps + 1, config.pair_dim)
        applications = max(config.inner_loops, config.untied_cells)
        self.loop = nn.Embedding(applications, config.pair_dim)
        self.condition_norm = (
            nn.LayerNorm(config.pair_dim) if applications > 1 else nn.Identity()
        )
        if applications > 1:
            self.reinject_gate = nn.Parameter(torch.full((applications - 1,), -4.0))
        else:
            self.register_parameter("reinject_gate", None)
        self.cells = nn.ModuleList(
            [TriangleDenoiserCell(config) for _ in range(config.untied_cells)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.pair_dim), nn.Linear(config.pair_dim, 1)
        )
        nn.init.constant_(self.head[-1].bias, -3.0)

    def encode(self, tokens):
        """Compute sequence-derived pair conditioning once per rollout batch."""
        valid = tokens != 0
        x = self.embedding(tokens)
        for block in self.sequence:
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(block, x, valid, use_reentrant=False)
            else:
                x = block(x, valid)
        x = self.seqnorm(x)
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        delta = positions[:, None] - positions[None, :]
        sep = delta.abs()
        bucket = torch.where(
            sep < 64, sep, 64 + ((sep.float() / 64).clamp_min(1).log2() * 8).long()
        ).clamp_max(128)
        signed_bucket = 128 + delta.sign() * bucket
        mask = valid[:, :, None] & valid[:, None, :]
        condition = (self.pair_init(x) + self.relative(signed_bucket)[None]) * mask[
            ..., None
        ]
        return condition, mask

    def denoise(self, encoded, noisy, timestep):
        """Predict clean contact logits from one discrete noisy map."""
        condition, mask = encoded
        if noisy.shape != mask.shape:
            raise ValueError("Noisy map shape does not match encoded sequence")
        if timestep.shape != (noisy.shape[0],):
            raise ValueError("Expected one timestep per protein")
        dynamic = self.noisy_state(noisy.long()) + self.time(timestep)[:, None, None]
        base = (condition + dynamic) * mask[..., None]
        z = base
        applications = max(self.config.inner_loops, self.config.untied_cells)
        for index in range(applications):
            if index:
                z = z + self.reinject_gate[index - 1].sigmoid() * self.condition_norm(
                    base
                )
            z = (z + self.loop.weight[index]) * mask[..., None]
            cell = self.cells[index] if self.config.untied_cells > 1 else self.cells[0]
            if self.training and self.config.gradient_checkpointing:
                z = checkpoint(cell, z, mask, use_reentrant=False)
            else:
                z = cell(z, mask)
        logits = self.head(z).squeeze(-1)
        return (logits + logits.transpose(1, 2)) * 0.5

    def forward(self, tokens, noisy, timestep):
        return self.denoise(self.encode(tokens), noisy, timestep)


class BinaryDiffusion(nn.Module):
    """Separation-conditioned binary D3PM schedule and exact reverse posterior."""

    def __init__(self, steps=8, priors=(0.025, 0.016, 0.005), cosine_offset=0.008):
        super().__init__()
        if steps < 2 or len(priors) != 3 or any(not 0 < p < 1 for p in priors):
            raise ValueError("Invalid binary diffusion schedule")
        self.steps = int(steps)
        prior = torch.tensor(priors, dtype=torch.float64)
        times = torch.arange(steps + 1, dtype=torch.float64) / steps
        alpha_bar = torch.cos(
            (times + cosine_offset) / (1 + cosine_offset) * math.pi / 2
        ).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_bar[-1] = 0.0
        q = torch.zeros(steps + 1, 3, 2, 2, dtype=torch.float64)
        qbar = torch.zeros_like(q)
        qbar[0] = torch.eye(2, dtype=torch.float64)[None]
        eye = torch.eye(2, dtype=torch.float64)
        for t in range(1, steps + 1):
            alpha = alpha_bar[t] / alpha_bar[t - 1]
            for b, p in enumerate(prior):
                stationary = torch.tensor([[1 - p, p], [1 - p, p]])
                q[t, b] = alpha * eye + (1 - alpha) * stationary
                qbar[t, b] = qbar[t - 1, b] @ q[t, b]
        posterior = torch.zeros(steps + 1, 3, 2, 2, 2, dtype=torch.float64)
        # [t, bucket, observed_xt, proposed_x0, previous_state]
        for t in range(1, steps + 1):
            for b in range(3):
                for xt in range(2):
                    for x0 in range(2):
                        weights = qbar[t - 1, b, x0] * q[t, b, :, xt]
                        posterior[t, b, xt, x0] = weights / weights.sum()
        self.register_buffer("priors", prior.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("qbar", qbar.float())
        self.register_buffer("posterior", posterior.float())

    @staticmethod
    def buckets(length, device):
        pos = torch.arange(length, device=device)
        sep = (pos[:, None] - pos[None, :]).abs()
        return torch.where(sep <= 11, 0, torch.where(sep <= 23, 1, 2)).long()

    @staticmethod
    def eligible(tokens):
        valid = tokens != 0
        length = tokens.shape[1]
        return (valid[:, :, None] & valid[:, None, :]) & torch.ones(
            length, length, dtype=torch.bool, device=tokens.device
        ).triu(6)

    @staticmethod
    def _symmetric_sample(probability, eligible, generator):
        draw = (
            torch.rand(
                probability.shape,
                device=probability.device,
                generator=generator,
                dtype=probability.dtype,
            )
            < probability
        )
        upper = draw & eligible
        return upper | upper.transpose(1, 2)

    def sample_forward(self, clean, tokens, timestep, generator):
        length = clean.shape[1]
        bucket = self.buckets(length, clean.device)
        clean_index = clean.long().clamp(0, 1)
        probability = self.qbar[timestep[:, None, None], bucket[None], clean_index, 1]
        return self._symmetric_sample(probability, self.eligible(tokens), generator)

    def sample_prior(self, tokens, generator):
        bucket = self.buckets(tokens.shape[1], tokens.device)
        probability = self.priors[bucket][None].expand(tokens.shape[0], -1, -1)
        return self._symmetric_sample(probability, self.eligible(tokens), generator)

    def sample_previous(
        self, noisy, clean_probability, tokens, timestep: int, generator
    ):
        if not 1 <= timestep <= self.steps:
            raise ValueError("Reverse timestep out of range")
        bucket = self.buckets(noisy.shape[1], noisy.device)
        observed = noisy.long()
        p_prev_given_0 = self.posterior[timestep, bucket[None], observed, 0, 1]
        p_prev_given_1 = self.posterior[timestep, bucket[None], observed, 1, 1]
        probability = (
            1 - clean_probability
        ) * p_prev_given_0 + clean_probability * p_prev_given_1
        return self._symmetric_sample(probability, self.eligible(tokens), generator)


@torch.inference_mode()
def sample_contact_maps(
    model,
    schedule,
    tokens,
    count,
    generator,
    temperature=1.0,
    logit_correction=0.0,
):
    """Sample a batch of complete maps and return final-step clean probabilities."""
    if tokens.shape[0] != 1 or count < 1 or temperature <= 0:
        raise ValueError(
            "Sampling expects one sequence, positive count and temperature"
        )
    base_condition, base_mask = model.encode(tokens)
    expanded_tokens = tokens.expand(count, -1)
    encoded = (
        base_condition.expand(count, -1, -1, -1),
        base_mask.expand(count, -1, -1),
    )
    noisy = schedule.sample_prior(expanded_tokens, generator)
    final_probability = None
    for t in range(schedule.steps, 0, -1):
        timestep = torch.full((count,), t, dtype=torch.long, device=tokens.device)
        logits = (
            model.denoise(encoded, noisy, timestep) - float(logit_correction)
        ) / temperature
        final_probability = logits.float().sigmoid()
        noisy = schedule.sample_previous(
            noisy, final_probability, expanded_tokens, t, generator
        )
    return noisy, final_probability
