"""Deep sequence-only Pairformer conditioning for binary contact diffusion.

The cached/noisy ablation moves the same noise/time embedding across the trunk:
after the trunk for cached conditioning, before block one for noisy conditioning.
All arms retain the same single stream, pair updates and four-block output head.
The single stream uses pair-biased attention and reaches the loss through its
projection into the denoiser. No MSA, pretrained feature or convolution is used.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from sparsa.model import ALPHABET, SequenceBlock
from sparsa.pair_trunk import ResidueToPair, TriangleAttention, TriangleUpdate, zero


class GatedTransition(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, 2 * expansion * width, bias=False)
        self.output = zero(nn.Linear(expansion * width, width, bias=False))

    def forward(self, x):
        a, b = self.input(self.norm(x)).chunk(2, dim=-1)
        return self.output(F.silu(a) * b)


class PairUpdate(nn.Module):
    def __init__(self, config, checkpoint_chunks=True):
        super().__init__()
        c = config.pair_dim
        self.outgoing = TriangleUpdate(c, config.triangle_dim)
        self.incoming = TriangleUpdate(c, config.triangle_dim, incoming=True)
        self.starting = TriangleAttention(
            c,
            config.pair_attention_heads,
            config.pair_attention_chunk,
            checkpoint_chunks=checkpoint_chunks,
        )
        self.ending = TriangleAttention(
            c,
            config.pair_attention_heads,
            config.pair_attention_chunk,
            ending=True,
            checkpoint_chunks=checkpoint_chunks,
        )
        self.transition = GatedTransition(c, config.transition_expansion)

    def forward(self, z, mask):
        z = self.incoming(self.outgoing(z, mask), mask)
        z = self.ending(self.starting(z, mask), mask)
        return (z + self.transition(z)) * mask[..., None]


class PairformerBlock(nn.Module):
    """Pair operations followed by pair-biased single attention and SwiGLU."""

    def __init__(self, config, checkpoint_chunks=True):
        super().__init__()
        single = config.sequence_dim
        self.heads = config.heads
        self.pair = PairUpdate(config, checkpoint_chunks=checkpoint_chunks)
        self.single_norm = nn.LayerNorm(single)
        self.pair_norm = nn.LayerNorm(config.pair_dim)
        self.pair_bias = nn.Linear(config.pair_dim, self.heads, bias=False)
        self.qkv = nn.Linear(single, single * 3, bias=False)
        self.gate = nn.Linear(single, single)
        self.output = zero(nn.Linear(single, single))
        self.transition = GatedTransition(single, config.transition_expansion)

    def forward(self, z, s, mask):
        z = self.pair(z, mask)
        valid = mask.any(-1)
        batch, length, width = s.shape
        x = self.single_norm(s)
        q, k, v = (
            self.qkv(x)
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        bias = self.pair_bias(self.pair_norm(z)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~valid[:, None, None, :], -torch.inf)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        y = y.transpose(1, 2).reshape(batch, length, width)
        s = s + self.output(y * self.gate(x).sigmoid())
        s = (s + self.transition(s)) * valid[..., None]
        return z, s


class PairformerDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.pairformer_mode not in {"cached", "noisy"}:
            raise ValueError("Pairformer mode must be cached or noisy")
        if config.pairformer_layers < 1 or config.denoiser_layers < 1:
            raise ValueError("Pairformer and denoiser depths must be positive")
        if (
            config.sequence_dim % config.heads
            or (config.sequence_dim // config.heads) % 2
        ):
            raise ValueError("Sequence heads require an even rotary dimension")
        if (
            config.self_conditioning
            or config.inner_loops != 1
            or config.untied_cells != 1
        ):
            raise ValueError(
                "Pairformer campaign uses independent blocks without self-conditioning"
            )
        self.config = config
        self.embedding = nn.Embedding(
            len(ALPHABET) + 1, config.sequence_dim, padding_idx=0
        )
        self.sequence = nn.ModuleList(
            SequenceBlock(config) for _ in range(config.sequence_layers)
        )
        self.seqnorm = nn.LayerNorm(config.sequence_dim)
        self.pair_init = ResidueToPair(
            config.sequence_dim,
            config.pair_dim,
            config.pair_outer_rank,
            zero_output=False,
        )
        self.relative = nn.Embedding(257, config.pair_dim)
        # The outer block checkpoint already bounds activation lifetime. Inner
        # chunk checkpoints would recompute attention a second time in backward.
        checkpoint_chunks = not config.gradient_checkpointing
        self.trunk = nn.ModuleList(
            PairformerBlock(config, checkpoint_chunks=checkpoint_chunks)
            for _ in range(config.pairformer_layers)
        )
        self.noisy_state = nn.Embedding(2, config.pair_dim)
        self.time = nn.Embedding(config.diffusion_steps + 1, config.pair_dim)
        self.single_norm = nn.LayerNorm(config.sequence_dim)
        self.single_left = nn.Linear(config.sequence_dim, config.pair_dim)
        self.single_right = nn.Linear(config.sequence_dim, config.pair_dim)
        self.denoiser = nn.ModuleList(
            PairUpdate(config, checkpoint_chunks=checkpoint_chunks)
            for _ in range(config.denoiser_layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.pair_dim), nn.Linear(config.pair_dim, 1)
        )
        nn.init.constant_(self.head[-1].bias, -3.0)

    def _run(self, module, *args):
        if self.training and self.config.gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _trunk(self, z, s, mask):
        for block in self.trunk:
            z, s = self._run(block, z, s, mask)
        return z, s

    def encode(self, tokens):
        valid = tokens != 0
        s = self.embedding(tokens)
        for block in self.sequence:
            s = self._run(block, s, valid)
        s = self.seqnorm(s)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        delta = positions[:, None] - positions[None, :]
        sep = delta.abs()
        bucket = torch.where(
            sep < 64, sep, 64 + ((sep.float() / 64).clamp_min(1).log2() * 8).long()
        ).clamp_max(128)
        mask = valid[:, :, None] & valid[:, None, :]
        z = (
            self.pair_init(s) + self.relative(128 + delta.sign() * bucket)[None]
        ) * mask[..., None]
        if self.config.pairformer_mode == "cached":
            z, s = self._trunk(z, s, mask)
        return z, mask, s

    def denoise(self, encoded, noisy, timestep, self_condition=None):
        z, mask, s = encoded
        if self_condition is not None:
            raise ValueError("Self-conditioning is disabled")
        if noisy.shape != mask.shape or timestep.shape != (noisy.shape[0],):
            raise ValueError("Noisy map or timestep shape mismatch")
        z = (
            z + self.noisy_state(noisy.long()) + self.time(timestep)[:, None, None]
        ) * mask[..., None]
        if self.config.pairformer_mode == "noisy":
            z, s = self._trunk(z, s, mask)
        single = self.single_norm(s)
        z = (
            z
            + self.single_left(single)[:, :, None]
            + self.single_right(single)[:, None, :]
        ) * mask[..., None]
        for block in self.denoiser:
            z = self._run(block, z, mask)
        logits = self.head(z).squeeze(-1)
        return (logits + logits.transpose(1, 2)) * 0.5

    def forward(self, tokens, noisy, timestep, self_condition=None):
        return self.denoise(self.encode(tokens), noisy, timestep, self_condition)
