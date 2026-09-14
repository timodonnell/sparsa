"""Randomly initialized sequence encoder and symmetric residue-pair reasoning.

The only biological input is an amino-acid token per residue. Rotary sequence
attention supplies global context; pair convolutions model local contact motifs,
while gated triangle multiplication propagates evidence through shared residues.
Padding is removed after every spatial operation so batches cannot communicate
through padded pair cells. The output is an unordered-pair Bernoulli logit.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
TOKEN_IDS = {aa: i + 1 for i, aa in enumerate(ALPHABET)}


def tokenize(sequence: str) -> torch.Tensor:
    sequence = "".join(sequence.split()).upper()
    if not sequence or any(aa not in TOKEN_IDS for aa in sequence):
        raise ValueError("Sequence must contain canonical amino acids or X")
    return torch.tensor([TOKEN_IDS[aa] for aa in sequence], dtype=torch.long)


@dataclass
class ModelConfig:
    sequence_dim: int = 384
    sequence_layers: int = 8
    heads: int = 8
    pair_dim: int = 64
    pair_layers: int = 12
    triangle_every: int = 3
    triangle_dim: int = 32
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    pair_family: str = "conv"
    pair_directional: bool = False
    pair_outer_rank: int = 32
    pair_attention_heads: int = 4
    pair_attention_every: int = 2
    pair_attention_chunk: int = 16
    pair_residue_dim: int = 256


def rotary(x: torch.Tensor) -> torch.Tensor:
    length, dim = x.shape[-2:]
    freq = torch.arange(0, dim, 2, device=x.device).float() / dim
    phase = (
        torch.arange(length, device=x.device).float()[:, None] * (10000**-freq)[None]
    )
    cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


class SequenceBlock(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.heads = c.heads
        self.norm = nn.LayerNorm(c.sequence_dim)
        self.qkv = nn.Linear(c.sequence_dim, c.sequence_dim * 3, bias=False)
        self.out = nn.Linear(c.sequence_dim, c.sequence_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(c.sequence_dim),
            nn.Linear(c.sequence_dim, c.sequence_dim * 4),
            nn.GELU(),
            nn.Linear(c.sequence_dim * 4, c.sequence_dim),
            nn.Dropout(c.dropout),
        )

    def forward(self, x, valid):
        b, length, dim = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .view(b, length, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        y = F.scaled_dot_product_attention(
            rotary(q), rotary(k), v, attn_mask=valid[:, None, None, :]
        )
        x = x + self.out(y.transpose(1, 2).reshape(b, length, dim))
        return (x + self.ff(x)) * valid[..., None]


class Triangle(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ab = nn.Linear(dim, hidden * 2)
        self.gates = nn.Linear(dim, hidden * 2)
        self.outnorm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, dim)
        self.gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, mask):
        x = self.norm(z)
        a, b = (self.ab(x) * self.gates(x).sigmoid() * mask[..., None]).chunk(2, -1)
        # Outgoing shared-neighbor and incoming shared-neighbor contractions.
        y = torch.einsum("bikc,bjkc->bijc", a, b) + torch.einsum(
            "bkic,bkjc->bijc", a, b
        )
        y = y / (2 * mask[:, :, 0].sum(-1).clamp_min(1).sqrt()[:, None, None, None])
        return z + self.out(self.outnorm(y)) * self.gate(x).sigmoid() * mask[..., None]


class PairBlock(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.directional = c.pair_directional
        dilation = [1, 2, 4, 8][index % 4]
        self.norm = nn.LayerNorm(c.pair_dim)
        self.conv = nn.Conv2d(
            c.pair_dim, c.pair_dim, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.LayerNorm(c.pair_dim)
        self.ff = nn.Sequential(
            nn.Linear(c.pair_dim, c.pair_dim * 2),
            nn.GELU(),
            nn.Linear(c.pair_dim * 2, c.pair_dim),
        )
        self.triangle = (
            Triangle(c.pair_dim, c.triangle_dim)
            if c.triangle_every and (index + 1) % c.triangle_every == 0
            else None
        )

    def forward(self, z, mask):
        x = (self.norm(z) * mask[..., None]).permute(0, 3, 1, 2)
        z = (z + F.gelu(self.conv(x).permute(0, 2, 3, 1))) * mask[..., None]
        z = (z + self.ff(self.norm2(z))) * mask[..., None]
        if self.triangle is not None:
            z = self.triangle(z, mask)
        return z if self.directional else (z + z.transpose(1, 2)) * 0.5


class ContactModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if (
            config.sequence_dim % config.heads
            or (config.sequence_dim // config.heads) % 2
        ):
            raise ValueError("Sequence head dimension must be even")
        self.config = config
        if config.pair_family not in {"conv", "triangle", "attention", "joint"}:
            raise ValueError("Unknown pair family")
        if config.pair_family != "conv" and not config.pair_directional:
            raise ValueError("New pair families require directed state")
        # Optional inference readout, selected using validation only. This is
        # separate from the training architecture for old checkpoint compatibility.
        self.relative_max_distance = None
        self.embedding = nn.Embedding(
            len(ALPHABET) + 1, config.sequence_dim, padding_idx=0
        )
        self.sequence = nn.ModuleList(
            [SequenceBlock(config) for _ in range(config.sequence_layers)]
        )
        self.seqnorm = nn.LayerNorm(config.sequence_dim)
        if config.pair_directional:
            from sparsa.pair_trunk import ResidueToPair

            self.pair_init = ResidueToPair(
                config.sequence_dim,
                config.pair_dim,
                config.pair_outer_rank,
                zero_output=False,
            )
            self.relative = nn.Embedding(257, config.pair_dim)
        else:
            self.left = nn.Linear(config.sequence_dim, config.pair_dim)
            self.product = nn.Linear(config.sequence_dim, config.pair_dim)
            self.relative = nn.Embedding(130, config.pair_dim)
        if config.pair_family == "conv":
            self.pairs = nn.ModuleList(
                [PairBlock(config, i) for i in range(config.pair_layers)]
            )
        else:
            from sparsa.pair_trunk import ReasoningBlock

            self.pairs = nn.ModuleList(
                [ReasoningBlock(config, i) for i in range(config.pair_layers)]
            )
        if config.pair_family == "joint":
            self.residue_init = nn.Linear(config.sequence_dim, config.pair_residue_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(config.pair_dim), nn.Linear(config.pair_dim, 1)
        )
        nn.init.constant_(self.head[-1].bias, -3.0)

    @torch.no_grad()
    def initialize_distant_buckets(self, max_trained_distance):
        """Extend a shorter-crop checkpoint with its learned edge-distance prior.

        Initially this reproduces distance capping, while separate distant rows
        can subsequently learn from longer training examples.
        """
        if not isinstance(max_trained_distance, int) or max_trained_distance < 0:
            raise ValueError("Expected a nonnegative trained separation")
        bucket = min(
            128,
            max_trained_distance
            if max_trained_distance < 64
            else 64 + int(math.log2(max_trained_distance / 64) * 8),
        )
        if self.config.pair_directional:
            self.relative.weight[129 + bucket :] = self.relative.weight[128 + bucket]
            self.relative.weight[: 128 - bucket] = self.relative.weight[128 - bucket]
        else:
            self.relative.weight[bucket + 1 :] = self.relative.weight[bucket]
        return bucket

    def forward(self, tokens):
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
        sep = (positions[:, None] - positions[None, :]).abs()
        # Logarithmic bins are still learned: unseen distances may need capping
        # at the largest separation observed in a training crop.
        if self.relative_max_distance is not None:
            sep = sep.clamp_max(self.relative_max_distance)
        bucket = torch.where(
            sep < 64, sep, 64 + ((sep.float() / 64).clamp_min(1).log2() * 8).long()
        ).clamp_max(128)
        mask = valid[:, :, None] & valid[:, None, :]
        if self.config.pair_directional:
            signed = (positions[:, None] - positions[None, :]).sign()
            z = self.pair_init(x) + self.relative(128 + signed * bucket)[None]
        else:
            a, p = self.left(x), self.product(x)
            z = (
                a[:, :, None]
                + a[:, None, :]
                + p[:, :, None] * p[:, None, :] / math.sqrt(self.config.pair_dim)
                + self.relative(bucket)[None]
            )
        z = z * mask[..., None]
        s = (
            self.residue_init(x) * valid[..., None]
            if self.config.pair_family == "joint"
            else None
        )
        for block in self.pairs:
            if self.config.pair_family == "joint":
                if self.training and self.config.gradient_checkpointing:
                    z, s = checkpoint(block, z, mask, s, use_reentrant=False)
                else:
                    z, s = block(z, mask, s)
            elif self.training and self.config.gradient_checkpointing:
                z = checkpoint(block, z, mask, use_reentrant=False)
            else:
                z = block(z, mask)
        logits = self.head(z).squeeze(-1)
        return (logits + logits.transpose(1, 2)) * 0.5
