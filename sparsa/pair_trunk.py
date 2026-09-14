"""Directed contact reasoning with triangle attention and residue feedback.

Only residue-derived hidden states enter these operators. Attention chunks the
fixed-node axis and checkpoints chunks during training to avoid retaining every
cubic attention activation for backward. Output symmetry belongs to the head.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def zero(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class ResidueToPair(nn.Module):
    def __init__(self, single, pair, rank, zero_output=True):
        super().__init__()
        self.rank = rank
        self.norm = nn.LayerNorm(single)
        self.left = nn.Linear(single, pair)
        self.right = nn.Linear(single, pair)
        self.a = nn.Linear(single, rank)
        self.b = nn.Linear(single, rank)
        # Contract one low-rank axis first: largest intermediate is [B,L,R,C].
        self.outer = nn.Parameter(torch.empty(rank, rank, pair))
        nn.init.normal_(self.outer, std=1 / rank)
        self.out = nn.Linear(pair, pair)
        if zero_output:
            zero(self.out)

    def forward(self, s):
        s = self.norm(s)
        a, b = self.a(s), self.b(s)
        left = torch.einsum("bia,adc->bidc", a, self.outer)
        product = torch.einsum("bidc,bjd->bijc", left, b) / math.sqrt(self.rank)
        return self.out(self.left(s)[:, :, None] + self.right(s)[:, None, :] + product)


class TriangleUpdate(nn.Module):
    def __init__(self, channels, hidden, incoming=False):
        super().__init__()
        self.incoming = incoming
        self.norm = nn.LayerNorm(channels)
        self.ab = nn.Linear(channels, hidden * 2)
        self.ab_gate = nn.Linear(channels, hidden * 2)
        self.outnorm = nn.LayerNorm(hidden)
        self.out = zero(nn.Linear(hidden, channels))
        self.gate = nn.Linear(channels, channels)

    def forward(self, z, mask):
        x = self.norm(z)
        a, b = (self.ab(x) * self.ab_gate(x).sigmoid() * mask[..., None]).chunk(2, -1)
        if self.incoming:
            a, b = a.transpose(1, 2), b.transpose(1, 2)
        y = torch.einsum("bikc,bjkc->bijc", a, b)
        n = mask.any(-1).sum(-1).clamp_min(1).sqrt()[:, None, None, None]
        return (z + self.out(self.outnorm(y / n)) * self.gate(x).sigmoid()) * mask[
            ..., None
        ]


class TriangleAttention(nn.Module):
    def __init__(self, channels, heads, chunk, ending=False):
        super().__init__()
        if channels % heads or chunk < 1:
            raise ValueError("Invalid triangle attention dimensions/chunk")
        self.heads, self.chunk, self.ending = heads, chunk, ending
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.bias = nn.Linear(channels, heads, bias=False)
        self.gate = nn.Linear(channels, channels)
        self.out = zero(nn.Linear(channels, channels))

    def _chunk(self, x, bias, mask):
        b, rows, length, channels = x.shape
        q, k, v = (
            self.qkv(x)
            .view(b, rows, length, 3, self.heads, channels // self.heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        # Shared third-edge bias [B,1,H,J,K], key validity [B,I,1,1,K].
        attn_mask = bias[:, None].masked_fill(~mask[:, :, None, None, :], -torch.inf)
        # CUDA fused SDPA backends require four-dimensional inputs. Keeping
        # a separate fixed-node dimension silently selects the cubic math path.
        q, k, v = [
            t.reshape(b * rows, self.heads, length, channels // self.heads)
            for t in (q, k, v)
        ]
        attn_mask = attn_mask.reshape(b * rows, self.heads, length, length)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.reshape(b, rows, self.heads, length, channels // self.heads)
        return y.permute(0, 1, 3, 2, 4).reshape(b, rows, length, channels)

    def forward(self, z, mask):
        if self.ending:
            z, mask = z.transpose(1, 2), mask.transpose(1, 2)
        x = self.norm(z) * mask[..., None]
        bias = self.bias(x).permute(0, 3, 1, 2)
        chunks = []
        for start in range(0, z.shape[1], self.chunk):
            args = (
                x[:, start : start + self.chunk],
                bias,
                mask[:, start : start + self.chunk],
            )
            if self.training and torch.is_grad_enabled():
                y = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                y = self._chunk(*args)
            chunks.append(y)
        y = torch.cat(chunks, 1)
        z = (z + self.out(y * self.gate(x).sigmoid())) * mask[..., None]
        return z.transpose(1, 2) if self.ending else z


class ResidueFeedback(nn.Module):
    def __init__(self, pair, single, outer_rank):
        super().__init__()
        if single % 8:
            raise ValueError("Residue state must divide into eight heads")
        self.norm = nn.LayerNorm(single)
        self.pairnorm = nn.LayerNorm(pair)
        self.bias = nn.Linear(pair, 8, bias=False)
        self.qkv = nn.Linear(single, single * 3, bias=False)
        self.out = zero(nn.Linear(single, single))
        self.ff = nn.Sequential(
            nn.LayerNorm(single),
            nn.Linear(single, single * 4),
            nn.GELU(),
            zero(nn.Linear(single * 4, single)),
        )
        self.to_pair = ResidueToPair(single, pair, outer_rank)
        self.pair_gate = nn.Linear(pair, pair)

    def forward(self, s, z, mask):
        valid = mask.any(-1)
        b, length, dim = s.shape
        q, k, v = (
            self.qkv(self.norm(s))
            .view(b, length, 3, 8, dim // 8)
            .permute(2, 0, 3, 1, 4)
        )
        bias = self.bias(self.pairnorm(z)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~valid[:, None, None, :], -torch.inf)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        s = s + self.out(y.transpose(1, 2).reshape(b, length, dim))
        s = (s + self.ff(s)) * valid[..., None]
        z = (z + self.to_pair(s) * self.pair_gate(z).sigmoid()) * mask[..., None]
        return s, z


class ReasoningBlock(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.joint = c.pair_family == "joint"
        self.norm = nn.LayerNorm(c.pair_dim)
        dilation = [1, 2, 4, 8][index % 4]
        self.conv = nn.Conv2d(
            c.pair_dim, c.pair_dim, 3, padding=dilation, dilation=dilation
        )
        self.outgoing = TriangleUpdate(c.pair_dim, c.triangle_dim)
        self.incoming = TriangleUpdate(c.pair_dim, c.triangle_dim, incoming=True)
        active = (index + 1) % c.pair_attention_every == 0
        self.attentions = nn.ModuleList(
            [
                TriangleAttention(
                    c.pair_dim, c.pair_attention_heads, c.pair_attention_chunk, end
                )
                for end in (False, True)
            ]
            if active and c.pair_family in {"attention", "joint"}
            else []
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(c.pair_dim),
            nn.Linear(c.pair_dim, c.pair_dim * 4),
            nn.GELU(),
            zero(nn.Linear(c.pair_dim * 4, c.pair_dim)),
        )
        self.feedback = (
            ResidueFeedback(c.pair_dim, c.pair_residue_dim, c.pair_outer_rank)
            if self.joint and active
            else None
        )

    def forward(self, z, mask, s=None):
        if self.feedback is not None:
            s, z = self.feedback(s, z, mask)
        x = (self.norm(z) * mask[..., None]).permute(0, 3, 1, 2)
        z = (z + F.gelu(self.conv(x).permute(0, 2, 3, 1))) * mask[..., None]
        z = self.incoming(self.outgoing(z, mask), mask)
        for attention in self.attentions:
            z = attention(z, mask)
        z = (z + self.ff(z)) * mask[..., None]
        return (z, s) if self.joint else z
