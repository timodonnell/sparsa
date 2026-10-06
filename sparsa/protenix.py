"""Sequence-only Protenix v1 backbone with a binary-contact recycle adapter.

The native 48-block trunk is vendored without mathematical changes. Reference
atom positions describe isolated ideal residues, never a protein structure.
MSA, templates, coordinate diffusion and confidence modules are not constructed.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# The pilot uses the upstream torch implementation, not its optional extension.
os.environ["LAYERNORM_TYPE"] = "torch"

from sparsa.model import ALPHABET
from sparsa.vendor.protenix.data.constants import (
    PROT_STD_RESIDUES_ONE_TO_THREE,
    STD_RESIDUES_WITH_GAP,
)
from sparsa.vendor.protenix.model.modules.embedders import (
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from sparsa.vendor.protenix.model.modules.pairformer import PairformerStack
from sparsa.vendor.protenix.model.modules.primitives import (
    LinearNoBias,
    rearrange_qk_to_dense_trunk,
)
from sparsa.vendor.protenix.model.triangular.layers import LayerNorm


@lru_cache(maxsize=1)
def residue_references():
    return json.loads(
        Path(__file__).with_name("protenix_residue_reference.json").read_text()
    )["residues"]


@torch.no_grad()
def sequence_features(token_ids, device):
    """Native chemical/query features for one unpadded protein (no homologs)."""
    refs = residue_references()
    if not token_ids or any(t < 1 or t > len(ALPHABET) for t in token_ids):
        raise ValueError("Expected a nonempty, unpadded amino-acid sequence")
    names = [PROT_STD_RESIDUES_ONE_TO_THREE[ALPHABET[t - 1]] for t in token_ids]
    positions, elements, charges, masks, chars, indices = [], [], [], [], [], []
    for index, name in enumerate(names):
        atoms = [a for a in refs[name] if a["name"] != "OXT" or index == len(names) - 1]
        xyz = torch.tensor([a["position"] for a in atoms], dtype=torch.float32)
        xyz = xyz - xyz.mean(0, keepdim=True)
        positions.extend(xyz.tolist())
        for atom in atoms:
            elements.append(atom["element"] - 1)
            charges.append(atom["charge"])
            masks.append(atom["mask"])
            chars.append(
                [max(0, min(63, ord(c) - 32)) for c in atom["name"].ljust(4)[:4]]
            )
            indices.append(index)
    tensor = lambda x, dtype: torch.tensor(x, device=device, dtype=dtype)
    pos = tensor(positions, torch.float32)
    atom_index = tensor(indices, torch.long)
    restype = F.one_hot(
        tensor([STD_RESIDUES_WITH_GAP[n] for n in names], torch.long), 32
    ).float()
    q, k, pad_info = rearrange_qk_to_dense_trunk(
        q=[pos, atom_index],
        k=[pos, atom_index],
        dim_q=[-2, -1],
        dim_k=[-2, -1],
        n_queries=32,
        n_keys=128,
        compute_mask=True,
    )
    residue_index = torch.arange(len(names), device=device)
    zeros = torch.zeros_like(residue_index)
    return {
        "atom_to_token_idx": atom_index,
        "ref_pos": pos,
        "ref_charge": tensor(charges, torch.float32),
        "ref_mask": tensor(masks, torch.float32),
        "ref_atom_name_chars": F.one_hot(tensor(chars, torch.long), 64).float(),
        "ref_element": F.one_hot(tensor(elements, torch.long), 128).float(),
        "d_lm": q[0][..., None, :] - k[0][..., None, :, :],
        "v_lm": (q[1][..., None].int() == k[1][..., None, :].int()).unsqueeze(-1),
        "pad_info": pad_info,
        "restype": restype,
        "profile": restype.clone(),
        "deletion_mean": torch.zeros(len(names), device=device),
        "asym_id": zeros,
        "residue_index": residue_index,
        "entity_id": zeros,
        "sym_id": zeros,
        "token_index": residue_index,
        # Native standard-polymer bonds are excluded from token_bonds.
        "token_bonds": torch.zeros(len(names), len(names), device=device),
    }


class ProtenixBackbone(nn.Module):
    """Native checkpoint key names, including its learned recycling layers."""

    def __init__(self, config):
        super().__init__()
        self.input_embedder = InputFeatureEmbedder()
        self.relative_position_encoding = RelativePositionEncoding()
        self.linear_no_bias_sinit = LinearNoBias(449, 384)
        self.linear_no_bias_zinit1 = LinearNoBias(384, 128)
        self.linear_no_bias_zinit2 = LinearNoBias(384, 128)
        self.linear_no_bias_token_bond = LinearNoBias(1, 128)
        self.linear_no_bias_z_cycle = LinearNoBias(128, 128)
        self.linear_no_bias_s = LinearNoBias(384, 384)
        self.layernorm_z_cycle = LayerNorm(128)
        self.layernorm_s = LayerNorm(384)
        self.pairformer_stack = PairformerStack(
            n_blocks=48,
            n_heads=16,
            c_z=128,
            c_s=384,
            dropout=config.dropout,
            blocks_per_ckpt=1 if config.gradient_checkpointing else None,
        )

    def encode_one(self, features):
        self.relative_position_encoding.generate_relp(features)
        s = self.linear_no_bias_sinit(self.input_embedder(features, inplace_safe=False))
        z = (
            self.linear_no_bias_zinit1(s)[..., None, :]
            + self.linear_no_bias_zinit2(s)[..., None, :, :]
        )
        z = z + self.relative_position_encoding(features["relp"])
        z = z + self.linear_no_bias_token_bond(features["token_bonds"][..., None])
        s = s + self.linear_no_bias_s(self.layernorm_s(torch.zeros_like(s)))
        return z, s


class ProtenixContactDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (
            config.pair_dim,
            config.sequence_dim,
            config.pairformer_layers,
            config.heads,
        ) != (128, 384, 48, 16):
            raise ValueError(
                "Protenix v1 requires its native 48 x (128 pair, 384 single) trunk"
            )
        if (
            config.self_conditioning
            or config.inner_loops != 1
            or config.untied_cells != 1
        ):
            raise ValueError(
                "The transfer pilot uses one pass and no self-conditioning"
            )
        self.config = config
        self.backbone = ProtenixBackbone(config)
        self.noisy_state = nn.Embedding(2, 128)
        self.time = nn.Embedding(config.diffusion_steps + 1, 128)
        self.separation = nn.Embedding(4, 128)
        self.adapter = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.adapter_gate = nn.Parameter(torch.tensor(0.1))
        # Read the final single state too: the final native single update has no
        # subsequent pair block, so a pair-only head would leave it untrained.
        self.single_readout = nn.Linear(384, 128, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.constant_(self.head[-1].bias, -3.0)

    def configure_trainability(self, train_backbone):
        self.backbone.requires_grad_(train_backbone)

    def optimizer_groups(self, adapter_lr, backbone_lr):
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("backbone.")
                ],
                "lr": adapter_lr,
                "base_lr": adapter_lr,
                "name": "adapter_head",
            },
            {
                "params": list(self.backbone.parameters()),
                "lr": backbone_lr,
                "base_lr": backbone_lr,
                "name": "backbone",
            },
        ]

    def initialize_backbone(self, state):
        metadata = state["initialization"]
        if (
            metadata["source_sha256"]
            != "2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04"
        ):
            raise ValueError("Unexpected pretrained checkpoint")
        self.backbone.load_state_dict(state["backbone"], strict=True)
        if sum(p.numel() for p in self.backbone.parameters()) != metadata["parameters"]:
            raise ValueError("Backbone parameter coverage mismatch")
        return metadata

    def encode(self, tokens):
        valid = tokens != 0
        lengths = valid.sum(-1).tolist()
        rows = tokens.detach().cpu().tolist()
        pairs, singles = [], []
        for row, length in zip(rows, lengths, strict=True):
            if not length or any(row[length:]) or 0 in row[:length]:
                raise ValueError("Only trailing sequence padding is supported")
            z, s = self.backbone.encode_one(
                sequence_features(row[:length], tokens.device)
            )
            pad = tokens.shape[1] - length
            pairs.append(F.pad(z, (0, 0, 0, pad, 0, pad)))
            singles.append(F.pad(s, (0, 0, 0, pad)))
        return torch.stack(pairs), torch.stack(singles), valid

    def denoise(self, encoded, noisy, timestep, self_condition=None):
        z_init, s_init, valid = encoded
        if self_condition is not None:
            raise ValueError("Self-conditioning is disabled")
        if noisy.shape != (
            valid.shape[0],
            valid.shape[1],
            valid.shape[1],
        ) or timestep.shape != (valid.shape[0],):
            raise ValueError("Noisy map/timestep shape mismatch")
        lengths = valid.sum(-1).tolist()
        outputs = [None] * len(lengths)
        for length in sorted(set(lengths)):
            members = [i for i, n in enumerate(lengths) if n == length]
            index = torch.tensor(members, device=noisy.device)
            z = z_init.index_select(0, index)[:, :length, :length]
            s = s_init.index_select(0, index)[:, :length]
            xt = noisy.index_select(0, index)[:, :length, :length]
            t = timestep.index_select(0, index)
            pos = torch.arange(length, device=noisy.device)
            sep = (pos[:, None] - pos[None, :]).abs()
            bucket = (sep >= 6).long() + (sep >= 12).long() + (sep >= 24).long()
            u = self.adapter(
                self.noisy_state(xt.long())
                + self.time(t)[:, None, None]
                + self.separation(bucket)
            )
            b = self.backbone
            baseline = b.linear_no_bias_z_cycle(
                b.layernorm_z_cycle(torch.zeros_like(u))
            )
            recycled = b.linear_no_bias_z_cycle(b.layernorm_z_cycle(u))
            z = z + baseline + self.adapter_gate * (recycled - baseline)
            pair_mask = torch.ones(z.shape[:-1], device=z.device, dtype=z.dtype)
            s, z = b.pairformer_stack(
                s,
                z,
                pair_mask,
                inplace_safe=False,
                chunk_size=self.config.pair_attention_chunk,
            )
            single = self.single_readout(s)
            z = z + single[:, :, None, :] + single[:, None, :, :]
            logits = self.head(z).squeeze(-1)
            logits = (logits + logits.transpose(1, 2)) * 0.5
            for offset, member in enumerate(members):
                pad = valid.shape[1] - length
                outputs[member] = F.pad(logits[offset], (0, pad, 0, pad))
        return torch.stack(outputs)

    def forward(self, tokens, noisy, timestep, self_condition=None):
        return self.denoise(self.encode(tokens), noisy, timestep, self_condition)
