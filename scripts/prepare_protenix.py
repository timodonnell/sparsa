"""Extract native v1 backbone weights and chemical references from official files.

Download the URLs recorded in reports/protenix_transfer/initialization.json and
sparsa/protenix_residue_reference.json first. Only trusted official files belong
here: both source artifacts use Python pickle serialization.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch

from sparsa.vendor.protenix.data.constants import PRO_STD_RESIDUES, RES_ATOMS_DICT

V1_SHA = "2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04"
KEEP = (
    "input_embedder",
    "relative_position_encoding",
    "linear_no_bias_sinit",
    "linear_no_bias_zinit1",
    "linear_no_bias_zinit2",
    "linear_no_bias_token_bond",
    "linear_no_bias_z_cycle",
    "linear_no_bias_s",
    "layernorm_z_cycle",
    "layernorm_s",
    "pairformer_stack",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ccd", type=Path, required=True)
    parser.add_argument("--backbone-out", type=Path, required=True)
    parser.add_argument("--references-out", type=Path, required=True)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != V1_SHA:
        raise ValueError("Source file is not the verified official v1 checkpoint")
    full = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    weights = {
        k.removeprefix("module."): v
        for k, v in full.items()
        if k.removeprefix("module.").split(".")[0] in KEEP
    }
    metadata = {
        "source_url": "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt",
        "source_sha256": digest,
        "upstream_commit": "85767b811c40ed46e73a9b39519cf6bfca8701ba",
        "parameters": sum(v.numel() for v in weights.values()),
        "keys": len(weights),
        "precision": "unchanged float32",
    }
    assert metadata["parameters"] == 148895680 and metadata["keys"] == 2840
    torch.save({"backbone": weights, "initialization": metadata}, args.backbone_out)
    with args.ccd.open("rb") as f:
        source_digest = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        molecules = pickle.load(f)
    residues = {}
    for name in PRO_STD_RESIDUES:
        mol = molecules[name]
        coordinates = mol.GetConformer(mol.ref_conf_id).GetPositions()
        atoms = []
        for atom_name in RES_ATOMS_DICT[name]:
            i = mol.atom_map[atom_name]
            atom = mol.GetAtomWithIdx(i)
            atoms.append(
                {
                    "name": atom_name,
                    "element": atom.GetAtomicNum(),
                    "charge": atom.GetFormalCharge(),
                    "mask": bool(mol.ref_mask[i]),
                    "position": coordinates[i].tolist(),
                }
            )
        residues[name] = atoms
    args.references_out.write_text(
        json.dumps(
            {
                "source_url": "https://protenix.tos-cn-beijing.volces.com/common/components.cif.rdkit_mol.pkl",
                "source_sha256": source_digest,
                "residues": residues,
            },
            indent=2,
        )
        + "\n"
    )
    with args.backbone_out.open("rb") as f:
        metadata["artifact_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
