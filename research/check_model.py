"""Executable model contract check, run in each isolated candidate snapshot."""

import argparse
import io
import json

import torch
import yaml

from sparsa.model import ALPHABET, ContactModel, ModelConfig, tokenize
from sparsa.train import contact_loss


def check(config, maximum):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    assert ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"
    assert tokenize(ALPHABET).tolist() == list(range(1, 22))
    # Count before allocating real parameter storage; a bad proposal should not
    # exhaust workstation RAM merely to discover that it exceeds the budget.
    with torch.device("meta"):
        shape_model = ContactModel(ModelConfig(**config))
    parameters = sum(p.numel() for p in shape_model.parameters())
    del shape_model
    if not 0 < parameters <= maximum:
        raise ValueError(f"Parameter limit: {parameters} > {maximum}")
    model = ContactModel(ModelConfig(**config)).eval()
    assert model.relative_max_distance is None
    assert callable(model.initialize_distant_buckets)
    tokens = tokenize("ACDEFGHIKLMNPQRSTV")[None]
    padded = torch.zeros(2, 25, dtype=torch.long)
    padded[0, : tokens.shape[1]] = tokens[0]
    padded[1] = torch.randint(1, 22, (25,))
    with torch.no_grad():
        result = model(tokens)
        assert result.shape == (1, tokens.shape[1], tokens.shape[1])
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, result.transpose(1, 2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            result,
            model(padded)[0:1, : tokens.shape[1], : tokens.shape[1]],
            atol=1e-5,
            rtol=1e-5,
        )
        # An architecture must retain sequence dependence.
        assert not torch.allclose(result, model(tokens.flip(1)))
    model.train()
    logits = model(tokens)
    target = torch.zeros_like(logits)
    target[:, 0, 8] = 1
    mask = torch.ones_like(logits, dtype=torch.bool).triu(6)
    loss = contact_loss(logits, target, mask, 4.0)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = ContactModel(ModelConfig(**config)).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens), model(tokens), atol=0, rtol=0)
    return {
        "parameters": parameters,
        "contract_passed": True,
        "loss": float(loss.detach()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-parameters", type=int, required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    print(json.dumps(check(cfg["model"], args.max_parameters)))


if __name__ == "__main__":
    main()
