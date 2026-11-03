#!/usr/bin/env python3
"""Tiny import + checkpoint smoke check. Does not run training."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for token in version.split("+")[0].split("."):
        digits = ""
        for char in token:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits or 0))
    return tuple(parts)


def check_minimums() -> None:
    import datasets
    import matplotlib
    import numpy
    import tiktoken
    import torch

    mins = {
        "torch": ("2.13.0", torch.__version__),
        "numpy": ("2.2.6", numpy.__version__),
        "datasets": ("5.0.1", datasets.__version__),
        "tiktoken": ("0.7.0", tiktoken.__version__),
        "matplotlib": ("3.8.4", matplotlib.__version__),
    }
    for name, (minimum, actual) in mins.items():
        if _version_tuple(actual) < _version_tuple(minimum):
            raise SystemExit(f"{name} {actual} is below required minimum {minimum}")
        print(f"ok {name} {actual} (>= {minimum})")


def smoke_model_and_checkpoint() -> None:
    import torch

    from main import GPT, GPTConfig, load_checkpoint, save_checkpoint

    device = torch.device("cpu")
    config = GPTConfig(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
    )
    model = GPT(config).to(device)
    model.eval()
    dummy = torch.randint(0, config.vocab_size, (1, 4), device=device)
    logits, loss = model(dummy, dummy)
    if logits.shape[-1] != config.vocab_size:
        raise SystemExit(f"unexpected logits shape {tuple(logits.shape)}")
    if loss is None:
        raise SystemExit("expected a loss from the dummy forward pass")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.pt")
        save_checkpoint(path, model, config, step=1, loss=float(loss.item()))
        loaded, checkpoint = load_checkpoint(path, device)
        restored = loaded(dummy)
        if restored[0].shape != logits.shape:
            raise SystemExit("loaded model produced a different logits shape")
        if "model_state_dict" not in checkpoint:
            raise SystemExit("checkpoint missing model_state_dict")
        if not isinstance(checkpoint["config"], dict):
            raise SystemExit("checkpoint config must be a primitive dict")
    print("ok checkpoint save/load with weights_only=True")
    print("ok tiny forward pass")


def main() -> None:
    check_minimums()
    smoke_model_and_checkpoint()
    print("smoke check passed")


if __name__ == "__main__":
    main()
