#!/usr/bin/env python3
"""Export a compact, portable W2Rep encoder checkpoint for publication."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from w2rep.utils.checkpoint import (
    encoder_spec_from_state,
    load_checkpoint_file,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", choices=("target", "encoder"), default="target")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    temporary = output.with_suffix(output.suffix + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = load_checkpoint_file(args.input)
    if args.key not in checkpoint:
        raise KeyError(f"Checkpoint has no {args.key!r} state dict")
    state = checkpoint[args.key]
    payload = {
        "target": state,
        "step": checkpoint.get("step"),
        "architecture": encoder_spec_from_state(state),
        "source_checkpoint_sha256": sha256_file(args.input),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(f"exported={output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
