from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from w2rep.models.w2rep import W2RepEncoder


class _LegacyDotDict(dict):
    """Allow tensor-only loading of checkpoints saved by the research code."""


def load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    """Load a W2Rep checkpoint without importing the historical source tree.

    Early paper checkpoints serialized their configuration as
    ``utils.config.DotDict``. PyTorch's restricted loader maps that harmless
    dict subclass to a plain local subclass while still rejecting arbitrary
    pickle globals.
    """
    path = Path(path).expanduser().resolve()
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals(
            [(_LegacyDotDict, "utils.config.DotDict")]
        ):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    else:  # New release checkpoints contain no custom pickle globals.
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError(
                "This legacy research checkpoint requires PyTorch's restricted "
                "safe_globals loader. Export a portable encoder checkpoint with "
                "PyTorch >= 2.6, or evaluate the already exported checkpoint."
            ) from error
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary-valued W2Rep checkpoint")
    return checkpoint


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def encoder_spec_from_state(state: dict[str, torch.Tensor]) -> dict[str, int]:
    state = _remove_module_prefix(state)
    patch_weight = state["patch_embed.proj.weight"]
    embed_dim = int(patch_weight.shape[0])
    patch_size = int(patch_weight.shape[-1])
    block_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    depth = max(block_indices) + 1
    head_lookup = {768: 12, 1024: 16}
    if embed_dim not in head_lookup:
        raise ValueError(
            f"Cannot infer attention heads for encoder dimension {embed_dim}; "
            "pass a supported W2Rep-B/L checkpoint."
        )
    return {
        "embed_dim": embed_dim,
        "patch_size": patch_size,
        "depth": depth,
        "num_heads": head_lookup[embed_dim],
        "z_tokens": int(state["z_tokens"].shape[1]),
    }


def load_encoder(
    checkpoint_path: str | Path,
    *,
    image_size: int = 224,
    checkpoint_key: str = "target",
    activation_checkpointing: bool = False,
) -> tuple[W2RepEncoder, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_checkpoint_file(checkpoint_path)
    if checkpoint_key not in checkpoint:
        available = [key for key in ("target", "encoder") if key in checkpoint]
        raise KeyError(
            f"Checkpoint has no {checkpoint_key!r} state dict; available={available}"
        )
    state = _remove_module_prefix(checkpoint[checkpoint_key])
    spec = encoder_spec_from_state(state)
    encoder = W2RepEncoder(
        img_size=image_size,
        patch_size=spec["patch_size"],
        embed_dim=spec["embed_dim"],
        depth=spec["depth"],
        num_heads=spec["num_heads"],
        z_tokens=spec["z_tokens"],
        max_frames=8,
        activation_checkpointing=activation_checkpointing,
        latent_reads_patches_only=False,
    )
    encoder.load_state_dict(state, strict=True)
    metadata = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "step": checkpoint.get("step"),
        **spec,
    }
    return encoder, metadata
