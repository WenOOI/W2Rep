from pathlib import Path

import torch

from w2rep.models.w2rep import W2RepEncoder
from w2rep.utils.checkpoint import load_encoder


def test_release_checkpoint_round_trip(tmp_path: Path):
    encoder = W2RepEncoder(
        img_size=32,
        patch_size=8,
        embed_dim=768,
        depth=1,
        num_heads=12,
        z_tokens=2,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save({"target": encoder.state_dict(), "step": 7}, path)
    loaded, metadata = load_encoder(path, image_size=32)
    loaded.load_state_dict(encoder.state_dict(), strict=True)
    assert metadata["step"] == 7
    assert metadata["patch_size"] == 8

