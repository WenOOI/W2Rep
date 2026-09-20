"""MMSegmentation backbone adapter for frozen W2Rep encoders."""

from __future__ import annotations

import contextlib

import torch

try:
    from mmengine.model import BaseModule
    from mmseg.registry import MODELS
except ImportError as error:  # pragma: no cover - optional dependency
    raise ImportError(
        "W2RepMMSegBackbone requires mmengine and mmsegmentation. "
        "Install the versions documented in w2rep/segmentation/README.md."
    ) from error

from w2rep.utils.checkpoint import load_encoder


@MODELS.register_module()
class W2RepMMSegBackbone(BaseModule):
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_key: str = "target",
        image_size: int = 224,
        out_indices: tuple[int, ...] | None = None,
        frozen: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if not checkpoint_path:
            raise ValueError("Set model.backbone.checkpoint_path to a W2Rep checkpoint")
        self.encoder, self.checkpoint_metadata = load_encoder(
            checkpoint_path,
            image_size=image_size,
            checkpoint_key=checkpoint_key,
        )
        depth = len(self.encoder.blocks)
        if out_indices is None:
            out_indices = (2, 5, 8, 11) if depth == 12 else (5, 11, 17, 23)
        self.out_indices = tuple(out_indices)
        self.embed_dim = self.encoder.embed_dim
        self.frozen = bool(frozen)
        if self.frozen:
            self.encoder.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gradient_context = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with gradient_context:
            features, (height, width) = self.encoder.forward_intermediates(
                images, self.out_indices
            )
        return tuple(
            feature.transpose(1, 2)
            .reshape(feature.shape[0], feature.shape[2], height, width)
            .contiguous()
            for feature in features
        )

