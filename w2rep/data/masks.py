"""Spatial block masks shared across every frame in a sampled clip."""

from __future__ import annotations

import math
from multiprocessing import Value
from typing import Any

import torch

from .video import collate_video_batch


class BlockMaskCollator:
    def __init__(
        self,
        *,
        grid_size: int,
        encoder_scale: tuple[float, float],
        predictor_scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        predictor_blocks: int,
        min_keep: int,
        seed: int = 0,
    ) -> None:
        self.height = self.width = grid_size
        self.total_patches = grid_size * grid_size
        self.encoder_scale = tuple(encoder_scale)
        self.predictor_scale = tuple(predictor_scale)
        self.aspect_ratio = tuple(aspect_ratio)
        self.predictor_blocks = predictor_blocks
        self.min_keep = min_keep
        self.seed = int(seed)
        self._counter = Value("q", -1)

    def _next_seed(self) -> int:
        with self._counter.get_lock():
            self._counter.value += 1
            return self.seed + self._counter.value

    def set_call_index(self, completed_calls: int) -> None:
        """Set the deterministic mask stream to the next unconsumed call."""
        with self._counter.get_lock():
            self._counter.value = int(completed_calls) - 1

    def _block_size(
        self,
        generator: torch.Generator,
        scale: tuple[float, float],
        aspect: tuple[float, float],
    ) -> tuple[int, int]:
        fraction = scale[0] + torch.rand((), generator=generator).item() * (
            scale[1] - scale[0]
        )
        ratio = aspect[0] + torch.rand((), generator=generator).item() * (
            aspect[1] - aspect[0]
        )
        area = max(1, int(self.total_patches * fraction))
        height = min(max(1, round(math.sqrt(area * ratio))), self.height - 1)
        width = min(max(1, round(math.sqrt(area / ratio))), self.width - 1)
        return int(height), int(width)

    def _sample_block(
        self,
        size: tuple[int, int],
        generator: torch.Generator,
        acceptable: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = size
        best: torch.Tensor | None = None
        for _ in range(100):
            top = int(
                torch.randint(
                    0, self.height - height + 1, (), generator=generator
                ).item()
            )
            left = int(
                torch.randint(
                    0, self.width - width + 1, (), generator=generator
                ).item()
            )
            block = torch.zeros(self.height, self.width, dtype=torch.bool)
            block[top : top + height, left : left + width] = True
            candidate = block if acceptable is None else block & acceptable
            indices = candidate.flatten().nonzero().flatten()
            if best is None or len(indices) > len(best):
                best = indices
            if len(indices) > self.min_keep:
                return indices, ~block
        if best is None or len(best) <= self.min_keep:
            raise RuntimeError(
                "Could not sample a valid spatial mask. Relax mask scales or min_keep."
            )
        return best, ~block

    def sample_masks(self, batch_size: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self._next_seed())
        predictor_size = self._block_size(
            generator, self.predictor_scale, self.aspect_ratio
        )
        encoder_size = self._block_size(generator, self.encoder_scale, (1.0, 1.0))
        all_predictor: list[list[torch.Tensor]] = []
        all_encoder: list[torch.Tensor] = []
        min_predictor = self.total_patches
        min_encoder = self.total_patches
        for _ in range(batch_size):
            predictor: list[torch.Tensor] = []
            complements: list[torch.Tensor] = []
            for _ in range(self.predictor_blocks):
                indices, complement = self._sample_block(
                    predictor_size, generator
                )
                predictor.append(indices)
                complements.append(complement)
                min_predictor = min(min_predictor, len(indices))
            acceptable = torch.stack(complements).all(dim=0)
            encoder, _ = self._sample_block(
                encoder_size, generator, acceptable=acceptable
            )
            all_predictor.append(predictor)
            all_encoder.append(encoder)
            min_encoder = min(min_encoder, len(encoder))
        return {
            "mask_encoder": torch.stack(
                [indices[:min_encoder] for indices in all_encoder]
            ),
            "mask_predictor": torch.stack(
                [
                    torch.stack([indices[:min_predictor] for indices in blocks])
                    for blocks in all_predictor
                ]
            ),
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        result = collate_video_batch(batch)
        result.update(self.sample_masks(len(batch)))
        return result
