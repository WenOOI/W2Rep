"""W2Rep encoder and cross-frame predictor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .vit import (
    Block,
    PatchEmbed,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    initialize_vit_module,
    rescale_residual_projections,
    trunc_normal_,
)


def build_3d_sincos(
    embed_dim: int,
    grid_size: tuple[int, int],
    frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a `[T, H*W, D]` separable time-space positional embedding."""
    if embed_dim % 4:
        raise ValueError("W2Rep requires an embedding dimension divisible by four")
    spatial = get_2d_sincos_pos_embed(embed_dim // 2, grid_size)
    temporal = get_1d_sincos_pos_embed(
        embed_dim // 2, np.arange(frames, dtype=np.float64)
    )
    height, width = grid_size
    result = torch.zeros(frames, height * width, embed_dim, dtype=torch.float32)
    result[:, :, : embed_dim // 2] = torch.from_numpy(temporal).float()[:, None]
    result[:, :, embed_dim // 2 :] = torch.from_numpy(spatial).float()[None]
    return result.to(device=device, dtype=dtype)


def _block_forward_with_attention_bias(
    block: Block,
    x: torch.Tensor,
    attention_bias: torch.Tensor,
) -> torch.Tensor:
    """Run a standard block while adding a pre-softmax attention bias."""
    batch, tokens, dim = x.shape
    heads = block.attn.num_heads
    head_dim = dim // heads
    qkv = block.attn.qkv(block.norm1(x)).reshape(
        batch, tokens, 3, heads, head_dim
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attention = (q.float() @ k.float().transpose(-2, -1)) * block.attn.scale
    attention = (attention + attention_bias.float()).softmax(dim=-1).to(x.dtype)
    attention = block.attn.attn_drop(attention)
    y = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
    y = block.attn.proj_drop(block.attn.proj(y))
    x = x + block.drop_path(y)
    return x + block.drop_path(block.mlp(block.norm2(x)))


class W2RepEncoder(nn.Module):
    """A ViT that supports image-only and video-with-latent forward passes.

    `with_z=False` is the retained downstream interface. `with_z=True` appends
    learnable auxiliary latents to the visible video patch sequence during
    pretraining.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        z_tokens: int = 16,
        max_frames: int = 8,
        latent_reads_patches_only: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        self.embed_dim = embed_dim
        self.grid = img_size // patch_size
        self.N = self.grid * self.grid
        self.K = z_tokens
        self.T = max_frames
        # Keep this historical attribute name for paper-checkpoint compatibility.
        self.asymmetric_z_attention = bool(latent_reads_patches_only)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.z_tokens = nn.Parameter(torch.zeros(1, z_tokens, embed_dim))
        # Initialize before constructing transformer blocks to preserve the
        # random-number stream used for the paper runs.
        trunc_normal_(self.z_tokens, std=0.02)
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm_z = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(initialize_vit_module)
        rescale_residual_projections(self.blocks)

    def _run_block(self, block: Block, x: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(
        self,
        pixels: torch.Tensor,
        masks_enc: torch.Tensor | None = None,
        *,
        with_z: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        if pixels.ndim != 5:
            raise ValueError("pixels must have shape [B,C,H,W] or [B,T,C,H,W]")
        batch, frames, channels, height, width = pixels.shape
        if channels != 3 or height % self.patch_embed.patch_size or width % self.patch_embed.patch_size:
            raise ValueError(f"Unsupported input shape: {tuple(pixels.shape)}")
        grid = (height // self.patch_embed.patch_size, width // self.patch_embed.patch_size)
        patches_per_frame = grid[0] * grid[1]
        x = self.patch_embed(pixels.reshape(batch * frames, channels, height, width))
        x = x.reshape(batch, frames, patches_per_frame, self.embed_dim)
        position = build_3d_sincos(
            self.embed_dim,
            grid,
            frames,
            device=x.device,
            dtype=x.dtype,
        )
        x = x + position[None]

        if masks_enc is not None:
            if masks_enc.ndim != 2 or masks_enc.shape[0] != batch:
                raise ValueError("masks_enc must have shape [B,K]")
            gather = masks_enc[:, None, :, None].expand(
                batch, frames, masks_enc.shape[1], self.embed_dim
            )
            x = torch.gather(x, 2, gather)

        visible_per_frame = x.shape[2]
        patch_sequence = x.reshape(batch, frames * visible_per_frame, self.embed_dim)
        if with_z:
            sequence = torch.cat(
                [patch_sequence, self.z_tokens.expand(batch, -1, -1)], dim=1
            )
            if self.asymmetric_z_attention:
                patch_count = patch_sequence.shape[1]
                bias = torch.zeros(
                    1,
                    1,
                    sequence.shape[1],
                    sequence.shape[1],
                    device=sequence.device,
                    dtype=sequence.dtype,
                )
                bias[:, :, :patch_count, patch_count:] = float("-inf")
                for block in self.blocks:
                    if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                        sequence = checkpoint(
                            lambda seq, attn_bias, current=block: _block_forward_with_attention_bias(
                                current, seq, attn_bias
                            ),
                            sequence,
                            bias,
                            use_reentrant=False,
                        )
                    else:
                        sequence = _block_forward_with_attention_bias(block, sequence, bias)
            else:
                for block in self.blocks:
                    sequence = self._run_block(block, sequence)
            patch_sequence = sequence[:, : frames * visible_per_frame]
            latent_sequence = sequence[:, frames * visible_per_frame :]
            z_out: torch.Tensor | None = self.norm_z(latent_sequence)
        else:
            for block in self.blocks:
                patch_sequence = self._run_block(block, patch_sequence)
            z_out = None

        patch_out = self.norm(patch_sequence).reshape(
            batch, frames, visible_per_frame, self.embed_dim
        )
        return {"patch_out": patch_out, "z_out": z_out}

    def forward_intermediates(
        self,
        pixels: torch.Tensor,
        out_indices: tuple[int, ...],
    ) -> tuple[list[torch.Tensor], tuple[int, int]]:
        """Return normalized image-token features from selected transformer blocks."""
        if pixels.ndim != 4:
            raise ValueError("forward_intermediates expects [B,C,H,W]")
        batch, channels, height, width = pixels.shape
        if channels != 3 or height % self.patch_embed.patch_size or width % self.patch_embed.patch_size:
            raise ValueError("W2Rep expects RGB input with patch-divisible height and width")
        grid = (height // self.patch_embed.patch_size, width // self.patch_embed.patch_size)
        x = self.patch_embed(pixels)
        position = build_3d_sincos(
            self.embed_dim,
            grid,
            1,
            device=x.device,
            dtype=x.dtype,
        )[0]
        if position.shape[0] != x.shape[1]:
            raise ValueError("Input height and width must be divisible by the patch size")
        x = x + position[None]
        requested = set(out_indices)
        if min(requested, default=0) < 0 or max(requested, default=0) >= len(self.blocks):
            raise ValueError(f"Invalid out_indices={out_indices} for depth={len(self.blocks)}")
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            x = self._run_block(block, x)
            if index in requested:
                features[index] = self.norm(x)
        return [features[index] for index in out_indices], grid


def _scalar_sincos(values: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError("Condition dimension must be even")
    half = dim // 2
    frequencies = torch.arange(half, device=values.device, dtype=torch.float32)
    frequencies = 1.0 / (10000.0 ** (frequencies / half))
    angles = values.float()[:, None] * frequencies[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ZCrossBlock(nn.Module):
    """Predictor block with offset-gated cross-attention to clip latents."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        z_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("Predictor dimension must be divisible by num_heads")
        self.block = Block(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
        z_dim = z_dim or dim
        self.norm_z = nn.LayerNorm(dim, eps=1e-6)
        self.q_z = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_z = nn.Linear(z_dim, 2 * dim, bias=qkv_bias)
        self.proj_z = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.n_gate = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q = self.q_z(self.norm_z(x)).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        kv = self.kv_z(z).reshape(
            batch, z.shape[1], 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        gate = self.n_gate(condition).unsqueeze(1)
        return self.block(x + gate * self.proj_z(output))


class W2RepPredictor(nn.Module):
    """Predict EMA target features from image features and video context."""

    def __init__(
        self,
        embed_dim: int = 768,
        pred_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        grid: int = 14,
        z_dim: int = 768,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.N = grid * grid
        self.pred_dim = pred_dim
        self.activation_checkpointing = bool(activation_checkpointing)
        self.predictor_embed = nn.Linear(embed_dim, pred_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        # Retained for strict loading of development checkpoints. The final
        # masked-position objective does not use this parameter.
        self.ctx_pred_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        # The learned queries are initialized before constructing the remaining
        # modules, matching the training code used for the paper checkpoints.
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.ctx_pred_token, std=0.02)
        position = get_2d_sincos_pos_embed(pred_dim, grid)
        self.register_buffer(
            "pos_embed", torch.from_numpy(position).float().unsqueeze(0)
        )
        self.z_in = nn.Linear(z_dim, pred_dim, bias=True)
        self.cond_mlp = nn.Sequential(
            nn.Linear(pred_dim, pred_dim),
            nn.SiLU(),
            nn.Linear(pred_dim, pred_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ZCrossBlock(
                    pred_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    z_dim=pred_dim,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(pred_dim, eps=1e-6)
        self.proj = nn.Linear(pred_dim, embed_dim, bias=True)
        self.apply(initialize_vit_module)
        for layer_index, block in enumerate(self.blocks, start=1):
            scale = math.sqrt(2.0 * layer_index)
            block.block.attn.proj.weight.data.div_(scale)
            block.block.mlp.fc2.weight.data.div_(scale)

    def forward(
        self,
        context: torch.Tensor,
        context_positions: torch.Tensor,
        target_positions: torch.Tensor,
        offset: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        batch = context.shape[0]
        x = self.predictor_embed(context)
        context_position = torch.gather(
            self.pos_embed.expand(batch, -1, -1),
            1,
            context_positions[:, :, None].expand(-1, -1, self.pred_dim),
        )
        x = x + context_position
        context_length = x.shape[1]
        queries = self.mask_token.expand(batch, target_positions.shape[1], -1)
        query_position = torch.gather(
            self.pos_embed.expand(batch, -1, -1),
            1,
            target_positions[:, :, None].expand(-1, -1, self.pred_dim),
        )
        x = torch.cat([x, queries + query_position], dim=1)
        z = self.z_in(z)
        condition = self.cond_mlp(_scalar_sincos(offset, self.pred_dim))
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, z, condition, use_reentrant=False)
            else:
                x = block(x, z, condition)
        return self.proj(self.norm(x)[:, context_length:])


@dataclass(frozen=True)
class ModelSize:
    embed_dim: int
    depth: int
    heads: int
    predictor_depth: int


MODEL_SIZES = {
    "vitb16": ModelSize(768, 12, 12, 6),
    "vitl16": ModelSize(1024, 24, 16, 12),
}


def build_w2rep(
    architecture: str,
    *,
    image_size: int = 224,
    patch_size: int = 16,
    frames: int = 8,
    z_tokens: int = 16,
    predictor_dim: int = 384,
    predictor_depth: int | None = None,
    activation_checkpointing: bool = False,
    latent_reads_patches_only: bool = False,
) -> tuple[W2RepEncoder, W2RepPredictor]:
    if architecture not in MODEL_SIZES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose {sorted(MODEL_SIZES)}")
    size = MODEL_SIZES[architecture]
    encoder = W2RepEncoder(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=size.embed_dim,
        depth=size.depth,
        num_heads=size.heads,
        z_tokens=z_tokens,
        max_frames=frames,
        activation_checkpointing=activation_checkpointing,
        latent_reads_patches_only=latent_reads_patches_only,
    )
    predictor = W2RepPredictor(
        embed_dim=size.embed_dim,
        pred_dim=predictor_dim,
        depth=predictor_depth or size.predictor_depth,
        num_heads=12,
        grid=image_size // patch_size,
        z_dim=size.embed_dim,
        activation_checkpointing=activation_checkpointing,
    )
    return encoder, predictor
