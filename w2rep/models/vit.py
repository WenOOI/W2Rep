"""Minimal Vision Transformer components used by W2Rep.

The module names intentionally follow common ViT conventions so that paper
checkpoints remain load-compatible. No external research repository is needed.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


def get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("The positional-embedding dimension must be even")
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    frequencies = np.arange(embed_dim // 2, dtype=np.float64)
    frequencies /= embed_dim / 2.0
    frequencies = 1.0 / (10000.0**frequencies)
    angles = np.einsum("m,d->md", positions, frequencies)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("The positional-embedding dimension must be even")
    if isinstance(grid_size, int):
        height = width = grid_size
    else:
        height, width = grid_size
    grid_w, grid_h = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid_w.reshape(-1))
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid_h.reshape(-1))
    return np.concatenate([emb_h, emb_w], axis=1)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} is not divisible by heads={num_heads}")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def initialize_vit_module(module: nn.Module, std: float = 0.02) -> None:
    # The paper training code explicitly reinitialized linear and normalization
    # layers while leaving PatchEmbed's Conv2d at PyTorch's default init.
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def rescale_residual_projections(blocks: nn.ModuleList) -> None:
    for layer_index, block in enumerate(blocks, start=1):
        scale = math.sqrt(2.0 * layer_index)
        block.attn.proj.weight.data.div_(scale)
        block.mlp.fc2.weight.data.div_(scale)
