#!/usr/bin/env python3
"""Render smooth, full-frame-rate W2Rep patch-attention diagnostics.

The script uses the retained EMA encoder and every decoded frame.  It writes:

* mean patch-token attention: final-layer attention averaged over all patch
  queries and heads, shown over spatial keys;
* cross-frame patch similarity: one fixed source patch compared with every
  patch of each independently encoded target frame.

The displayed heatmaps are bicubic visualizations of the native token grid;
the metadata records that grid explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from w2rep.models.w2rep import build_3d_sincos
from w2rep.utils.checkpoint import load_encoder


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--source-frame", type=int, default=11)
    parser.add_argument("--query-x", type=float, default=0.46)
    parser.add_argument("--query-y", type=float, default=0.54)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 12.0)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, fps


def center_crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    crop = frame[top : top + side, left : left + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


def pixels(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    value = np.stack(frames).astype(np.float32) / 255.0
    value = (value - MEAN) / STD
    return torch.from_numpy(value).permute(0, 3, 1, 2).to(device)


@torch.inference_mode()
def features_and_mean_attention(
    encoder: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized patch features and mean incoming patch attention."""
    batch, channels, height, width = images.shape
    grid_h = height // encoder.patch_embed.patch_size
    grid_w = width // encoder.patch_embed.patch_size
    sequence = encoder.patch_embed(images)
    position = build_3d_sincos(
        encoder.embed_dim,
        (grid_h, grid_w),
        1,
        device=sequence.device,
        dtype=sequence.dtype,
    )[0]
    sequence = sequence + position[None]
    for block in encoder.blocks[:-1]:
        sequence = block(sequence)

    block = encoder.blocks[-1]
    normalized = block.norm1(sequence)
    tokens = normalized.shape[1]
    heads = block.attn.num_heads
    head_dim = encoder.embed_dim // heads
    qkv = block.attn.qkv(normalized).reshape(
        batch, tokens, 3, heads, head_dim
    ).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    attention = (
        (q.float() @ k.float().transpose(-2, -1)) * block.attn.scale
    ).softmax(dim=-1)
    # Average over heads and patch queries.  The remaining key axis forms the
    # spatial map; all tokens are patches because this is the retained
    # image-only path and the architecture has no class token.
    mean_attention = attention.mean(dim=(1, 2))
    sequence = block(sequence)
    features = F.normalize(encoder.norm(sequence).float(), dim=-1)
    return features.cpu(), mean_attention.cpu()


def robust_unit(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    low, high = np.percentile(values, [low_q, high_q])
    return np.clip((values - low) / max(float(high - low), 1e-12), 0.0, 1.0)


def heatmaps(values: np.ndarray, size: int) -> list[np.ndarray]:
    unit = robust_unit(values)
    rendered: list[np.ndarray] = []
    for value in unit:
        upsampled = cv2.resize(value, (size, size), interpolation=cv2.INTER_CUBIC)
        upsampled = np.clip(upsampled, 0.0, 1.0)
        colored = cv2.applyColorMap(
            (upsampled * 255).round().astype(np.uint8), cv2.COLORMAP_MAGMA
        )
        rendered.append(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    return rendered


def write_webm(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not initialize VP8 writer")
    for rgb in frames:
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count != len(frames):
        raise RuntimeError(f"Decode verification failed: {count}/{len(frames)}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    source_path = args.video.expanduser().resolve()
    raw, fps = read_video(source_path)
    rgb = [center_crop_resize(frame, args.image_size) for frame in raw]
    tensor = pixels(rgb, device)

    encoder, checkpoint_meta = load_encoder(
        args.checkpoint,
        image_size=args.image_size,
        checkpoint_key="target",
    )
    encoder = encoder.eval().requires_grad_(False).to(device)
    feature_parts: list[torch.Tensor] = []
    attention_parts: list[torch.Tensor] = []
    for start in range(0, len(tensor), args.batch_size):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            feature, attention = features_and_mean_attention(
                encoder, tensor[start : start + args.batch_size]
            )
        feature_parts.append(feature)
        attention_parts.append(attention)
    features = torch.cat(feature_parts)
    attention = torch.cat(attention_parts)
    del encoder, tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    grid = int(round(attention.shape[1] ** 0.5))
    if grid * grid != attention.shape[1]:
        raise RuntimeError("Expected a square patch grid")
    attention_np = attention.reshape(len(rgb), grid, grid).numpy()
    attention_heat = heatmaps(attention_np, args.image_size)
    attention_video = [
        np.concatenate([frame, heat], axis=1)
        for frame, heat in zip(rgb, attention_heat)
    ]

    source_index = min(max(args.source_frame, 0), len(rgb) - 1)
    query_column = min(grid - 1, max(0, int(args.query_x * grid)))
    query_row = min(grid - 1, max(0, int(args.query_y * grid)))
    query_index = query_row * grid + query_column
    query = features[source_index, query_index]
    similarity = torch.einsum("tnd,d->tn", features, query).reshape(
        len(rgb), grid, grid
    ).numpy()
    similarity_heat = heatmaps(similarity, args.image_size)
    source = rgb[source_index].copy()
    patch = args.image_size / grid
    x0, y0 = int(query_column * patch), int(query_row * patch)
    x1, y1 = int((query_column + 1) * patch), int((query_row + 1) * patch)
    cv2.rectangle(source, (x0, y0), (x1, y1), (38, 224, 220), 4)
    similarity_video = [
        np.concatenate([source, target, heat], axis=1)
        for target, heat in zip(rgb, similarity_heat)
    ]

    outputs = {
        "mean_patch_attention": attention_video,
        "cross_frame_similarity": similarity_video,
    }
    output_meta: dict[str, object] = {}
    for name, frames in outputs.items():
        video_path = args.output_dir / f"{name}.webm"
        poster_path = args.output_dir / f"{name}.png"
        write_webm(video_path, frames, fps)
        Image.fromarray(frames[0]).save(poster_path)
        output_meta[name] = {
            "video": video_path.name,
            "sha256": sha256(video_path),
            "poster": poster_path.name,
            "frames": len(frames),
            "fps": fps,
        }
        print(video_path, flush=True)

    metadata = {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "source_resolution": [int(raw[0].shape[1]), int(raw[0].shape[0])],
        "decoded_frames": len(raw),
        "source_fps": fps,
        "model_input_resolution": [args.image_size, args.image_size],
        "native_feature_grid": [grid, grid],
        "display_interpolation": "bicubic",
        "checkpoint": checkpoint_meta,
        "attention_definition": (
            "Final-layer attention averaged over all heads and all patch "
            "queries; the displayed spatial axis consists of patch keys."
        ),
        "source_query": {
            "frame": source_index,
            "row": query_row,
            "column": query_column,
            "index": query_index,
        },
        "outputs": output_meta,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
