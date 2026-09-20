"""Portable video-manifest datasets for W2Rep."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_video(
    pixels: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    pixels = pixels.float().div(255.0)
    shape = [1] * pixels.ndim
    shape[-3] = 3
    mean_tensor = torch.as_tensor(mean, device=pixels.device, dtype=pixels.dtype)
    std_tensor = torch.as_tensor(std, device=pixels.device, dtype=pixels.dtype)
    return (pixels - mean_tensor.view(shape)) / std_tensor.view(shape)


def decode_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frames: list[np.ndarray] = []
    while True:
        success, bgr = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def load_json_manifest(
    manifest: str | Path,
    split: str,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    manifest = Path(manifest).expanduser().resolve()
    payload = json.loads(manifest.read_text())
    entries = payload.get(split) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"Manifest {manifest} has no list-valued split {split!r}")
    root = Path(root).expanduser()
    resolved = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"Invalid manifest entry {index}: {entry!r}")
        item = dict(entry)
        path = Path(item["path"]).expanduser()
        item["path"] = str(path if path.is_absolute() else root / path)
        item.setdefault("id", str(index))
        item.setdefault("label", item.get("template_id", -1))
        resolved.append(item)
    return resolved


def load_csv_manifest(
    manifest: str | Path,
    split: str,
    root: str | Path = ".",
) -> list[dict[str, Any]]:
    manifest = Path(manifest).expanduser().resolve()
    root = Path(root).expanduser()
    entries: list[dict[str, Any]] = []
    with manifest.open(newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            if row.get("split", split) != split:
                continue
            path = Path(row["path"]).expanduser()
            entries.append(
                {
                    **row,
                    "path": str(path if path.is_absolute() else root / path),
                    "id": row.get("id", str(row_index)),
                    "label": int(row["label"]),
                }
            )
    if not entries:
        raise ValueError(f"No {split!r} rows found in {manifest}")
    return entries


class VideoClipDataset(Dataset):
    """Sample one geometrically consistent clip and source/target indices."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        root: str | Path = ".",
        split: str = "train",
        frames: int = 8,
        stride: int = 3,
        image_size: int = 224,
        cross_frame_targets: int = 3,
        train: bool = True,
        crop_scale: tuple[float, float] = (0.6, 1.0),
        horizontal_flip: bool = True,
        max_decode_retries: int = 8,
        temporal_sampling: str = "stride",
        eval_spatial: str = "center_crop",
    ) -> None:
        suffix = Path(manifest).suffix.lower()
        loader = load_csv_manifest if suffix == ".csv" else load_json_manifest
        self.entries = loader(manifest, split, root)
        self.frames = frames
        self.stride = stride
        self.image_size = image_size
        self.cross_frame_targets = cross_frame_targets
        self.train = train
        self.crop_scale = tuple(crop_scale)
        self.horizontal_flip = horizontal_flip
        self.max_decode_retries = max_decode_retries
        self.temporal_sampling = temporal_sampling
        self.eval_spatial = eval_spatial
        if frames < 2:
            raise ValueError("A video clip must contain at least two sampled frames")
        if cross_frame_targets > frames - 1:
            raise ValueError("cross_frame_targets cannot exceed frames - 1")
        if temporal_sampling not in {"stride", "uniform"}:
            raise ValueError("temporal_sampling must be 'stride' or 'uniform'")
        if eval_spatial not in {"center_crop", "direct_resize"}:
            raise ValueError("eval_spatial must be 'center_crop' or 'direct_resize'")

    def __len__(self) -> int:
        return len(self.entries)

    def _sample_frame_indices(self, length: int) -> list[int]:
        if not self.train and self.temporal_sampling == "uniform":
            return np.linspace(0, length - 1, self.frames, dtype=int).tolist()
        span = (self.frames - 1) * self.stride
        if length - 1 >= span:
            start = random.randint(0, length - 1 - span) if self.train else (length - 1 - span) // 2
            return [start + index * self.stride for index in range(self.frames)]
        return [min(index * self.stride, length - 1) for index in range(self.frames)]

    def _crop_parameters(self, height: int, width: int) -> tuple[tuple[int, int, int, int], bool]:
        if not self.train:
            if self.eval_spatial == "direct_resize":
                return (0, 0, height, width), False
            side = min(height, width)
            return ((height - side) // 2, (width - side) // 2, side, side), False
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*self.crop_scale) * area
            aspect = random.uniform(3.0 / 4.0, 4.0 / 3.0)
            crop_width = int(round(np.sqrt(target_area * aspect)))
            crop_height = int(round(np.sqrt(target_area / aspect)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                flip = self.horizontal_flip and random.random() < 0.5
                return (top, left, crop_height, crop_width), flip
        side = min(height, width)
        return ((height - side) // 2, (width - side) // 2, side, side), False

    def _transform(
        self,
        frame: np.ndarray,
        crop: tuple[int, int, int, int],
        flip: bool,
    ) -> torch.Tensor:
        top, left, height, width = crop
        frame = frame[top : top + height, left : left + width]
        frame = cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        if flip:
            frame = frame[:, ::-1]
        frame = np.ascontiguousarray(frame.transpose(2, 0, 1))
        return torch.from_numpy(frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        attempts = self.max_decode_retries if self.train else 1
        for retry in range(attempts):
            entry_index = index if retry == 0 else random.randrange(len(self.entries))
            entry = self.entries[entry_index]
            decoded = decode_video(Path(entry["path"]))
            if len(decoded) >= 2:
                break
        else:
            raise RuntimeError(
                f"Failed to decode a valid video after {attempts} attempts; "
                f"first path was {self.entries[index]['path']}"
            )
        height, width = decoded[0].shape[:2]
        crop, flip = self._crop_parameters(height, width)
        indices = self._sample_frame_indices(len(decoded))
        clip = torch.stack(
            [self._transform(decoded[frame_index], crop, flip) for frame_index in indices]
        )
        source = random.randrange(self.frames)
        candidates = [frame_index for frame_index in range(self.frames) if frame_index != source]
        targets = random.sample(candidates, self.cross_frame_targets)
        return {
            "clip_px": clip,
            "source_index": source,
            "target_indices": targets,
            "label": int(entry.get("label", -1)),
            "video_id": str(entry["id"]),
            "path": str(entry["path"]),
        }


def collate_video_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clip_px": torch.stack([item["clip_px"] for item in batch]),
        "source_index": torch.tensor(
            [item["source_index"] for item in batch], dtype=torch.long
        ),
        "target_indices": torch.tensor(
            [item["target_indices"] for item in batch], dtype=torch.long
        ),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "video_id": [item["video_id"] for item in batch],
        "path": [item["path"] for item in batch],
    }
