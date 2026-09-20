#!/usr/bin/env python3
"""Frozen linear evaluation for image or video classification datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from w2rep.data.video import VideoClipDataset, normalize_video
from w2rep.utils.checkpoint import load_encoder


class ImageFolderView(Dataset):
    def __init__(self, root: Path, image_size: int, spatial_preprocess: str) -> None:
        self.dataset = datasets.ImageFolder(root)
        self.image_size = image_size
        self.spatial_preprocess = spatial_preprocess
        self.center_crop = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.PILToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.dataset.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.spatial_preprocess == "center_crop":
                pixels = self.center_crop(image)
            else:
                array = cv2.resize(
                    np.asarray(image),
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                pixels = torch.from_numpy(
                    np.ascontiguousarray(array.transpose(2, 0, 1))
                )
        return pixels, label


class VideoView(Dataset):
    def __init__(
        self,
        manifest: Path,
        root: Path,
        split: str,
        frames: int,
        stride: int,
        image_size: int,
        temporal_sampling: str,
        spatial_preprocess: str,
    ) -> None:
        self.dataset = VideoClipDataset(
            manifest,
            root=root,
            split=split,
            frames=frames,
            stride=stride,
            image_size=image_size,
            cross_frame_targets=1,
            train=False,
            horizontal_flip=False,
            temporal_sampling=temporal_sampling,
            eval_spatial=spatial_preprocess,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[index]
        return item["clip_px"], item["label"]


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def encode_batch(
    encoder: nn.Module,
    pixels: torch.Tensor,
    encoding: str,
) -> torch.Tensor:
    pixels = normalize_video(pixels)
    if pixels.ndim == 4:
        pixels = pixels[:, None]
    batch, frames = pixels.shape[:2]
    if encoding == "independent" and frames > 1:
        flattened = pixels.flatten(0, 1)[:, None]
        patch = encoder(flattened, with_z=False)["patch_out"]
        return patch.mean(dim=2).reshape(batch, frames, -1).mean(dim=1)
    if encoding not in {"independent", "joint"}:
        raise ValueError(f"Unknown encoding mode: {encoding}")
    patch = encoder(pixels, with_z=False)["patch_out"]
    return patch.mean(dim=(1, 2))


@torch.inference_mode()
def extract_features(
    encoder: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    encoding: str,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features = torch.empty(len(dataset), feature_dim, dtype=torch.float32)
    labels = torch.empty(len(dataset), dtype=torch.long)
    cursor = 0
    encoder.eval()
    for batch_index, (pixels, target) in enumerate(loader):
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            descriptor = encode_batch(encoder, pixels, encoding).float().cpu()
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("Encoder produced non-finite features")
        stop = cursor + descriptor.shape[0]
        features[cursor:stop] = descriptor
        labels[cursor:stop] = target
        cursor = stop
        if (batch_index + 1) % 100 == 0:
            print(f"[features] {cursor}/{len(dataset)}", flush=True)
    if cursor != len(dataset):
        raise RuntimeError(f"Feature coverage mismatch: {cursor} != {len(dataset)}")
    return features, labels


@torch.inference_mode()
def evaluate_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, float]:
    correct1 = correct5 = total = 0
    class_correct = torch.zeros(head.out_features, dtype=torch.long)
    class_total = torch.zeros(head.out_features, dtype=torch.long)
    for start in range(0, len(features), batch_size):
        feature = features[start : start + batch_size].to(device)
        target = labels[start : start + batch_size].to(device)
        logits = head(feature)
        topk = logits.topk(min(5, head.out_features), dim=1).indices
        prediction = topk[:, 0]
        matches = prediction.eq(target)
        correct1 += int(matches.sum())
        correct5 += int(topk.eq(target[:, None]).any(dim=1).sum())
        total += len(target)
        class_total.scatter_add_(0, target.cpu(), torch.ones_like(target.cpu()))
        class_correct.scatter_add_(0, target.cpu(), matches.long().cpu())
    observed = class_total > 0
    macro = (class_correct[observed].float() / class_total[observed]).mean().item()
    return {
        "top1_percent": 100.0 * correct1 / total,
        "top5_percent": 100.0 * correct5 / total,
        "macro_accuracy_percent": 100.0 * macro,
    }


def train_linear_heads(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int,
    seeds: list[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    classes = sorted(train_labels.unique().tolist())
    if classes != list(range(len(classes))):
        raise ValueError("Labels must be contiguous integers beginning at zero")
    if not set(val_labels.unique().tolist()).issubset(classes):
        raise ValueError("Validation contains labels absent from training")
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-6)
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        head = nn.Linear(train_features.shape[1], len(classes)).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        curve: list[float] = []
        for epoch in range(epochs):
            head.train()
            generator = torch.Generator().manual_seed(seed + epoch * 1009)
            order = torch.randperm(len(train_features), generator=generator)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                feature = train_features[indices].to(device, non_blocking=True)
                target = train_labels[indices].to(device, non_blocking=True)
                loss = F.cross_entropy(head(feature), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            head.eval()
            metrics = evaluate_head(
                head, val_features, val_labels, device=device
            )
            curve.append(metrics["top1_percent"])
            print(
                f"[probe] seed={seed} epoch={epoch + 1}/{epochs} "
                f"top1={metrics['top1_percent']:.3f}",
                flush=True,
            )
        head_path = output_dir / f"linear_head_seed{seed}.pt"
        atomic_torch_save(
            {
                "head": {key: value.cpu() for key, value in head.state_dict().items()},
                "feature_mean": mean,
                "feature_std": std,
                "seed": seed,
                "epochs": epochs,
                "metrics": metrics,
            },
            head_path,
        )
        runs.append(
            {
                "seed": seed,
                "metrics": metrics,
                "validation_top1_curve": curve,
                "head": str(head_path),
            }
        )
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    for task in ("image", "video"):
        child = subparsers.add_parser(task)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--checkpoint-key", choices=("target", "encoder"), default="target")
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--image-size", type=int, default=224)
        child.add_argument("--encoding", choices=("independent", "joint"), default="independent")
        child.add_argument("--feature-batch-size", type=int, default=128)
        child.add_argument("--workers", type=int, default=8)
        child.add_argument("--epochs", type=int, default=50)
        child.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
        child.add_argument("--head-batch-size", type=int, default=1024)
        child.add_argument("--learning-rate", type=float, default=1e-3)
        child.add_argument("--weight-decay", type=float, default=1e-4)
    image = subparsers.choices["image"]
    image.add_argument("--data-root", type=Path, required=True)
    image.add_argument("--train-split", default="train")
    image.add_argument("--val-split", default="val")
    image.add_argument(
        "--spatial-preprocess",
        choices=("direct_resize", "center_crop"),
        default="direct_resize",
    )
    video = subparsers.choices["video"]
    video.add_argument("--manifest", type=Path, required=True)
    video.add_argument("--data-root", type=Path, default=Path("."))
    video.add_argument("--train-split", default="train")
    video.add_argument("--val-split", default="val")
    video.add_argument("--frames", type=int, default=8)
    video.add_argument("--stride", type=int, default=3)
    video.add_argument(
        "--temporal-sampling", choices=("uniform", "stride"), default="uniform"
    )
    video.add_argument(
        "--spatial-preprocess",
        choices=("direct_resize", "center_crop"),
        default="direct_resize",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    cache_path = output_dir / "features.pt"
    if result_path.exists() or result_path.with_suffix(".json.partial").exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {result_path}")
    encoder, checkpoint = load_encoder(
        args.checkpoint,
        image_size=args.image_size,
        checkpoint_key=args.checkpoint_key,
    )
    encoder = encoder.to(device).eval().requires_grad_(False)
    feature_dim = checkpoint["embed_dim"]
    identity = {
        "checkpoint": checkpoint,
        "task": args.task,
        "image_size": args.image_size,
        "encoding": args.encoding,
        "data_root": str(args.data_root.expanduser().resolve()),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "manifest": str(args.manifest.expanduser().resolve()) if args.task == "video" else None,
        "frames": args.frames if args.task == "video" else 1,
        "stride": args.stride if args.task == "video" else None,
        "temporal_sampling": args.temporal_sampling if args.task == "video" else None,
        "spatial_preprocess": args.spatial_preprocess,
    }
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache.get("identity") != identity:
            raise RuntimeError("Feature cache identity does not match this evaluation")
        train_features = cache["train_features"]
        train_labels = cache["train_labels"]
        val_features = cache["val_features"]
        val_labels = cache["val_labels"]
    else:
        if args.task == "image":
            train_dataset = ImageFolderView(
                args.data_root / args.train_split,
                args.image_size,
                args.spatial_preprocess,
            )
            val_dataset = ImageFolderView(
                args.data_root / args.val_split,
                args.image_size,
                args.spatial_preprocess,
            )
        else:
            train_dataset = VideoView(
                args.manifest,
                args.data_root,
                args.train_split,
                args.frames,
                args.stride,
                args.image_size,
                args.temporal_sampling,
                args.spatial_preprocess,
            )
            val_dataset = VideoView(
                args.manifest,
                args.data_root,
                args.val_split,
                args.frames,
                args.stride,
                args.image_size,
                args.temporal_sampling,
                args.spatial_preprocess,
            )
        train_features, train_labels = extract_features(
            encoder,
            train_dataset,
            batch_size=args.feature_batch_size,
            workers=args.workers,
            device=device,
            encoding=args.encoding,
            feature_dim=feature_dim,
        )
        val_features, val_labels = extract_features(
            encoder,
            val_dataset,
            batch_size=args.feature_batch_size,
            workers=args.workers,
            device=device,
            encoding=args.encoding,
            feature_dim=feature_dim,
        )
        atomic_torch_save(
            {
                "identity": identity,
                "train_features": train_features,
                "train_labels": train_labels,
                "val_features": val_features,
                "val_labels": val_labels,
            },
            cache_path,
        )
    runs = train_linear_heads(
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=args.epochs,
        seeds=list(args.seeds),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.head_batch_size,
        device=device,
        output_dir=output_dir,
    )
    top1 = np.asarray([run["metrics"]["top1_percent"] for run in runs])
    result = {
        "status": "complete",
        "identity": identity,
        "protocol": {
            "epochs": args.epochs,
            "seeds": list(args.seeds),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "head_batch_size": args.head_batch_size,
            "selection": "fixed final epoch",
        },
        "runs": runs,
        "summary": {
            "top1_mean_percent": float(top1.mean()),
            "top1_std_percent": float(top1.std(ddof=0)),
        },
    }
    atomic_json_save(result, result_path)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
