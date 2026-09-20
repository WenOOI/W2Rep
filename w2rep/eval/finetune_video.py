#!/usr/bin/env python3
"""Full fine-tuning of a W2Rep encoder on a labeled video dataset."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from w2rep.data.video import (
    decode_video,
    load_csv_manifest,
    load_json_manifest,
    normalize_video,
)
from w2rep.utils.checkpoint import load_checkpoint_file, load_encoder


class VideoFineTuneDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        root: Path,
        split: str,
        frames: int,
        image_size: int,
        train: bool,
        crop_scale: tuple[float, float] = (0.6, 1.0),
    ) -> None:
        loader = load_csv_manifest if manifest.suffix.lower() == ".csv" else load_json_manifest
        self.entries = loader(manifest, split, root)
        self.frames = frames
        self.image_size = image_size
        self.train = train
        self.crop_scale = crop_scale

    def __len__(self) -> int:
        return len(self.entries)

    def _frame_indices(self, length: int) -> list[int]:
        if self.train:
            boundaries = np.linspace(0, length, self.frames + 1, dtype=int)
            return [
                min(start, length - 1)
                if stop <= start
                else random.randrange(start, stop)
                for start, stop in zip(boundaries[:-1], boundaries[1:])
            ]
        return np.linspace(0, length - 1, self.frames, dtype=int).tolist()

    def _crop(self, height: int, width: int) -> tuple[int, int, int, int]:
        area = height * width
        for _ in range(10):
            target = random.uniform(*self.crop_scale) * area
            aspect = random.uniform(0.75, 4.0 / 3.0)
            crop_width = int(round(math.sqrt(target * aspect)))
            crop_height = int(round(math.sqrt(target / aspect)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                return top, left, crop_height, crop_width
        side = min(height, width)
        return (height - side) // 2, (width - side) // 2, side, side

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        attempts = 16 if self.train else 1
        for offset in range(attempts):
            entry = self.entries[(index + offset) % len(self.entries)]
            decoded = decode_video(Path(entry["path"]))
            if decoded:
                break
        else:
            raise RuntimeError(
                f"Could not decode a video after {attempts} attempts from index {index}"
            )
        indices = self._frame_indices(len(decoded))
        crop = self._crop(*decoded[indices[0]].shape[:2]) if self.train else None
        frames = []
        for frame_index in indices:
            frame = decoded[frame_index]
            if crop is not None:
                top, left, height, width = crop
                frame = frame[top : top + height, left : left + width]
            frame = cv2.resize(
                frame,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            frames.append(
                torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
            )
        return torch.stack(frames), int(entry["label"])


class DistributedSliceSampler(Sampler[int]):
    """Shard validation samples without padding duplicates."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = list(range(rank, size, world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class VideoClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, dimension: int, classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dimension, classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        patches = self.encoder(clips, with_z=False)["patch_out"]
        return self.head(patches.mean(dim=(1, 2)))


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Full video fine-tuning requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def layer_id(name: str, depth: int) -> int:
    if name.startswith("encoder.patch_embed"):
        return 0
    if name.startswith("encoder.blocks."):
        return int(name.split(".")[2]) + 1
    return depth + 1


def build_optimizer(
    model: nn.Module,
    *,
    depth: int,
    peak_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> torch.optim.Optimizer:
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer = layer_id(name, depth)
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        key = (layer, no_decay)
        groups.setdefault(
            key,
            {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": layer_decay ** (depth + 1 - layer),
            },
        )["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), lr=peak_lr, betas=(0.9, 0.999))


def scheduled_learning_rate(
    update: int,
    total_updates: int,
    warmup_updates: int,
    peak: float,
    minimum: float,
) -> float:
    if update < warmup_updates:
        return minimum + (peak - minimum) * (update + 1) / max(1, warmup_updates)
    progress = min(
        1.0,
        (update - warmup_updates) / max(1, total_updates - warmup_updates - 1),
    )
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer: torch.optim.Optimizer, base_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * group["lr_scale"]


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: int,
) -> dict[str, float | int]:
    model.eval()
    correct1 = torch.zeros((), dtype=torch.long, device=device)
    correct5 = torch.zeros((), dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.long, device=device)
    class_correct = torch.zeros(classes, dtype=torch.long, device=device)
    class_total = torch.zeros(classes, dtype=torch.long, device=device)
    for clips, labels in loader:
        clips = normalize_video(clips.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(clips)
        topk = logits.topk(min(5, classes), dim=1).indices
        prediction = topk[:, 0]
        matches = prediction.eq(labels)
        correct1 += matches.sum()
        correct5 += topk.eq(labels[:, None]).any(dim=1).sum()
        total += len(labels)
        class_total.scatter_add_(0, labels, torch.ones_like(labels))
        class_correct.scatter_add_(0, labels, matches.long())
    if dist.is_initialized():
        for tensor in (correct1, correct5, total, class_correct, class_total):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    observed = class_total > 0
    return {
        "samples": int(total.item()),
        "top1_percent": float(correct1.item() / total.item() * 100.0),
        "top5_percent": float(correct5.item() / total.item() * 100.0),
        "macro_accuracy_percent": float(
            (class_correct[observed].float() / class_total[observed]).mean().item()
            * 100.0
        ),
    }


def atomic_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-key", choices=("target", "encoder"), default="target")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-lr-at-256", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = initialize_distributed()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VideoFineTuneDataset(
        args.manifest,
        root=args.data_root,
        split=args.train_split,
        frames=args.frames,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = VideoFineTuneDataset(
        args.manifest,
        root=args.data_root,
        split=args.val_split,
        frames=args.frames,
        image_size=args.image_size,
        train=False,
    )
    labels = sorted({int(entry["label"]) for entry in train_dataset.entries})
    if labels != list(range(len(labels))):
        raise ValueError("Training labels must be contiguous integers beginning at zero")
    classes = len(labels)
    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    train_generator = torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSliceSampler(len(val_dataset), rank, world_size),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    encoder, checkpoint_metadata = load_encoder(
        args.checkpoint,
        image_size=args.image_size,
        checkpoint_key=args.checkpoint_key,
        activation_checkpointing=True,
    )
    encoder.z_tokens.requires_grad_(False)
    encoder.norm_z.requires_grad_(False)
    model = VideoClassifier(encoder, encoder.embed_dim, classes).to(device)
    depth = len(encoder.blocks)
    effective_batch = args.batch_size * world_size * args.accumulation_steps
    peak_lr = args.base_lr_at_256 * effective_batch / 256.0
    optimizer = build_optimizer(
        model,
        depth=depth,
        peak_lr=peak_lr,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
    )
    start_epoch = global_update = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = load_checkpoint_file(args.resume)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        global_update = int(checkpoint["global_update"])
        history = list(checkpoint.get("history", []))
    model_ddp: nn.Module = (
        DistributedDataParallel(model, device_ids=[local_rank])
        if world_size > 1
        else model
    )
    updates_per_epoch = len(train_loader) // args.accumulation_steps
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = updates_per_epoch * args.warmup_epochs

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_generator.manual_seed(args.seed + epoch * 1_000_003 + rank * 1_009)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch_index, (clips, labels_tensor) in enumerate(train_loader):
            if batch_index >= updates_per_epoch * args.accumulation_steps:
                break
            clips = normalize_video(clips.to(device, non_blocking=True))
            labels_tensor = labels_tensor.to(device, non_blocking=True)
            synchronize = (batch_index + 1) % args.accumulation_steps == 0
            sync_context = (
                contextlib.nullcontext()
                if synchronize or world_size == 1
                else model_ddp.no_sync()
            )
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model_ddp(clips)
                    loss = F.cross_entropy(
                        logits,
                        labels_tensor,
                        label_smoothing=args.label_smoothing,
                    )
                    scaled_loss = loss / args.accumulation_steps
                scaled_loss.backward()
            running_loss += float(loss.detach())
            if synchronize:
                base_lr = scheduled_learning_rate(
                    global_update,
                    total_updates,
                    warmup_updates,
                    peak_lr,
                    args.min_lr,
                )
                set_learning_rate(optimizer, base_lr)
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "mean_train_loss": running_loss / (updates_per_epoch * args.accumulation_steps),
            "global_update": global_update,
        }
        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
            record["validation"] = validate(model_ddp, val_loader, device, classes)
        history.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            if rank == 0:
                path = (
                    output_dir / "ckpt_final.pt"
                    if epoch + 1 == args.epochs
                    else output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
                )
                atomic_save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "global_update": global_update,
                        "history": history,
                        "checkpoint": checkpoint_metadata,
                        "protocol": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                        },
                    },
                    path,
                )

    if rank == 0:
        result_path = output_dir / "result.json"
        temporary = result_path.with_suffix(".json.partial")
        if result_path.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {result_path}")
        temporary.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "checkpoint": checkpoint_metadata,
                    "history": history,
                    "final": history[-1].get("validation"),
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        os.replace(temporary, result_path)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
