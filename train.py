#!/usr/bin/env python3
"""Distributed W2Rep pretraining."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from w2rep.data import BlockMaskCollator, VideoClipDataset, normalize_video
from w2rep.data.video import collate_video_batch
from w2rep.models import build_w2rep
from w2rep.utils.config import Config, load_config
from w2rep.utils.checkpoint import load_checkpoint_file
from w2rep.utils.distributed import (
    finish_distributed,
    initialize_distributed,
    is_main_process,
    reduce_mean,
)
from w2rep.utils.schedules import (
    CosineWeightDecaySchedule,
    WarmupCosineSchedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help="Allow resume when the YAML hash differs. Use only for intentional migrations.",
    )
    return parser.parse_args()


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def update_ema(target: nn.Module, student: nn.Module, momentum: float) -> None:
    for target_parameter, student_parameter in zip(
        target.parameters(), student.parameters()
    ):
        target_parameter.mul_(momentum).add_(
            student_parameter.detach(), alpha=1.0 - momentum
        )
    for target_buffer, student_buffer in zip(target.buffers(), student.buffers()):
        target_buffer.copy_(student_buffer)


def atomic_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint artifact: {path}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    *,
    encoder: nn.Module,
    target: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_schedule: WarmupCosineSchedule,
    wd_schedule: CosineWeightDecaySchedule,
    config: Config,
    config_path: Path,
    config_sha256: str,
    step: int,
    epoch: int,
    batches_in_epoch: int,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    local_rng = rng_state()
    if dist.is_initialized():
        states: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(states, local_rng)
    else:
        states = [local_rng]
    if rank != 0:
        return None
    optimizer_state = optimizer.state_dict()
    lr_state = lr_schedule.state_dict()
    wd_state = wd_schedule.state_dict()
    plain_config = config.plain()
    return {
        "format_version": 1,
        "encoder": encoder.state_dict(),
        "target": target.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer_state,
        # Legacy aliases make the release checkpoint usable with old scripts.
        "optim": optimizer_state,
        "lr_schedule": lr_state,
        "sched": {"_step": lr_schedule.step_count},
        "weight_decay_schedule": wd_state,
        "wd_sched": {"_step": wd_schedule.step_count},
        "step": step,
        "epoch": epoch,
        "batches_in_epoch": batches_in_epoch,
        "rng_states": states,
        "world_size": world_size,
        "config": plain_config,
        "cfg": plain_config,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
    }


def load_training_state(
    path: Path,
    *,
    encoder: nn.Module,
    target: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_schedule: WarmupCosineSchedule,
    wd_schedule: CosineWeightDecaySchedule,
    config_sha256: str,
    allow_config_change: bool,
    rank: int,
    world_size: int,
) -> tuple[int, int, int]:
    checkpoint = load_checkpoint_file(path)
    saved_hash = checkpoint.get("config_sha256")
    if saved_hash and saved_hash != config_sha256 and not allow_config_change:
        raise RuntimeError(
            "The resume checkpoint was produced by a different YAML file. "
            "Pass --allow-config-change only if this is intentional."
        )
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    target.load_state_dict(checkpoint["target"], strict=True)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    optimizer_state = checkpoint.get("optimizer", checkpoint.get("optim"))
    if optimizer_state is None:
        raise RuntimeError("Resume checkpoint has no optimizer state")
    optimizer.load_state_dict(optimizer_state)
    lr_state = checkpoint.get("lr_schedule", checkpoint.get("sched"))
    wd_state = checkpoint.get(
        "weight_decay_schedule", checkpoint.get("wd_sched")
    )
    if lr_state is None or wd_state is None:
        raise RuntimeError("Resume checkpoint has incomplete schedule state")
    lr_schedule.load_state_dict(lr_state)
    wd_schedule.load_state_dict(wd_state)
    saved_world = checkpoint.get("world_size")
    states = checkpoint.get("rng_states")
    if states is not None and saved_world == world_size and rank < len(states):
        restore_rng_state(states[rank])
    elif is_main_process():
        print(
            "[resume] RNG state was not restored because the checkpoint world size "
            "does not match; model/optimizer/schedules were restored.",
            flush=True,
        )
    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("batches_in_epoch", 0)),
    )


def make_optimizer(
    encoder: nn.Module,
    predictor: nn.Module,
    config: Config,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for module in (encoder, predictor):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(
                parameter
            )
    groups = [
        {"params": decay},
        {
            "params": no_decay,
            "weight_decay": 0.0,
            "exclude_weight_decay": True,
        },
    ]
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.train.peak_lr,
        betas=tuple(config.train.betas),
        weight_decay=config.train.weight_decay,
    )
    return optimizer, decay + no_decay


def clip_visible_to_z(
    clip: torch.Tensor,
    source_index: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    mode: str,
    seed: int,
    step: int,
    rank: int,
) -> torch.Tensor:
    """Optionally remove matched frame content for leakage-control ablations."""
    if mode == "all":
        return clip
    batch, frames = clip.shape[:2]
    if mode == "exclude_targets":
        excluded = target_indices
    elif mode == "exclude_random":
        rows = []
        for sample_index in range(batch):
            occupied = {
                int(source_index[sample_index]),
                *[int(value) for value in target_indices[sample_index]],
            }
            candidates = [index for index in range(frames) if index not in occupied]
            if len(candidates) < target_indices.shape[1]:
                raise RuntimeError("Not enough non-source/non-target frames for random exclusion")
            generator = torch.Generator(device="cpu").manual_seed(
                seed * 1_000_003 + step * 97_409 + rank * 10_007 + sample_index
            )
            order = torch.randperm(len(candidates), generator=generator)
            rows.append([candidates[index] for index in order[: target_indices.shape[1]]])
        excluded = torch.tensor(rows, dtype=torch.long, device=clip.device)
    else:
        raise ValueError(f"Unknown z-context mode: {mode}")
    result = clip.clone()
    batch_index = torch.arange(batch, device=clip.device)[:, None]
    result[batch_index, excluded] = 0.0
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config, config_sha256 = load_config(config_path)
    rank, world_size, local_rank, device = initialize_distributed()
    seed_everything(config.train.seed, rank)

    dataset = VideoClipDataset(
        config.data.manifest,
        root=config.data.root,
        split=config.data.split,
        frames=config.data.frames,
        stride=config.data.stride,
        image_size=config.data.image_size,
        cross_frame_targets=config.data.cross_frame_targets,
        train=True,
        crop_scale=tuple(config.data.random_resized_crop_scale),
        horizontal_flip=config.data.horizontal_flip,
    )
    collator = BlockMaskCollator(
        grid_size=config.data.image_size // config.model.patch_size,
        encoder_scale=tuple(config.mask.encoder_scale),
        predictor_scale=tuple(config.mask.predictor_scale),
        aspect_ratio=tuple(config.mask.aspect_ratio),
        predictor_blocks=config.mask.predictor_blocks,
        min_keep=config.mask.min_keep,
        seed=config.train.seed,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.train.seed,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    loader_generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size_per_gpu,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.train.workers,
        collate_fn=collate_video_batch,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2 if config.train.workers > 0 else None,
        generator=loader_generator,
    )

    encoder, predictor = build_w2rep(
        config.model.architecture,
        image_size=config.data.image_size,
        patch_size=config.model.patch_size,
        frames=config.data.frames,
        z_tokens=config.model.z_tokens,
        predictor_dim=config.model.predictor_dim,
        predictor_depth=config.model.predictor_depth,
        activation_checkpointing=config.model.activation_checkpointing,
        latent_reads_patches_only=config.model.latent_reads_patches_only,
    )
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    target = copy.deepcopy(encoder).to(device).requires_grad_(False).eval()
    optimizer, trainable_parameters = make_optimizer(encoder, predictor, config)
    lr_schedule = WarmupCosineSchedule(
        optimizer,
        config.train.warmup_steps,
        config.train.start_lr,
        config.train.peak_lr,
        config.train.total_steps,
        config.train.final_lr,
    )
    wd_schedule = CosineWeightDecaySchedule(
        optimizer,
        config.train.weight_decay,
        config.train.final_weight_decay,
        config.train.total_steps,
    )

    step = epoch = batches_in_epoch = 0
    if args.resume:
        step, epoch, batches_in_epoch = load_training_state(
            args.resume.expanduser().resolve(),
            encoder=encoder,
            target=target,
            predictor=predictor,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            config_sha256=config_sha256,
            allow_config_change=args.allow_config_change,
            rank=rank,
            world_size=world_size,
        )
        target.to(device)

    encoder_ddp: nn.Module = (
        DistributedDataParallel(
            encoder, device_ids=[local_rank], static_graph=True
        )
        if world_size > 1
        else encoder
    )
    predictor_ddp: nn.Module = (
        DistributedDataParallel(
            predictor, device_ids=[local_rank], static_graph=True
        )
        if world_size > 1
        else predictor
    )
    encoder.train()
    predictor.train()

    output_dir = Path(config.output_dir).expanduser().resolve()
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = output_dir / "resolved_config.json"
        resolved_config_text = json.dumps(config.plain(), indent=2) + "\n"
        if resolved_config_path.exists():
            if resolved_config_path.read_text() != resolved_config_text:
                raise FileExistsError(
                    f"Existing resolved config differs: {resolved_config_path}"
                )
        else:
            resolved_config_path.write_text(resolved_config_text)
        encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
        predictor_parameters = sum(parameter.numel() for parameter in predictor.parameters())
        print(
            f"[model] encoder={encoder_parameters / 1e6:.1f}M "
            f"predictor={predictor_parameters / 1e6:.1f}M",
            flush=True,
        )
        print(
            "[objective] masked positions; "
            f"same_frame={config.objective.same_frame}, "
            f"cross_frame={config.objective.cross_frame}, "
            f"signed_offset={config.objective.signed_offset}, "
            f"clip_latents={config.objective.use_clip_latents}, "
            f"z_l2={config.objective.z_l2}",
            flush=True,
        )
    writer = None
    if is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(output_dir / "tensorboard")
        except ImportError:
            print("[logging] tensorboard is unavailable; continuing without it", flush=True)

    amp_name = str(config.train.amp).lower()
    amp_enabled = amp_name == "bf16"
    start_time = time.time()

    def new_iterator(current_epoch: int):
        if sampler is not None:
            sampler.set_epoch(current_epoch)
        loader_generator.manual_seed(config.train.seed + current_epoch * 1_000_003)
        return iter(loader)

    data_iterator = new_iterator(epoch)
    if batches_in_epoch:
        if is_main_process():
            print(
                f"[resume] advancing {batches_in_epoch} batches in epoch {epoch}",
                flush=True,
            )
        for _ in range(batches_in_epoch):
            try:
                next(data_iterator)
            except StopIteration as error:
                raise RuntimeError("Saved batch offset exceeds current epoch length") from error
    collator.set_call_index(epoch * len(loader) + batches_in_epoch)

    while step < config.train.total_steps:
        try:
            batch = next(data_iterator)
            batches_in_epoch += 1
        except StopIteration:
            epoch += 1
            batches_in_epoch = 0
            data_iterator = new_iterator(epoch)
            batch = next(data_iterator)
            batches_in_epoch = 1

        clip = normalize_video(
            batch["clip_px"].to(device, non_blocking=True)
        )
        spatial_masks = collator.sample_masks(clip.shape[0])
        source_index = batch["source_index"].to(device, non_blocking=True)
        cross_indices = batch["target_indices"].to(device, non_blocking=True)
        mask_encoder = spatial_masks["mask_encoder"].to(device, non_blocking=True)
        mask_predictor = spatial_masks["mask_predictor"].to(device, non_blocking=True)
        batch_size = clip.shape[0]
        batch_index = torch.arange(batch_size, device=device)
        source = clip[batch_index, source_index]
        prediction_positions = mask_predictor.flatten(1)
        z_context_mode = config.objective.get("z_context", "all")

        target_columns: list[torch.Tensor] = []
        if config.objective.same_frame:
            target_columns.append(source_index)
        if config.objective.cross_frame:
            target_columns.extend(cross_indices.unbind(dim=1))
        if not target_columns:
            raise RuntimeError("No active target terms")

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            if config.objective.use_clip_latents:
                clip_for_z = clip_visible_to_z(
                    clip,
                    source_index,
                    cross_indices,
                    mode=z_context_mode,
                    seed=config.train.seed,
                    step=step,
                    rank=rank,
                )
                z = encoder_ddp(
                    clip_for_z, mask_encoder, with_z=True
                )["z_out"]
                if z is None:
                    raise RuntimeError("Encoder did not return auxiliary clip latents")
            else:
                z = torch.zeros(
                    batch_size,
                    config.model.z_tokens,
                    encoder.embed_dim,
                    device=device,
                    dtype=clip.dtype,
                )
            source_features = encoder_ddp(
                source[:, None], mask_encoder, with_z=False
            )["patch_out"][:, 0]
            loss_terms: list[torch.Tensor] = []
            cosine_terms: list[torch.Tensor] = []
            for target_index in target_columns:
                offset = (target_index - source_index).float()
                if not config.objective.signed_offset:
                    offset = torch.zeros_like(offset)
                z_for_target = z * (target_index != source_index)[:, None, None]
                prediction = predictor_ddp(
                    source_features,
                    mask_encoder,
                    prediction_positions,
                    offset,
                    z_for_target,
                )
                with torch.no_grad():
                    target_frame = clip[batch_index, target_index]
                    target_features = target(
                        target_frame[:, None], with_z=False
                    )["patch_out"][:, 0]
                    target_features = F.layer_norm(
                        target_features, (target_features.shape[-1],)
                    )
                    target_features = torch.gather(
                        target_features,
                        1,
                        prediction_positions[:, :, None].expand(
                            -1, -1, target_features.shape[-1]
                        ),
                    )
                loss_terms.append(
                    F.smooth_l1_loss(prediction.float(), target_features.float())
                )
                cosine_terms.append(
                    F.cosine_similarity(
                        prediction.float(), target_features.float(), dim=-1
                    ).mean()
                )
            prediction_loss = torch.stack(loss_terms).mean()
            z_penalty = z.float().square().mean() if config.objective.z_l2 else z.new_zeros(())
            loss = prediction_loss + config.objective.z_l2 * z_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            trainable_parameters, config.train.gradient_clip
        )
        optimizer.step()
        learning_rate = lr_schedule.step()
        weight_decay = wd_schedule.step()
        progress = step / max(1, config.train.total_steps)
        momentum = config.ema.start_momentum + progress * (
            config.ema.final_momentum - config.ema.start_momentum
        )
        update_ema(target, encoder, momentum)
        step += 1

        if step % config.train.log_every == 0:
            mean_loss = reduce_mean(loss).item()
            mean_prediction_loss = reduce_mean(prediction_loss).item()
            mean_cosine = reduce_mean(torch.stack(cosine_terms).mean()).item()
            if is_main_process():
                elapsed = max(time.time() - start_time, 1e-6)
                samples_per_second = (
                    config.train.log_every * batch_size * world_size / elapsed
                )
                message = (
                    f"step={step}/{config.train.total_steps} "
                    f"loss={mean_loss:.5f} prediction={mean_prediction_loss:.5f} "
                    f"cosine={mean_cosine:.4f} z_l2={z_penalty.item():.4f} "
                    f"grad_norm={float(gradient_norm):.3f} lr={learning_rate:.3e} "
                    f"wd={weight_decay:.4f} ema={momentum:.6f} "
                    f"samples/s={samples_per_second:.1f}"
                )
                print(message, flush=True)
                if writer is not None:
                    writer.add_scalar("train/loss", mean_loss, step)
                    writer.add_scalar("train/prediction_loss", mean_prediction_loss, step)
                    writer.add_scalar("train/cosine", mean_cosine, step)
                    writer.add_scalar("train/z_l2_unscaled", z_penalty.item(), step)
                    writer.add_scalar("train/learning_rate", learning_rate, step)
                    writer.add_scalar("train/weight_decay", weight_decay, step)
                start_time = time.time()

        should_save = (
            step % config.train.checkpoint_every == 0
            and step < config.train.total_steps
        )
        if should_save:
            payload = checkpoint_payload(
                encoder=encoder,
                target=target,
                predictor=predictor,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                wd_schedule=wd_schedule,
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                step=step,
                epoch=epoch,
                batches_in_epoch=batches_in_epoch,
                rank=rank,
                world_size=world_size,
            )
            if is_main_process():
                path = output_dir / f"checkpoint_{step:07d}.pt"
                atomic_checkpoint(payload, path)
                print(f"[checkpoint] {path}", flush=True)

    payload = checkpoint_payload(
        encoder=encoder,
        target=target,
        predictor=predictor,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        wd_schedule=wd_schedule,
        config=config,
        config_path=config_path,
        config_sha256=config_sha256,
        step=step,
        epoch=epoch,
        batches_in_epoch=batches_in_epoch,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process():
        final_path = output_dir / "ckpt_final.pt"
        atomic_checkpoint(payload, final_path)
        print(f"[done] {final_path}", flush=True)
        if writer is not None:
            writer.close()
    finish_distributed()


if __name__ == "__main__":
    main()
