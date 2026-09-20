from __future__ import annotations

import math

from torch.optim import Optimizer


class WarmupCosineSchedule:
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float,
        peak_lr: float,
        total_steps: int,
        final_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.decay_steps = max(1, total_steps - warmup_steps)
        self.final_lr = final_lr
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            progress = self.step_count / max(1, self.warmup_steps)
            value = self.start_lr + progress * (self.peak_lr - self.start_lr)
        else:
            progress = min(1.0, (self.step_count - self.warmup_steps) / self.decay_steps)
            value = self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step_count = int(state.get("step_count", state.get("_step", 0)))


class CosineWeightDecaySchedule:
    def __init__(
        self,
        optimizer: Optimizer,
        start: float,
        final: float,
        total_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.start = start
        self.final = final
        self.total_steps = max(1, total_steps)
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        progress = min(1.0, self.step_count / self.total_steps)
        value = self.final + 0.5 * (self.start - self.final) * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in self.optimizer.param_groups:
            if not group.get("exclude_weight_decay", group.get("WD_exclude", False)):
                group["weight_decay"] = value
        return value

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step_count = int(state.get("step_count", state.get("_step", 0)))

