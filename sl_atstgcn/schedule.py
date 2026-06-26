"""Learning-rate schedulers used by the training scripts."""
from __future__ import annotations

import math
from typing import Protocol

import torch


class BatchScheduler(Protocol):
    def batch_step(self) -> float:
        ...


class CyclicLR:
    """Per-batch cyclical learning rate for PyTorch optimizers.

    The paper selects dataset-specific LR ranges by LR range tests; those ranges
    are provided in config files and are applied here. ``mode='triangular'``
    matches the original behavior. ``mode='triangular2'`` follows Smith's
    amplitude-halving variant and is useful for longer fine-tuning runs where
    repeated high-LR peaks destabilize validation accuracy.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int,
        *,
        mode: str = "triangular",
        gamma: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.max_lr = float(max_lr)
        self.step_size = int(max(1, step_size))
        self.mode = str(mode).lower()
        self.gamma = float(gamma)
        self.iterations = 0
        self.history: dict[str, list[float]] = {"lr": []}

    def clr(self) -> float:
        cycle = math.floor(1 + self.iterations / (2 * self.step_size))
        x = abs(self.iterations / self.step_size - 2 * cycle + 1)
        scale = max(0.0, 1.0 - x)
        if self.mode == "triangular":
            amplitude = scale
        elif self.mode == "triangular2":
            amplitude = scale / (2.0 ** (cycle - 1))
        elif self.mode == "exp_range":
            amplitude = scale * (self.gamma**self.iterations)
        else:
            raise ValueError(f"Unknown cyclic LR mode: {self.mode!r}")
        return self.base_lr + (self.max_lr - self.base_lr) * amplitude

    def batch_step(self) -> float:
        lr = self.clr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        self.history["lr"].append(lr)
        self.iterations += 1
        return lr


class CosineWarmupLR:
    """Per-batch warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(max(0, warmup_steps))
        self.total_steps = int(max(1, total_steps))
        self.iterations = 0
        self.history: dict[str, list[float]] = {"lr": []}

    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            pct = step / max(1, self.warmup_steps)
            return self.base_lr + (self.max_lr - self.base_lr) * pct
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        pct = min(1.0, max(0.0, (step - self.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * pct))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def batch_step(self) -> float:
        lr = self.lr_at(self.iterations)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        self.history["lr"].append(lr)
        self.iterations += 1
        return lr


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    *,
    steps_per_epoch: int,
) -> BatchScheduler:
    """Build the configured per-batch learning-rate scheduler."""
    scheduler_name = str(cfg.get("scheduler", "cyclic")).lower()
    steps_per_epoch = int(max(1, steps_per_epoch))
    if scheduler_name in {"cyclic", "clr"}:
        return CyclicLR(
            optimizer,
            base_lr=float(cfg["base_lr"]),
            max_lr=float(cfg["max_lr"]),
            step_size=int(cfg.get("clr_step_size", steps_per_epoch * int(cfg.get("clr_half_cycle_epochs", 2)))),
            mode=str(cfg.get("clr_mode", cfg.get("cyclic_mode", "triangular"))),
            gamma=float(cfg.get("clr_gamma", 1.0)),
        )
    if scheduler_name in {"cosine", "cosine_warmup"}:
        total_steps = steps_per_epoch * int(cfg["epochs"])
        warmup_steps = steps_per_epoch * int(cfg.get("warmup_epochs", 5))
        return CosineWarmupLR(
            optimizer,
            base_lr=float(cfg.get("base_lr", cfg["max_lr"])),
            max_lr=float(cfg["max_lr"]),
            min_lr=float(cfg.get("min_lr", cfg.get("final_lr", 1.0e-4))),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name!r}")
