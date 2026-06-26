"""Augmentation configuration shared by skeleton sequence dataloaders."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True
    scale: bool = True
    flip: bool = True
    speed: bool = True
    scale_min: float = 0.5
    scale_max: float = 1.0
    flip_prob: float = 0.5
    speed_min_frames: int = 48
    speed_max_frames: int = 74
    swap_sides_on_flip: bool = True
    shift: bool = False
    shift_max: float = 0.03
    noise: bool = False
    noise_std: float = 0.005
    rotate: bool = False
    rotate_degrees: float = 0.0
    temporal_crop: bool = False
    temporal_crop_min_ratio: float = 0.85
    temporal_crop_max_ratio: float = 1.0
    time_mask: bool = False
    time_mask_prob: float = 0.25
    time_mask_max_ratio: float = 0.12
    joint_mask: bool = False
    joint_mask_prob: float = 0.20
    joint_mask_max_width: int = 8

    def __post_init__(self) -> None:
        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("Scale factors must be positive")
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min cannot be greater than scale_max")
        if self.speed_min_frames < 1 or self.speed_max_frames < 1:
            raise ValueError("Speed frame bounds must be positive")
        if self.speed_min_frames > self.speed_max_frames:
            raise ValueError("speed_min_frames cannot be greater than speed_max_frames")
        if self.shift_max < 0:
            raise ValueError("shift_max cannot be negative")
        if self.noise_std < 0:
            raise ValueError("noise_std cannot be negative")
        if self.rotate_degrees < 0:
            raise ValueError("rotate_degrees cannot be negative")
        if not 0.0 < self.temporal_crop_min_ratio <= 1.0:
            raise ValueError("temporal_crop_min_ratio must be in (0, 1]")
        if not 0.0 < self.temporal_crop_max_ratio <= 1.0:
            raise ValueError("temporal_crop_max_ratio must be in (0, 1]")
        if self.temporal_crop_min_ratio > self.temporal_crop_max_ratio:
            raise ValueError("temporal_crop_min_ratio cannot be greater than temporal_crop_max_ratio")
        if not 0.0 <= self.time_mask_prob <= 1.0:
            raise ValueError("time_mask_prob must be in [0, 1]")
        if self.time_mask_max_ratio < 0:
            raise ValueError("time_mask_max_ratio cannot be negative")
        if not 0.0 <= self.joint_mask_prob <= 1.0:
            raise ValueError("joint_mask_prob must be in [0, 1]")
        if self.joint_mask_max_width < 1:
            raise ValueError("joint_mask_max_width must be positive")


def infer_valid_temporal_length(sequence: np.ndarray) -> int:
    """Infer the non-padded temporal extent of a skeleton sequence."""
    if sequence.ndim != 3 or sequence.shape[2] < 2:
        raise ValueError(f"Expected skeleton shape (T, V, C>=2), got {sequence.shape}")
    valid_rows = np.any((sequence[:, :, 0] != 0.0) | (sequence[:, :, 1] != 0.0), axis=1)
    valid_indices = np.flatnonzero(valid_rows)
    if len(valid_indices) == 0:
        return int(sequence.shape[0])
    return int(valid_indices[-1]) + 1
