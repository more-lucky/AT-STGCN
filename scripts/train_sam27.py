#!/usr/bin/env python
"""Train a standalone SAM-SLR-style 27-keypoint skeleton model.

This script intentionally does not modify or depend on the project's 68-point
model classes. It reads 27-point or 68-point skeleton manifests and maps each
sample to the SAM-SLR 27-node layout on the fly:

    body: nose, left/right shoulders, left/right elbows, left/right wrists
    hands: per hand [wrist, thumb_tip, index_mcp, index_tip, middle_mcp,
            middle_tip, ring_mcp, ring_tip, pinky_mcp, pinky_tip]

The original SAM-SLR data contains separate upper-body wrist points. The current
project's 68-point format does not, so the upper-body wrist nodes are filled
from the corresponding hand wrist nodes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sl_atstgcn.data import attach_label_ids, read_manifest
from sl_atstgcn.keypoints import JOINT_INDEX


SAM27_HAND_OFFSETS = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
SAM27_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist_from_hand",
    "right_wrist_from_hand",
    "left_hand_wrist",
    "left_hand_thumb_tip",
    "left_hand_index_finger_mcp",
    "left_hand_index_finger_tip",
    "left_hand_middle_finger_mcp",
    "left_hand_middle_finger_tip",
    "left_hand_ring_finger_mcp",
    "left_hand_ring_finger_tip",
    "left_hand_pinky_mcp",
    "left_hand_pinky_tip",
    "right_hand_wrist",
    "right_hand_thumb_tip",
    "right_hand_index_finger_mcp",
    "right_hand_index_finger_tip",
    "right_hand_middle_finger_mcp",
    "right_hand_middle_finger_tip",
    "right_hand_ring_finger_mcp",
    "right_hand_ring_finger_tip",
    "right_hand_pinky_mcp",
    "right_hand_pinky_tip",
)


def _hand_indices(prefix: str) -> list[int]:
    base_names = (
        "wrist",
        "thumb_tip",
        "index_finger_mcp",
        "index_finger_tip",
        "middle_finger_mcp",
        "middle_finger_tip",
        "ring_finger_mcp",
        "ring_finger_tip",
        "pinky_mcp",
        "pinky_tip",
    )
    return [JOINT_INDEX[f"{prefix}_{name}"] for name in base_names]


SAM27_FROM_68 = np.asarray(
    [
        JOINT_INDEX["nose"],
        JOINT_INDEX["left_shoulder"],
        JOINT_INDEX["right_shoulder"],
        JOINT_INDEX["left_elbow"],
        JOINT_INDEX["right_elbow"],
        JOINT_INDEX["left_hand_wrist"],
        JOINT_INDEX["right_hand_wrist"],
        *_hand_indices("left_hand"),
        *_hand_indices("right_hand"),
    ],
    dtype=np.int64,
)

SAM27_MIRROR = np.asarray(
    [
        0,
        2,
        1,
        4,
        3,
        6,
        5,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ],
    dtype=np.int64,
)

# Parent -> child edges from SAM-SLR's sign_27 graph after its index shift.
SAM27_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (7, 8),
    (7, 9),
    (7, 11),
    (7, 13),
    (7, 15),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (17, 19),
    (17, 21),
    (17, 23),
    (17, 25),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (5, 7),
    (6, 17),
)

SAM27_PARENT = np.arange(27, dtype=np.int64)
for parent, child in SAM27_EDGES:
    SAM27_PARENT[child] = parent

SAM27_PARTS = {
    "body": tuple(range(0, 7)),
    "left_hand": tuple(range(7, 17)),
    "right_hand": tuple(range(17, 27)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config for SAM-style 27-point training.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return dict(cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict) -> torch.device:
    requested = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU")
        requested = "cpu"
    return torch.device(requested)


def parse_int_sequence(value, default=(1,)) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = [value]
    parsed = tuple(max(1, int(item)) for item in items)
    return parsed or tuple(default)


def _as_sequence_tvc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {arr.shape}")
    if arr.shape[1] in {27, 68}:
        return arr
    if arr.shape[0] in {2, 3} and arr.shape[2] in {27, 68}:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Cannot interpret skeleton shape {arr.shape}")


def to_sam27_sequence(arr: np.ndarray) -> np.ndarray:
    seq = _as_sequence_tvc(arr)
    if seq.shape[1] == 27:
        compact = seq[:, :, : min(3, seq.shape[2])]
    elif seq.shape[1] == 68:
        compact = seq[:, SAM27_FROM_68, : min(3, seq.shape[2])]
    else:
        raise ValueError(f"Expected skeleton width 27 or 68; got {seq.shape}")
    if compact.shape[2] == 2:
        zeros = np.zeros((*compact.shape[:2], 1), dtype=np.float32)
        compact = np.concatenate([compact, zeros], axis=2)
    return np.ascontiguousarray(compact[:, :, :3], dtype=np.float32)


def infer_valid_length(sequence: np.ndarray) -> int:
    valid_rows = np.any((sequence[:, :, 0] != 0.0) | (sequence[:, :, 1] != 0.0), axis=1)
    valid = np.flatnonzero(valid_rows)
    return int(valid[-1]) + 1 if len(valid) else int(sequence.shape[0])


def resize_or_pad(sequence: np.ndarray, length: int) -> np.ndarray:
    target = int(length)
    if target <= 0:
        raise ValueError(f"sequence_length must be positive, got {length}")
    if sequence.shape[0] == target:
        return sequence.astype(np.float32, copy=False)
    if sequence.shape[0] <= 0:
        return np.zeros((target, 27, 3), dtype=np.float32)
    if sequence.shape[0] > target:
        indices = np.linspace(0, sequence.shape[0] - 1, target).round().astype(np.int64)
        return sequence[indices].astype(np.float32, copy=False)
    out = np.zeros((target, 27, 3), dtype=np.float32)
    out[: sequence.shape[0]] = sequence
    return out


def resample_sequence(sequence: np.ndarray, length: int) -> np.ndarray:
    target = int(max(1, length))
    if sequence.shape[0] == target:
        return sequence.astype(np.float32, copy=True)
    if sequence.shape[0] <= 0:
        return np.zeros((target, 27, 3), dtype=np.float32)
    source_x = np.arange(sequence.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, float(sequence.shape[0] - 1), target, dtype=np.float32)
    out = np.zeros((target, 27, 3), dtype=np.float32)
    for joint in range(27):
        for channel in range(sequence.shape[2]):
            out[:, joint, channel] = np.interp(target_x, source_x, sequence[:, joint, channel]).astype(np.float32)
    return out


def repair_missing(sequence: np.ndarray, min_valid: int = 2) -> np.ndarray:
    out = sequence.astype(np.float32, copy=True)
    valid_t = infer_valid_length(out)
    frame_index = np.arange(valid_t, dtype=np.float32)
    valid = (out[:valid_t, :, 0] != 0.0) | (out[:valid_t, :, 1] != 0.0)
    min_valid = max(1, int(min_valid))
    for joint in range(27):
        mask = valid[:, joint]
        if int(mask.sum()) < min_valid or bool(mask.all()):
            continue
        source_x = frame_index[mask]
        for channel in (0, 1):
            source_y = out[:valid_t, joint, channel][mask]
            out[:valid_t, joint, channel] = np.interp(frame_index, source_x, source_y).astype(np.float32)
    return out


@dataclass(frozen=True)
class Augment:
    enabled: bool = False
    scale: bool = False
    shift: bool = False
    noise: bool = False
    rotate: bool = False
    temporal_crop: bool = False
    flip: bool = False
    speed: bool = False
    time_mask: bool = False
    joint_mask: bool = False
    scale_min: float = 0.9
    scale_max: float = 1.1
    shift_max: float = 0.02
    noise_std: float = 0.003
    rotate_degrees: float = 5.0
    temporal_crop_min_ratio: float = 0.9
    temporal_crop_max_ratio: float = 1.0
    flip_prob: float = 0.5
    speed_min_frames: int = 56
    speed_max_frames: int = 72
    time_mask_prob: float = 0.2
    time_mask_max_ratio: float = 0.1
    joint_mask_prob: float = 0.2
    joint_mask_max_width: int = 4

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Augment":
        if not isinstance(cfg, dict):
            return cls(enabled=False)
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in cfg.items() if key in fields}
        return cls(**values)


def apply_affine_xy(out: np.ndarray, matrix: np.ndarray) -> None:
    valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    xy = out[:, :, :2]
    centered = xy - 0.5
    transformed = np.einsum("ij,tvj->tvi", matrix.astype(np.float32), centered) + 0.5
    out[:, :, :2] = np.where(valid[:, :, None], transformed, xy)


def augment_sequence(sequence: np.ndarray, cfg: Augment, rng: np.random.Generator) -> np.ndarray:
    out = sequence.astype(np.float32, copy=True)
    target_len = int(out.shape[0])
    valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    if cfg.flip and rng.random() < float(cfg.flip_prob):
        out[:, :, 0] = np.where(valid, 1.0 - out[:, :, 0], out[:, :, 0])
        out = out[:, SAM27_MIRROR, :]
        valid = valid[:, SAM27_MIRROR]
    if cfg.scale:
        factor = float(rng.uniform(float(cfg.scale_min), float(cfg.scale_max)))
        matrix = np.asarray([[factor, 0.0], [0.0, factor]], dtype=np.float32)
        apply_affine_xy(out, matrix)
    if cfg.rotate and abs(float(cfg.rotate_degrees)) > 0:
        angle = math.radians(float(rng.uniform(-float(cfg.rotate_degrees), float(cfg.rotate_degrees))))
        c, s = math.cos(angle), math.sin(angle)
        matrix = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        apply_affine_xy(out, matrix)
    if cfg.shift and float(cfg.shift_max) > 0:
        delta = rng.uniform(-float(cfg.shift_max), float(cfg.shift_max), size=(1, 1, 2)).astype(np.float32)
        out[:, :, :2] = np.where(valid[:, :, None], out[:, :, :2] + delta, out[:, :, :2])
    if cfg.noise and float(cfg.noise_std) > 0:
        noise = rng.normal(0.0, float(cfg.noise_std), size=out[:, :, :2].shape).astype(np.float32)
        out[:, :, :2] = np.where(valid[:, :, None], out[:, :, :2] + noise, out[:, :, :2])
    if cfg.temporal_crop:
        valid_t = infer_valid_length(out)
        if valid_t > 2:
            ratio = float(rng.uniform(float(cfg.temporal_crop_min_ratio), float(cfg.temporal_crop_max_ratio)))
            crop_t = max(2, min(valid_t, int(round(valid_t * ratio))))
            if crop_t < valid_t:
                start = int(rng.integers(0, valid_t - crop_t + 1))
                out = resize_or_pad(out[start : start + crop_t], target_len)
                valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    if cfg.speed:
        valid_t = infer_valid_length(out)
        if valid_t > 2:
            low = max(2, int(cfg.speed_min_frames))
            high = max(low, int(cfg.speed_max_frames))
            new_t = int(rng.integers(low, high + 1))
            resized = resample_sequence(out[:valid_t], new_t)
            out = resize_or_pad(resized, target_len)
            valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    if cfg.time_mask and rng.random() < float(cfg.time_mask_prob):
        valid_t = infer_valid_length(out)
        width = max(1, int(round(valid_t * float(cfg.time_mask_max_ratio))))
        start = int(rng.integers(0, max(1, valid_t - width + 1)))
        out[start : start + width] = 0.0
    if cfg.joint_mask and rng.random() < float(cfg.joint_mask_prob):
        width = max(1, min(int(cfg.joint_mask_max_width), 27))
        start = int(rng.integers(0, 27 - width + 1))
        out[:, start : start + width] = 0.0
    return out.astype(np.float32, copy=False)


class SAM27Dataset(Dataset):
    def __init__(
        self,
        df,
        *,
        sequence_length: int,
        training: bool,
        augment: Augment,
        seed: int,
        repair: bool,
        repair_min_valid: int,
    ) -> None:
        path_column = "keypoints_path" if "keypoints_path" in df.columns else "path"
        self.paths = df[path_column].astype(str).tolist()
        self.labels = df["label_id"].astype(np.int64).to_numpy()
        self.sequence_length = int(sequence_length)
        self.training = bool(training)
        self.augment = augment
        self.seed = int(seed)
        self.repair = bool(repair)
        self.repair_min_valid = int(repair_min_valid)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        sequence = to_sam27_sequence(np.load(self.paths[index]))
        sequence = resize_or_pad(sequence, self.sequence_length)
        if self.repair:
            sequence = repair_missing(sequence, min_valid=self.repair_min_valid)
        if self.training and self.augment.enabled:
            worker_seed = int(torch.initial_seed() % (2**32 - 1))
            draw_seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
            rng_seed = (worker_seed ^ draw_seed ^ (self.seed + int(index) * 7919)) % (2**32 - 1)
            sequence = augment_sequence(sequence, self.augment, np.random.default_rng(rng_seed))
        tensor = torch.from_numpy(np.ascontiguousarray(sequence.transpose(2, 0, 1), dtype=np.float32))
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


def make_loader(df, cfg: dict, *, training: bool, seed: int) -> DataLoader:
    dataset = SAM27Dataset(
        df,
        sequence_length=int(cfg.get("sequence_length", 64)),
        training=training,
        augment=Augment.from_config(cfg.get("augmentation", {})),
        seed=seed,
        repair=bool(cfg.get("repair_missing_keypoints", True)),
        repair_min_valid=int(cfg.get("repair_min_valid", 2)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = bool(training)
    if training and bool(cfg.get("balanced_sampling", False)):
        counts = np.maximum(np.bincount(dataset.labels), 1)
        weights = 1.0 / counts[dataset.labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", torch.cuda.is_available())),
        generator=generator if training and sampler is None else None,
    )


def build_adjacency(*, hops: int = 2) -> torch.Tensor:
    num_joints = 27
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    for parent, child in SAM27_EDGES:
        adjacency[parent, child] = 1.0
        adjacency[child, parent] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    norm = degree.rsqrt()
    adjacency = norm[:, None] * adjacency * norm[None, :]
    matrices = [torch.eye(num_joints, dtype=torch.float32)]
    current = adjacency
    for _ in range(max(1, int(hops))):
        matrices.append(current)
        current = torch.matmul(current, adjacency).clamp(0.0, 1.0)
    return torch.stack(matrices, dim=0)


class GraphConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        adjacency_hops: int,
        adaptive: bool,
        edge_importance: bool,
        adaptive_scale: float,
    ) -> None:
        super().__init__()
        adjacency = build_adjacency(hops=adjacency_hops)
        self.register_buffer("adjacency", adjacency)
        self.adaptive_scale = float(adaptive_scale)
        self.adaptive = nn.Parameter(torch.zeros_like(adjacency)) if adaptive else None
        self.edge_importance = nn.Parameter(torch.ones_like(adjacency)) if edge_importance else None
        self.proj = nn.Conv2d(in_channels * adjacency.size(0), out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.adjacency
        if self.adaptive is not None:
            adjacency = adjacency + torch.tanh(self.adaptive) * self.adaptive_scale
        if self.edge_importance is not None:
            adjacency = adjacency * self.edge_importance
        supports = [torch.einsum("nctv,vw->nctw", x, adj) for adj in adjacency]
        return self.proj(torch.cat(supports, dim=1))


class TemporalConv(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int, dilations: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.pre = nn.Sequential(nn.BatchNorm2d(channels), nn.GELU())
        branches = []
        for dilation in dilations:
            dilation = int(max(1, dilation))
            padding = (int(kernel_size) // 2) * dilation
            branches.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(int(kernel_size), 1),
                    padding=(padding, 0),
                    dilation=(dilation, 1),
                    bias=False,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Conv2d(channels * len(branches), channels, kernel_size=1, bias=False) if len(branches) > 1 else nn.Identity()
        self.post = nn.Sequential(nn.BatchNorm2d(channels), nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.post(self.fuse(x))


class GraphBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        adjacency_hops: int,
        temporal_kernel: int,
        temporal_dilations: tuple[int, ...],
        adaptive_graph: bool,
        edge_importance: bool,
        adaptive_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gcn = GraphConv(
            in_channels,
            out_channels,
            adjacency_hops=adjacency_hops,
            adaptive=adaptive_graph,
            edge_importance=edge_importance,
            adaptive_scale=adaptive_scale,
        )
        self.tcn = TemporalConv(
            out_channels,
            kernel_size=temporal_kernel,
            dilations=temporal_dilations,
            dropout=dropout,
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.tcn(self.gcn(x)) + self.residual(x))


class STCAttention(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 16, temporal_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        padding = int(temporal_kernel) // 2
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.temporal = nn.Conv2d(1, 1, kernel_size=(temporal_kernel, 1), padding=(padding, 0), bias=False)
        self.spatial = nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(self.channel(x))
        summary = x.mean(dim=1, keepdim=True)
        x = x * torch.sigmoid(self.temporal(summary))
        x = x * torch.sigmoid(self.spatial(summary))
        return x


class AttentionPool(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, int(channels) // int(reduction))
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        weights = torch.softmax(self.score(x).flatten(2), dim=-1).view(n, 1, t, v)
        return torch.sum(x * weights, dim=(2, 3), keepdim=True)


class SAM27Encoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_channels: int,
        blocks: int,
        adjacency_hops: int,
        dropout: float,
        temporal_kernel: int,
        temporal_dilations: tuple[int, ...],
        adaptive_graph: bool,
        edge_importance: bool,
        adaptive_scale: float,
        stc_attention: bool,
        center_joints: bool,
        scale_normalize: bool,
        hand_weight: float,
        include_absolute_xy: bool,
        include_validity: bool,
        include_temporal_position: bool,
        include_root_motion: bool,
        include_acceleration: bool,
        use_bone_features: bool,
        use_motion_features: bool,
        pooling: str,
        part_pooling: bool,
        part_pooling_scale: float,
    ) -> None:
        super().__init__()
        self.center_joints = bool(center_joints)
        self.scale_normalize = bool(scale_normalize)
        self.include_absolute_xy = bool(include_absolute_xy)
        self.include_validity = bool(include_validity)
        self.include_temporal_position = bool(include_temporal_position)
        self.include_root_motion = bool(include_root_motion)
        self.include_acceleration = bool(include_acceleration)
        self.use_bone_features = bool(use_bone_features)
        self.use_motion_features = bool(use_motion_features)
        self.pooling = str(pooling).strip().lower()
        if self.pooling not in {"avg", "avgmax", "avgattn", "avgmaxattn"}:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        self.use_attention_pooling = self.pooling in {"avgattn", "avgmaxattn"}
        self.part_pooling = bool(part_pooling)
        self.part_pooling_scale = float(part_pooling_scale)
        self.register_buffer("parent_indices", torch.as_tensor(SAM27_PARENT, dtype=torch.long), persistent=False)
        joint_weights = torch.ones(27, dtype=torch.float32)
        joint_weights[7:] = float(hand_weight)
        self.register_buffer("joint_weights", joint_weights, persistent=False)
        self.part_names = tuple(SAM27_PARTS)
        for name, indices in SAM27_PARTS.items():
            self.register_buffer(f"part_{name}", torch.as_tensor(indices, dtype=torch.long), persistent=False)

        input_channels = 2
        if self.use_bone_features:
            input_channels += 2
        if self.use_motion_features:
            input_channels += 2
            if self.use_bone_features:
                input_channels += 2
        if self.include_acceleration:
            input_channels += 2
            if self.use_bone_features:
                input_channels += 2
        if self.include_absolute_xy:
            input_channels += 2
        if self.include_root_motion:
            input_channels += 2
        if self.include_validity:
            input_channels += 1
        if self.include_temporal_position:
            input_channels += 1

        self.input_bn = nn.BatchNorm2d(input_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, int(blocks))):
            layers.append(
                GraphBlock(
                    in_channels,
                    int(hidden_channels),
                    adjacency_hops=int(adjacency_hops),
                    temporal_kernel=int(temporal_kernel),
                    temporal_dilations=tuple(temporal_dilations),
                    adaptive_graph=bool(adaptive_graph),
                    edge_importance=bool(edge_importance),
                    adaptive_scale=float(adaptive_scale),
                    dropout=float(dropout),
                )
            )
            in_channels = int(hidden_channels)
        if stc_attention:
            layers.append(STCAttention(int(hidden_channels), reduction=16, temporal_kernel=7))
        self.blocks = nn.Sequential(*layers)
        self.attention_pool = AttentionPool(int(hidden_channels)) if self.use_attention_pooling else None
        pool_multiplier = 1
        if self.pooling in {"avgmax", "avgmaxattn"}:
            pool_multiplier += 1
        if self.use_attention_pooling:
            pool_multiplier += 1
        region_count = 1 + (len(SAM27_PARTS) if self.part_pooling else 0)
        self.out_channels = int(hidden_channels) * pool_multiplier * region_count

    @staticmethod
    def _temporal_delta(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(values)
        if values.size(2) > 1:
            both_valid = valid[:, 1:] & valid[:, :-1]
            delta[:, :, 1:] = torch.where(
                both_valid[:, None],
                values[:, :, 1:] - values[:, :, :-1],
                torch.zeros_like(values[:, :, 1:]),
            )
        return delta

    def _joint_xy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(3) != 27:
            raise ValueError(f"Expected SAM27 tensor (N,C,T,27), got {tuple(x.shape)}")
        raw_xy = x[:, :2]
        valid = (raw_xy[:, 0] != 0.0) | (raw_xy[:, 1] != 0.0)
        shoulder_mid = (raw_xy[:, :, :, 1:2] + raw_xy[:, :, :, 2:3]) * 0.5
        joints = raw_xy
        if self.center_joints:
            joints = torch.where(valid[:, None], joints - shoulder_mid, torch.zeros_like(joints))
        if self.scale_normalize:
            shoulder_distance = torch.linalg.vector_norm(raw_xy[:, :, :, 1] - raw_xy[:, :, :, 2], dim=1)
            shoulder_distance = shoulder_distance.mean(dim=1).clamp_min(0.05)
            joints = joints / shoulder_distance[:, None, None, None]
        return joints, valid, raw_xy, shoulder_mid

    def _pool_region(self, features: torch.Tensor) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool2d(features, (1, 1))]
        if self.pooling in {"avgmax", "avgmaxattn"}:
            pooled.append(F.adaptive_max_pool2d(features, (1, 1)))
        if self.attention_pool is not None:
            pooled.append(self.attention_pool(features))
        return torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joints, valid, raw_xy, shoulder_mid = self._joint_xy(x)
        feature_parts = [joints]
        bones = None
        bone_valid = None
        if self.use_bone_features:
            parent_joints = joints.index_select(dim=3, index=self.parent_indices)
            parent_valid = valid.index_select(dim=2, index=self.parent_indices)
            bone_valid = valid & parent_valid
            bones = torch.where(bone_valid[:, None], joints - parent_joints, torch.zeros_like(joints))
            feature_parts.append(bones)
        joint_motion = None
        bone_motion = None
        if self.use_motion_features:
            joint_motion = self._temporal_delta(joints, valid)
            feature_parts.append(joint_motion)
            if self.use_bone_features:
                bone_motion = self._temporal_delta(bones, bone_valid)
                feature_parts.append(bone_motion)
        if self.include_acceleration:
            if joint_motion is None:
                joint_motion = self._temporal_delta(joints, valid)
            feature_parts.append(self._temporal_delta(joint_motion, valid))
            if self.use_bone_features:
                if bone_motion is None:
                    bone_motion = self._temporal_delta(bones, bone_valid)
                feature_parts.append(self._temporal_delta(bone_motion, bone_valid))
        if self.include_absolute_xy:
            feature_parts.append(torch.where(valid[:, None], raw_xy.mul(2.0).sub(1.0), torch.zeros_like(raw_xy)))
        if self.include_root_motion:
            root_valid = valid[:, :, 1:2] & valid[:, :, 2:3]
            root_motion = self._temporal_delta(shoulder_mid, root_valid).expand(-1, -1, -1, 27)
            feature_parts.append(root_motion)
        if self.include_validity:
            feature_parts.append(valid[:, None].to(dtype=joints.dtype))
        if self.include_temporal_position:
            positions = torch.linspace(-1.0, 1.0, x.size(2), device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
            positions = positions.expand(x.size(0), 1, x.size(2), 27)
            feature_parts.append(torch.where(valid[:, None], positions, torch.zeros_like(positions)))
        features = torch.cat(feature_parts, dim=1)
        features = features * self.joint_weights.view(1, 1, 1, -1)
        features = self.blocks(self.input_bn(features))
        pooled = [self._pool_region(features)]
        if self.part_pooling:
            for name in self.part_names:
                indices = getattr(self, f"part_{name}")
                pooled.append(self._pool_region(features.index_select(dim=3, index=indices)) * self.part_pooling_scale)
        return torch.flatten(torch.cat(pooled, dim=1), 1)


def cosine_margin_logits(
    features: torch.Tensor,
    weight: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    classifier_type: str,
    training: bool,
    margin: float,
    scale: float,
) -> torch.Tensor:
    cosine = F.linear(F.normalize(features, dim=1), F.normalize(weight, dim=1)).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    resolved = str(classifier_type).strip().lower()
    if labels is not None and training and float(margin) > 0.0 and resolved in {"cosface", "arcface"}:
        labels = labels.view(-1, 1)
        target_cosine = cosine.gather(1, labels)
        if resolved == "arcface":
            sine = torch.sqrt((1.0 - target_cosine.square()).clamp_min(1.0e-7))
            phi = target_cosine * math.cos(margin) - sine * math.sin(margin)
            threshold = math.cos(math.pi - margin)
            correction = math.sin(math.pi - margin) * margin
            target_logits = torch.where(target_cosine > threshold, phi, target_cosine - correction)
        else:
            target_logits = target_cosine - float(margin)
        cosine = cosine.scatter(1, labels, target_logits)
    return cosine * float(scale)


class SAM27Classifier(nn.Module):
    def __init__(self, *, num_classes: int, cfg: dict) -> None:
        super().__init__()
        self.encoder = SAM27Encoder(
            hidden_channels=int(cfg.get("skeleton_hidden_channels", 192)),
            blocks=int(cfg.get("skeleton_blocks", 4)),
            adjacency_hops=int(cfg.get("skeleton_adjacency_hops", 2)),
            dropout=float(cfg.get("skeleton_dropout", 0.18)),
            temporal_kernel=int(cfg.get("skeleton_temporal_kernel", 7)),
            temporal_dilations=parse_int_sequence(cfg.get("skeleton_temporal_dilations", (1, 2, 3))),
            adaptive_graph=bool(cfg.get("skeleton_adaptive_graph", True)),
            edge_importance=bool(cfg.get("skeleton_edge_importance", True)),
            adaptive_scale=float(cfg.get("skeleton_adaptive_scale", 0.08)),
            stc_attention=bool(cfg.get("skeleton_stc_attention", True)),
            center_joints=bool(cfg.get("skeleton_center_joints", True)),
            scale_normalize=bool(cfg.get("skeleton_scale_normalize", True)),
            hand_weight=float(cfg.get("skeleton_hand_weight", 1.25)),
            include_absolute_xy=bool(cfg.get("skeleton_include_absolute_xy", True)),
            include_validity=bool(cfg.get("skeleton_include_validity", True)),
            include_temporal_position=bool(cfg.get("skeleton_include_temporal_position", True)),
            include_root_motion=bool(cfg.get("skeleton_include_root_motion", True)),
            include_acceleration=bool(cfg.get("skeleton_include_acceleration", True)),
            use_bone_features=bool(cfg.get("skeleton_use_bone_features", True)),
            use_motion_features=bool(cfg.get("skeleton_use_motion_features", True)),
            pooling=str(cfg.get("skeleton_pooling", "avgmaxattn")),
            part_pooling=bool(cfg.get("skeleton_part_pooling", True)),
            part_pooling_scale=float(cfg.get("skeleton_part_pooling_scale", 1.0)),
        )
        self.feature_norm = nn.LayerNorm(self.encoder.out_channels) if bool(cfg.get("feature_layer_norm", True)) else nn.Identity()
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.5)))
        self.classifier_type = str(cfg.get("classifier_type", "arcface")).strip().lower()
        self.logit_scale = float(cfg.get("logit_scale", 32.0))
        self.classifier_margin = float(cfg.get("classifier_margin", 0.18))
        self.classifier = nn.Linear(
            self.encoder.out_channels,
            int(num_classes),
            bias=self.classifier_type == "linear",
        )
        center_cfg = cfg.get("center_loss", {})
        self.center_loss_weight = float(center_cfg.get("weight", 0.0)) if bool(center_cfg.get("enabled", False)) else 0.0
        if self.center_loss_weight > 0:
            self.class_centers = nn.Parameter(torch.empty(int(num_classes), self.encoder.out_channels))
            nn.init.normal_(self.class_centers, std=0.02)
        else:
            self.register_parameter("class_centers", None)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(self.encoder(x))

    def center_clustering_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.class_centers is None or self.center_loss_weight <= 0:
            return features.new_zeros(())
        features = F.normalize(features, dim=1)
        centers = F.normalize(self.class_centers, dim=1)
        target_centers = centers.index_select(0, labels)
        return (1.0 - torch.sum(features * target_centers, dim=1)).mean() * self.center_loss_weight

    def classify_features(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.classifier_type in {"cosine", "cosface", "arcface"}:
            return cosine_margin_logits(
                x,
                self.classifier.weight,
                labels=labels,
                classifier_type=self.classifier_type,
                training=self.training,
                margin=self.classifier_margin,
                scale=self.logit_scale,
            )
        return self.classifier(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, *, return_features: bool = False):
        features = self.extract_features(x)
        logits = self.classify_features(self.dropout(features), labels=labels)
        return (logits, features) if return_features else logits


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            model_value = state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def smooth_one_hot(labels: torch.Tensor, *, num_classes: int, smoothing: float) -> torch.Tensor:
    smoothing = float(smoothing)
    off = smoothing / max(1, int(num_classes) - 1)
    on = 1.0 - smoothing
    target = torch.full((labels.size(0), int(num_classes)), off, device=labels.device, dtype=torch.float32)
    target.scatter_(1, labels.view(-1, 1), on)
    return target


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sum(-targets * torch.log_softmax(logits, dim=1), dim=1).mean()


def maybe_mixup(inputs: torch.Tensor, targets: torch.Tensor, *, alpha: float, prob: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if alpha <= 0 or prob <= 0 or inputs.size(0) < 2 or torch.rand((), device=inputs.device).item() > prob:
        return inputs, targets, False
    lam = torch.distributions.Beta(float(alpha), float(alpha)).sample().to(inputs.device)
    lam = torch.maximum(lam, 1.0 - lam)
    order = torch.randperm(inputs.size(0), device=inputs.device)
    return inputs.mul(lam).add(inputs[order], alpha=float(1.0 - lam)), targets.mul(lam).add(targets[order], alpha=float(1.0 - lam)), True


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    k = max(1, min(int(k), int(logits.size(1))))
    pred = logits.topk(k=k, dim=1).indices
    return int(pred.eq(labels.view(-1, 1)).any(dim=1).sum().item())


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def epoch_lr(cfg: dict, epoch: int) -> float:
    max_lr = float(cfg.get("max_lr", cfg.get("base_lr", 1.0e-3)))
    min_lr = float(cfg.get("min_lr", 1.0e-5))
    warmup = int(cfg.get("warmup_epochs", 0))
    epochs = max(1, int(cfg.get("epochs", 100)))
    if warmup > 0 and epoch <= warmup:
        base_lr = float(cfg.get("base_lr", max_lr * 0.1))
        return base_lr + (max_lr - base_lr) * (epoch / max(1, warmup))
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def create_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("max_lr", cfg.get("base_lr", 1.0e-3)))
    weight_decay = float(cfg.get("weight_decay", 1.0e-4))
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(cfg.get("momentum", 0.9)), nesterov=True, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: SAM27Classifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    cfg: dict,
    num_classes: int,
    epoch: int,
    ema: ModelEMA | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_seen = 0
    mixup_cfg = cfg.get("mixup", {})
    mixup_active = bool(mixup_cfg.get("enabled", False)) and int(mixup_cfg.get("start_epoch", 1)) <= epoch <= int(mixup_cfg.get("end_epoch", 10**9))
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = smooth_one_hot(labels, num_classes=num_classes, smoothing=float(cfg.get("label_smoothing", 0.0)))
        mixed = False
        if mixup_active:
            inputs, targets, mixed = maybe_mixup(
                inputs,
                targets,
                alpha=float(mixup_cfg.get("alpha", 0.2)),
                prob=float(mixup_cfg.get("prob", 1.0)),
            )
        optimizer.zero_grad(set_to_none=True)
        margin_labels = None if mixed else labels
        need_features = bool(model.center_loss_weight > 0 and not mixed)
        if need_features:
            logits, features = model(inputs, labels=margin_labels, return_features=True)
        else:
            logits = model(inputs, labels=margin_labels)
            features = None
        loss = soft_cross_entropy(logits, targets)
        if need_features and features is not None:
            loss = loss + model.center_clustering_loss(features, labels)
        loss.backward()
        grad_clip = float(cfg.get("grad_clip_norm", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_top1 += topk_correct(logits.detach(), labels, 1)
        total_top5 += topk_correct(logits.detach(), labels, 5)
        total_seen += batch_size
    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "top5": total_top5 / max(1, total_seen),
    }


def tta_inputs(inputs: torch.Tensor, *, flip: bool, scale: float) -> torch.Tensor:
    out = inputs.clone()
    valid = (out[:, 0] != 0.0) | (out[:, 1] != 0.0)
    if abs(float(scale) - 1.0) > 1.0e-6:
        out[:, :2] = torch.where(valid[:, None], (out[:, :2] - 0.5) * float(scale) + 0.5, out[:, :2])
    if flip:
        out[:, 0] = torch.where(valid, 1.0 - out[:, 0], out[:, 0])
        mirror = torch.as_tensor(SAM27_MIRROR, device=out.device, dtype=torch.long)
        out = out.index_select(dim=3, index=mirror)
    return out


@torch.no_grad()
def evaluate(model: SAM27Classifier, loader: DataLoader, *, device: torch.device, cfg: dict) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_seen = 0
    tta_cfg = cfg.get("eval_tta", {})
    flips = [False, True] if bool(tta_cfg.get("flip", False)) else [False]
    scales = [float(x) for x in tta_cfg.get("scales", [1.0])]
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits_sum = None
        for scale in scales:
            for flip in flips:
                logits = model(tta_inputs(inputs, flip=flip, scale=scale), labels=None)
                logits_sum = logits if logits_sum is None else logits_sum + logits
        logits = logits_sum / float(len(flips) * len(scales))
        targets = smooth_one_hot(labels, num_classes=logits.size(1), smoothing=0.0)
        loss = soft_cross_entropy(logits, targets)
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_top1 += topk_correct(logits, labels, 1)
        total_top5 += topk_correct(logits, labels, 5)
        total_seen += batch_size
    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "top5": total_top5 / max(1, total_seen),
    }


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, label_map: dict, metrics: dict) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": cfg,
        "label_map": label_map,
        "metrics": metrics,
        "sam27_names": SAM27_NAMES,
        "sam27_from_68": SAM27_FROM_68.tolist(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(cfg)
    out_dir = Path(cfg.get("output_dir", "runs/sam27"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    df = read_manifest(cfg["manifest"])
    df, label_map = attach_label_ids(df)
    train_split = str(cfg.get("train_split", "train"))
    val_split = str(cfg.get("val_split", "val"))
    train_df = df[df["split"].astype(str) == train_split].copy()
    val_df = df[df["split"].astype(str).isin([val_split, "validation"] if val_split == "val" else [val_split])].copy()
    if train_df.empty or val_df.empty:
        raise ValueError(f"Need non-empty train/val splits, got train={len(train_df)} val={len(val_df)}")
    num_classes = len(label_map.id_to_label)
    label_payload = {str(idx): label for idx, label in enumerate(label_map.id_to_label)}
    with (out_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_payload, f, ensure_ascii=False, indent=2)
    with (out_dir / "sam27_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "names": SAM27_NAMES,
                "indices_from_project_68": SAM27_FROM_68.tolist(),
                "mirror_indices": SAM27_MIRROR.tolist(),
                "edges_parent_child": list(SAM27_EDGES),
                "note": "Upper-body wrist nodes use hand wrist coordinates because the project 68-point format has no separate pose wrists.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_loader = make_loader(train_df, cfg, training=True, seed=seed)
    val_loader = make_loader(val_df, cfg, training=False, seed=seed + 1)
    model = SAM27Classifier(num_classes=num_classes, cfg=cfg).to(device)
    optimizer = create_optimizer(model, cfg)
    ema_cfg = cfg.get("ema", {})
    ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999))) if bool(ema_cfg.get("enabled", False)) else None

    history = []
    best_top1 = -1.0
    best_metrics: dict[str, float] = {}
    epochs = int(cfg.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        lr = epoch_lr(cfg, epoch)
        set_optimizer_lr(optimizer, lr)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            cfg=cfg,
            num_classes=num_classes,
            epoch=epoch,
            ema=ema,
        )
        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate(eval_model, val_loader, device=device, cfg=cfg)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{epochs} lr={lr:.6g} "
            f"train_top1={row['train_top1']:.4f} val_top1={row['val_top1']:.4f} val_top5={row['val_top5']:.4f}"
        )
        if val_metrics["top1"] > best_top1:
            best_top1 = float(val_metrics["top1"])
            best_metrics = dict(row)
            save_checkpoint(out_dir / "best.pt", eval_model, cfg, label_payload, best_metrics)
        save_checkpoint(out_dir / "last.pt", eval_model, cfg, label_payload, row)

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"best val_top1={best_metrics.get('val_top1', 0.0):.4f} val_top5={best_metrics.get('val_top5', 0.0):.4f}")


if __name__ == "__main__":
    main()
