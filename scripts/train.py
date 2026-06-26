#!/usr/bin/env python
"""Train the skeleton-only AT-STGCN model used by the paper."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path


def _sanitize_thread_env() -> None:
    """Avoid libgomp failures from invalid thread-count environment values."""
    value = os.environ.get("OMP_NUM_THREADS")
    if value is None:
        return
    value = value.strip()
    if not value.isdecimal() or int(value) <= 0:
        os.environ.pop("OMP_NUM_THREADS", None)


_sanitize_thread_env()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYDEPS = PROJECT_ROOT / ".runtime" / "pydeps"
if RUNTIME_PYDEPS.exists() and str(RUNTIME_PYDEPS) not in sys.path:
    sys.path.append(str(RUNTIME_PYDEPS))

import numpy as np

import torch
import yaml
from torch import nn

try:
    torch.from_numpy(np.zeros((1,), dtype=np.float32))
except RuntimeError as exc:
    raise RuntimeError(
        "PyTorch cannot use the installed NumPy package in this environment. "
        f"Detected NumPy {np.__version__}. If your PyTorch build was compiled "
        "against NumPy 1.x, install NumPy 1.26.4, for example: "
        "python -m pip install --force-reinstall 'numpy==1.26.4'"
    ) from exc

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.augment import AugmentConfig
from sl_atstgcn.at_stgcn import AT_STGCN_MODEL_TYPES, build_at_stgcn_classifier, create_optimizer, save_checkpoint
from sl_atstgcn.data import LabelMap, attach_label_ids, make_skeleton_dataloader, read_manifest
from sl_atstgcn.graph import SKELETON_MIRROR_JOINT_INDICES
from sl_atstgcn.schedule import BatchScheduler, build_scheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config, see configs/*.yaml")
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError("Empty config")
    return cfg


def parse_int_sequence(value, default=(1,)) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    parsed = tuple(max(1, int(item)) for item in values)
    return parsed or tuple(default)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict) -> torch.device:
    requested = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; using CPU")
        requested = "cpu"
    return torch.device(requested)


def topk_correct_from_logits(logits: torch.Tensor, labels: torch.Tensor, *, k: int = 1) -> int:
    k = max(1, min(int(k), int(logits.size(1))))
    topk = logits.topk(k=k, dim=1).indices
    return int(topk.eq(labels.view(-1, 1)).any(dim=1).sum().item())


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return topk_correct_from_logits(logits, labels, k=1)


def smooth_one_hot(labels: torch.Tensor, *, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    smoothing = float(smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("label smoothing must be in [0, 1)")
    off_value = smoothing / max(1, int(num_classes) - 1)
    on_value = 1.0 - smoothing
    target = torch.full((labels.size(0), int(num_classes)), off_value, device=labels.device, dtype=torch.float32)
    target.scatter_(1, labels.view(-1, 1), on_value)
    return target


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    per_class_loss = -targets * log_probs
    if float(focal_gamma) > 0.0:
        probs = torch.exp(log_probs)
        per_class_loss = per_class_loss * torch.pow(1.0 - probs, float(focal_gamma))
    if class_weights is not None:
        per_class_loss = per_class_loss * class_weights.view(1, -1).to(device=logits.device, dtype=logits.dtype)
    return torch.sum(per_class_loss, dim=1).mean()


def maybe_mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixed_images, mixed_targets, _ = maybe_mixup_with_extra(
        images,
        targets,
        extra_targets=None,
        alpha=alpha,
        prob=prob,
    )
    return mixed_images, mixed_targets


def maybe_mixup_with_extra(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    extra_targets: torch.Tensor | None,
    alpha: float,
    prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if alpha <= 0.0 or prob <= 0.0 or images.size(0) < 2:
        return images, targets, extra_targets
    if torch.rand((), device=images.device).item() > float(prob):
        return images, targets, extra_targets
    lam = torch.distributions.Beta(float(alpha), float(alpha)).sample().to(images.device)
    lam = torch.maximum(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = images.mul(lam).add(images[index], alpha=float(1.0 - lam))
    mixed_targets = targets.mul(lam).add(targets[index], alpha=float(1.0 - lam))
    mixed_extra = None
    if extra_targets is not None:
        mixed_extra = extra_targets.mul(lam).add(extra_targets[index], alpha=float(1.0 - lam))
    return mixed_images, mixed_targets, mixed_extra


def distillation_cross_entropy(
    logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    temperature: float,
    scale_by_temperature: bool = True,
) -> torch.Tensor:
    temperature = max(float(temperature), 1.0e-6)
    teacher_probs = teacher_probs.clamp_min(1.0e-8)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    loss = soft_cross_entropy(logits / temperature, teacher_probs)
    if scale_by_temperature:
        loss = loss * (temperature * temperature)
    return loss


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Batch supervised contrastive loss; returns zero when no positive pairs exist."""
    if features.size(0) < 2:
        return features.new_zeros(())
    temperature = max(float(temperature), 1.0e-6)
    features = torch.nn.functional.normalize(features, dim=1)
    logits = torch.matmul(features, features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.t()).to(dtype=features.dtype, device=features.device)
    self_mask = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
    positive_mask = positive_mask.masked_fill(self_mask, 0.0)
    positive_counts = positive_mask.sum(dim=1)
    valid_anchors = positive_counts > 0
    if not torch.any(valid_anchors):
        return features.new_zeros(())
    logits_mask = (~self_mask).to(dtype=features.dtype)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    return -mean_log_prob[valid_anchors].mean()


def _manifest_path_key(value: str | Path) -> str:
    return str(value).replace("\\", "/").lower()


def load_distillation_soft_targets(
    path: str | Path,
    df,
    *,
    num_classes: int,
) -> np.ndarray:
    with np.load(str(path), allow_pickle=True) as payload:
        if "paths" not in payload or "probs" not in payload:
            raise ValueError(f"Distillation soft-label file {path} must contain 'paths' and 'probs' arrays")
        paths = [str(x) for x in payload["paths"].tolist()]
        probs = np.asarray(payload["probs"], dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != int(num_classes):
        raise ValueError(f"Expected distillation probs shape (N, {num_classes}), got {probs.shape}")
    if len(paths) != probs.shape[0]:
        raise ValueError(f"Distillation paths {len(paths)} != probs rows {probs.shape[0]}")
    lookup = {_manifest_path_key(path_value): probs[index] for index, path_value in enumerate(paths)}
    rows = []
    missing = []
    for path_value in df["path"].astype(str):
        key = _manifest_path_key(path_value)
        target = lookup.get(key)
        if target is None:
            missing.append(path_value)
        else:
            rows.append(target)
    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        raise ValueError(f"Missing {len(missing)} distillation targets; first missing paths: {preview}")
    soft_targets = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
    row_sums = soft_targets.sum(axis=1, keepdims=True)
    soft_targets = soft_targets / np.clip(row_sums, 1.0e-8, None)
    return soft_targets


def semantic_horizontal_flip_batch(images: torch.Tensor) -> torch.Tensor:
    flipped = images.clone()
    valid = (flipped[:, 0] != 0.0) | (flipped[:, 1] != 0.0)
    flipped[:, 0][valid] = 1.0 - flipped[:, 0][valid]
    if images.size(3) == len(SKELETON_MIRROR_JOINT_INDICES):
        indices = SKELETON_MIRROR_JOINT_INDICES
    else:
        indices = list(range(int(images.size(3))))
    column_indices = torch.as_tensor(indices, device=images.device, dtype=torch.long)
    return flipped.index_select(dim=3, index=column_indices)


def _valid_xy_mask(images: torch.Tensor) -> torch.Tensor:
    return (images[:, 0] != 0.0) | (images[:, 1] != 0.0)


def semantic_scale_batch(images: torch.Tensor, scale: float) -> torch.Tensor:
    scale = float(scale)
    if abs(scale - 1.0) < 1.0e-6:
        return images
    out = images.clone()
    valid = _valid_xy_mask(out)
    valid_f = valid.to(dtype=out.dtype)
    counts = valid_f.flatten(1).sum(dim=1).clamp_min(1.0)
    for channel in (0, 1):
        coords = out[:, channel]
        centers = (coords * valid_f).flatten(1).sum(dim=1) / counts
        scaled = (coords - centers[:, None, None]) * scale + centers[:, None, None]
        out[:, channel] = torch.where(valid, scaled.clamp(0.0, 1.0), coords)
    if out.size(1) > 2:
        out[:, 2] = torch.where(valid, (out[:, 2] * abs(scale)).clamp(0.0, 1.0), out[:, 2])
    return out


def semantic_shift_batch(images: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    dx = float(dx)
    dy = float(dy)
    if abs(dx) < 1.0e-8 and abs(dy) < 1.0e-8:
        return images
    out = images.clone()
    valid = _valid_xy_mask(out)
    out[:, 0] = torch.where(valid, (out[:, 0] + dx).clamp(0.0, 1.0), out[:, 0])
    out[:, 1] = torch.where(valid, (out[:, 1] + dy).clamp(0.0, 1.0), out[:, 1])
    return out


def _float_list(value, *, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(x) for x in value]


def _shift_list(value) -> list[tuple[float, float]]:
    if value is None:
        return [(0.0, 0.0)]
    shifts: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            shifts.append((float(item.get("x", 0.0)), float(item.get("y", 0.0))))
        else:
            dx, dy = item
            shifts.append((float(dx), float(dy)))
    return shifts or [(0.0, 0.0)]


def tta_logits(model: nn.Module, images: torch.Tensor, tta_config: dict | None = None) -> torch.Tensor:
    if not tta_config:
        return model(images)
    scales = _float_list(tta_config.get("scales", tta_config.get("scale_values")), default=[1.0])
    shifts = _shift_list(tta_config.get("shifts"))
    use_flip = bool(tta_config.get("flip", False))
    logits_sum: torch.Tensor | None = None
    count = 0
    seen: set[tuple[float, float, float, bool]] = set()
    for scale in scales:
        for dx, dy in shifts:
            transformed = semantic_scale_batch(images, scale)
            transformed = semantic_shift_batch(transformed, dx, dy)
            flip_options = (False, True) if use_flip else (False,)
            for flip in flip_options:
                key = (round(float(scale), 6), round(float(dx), 6), round(float(dy), 6), bool(flip))
                if key in seen:
                    continue
                seen.add(key)
                batch = semantic_horizontal_flip_batch(transformed) if flip else transformed
                logits = model(batch)
                logits_sum = logits if logits_sum is None else logits_sum + logits
                count += 1
    if logits_sum is None:
        return model(images)
    return logits_sum / max(1, count)


class ModelEMA:
    """Exponential moving average of model weights for stabler validation."""

    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


class SAMOptimizer:
    """Sharpness-Aware Minimization wrapper around an existing optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer, *, rho: float = 0.05, adaptive: bool = False) -> None:
        self.optimizer = optimizer
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.state = optimizer.state
        self.param_groups = optimizer.param_groups

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        norms = []
        shared_device = self.param_groups[0]["params"][0].device
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                scale = torch.abs(param) if self.adaptive else 1.0
                norms.append((scale * param.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, *, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1.0e-12)
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                perturb = param.grad * scale.to(param)
                if self.adaptive:
                    perturb = perturb * torch.pow(param, 2)
                self.state[param]["sam_perturb"] = perturb
                param.add_(perturb)
        if zero_grad:
            self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, *, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                perturb = self.state[param].pop("sam_perturb", None)
                if perturb is not None:
                    param.sub_(perturb)
        self.optimizer.step()
        if zero_grad:
            self.optimizer.zero_grad(set_to_none=True)


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("Cannot average an empty state dict list")
    averaged: dict[str, torch.Tensor] = {}
    reference = state_dicts[0]
    for key, ref_value in reference.items():
        values = [state[key] for state in state_dicts]
        if ref_value.dtype.is_floating_point:
            stacked = torch.stack([value.to(dtype=torch.float32) for value in values], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=ref_value.dtype)
        else:
            averaged[key] = ref_value.clone()
    return averaged


def effective_number_class_weights(labels: np.ndarray, *, num_classes: int, beta: float = 0.999) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    beta = min(max(float(beta), 0.0), 0.999999)
    if beta <= 0.0:
        weights = np.ones_like(counts, dtype=np.float64)
    else:
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / np.mean(weights)
    return torch.as_tensor(weights, dtype=torch.float32)


class TopKModelSoup:
    """Keep the top validation checkpoints and average them after training."""

    def __init__(self, *, top_k: int = 5, start_epoch: int = 1) -> None:
        self.top_k = int(max(1, top_k))
        self.start_epoch = int(max(1, start_epoch))
        self.entries: list[dict] = []

    def maybe_add(self, *, metric: float, epoch: int, model: nn.Module, metrics: dict[str, float]) -> None:
        if int(epoch) < self.start_epoch:
            return
        if len(self.entries) >= self.top_k:
            worst = min(self.entries, key=lambda item: (item["metric"], item["epoch"]))
            if (float(metric), int(epoch)) <= (float(worst["metric"]), int(worst["epoch"])):
                return
        self.entries.append(
            {
                "metric": float(metric),
                "epoch": int(epoch),
                "metrics": dict(metrics),
                "state": clone_state_dict_to_cpu(model),
            }
        )
        self.entries.sort(key=lambda item: (item["metric"], item["epoch"]), reverse=True)
        del self.entries[self.top_k :]

    def build_model(self, template: nn.Module, device: torch.device) -> nn.Module:
        if not self.entries:
            raise ValueError("No checkpoints collected for model soup")
        soup_model = copy.deepcopy(template).to(device)
        averaged_state = average_state_dicts([entry["state"] for entry in self.entries])
        soup_model.load_state_dict(averaged_state)
        soup_model.eval()
        return soup_model

    def summary(self) -> list[dict[str, float]]:
        return [
            {"epoch": float(entry["epoch"]), "metric": float(entry["metric"])}
            for entry in self.entries
        ]


def build_model_from_config(cfg: dict, *, num_classes: int, image_height: int) -> tuple[nn.Module, dict]:
    model_type = str(cfg.get("model_type", cfg.get("input_type", "skeleton"))).strip().lower()
    if model_type not in AT_STGCN_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. This paper-facing trainer is skeleton-only; "
            "use one of the AT-STGCN skeleton model_type values."
        )
    classifier_type = str(cfg.get("classifier_type", "linear"))
    logit_scale = float(cfg.get("logit_scale", 30.0))
    classifier_margin = float(cfg.get("classifier_margin", 0.0))
    feature_layer_norm = bool(cfg.get("feature_layer_norm", False))
    center_loss_cfg = cfg.get("center_loss", {})
    if isinstance(center_loss_cfg, dict):
        center_loss_weight = float(center_loss_cfg.get("weight", 0.0 if not center_loss_cfg.get("enabled", False) else 0.02))
    else:
        center_loss_weight = 0.0
    sequence_length = int(cfg.get("sequence_length", image_height))
    skeleton_hidden_channels = int(cfg.get("skeleton_hidden_channels", cfg.get("hidden_channels", 192)))
    skeleton_blocks = int(cfg.get("skeleton_blocks", cfg.get("blocks", 4)))
    skeleton_adjacency_hops = int(cfg.get("skeleton_adjacency_hops", cfg.get("adjacency_hops", 2)))
    skeleton_dropout = float(cfg.get("skeleton_dropout", 0.10))
    skeleton_temporal_kernel = int(cfg.get("skeleton_temporal_kernel", 5))
    skeleton_temporal_dilations = parse_int_sequence(cfg.get("skeleton_temporal_dilations", (1,)))
    skeleton_adaptive_graph = bool(cfg.get("skeleton_adaptive_graph", False))
    skeleton_edge_importance = bool(cfg.get("skeleton_edge_importance", False))
    skeleton_adaptive_scale = float(cfg.get("skeleton_adaptive_scale", 0.10))
    skeleton_relation_graph = bool(cfg.get("skeleton_relation_graph", False))
    skeleton_relation_scale = float(cfg.get("skeleton_relation_scale", 0.05))
    skeleton_relation_channels = int(cfg.get("skeleton_relation_channels", 32))
    skeleton_stc_attention = bool(cfg.get("skeleton_stc_attention", True))
    skeleton_center_joints = bool(cfg.get("skeleton_center_joints", True))
    skeleton_scale_normalize = bool(cfg.get("skeleton_scale_normalize", True))
    skeleton_hand_weight = float(cfg.get("skeleton_hand_weight", 1.20))
    skeleton_include_absolute_xy = bool(cfg.get("skeleton_include_absolute_xy", False))
    skeleton_include_validity = bool(cfg.get("skeleton_include_validity", False))
    skeleton_include_temporal_position = bool(cfg.get("skeleton_include_temporal_position", False))
    skeleton_include_root_motion = bool(cfg.get("skeleton_include_root_motion", False))
    skeleton_include_acceleration = bool(cfg.get("skeleton_include_acceleration", False))
    skeleton_use_bone_features = bool(cfg.get("skeleton_use_bone_features", True))
    skeleton_use_motion_features = bool(cfg.get("skeleton_use_motion_features", True))
    skeleton_pooling = str(cfg.get("skeleton_pooling", "avg"))
    skeleton_part_pooling = bool(cfg.get("skeleton_part_pooling", False))
    skeleton_part_pooling_scale = float(cfg.get("skeleton_part_pooling_scale", 1.0))
    model = build_at_stgcn_classifier(
        num_classes=num_classes,
        dropout=float(cfg["dropout"]),
        input_channels=int(cfg.get("input_channels", 3)),
        hidden_channels=skeleton_hidden_channels,
        blocks=skeleton_blocks,
        adjacency_hops=skeleton_adjacency_hops,
        skeleton_dropout=skeleton_dropout,
        skeleton_temporal_kernel=skeleton_temporal_kernel,
        skeleton_temporal_dilations=skeleton_temporal_dilations,
        skeleton_adaptive_graph=skeleton_adaptive_graph,
        skeleton_edge_importance=skeleton_edge_importance,
        skeleton_adaptive_scale=skeleton_adaptive_scale,
        skeleton_relation_graph=skeleton_relation_graph,
        skeleton_relation_scale=skeleton_relation_scale,
        skeleton_relation_channels=skeleton_relation_channels,
        stc_attention=skeleton_stc_attention,
        center_joints=skeleton_center_joints,
        scale_normalize=skeleton_scale_normalize,
        hand_weight=skeleton_hand_weight,
        include_absolute_xy=skeleton_include_absolute_xy,
        include_validity=skeleton_include_validity,
        include_temporal_position=skeleton_include_temporal_position,
        include_root_motion=skeleton_include_root_motion,
        include_acceleration=skeleton_include_acceleration,
        use_bone_features=skeleton_use_bone_features,
        use_motion_features=skeleton_use_motion_features,
        pooling=skeleton_pooling,
        part_pooling=skeleton_part_pooling,
        part_pooling_scale=skeleton_part_pooling_scale,
        classifier_type=classifier_type,
        logit_scale=logit_scale,
        classifier_margin=classifier_margin,
        center_loss_weight=center_loss_weight,
        feature_layer_norm=bool(cfg.get("feature_layer_norm", True)),
    )
    model_config = {
        "model_type": "skeleton",
        "num_classes": num_classes,
        "image_height": sequence_length,
        "sequence_length": sequence_length,
        "input_channels": int(cfg.get("input_channels", 3)),
        "dropout": float(cfg["dropout"]),
        "classifier_type": classifier_type,
        "logit_scale": logit_scale,
        "classifier_margin": classifier_margin,
        "skeleton_hidden_channels": skeleton_hidden_channels,
        "skeleton_blocks": skeleton_blocks,
        "skeleton_adjacency_hops": skeleton_adjacency_hops,
        "skeleton_dropout": skeleton_dropout,
        "skeleton_temporal_kernel": skeleton_temporal_kernel,
        "skeleton_temporal_dilations": list(skeleton_temporal_dilations),
        "skeleton_adaptive_graph": skeleton_adaptive_graph,
        "skeleton_edge_importance": skeleton_edge_importance,
        "skeleton_adaptive_scale": skeleton_adaptive_scale,
        "skeleton_relation_graph": skeleton_relation_graph,
        "skeleton_relation_scale": skeleton_relation_scale,
        "skeleton_relation_channels": skeleton_relation_channels,
        "skeleton_stc_attention": skeleton_stc_attention,
        "skeleton_center_joints": skeleton_center_joints,
        "skeleton_scale_normalize": skeleton_scale_normalize,
        "skeleton_hand_weight": skeleton_hand_weight,
        "skeleton_include_absolute_xy": skeleton_include_absolute_xy,
        "skeleton_include_validity": skeleton_include_validity,
        "skeleton_include_temporal_position": skeleton_include_temporal_position,
        "skeleton_include_root_motion": skeleton_include_root_motion,
        "skeleton_include_acceleration": skeleton_include_acceleration,
        "skeleton_use_bone_features": skeleton_use_bone_features,
        "skeleton_use_motion_features": skeleton_use_motion_features,
        "skeleton_pooling": skeleton_pooling,
        "skeleton_part_pooling": skeleton_part_pooling,
        "skeleton_part_pooling_scale": skeleton_part_pooling_scale,
        "feature_layer_norm": bool(cfg.get("feature_layer_norm", True)),
        "center_loss_weight": center_loss_weight,
        "repair_missing_keypoints": bool(cfg.get("repair_missing_keypoints", False)),
        "repair_min_valid": int(cfg.get("repair_min_valid", 2)),
    }
    return model, model_config


def apply_manifest_quality_filter(df, cfg: dict):
    quality_cfg = cfg.get("quality_filter", {})
    if not isinstance(quality_cfg, dict) or not bool(quality_cfg.get("enabled", False)):
        return df
    if "valid_pose_frames" not in df.columns:
        print("WARNING: quality_filter enabled but manifest has no valid_pose_frames column")
        return df
    min_frames = int(quality_cfg.get("min_valid_pose_frames", 1))
    splits = {str(x).lower() for x in quality_cfg.get("splits", ["train"])}
    split_values = df["split"].astype(str).str.lower()
    valid_frames = df["valid_pose_frames"].fillna(0).astype(int)
    keep = (~split_values.isin(splits)) | (valid_frames >= min_frames)
    dropped = df.loc[~keep]
    if len(dropped) > 0:
        counts = dropped["split"].astype(str).str.lower().value_counts().to_dict()
        print(f"Quality filter dropped {len(dropped)} rows with valid_pose_frames < {min_frames}: {counts}")
    return df.loc[keep].reset_index(drop=True)


def train_one_epoch(
    model: nn.Module,
    loader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: BatchScheduler,
    device: torch.device,
    ema: ModelEMA | None = None,
    sam: SAMOptimizer | None = None,
    grad_clip_norm: float | None = None,
    num_classes: int,
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    mixup_config: dict | None = None,
    distillation_alpha: float = 0.0,
    distillation_temperature: float = 1.0,
    distillation_scale_by_temperature: bool = True,
    contrastive_config: dict | None = None,
    epoch: int = 1,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_seen = 0
    last_lr = optimizer.param_groups[0]["lr"]

    for batch in loader:
        if len(batch) == 3:
            images, labels, teacher_targets = batch
        else:
            images, labels = batch
            teacher_targets = None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if teacher_targets is not None:
            teacher_targets = teacher_targets.to(device, non_blocking=True, dtype=torch.float32)
        last_lr = scheduler.batch_step()
        targets = smooth_one_hot(labels, num_classes=num_classes, smoothing=label_smoothing)
        mixup_active = False
        if mixup_config is not None and bool(mixup_config.get("enabled", False)):
            start_epoch = int(mixup_config.get("start_epoch", 1))
            end_epoch = int(mixup_config.get("end_epoch", 10**9))
            mixup_active = start_epoch <= int(epoch) <= end_epoch
        if mixup_active:
            images, targets, teacher_targets = maybe_mixup_with_extra(
                images,
                targets,
                extra_targets=teacher_targets,
                alpha=float(mixup_config.get("alpha", 0.2)),
                prob=float(mixup_config.get("prob", 1.0)),
            )

        optimizer.zero_grad(set_to_none=True)
        margin_labels = None if mixup_active else labels
        use_center_loss = (
            not mixup_active
            and hasattr(model, "center_loss_weight")
            and float(getattr(model, "center_loss_weight", 0.0)) > 0.0
        )
        contrastive_config = contrastive_config if isinstance(contrastive_config, dict) else {}
        contrastive_active = (
            not mixup_active
            and bool(contrastive_config.get("enabled", False))
            and float(contrastive_config.get("weight", 0.0)) > 0.0
            and int(contrastive_config.get("start_epoch", 1)) <= int(epoch)
            and int(epoch) <= int(contrastive_config.get("end_epoch", 10**9))
        )
        need_features = bool(use_center_loss or contrastive_active)
        def forward_loss() -> tuple[torch.Tensor, torch.Tensor]:
            if need_features:
                batch_logits, features = model(images, labels=margin_labels, return_features=True)
            else:
                batch_logits = model(images, labels=margin_labels)
                features = None
            batch_loss = soft_cross_entropy(
                batch_logits,
                targets,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
            )
            if teacher_targets is not None and float(distillation_alpha) > 0.0:
                teacher_loss = distillation_cross_entropy(
                    batch_logits,
                    teacher_targets,
                    temperature=float(distillation_temperature),
                    scale_by_temperature=bool(distillation_scale_by_temperature),
                )
                alpha = min(max(float(distillation_alpha), 0.0), 1.0)
                batch_loss = batch_loss * (1.0 - alpha) + teacher_loss * alpha
            if use_center_loss and features is not None:
                batch_loss = batch_loss + model.center_clustering_loss(features, labels)
            if contrastive_active and features is not None:
                batch_loss = batch_loss + supervised_contrastive_loss(
                    features,
                    labels,
                    temperature=float(contrastive_config.get("temperature", 0.07)),
                ) * float(contrastive_config.get("weight", 0.0))
            return batch_loss, batch_logits

        loss, logits = forward_loss()
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        if sam is not None:
            sam.first_step(zero_grad=True)
            second_loss, _ = forward_loss()
            second_loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            sam.second_step(zero_grad=True)
        else:
            optimizer.step()
        if ema is not None:
            ema.update(model)

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        detached_logits = logits.detach()
        total_top1_correct += topk_correct_from_logits(detached_logits, labels, k=1)
        total_top5_correct += topk_correct_from_logits(detached_logits, labels, k=5)
        total_seen += batch_size

    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1_correct / max(1, total_seen),
        "top5": total_top5_correct / max(1, total_seen),
        "lr": float(last_lr),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader,
    *,
    criterion: nn.Module,
    device: torch.device,
    tta_flip: bool = False,
    tta_config: dict | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        resolved_tta = tta_config if tta_config is not None else ({"flip": True} if tta_flip else None)
        logits = tta_logits(model, images, resolved_tta)
        loss = criterion(logits, labels)
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_top1_correct += topk_correct_from_logits(logits, labels, k=1)
        total_top5_correct += topk_correct_from_logits(logits, labels, k=5)
        total_seen += batch_size
    return {
        "val_loss": total_loss / max(1, total_seen),
        "val_top1": total_top1_correct / max(1, total_seen),
        "val_top5": total_top5_correct / max(1, total_seen),
    }


def write_history_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(cfg)
    print(f"Using device: {device}")

    df = read_manifest(cfg["manifest"])
    df = apply_manifest_quality_filter(df, cfg)
    df, label_map = attach_label_ids(df)
    label_map.save(out_dir / "label_map.json")

    train_df = df[df["split"].str.lower().isin(["train", "training"])].reset_index(drop=True)
    val_df = df[df["split"].str.lower().isin(["val", "valid", "validation"])].reset_index(drop=True)
    if len(train_df) == 0:
        raise ValueError("No train rows found in manifest")
    if len(val_df) == 0:
        print("WARNING: no validation split found; training without validation metrics")

    image_height = int(cfg["image_height"])
    model_type = str(cfg.get("model_type", cfg.get("input_type", "skeleton"))).strip().lower()
    use_skeleton_input = model_type in AT_STGCN_MODEL_TYPES
    if not use_skeleton_input:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. scripts/train.py now trains only skeleton-only AT-STGCN models."
        )
    sequence_length = int(cfg.get("sequence_length", image_height))
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", device.type == "cuda"))
    repair_missing = bool(cfg.get("repair_missing_keypoints", False))
    repair_min_valid = int(cfg.get("repair_min_valid", 2))
    aug_cfg = AugmentConfig(**cfg.get("augmentation", {}))
    num_classes = len(label_map.label_to_id)
    distillation_cfg = cfg.get("distillation", {})
    distillation_cfg = distillation_cfg if isinstance(distillation_cfg, dict) else {}
    train_soft_targets = None
    if bool(distillation_cfg.get("enabled", False)):
        soft_label_path = distillation_cfg.get("soft_labels")
        if not soft_label_path:
            raise ValueError("distillation.enabled=true requires distillation.soft_labels")
        train_soft_targets = load_distillation_soft_targets(
            soft_label_path,
            train_df,
            num_classes=num_classes,
        )
        print(
            "Loaded distillation soft labels "
            f"from {soft_label_path} for {len(train_soft_targets)} train samples"
        )
    train_loader = make_skeleton_dataloader(
        train_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        training=True,
        augment_config=aug_cfg,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        class_balanced=bool(cfg.get("balanced_sampling", False)),
        repair_missing=repair_missing,
        repair_min_valid=repair_min_valid,
        soft_targets=train_soft_targets,
    )
    val_loader = None
    if len(val_df) > 0:
        val_loader = make_skeleton_dataloader(
            val_df,
            sequence_length=sequence_length,
            batch_size=batch_size,
            training=False,
            augment_config=None,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            repair_missing=repair_missing,
            repair_min_valid=repair_min_valid,
        )

    model, model_config = build_model_from_config(cfg, num_classes=num_classes, image_height=image_height)
    init_checkpoint = cfg.get("init_checkpoint", None)
    if init_checkpoint:
        init_path = Path(str(init_checkpoint))
        if not init_path.is_absolute():
            init_path = PROJECT_ROOT / init_path
        checkpoint = torch.load(init_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            source_epoch = checkpoint.get("epoch", None)
            source_metrics = checkpoint.get("metrics", None)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            source_epoch = None
            source_metrics = None
        else:
            raise ValueError(f"Unsupported init checkpoint format in {init_path}")
        strict = bool(cfg.get("init_checkpoint_strict", True))
        incompatible = model.load_state_dict(state_dict, strict=strict)
        print(
            "Initialized model from "
            f"{init_path} strict={strict} source_epoch={source_epoch} "
            f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}"
        )
        model_config["init_checkpoint"] = str(init_path)
        if source_epoch is not None:
            model_config["init_checkpoint_epoch"] = int(source_epoch)
        if isinstance(source_metrics, dict):
            model_config["init_checkpoint_metrics"] = {
                key: float(value) for key, value in source_metrics.items() if isinstance(value, (int, float))
            }
    model = model.to(device)
    distillation_alpha = 0.0
    distillation_temperature = 1.0
    distillation_scale_by_temperature = True
    if train_soft_targets is not None:
        distillation_alpha = float(distillation_cfg.get("alpha", 0.65))
        distillation_temperature = float(distillation_cfg.get("temperature", 2.0))
        distillation_scale_by_temperature = bool(distillation_cfg.get("scale_by_temperature", True))
        model_config["distillation"] = {
            "enabled": True,
            "soft_labels": str(distillation_cfg.get("soft_labels")),
            "alpha": distillation_alpha,
            "temperature": distillation_temperature,
            "scale_by_temperature": distillation_scale_by_temperature,
            "teacher": str(distillation_cfg.get("teacher", "ensemble")),
        }
    ema_cfg = cfg.get("ema", {})
    ema = None
    if bool(ema_cfg.get("enabled", False)):
        ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999)))
        model_config["ema"] = {"enabled": True, "decay": float(ema_cfg.get("decay", 0.999))}
    eval_tta_cfg = cfg.get("eval_tta", {})
    eval_tta_cfg = eval_tta_cfg if isinstance(eval_tta_cfg, dict) else {}
    if eval_tta_cfg:
        model_config["eval_tta"] = dict(eval_tta_cfg)
    optimizer = create_optimizer(
        model,
        learning_rate=float(cfg["base_lr"]),
        weight_decay=float(cfg["weight_decay"]),
        momentum=float(cfg.get("momentum", 0.98)),
        optimizer_name=str(cfg.get("optimizer", "sgd")),
        backbone_lr_scale=float(cfg.get("backbone_lr_scale", 1.0)),
        attention_lr_scale=float(cfg.get("attention_lr_scale", 1.0)),
        classifier_lr_scale=float(cfg.get("classifier_lr_scale", cfg.get("head_lr_scale", 1.0))),
        no_weight_decay_norm_bias=bool(cfg.get("no_weight_decay_norm_bias", False)),
    )
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_cfg = cfg.get("loss", {})
    loss_cfg = loss_cfg if isinstance(loss_cfg, dict) else {}
    focal_gamma = float(loss_cfg.get("focal_gamma", 0.0))
    class_weight_cfg = cfg.get("class_weighting", {})
    class_weight_cfg = class_weight_cfg if isinstance(class_weight_cfg, dict) else {}
    class_weights = None
    if bool(class_weight_cfg.get("enabled", False)):
        class_weights = effective_number_class_weights(
            train_df["label_id"].to_numpy(),
            num_classes=num_classes,
            beta=float(class_weight_cfg.get("beta", 0.999)),
        ).to(device)
        model_config["class_weighting"] = {
            "enabled": True,
            "beta": float(class_weight_cfg.get("beta", 0.999)),
        }
    if focal_gamma > 0.0:
        model_config["loss"] = {"focal_gamma": focal_gamma}
    mixup_config = cfg.get("mixup", {})
    contrastive_config = cfg.get("contrastive_loss", {})
    contrastive_config = contrastive_config if isinstance(contrastive_config, dict) else {}
    if bool(contrastive_config.get("enabled", False)) and float(contrastive_config.get("weight", 0.0)) > 0.0:
        model_config["contrastive_loss"] = {
            "enabled": True,
            "weight": float(contrastive_config.get("weight", 0.0)),
            "temperature": float(contrastive_config.get("temperature", 0.07)),
            "start_epoch": int(contrastive_config.get("start_epoch", 1)),
            "end_epoch": int(contrastive_config.get("end_epoch", 10**9)),
        }
    sam_cfg = cfg.get("sam", {})
    sam_cfg = sam_cfg if isinstance(sam_cfg, dict) else {}
    sam = None
    if bool(sam_cfg.get("enabled", False)):
        sam = SAMOptimizer(
            optimizer,
            rho=float(sam_cfg.get("rho", 0.05)),
            adaptive=bool(sam_cfg.get("adaptive", False)),
        )
        model_config["sam"] = {
            "enabled": True,
            "rho": float(sam_cfg.get("rho", 0.05)),
            "adaptive": bool(sam_cfg.get("adaptive", False)),
        }
    soup_cfg = cfg.get("model_soup", {})
    model_soup = None
    if val_loader is not None and bool(soup_cfg.get("enabled", False)):
        model_soup = TopKModelSoup(
            top_k=int(soup_cfg.get("top_k", 5)),
            start_epoch=int(soup_cfg.get("start_epoch", max(1, int(cfg["epochs"]) // 2))),
        )
        model_config["model_soup"] = {
            "enabled": True,
            "top_k": model_soup.top_k,
            "start_epoch": model_soup.start_epoch,
        }

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Trainable parameters: {trainable_params:,}")

    steps_per_epoch = max(1, int((len(train_df) + batch_size - 1) // batch_size))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)
    grad_clip_norm = cfg.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)

    history_rows: list[dict[str, float]] = []
    history: dict[str, list[float]] = {"loss": [], "top1": [], "top5": [], "lr": []}
    if val_loader is not None:
        history.update({"val_loss": [], "val_top1": [], "val_top5": []})

    monitor_key = "val_top1" if val_loader is not None else "top1"
    best_metric = float("-inf")
    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            ema=ema,
            sam=sam,
            grad_clip_norm=grad_clip_norm,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            mixup_config=mixup_config,
            distillation_alpha=distillation_alpha,
            distillation_temperature=distillation_temperature,
            distillation_scale_by_temperature=distillation_scale_by_temperature,
            contrastive_config=contrastive_config,
            epoch=epoch,
        )
        metrics = {"epoch": float(epoch), **train_metrics}
        if val_loader is not None:
            eval_model = ema.module if ema is not None else model
            metrics.update(
                evaluate_one_epoch(
                    eval_model,
                    val_loader,
                    criterion=criterion,
                    device=device,
                    tta_config=eval_tta_cfg,
                )
            )

        current_metric = float(metrics[monitor_key])
        checkpoint_model = ema.module if ema is not None else model
        if model_soup is not None:
            model_soup.maybe_add(
                metric=current_metric,
                epoch=epoch,
                model=checkpoint_model,
                metrics=metrics,
            )
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                out_dir / "best.pt",
                model=checkpoint_model,
                model_config=model_config,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )

        history_rows.append(metrics)
        for key, value in metrics.items():
            if key == "epoch":
                continue
            history.setdefault(key, []).append(float(value))
        metric_text = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "epoch")
        print(f"Epoch {epoch}/{epochs} {metric_text}")

    checkpoint_model = ema.module if ema is not None else model
    save_checkpoint(
        out_dir / "last.pt",
        model=checkpoint_model,
        model_config=model_config,
        optimizer=optimizer,
        epoch=epochs,
        metrics=history_rows[-1] if history_rows else None,
    )
    if model_soup is not None and model_soup.entries:
        soup_model = model_soup.build_model(checkpoint_model, device)
        soup_metrics = evaluate_one_epoch(
            soup_model,
            val_loader,
            criterion=criterion,
            device=device,
            tta_config=eval_tta_cfg,
        )
        soup_metric = float(soup_metrics[monitor_key])
        soup_payload = {
            "epoch": float(epochs),
            "soup_val_loss": float(soup_metrics["val_loss"]),
            "soup_val_top1": float(soup_metrics["val_top1"]),
            "soup_val_top5": float(soup_metrics["val_top5"]),
            "members": model_soup.summary(),
        }
        with (out_dir / "model_soup.json").open("w", encoding="utf-8") as f:
            json.dump(soup_payload, f, indent=2)
        save_checkpoint(
            out_dir / "soup.pt",
            model=soup_model,
            model_config=model_config,
            optimizer=optimizer,
            epoch=epochs,
            metrics={**soup_metrics, "epoch": float(epochs)},
        )
        if soup_metric > best_metric and bool(soup_cfg.get("promote_if_better", True)):
            best_metric = soup_metric
            save_checkpoint(
                out_dir / "best.pt",
                model=soup_model,
                model_config=model_config,
                optimizer=optimizer,
                epoch=epochs,
                metrics={**soup_metrics, "epoch": float(epochs), "model_soup": 1.0},
            )
        print(
            "Model soup "
            f"top_k={len(model_soup.entries)} val_loss={soup_metrics['val_loss']:.4f} "
            f"val_top1={soup_metrics['val_top1']:.4f} "
            f"val_top5={soup_metrics['val_top5']:.4f}"
        )
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    write_history_csv(out_dir / "history.csv", history_rows)
    print(f"Best {monitor_key}={best_metric:.4f}")
    print(f"Training complete. Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
