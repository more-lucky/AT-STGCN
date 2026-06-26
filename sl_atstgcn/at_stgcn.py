"""Public AT-STGCN API for the skeleton-only paper implementation.

Training, evaluation, prediction, and experiment scripts import from this
module so the paper-facing surface stays skeleton-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import (
    SkeletonOnlyClassifier,
    SkeletonSequenceFeatureEncoder,
    build_skeleton_classifier,
    create_optimizer,
    load_skeleton_checkpoint,
    save_checkpoint,
)


AT_STGCN_MODEL_TYPES = {"skeleton", "skeleton_only", "stgcn", "st-gcn", "at_stgcn", "at-stgcn"}


def is_at_stgcn_model_type(value: object) -> bool:
    return str(value).strip().lower() in AT_STGCN_MODEL_TYPES


def build_at_stgcn_classifier(*args: Any, **kwargs: Any) -> SkeletonOnlyClassifier:
    """Build the paper-facing skeleton-only AT-STGCN classifier."""
    return build_skeleton_classifier(*args, **kwargs)


def load_at_stgcn_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    num_classes: int | None = None,
    image_height: int | None = None,
    dropout: float | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a skeleton-only AT-STGCN checkpoint."""
    model, checkpoint = load_skeleton_checkpoint(
        path,
        device=device,
        num_classes=num_classes,
        image_height=image_height,
        dropout=dropout,
    )
    config = dict(checkpoint.get("model_config", {}))
    model_type = config.get("model_type", "skeleton")
    if not is_at_stgcn_model_type(model_type):
        raise ValueError(
            f"{path} is a non-AT-STGCN {model_type!r} checkpoint. "
            "This entry point accepts only skeleton-only AT-STGCN checkpoints."
        )
    return model, checkpoint
