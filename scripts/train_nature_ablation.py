#!/usr/bin/env python
"""Train optional Nature-inspired ASL skeleton ablation models.

This script reuses ``scripts/train.py`` without editing it.  It monkey-patches
only the model factory for configs that set one of these variants:

1. ``nature_1_feature_gate``
2. ``nature_2_adaptive_temporal``
3. ``nature_3_dual_graph``
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.model_nature_ablation import build_nature_model_from_config


NATURE_VARIANTS = {
    "1",
    "2",
    "3",
    "feature_gate",
    "adaptive_temporal",
    "dual_graph",
    "nature_1_feature_gate",
    "nature_2_adaptive_temporal",
    "nature_3_dual_graph",
}


def _load_base_train_module():
    spec = importlib.util.spec_from_file_location("_sl_atstgcn_base_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base training script: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    train_module = _load_base_train_module()
    original_build_model_from_config = train_module.build_model_from_config

    def patched_build_model_from_config(cfg: dict, *, num_classes: int, image_height: int):
        variant = str(cfg.get("model_variant", "")).strip().lower()
        if variant in NATURE_VARIANTS:
            return build_nature_model_from_config(cfg, num_classes=num_classes, image_height=image_height)
        return original_build_model_from_config(cfg, num_classes=num_classes, image_height=image_height)

    train_module.build_model_from_config = patched_build_model_from_config
    train_module.main()


if __name__ == "__main__":
    main()
