#!/usr/bin/env python
"""Run AT-STGCN inference on one raw video using MediaPipe keypoints."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from sl_atstgcn.data import LabelMap
from sl_atstgcn.data import (
    prepare_skeleton_sequence,
    repair_missing_skeleton_keypoints,
)
from sl_atstgcn.extractor import ExtractionConfig, extract_selected_keypoint_sequence
from sl_atstgcn.at_stgcn import AT_STGCN_MODEL_TYPES, load_at_stgcn_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label-map", required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; using CPU")
        requested = "cpu"
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    label_map = LabelMap.load(args.label_map)
    seq = extract_selected_keypoint_sequence(args.video, ExtractionConfig())

    model, checkpoint = load_at_stgcn_checkpoint(
        args.model,
        device=device,
        num_classes=len(label_map.label_to_id),
        image_height=args.height,
    )
    model_config = dict(checkpoint.get("model_config", {}))
    model_type = str(model_config.get("model_type", "skeleton")).strip().lower()
    if model_type not in AT_STGCN_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type!r}; prediction is skeleton-only.")
    repair_min_valid = int(model_config.get("repair_min_valid", 2))
    sequence_length = int(model_config.get("sequence_length", model_config.get("image_height", args.height)))
    sequence = prepare_skeleton_sequence(seq, sequence_length=sequence_length)
    if bool(model_config.get("repair_missing_keypoints", False)):
        sequence = repair_missing_skeleton_keypoints(sequence, min_valid=repair_min_valid)
    tensor = torch.from_numpy(np.ascontiguousarray(sequence.transpose(2, 0, 1))).unsqueeze(0).float().to(device)
    model.eval()
    calibration_cfg = model_config.get("inference_calibration", {})
    calibration_cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    temperature = max(float(calibration_cfg.get("temperature", 1.0)), 1.0e-6)
    prototype_cfg = calibration_cfg.get("prototype_blend", {})
    prototype_cfg = prototype_cfg if isinstance(prototype_cfg, dict) else {}
    memory_cfg = calibration_cfg.get("memory_blend", {})
    memory_cfg = memory_cfg if isinstance(memory_cfg, dict) else {}
    with torch.no_grad():
        logits = model(tensor)
        probs_t = torch.softmax(logits / temperature, dim=1)
        feature = None
        if bool(prototype_cfg.get("enabled", False)):
            prototypes = checkpoint.get("inference_prototypes")
            if prototypes is not None:
                feature = torch.nn.functional.normalize(model.extract_features(tensor), dim=1)
                prototypes_t = torch.nn.functional.normalize(
                    torch.as_tensor(prototypes, device=device, dtype=torch.float32),
                    dim=1,
                )
                proto_probs = torch.softmax(feature @ prototypes_t.t() * float(prototype_cfg.get("scale", 1.0)), dim=1)
                alpha = min(max(float(prototype_cfg.get("alpha", 0.0)), 0.0), 1.0)
                probs_t = probs_t.mul(1.0 - alpha).add(proto_probs, alpha=alpha)
        if bool(memory_cfg.get("enabled", False)):
            memory_features = checkpoint.get("inference_memory_features")
            memory_labels = checkpoint.get("inference_memory_labels")
            if memory_features is not None and memory_labels is not None:
                if feature is None:
                    feature = torch.nn.functional.normalize(model.extract_features(tensor), dim=1)
                memory_features_t = torch.nn.functional.normalize(
                    torch.as_tensor(memory_features, device=device, dtype=torch.float32),
                    dim=1,
                )
                memory_labels_t = torch.as_tensor(memory_labels, device=device, dtype=torch.long).view(-1)
                top_k_memory = max(1, min(int(memory_cfg.get("top_k", 5)), int(memory_features_t.size(0))))
                values, indices = (feature @ memory_features_t.t()).topk(k=top_k_memory, dim=1)
                labels = memory_labels_t.index_select(0, indices.reshape(-1)).view_as(indices)
                memory_logits = torch.zeros(1, len(label_map.label_to_id), device=device, dtype=values.dtype)
                memory_logits.scatter_add_(1, labels, values)
                memory_probs = torch.softmax(memory_logits * float(memory_cfg.get("scale", 1.0)), dim=1)
                alpha = min(max(float(memory_cfg.get("alpha", 0.0)), 0.0), 1.0)
                probs_t = probs_t.mul(1.0 - alpha).add(memory_probs, alpha=alpha)
        if calibration_cfg.get("final_class_bias") is not None:
            bias = torch.as_tensor(calibration_cfg["final_class_bias"], device=device, dtype=torch.float32).view(1, -1)
            probs_t = torch.softmax(torch.log(probs_t.clamp_min(1.0e-12)) + bias, dim=1)
        probs = probs_t.cpu().numpy()[0]
    top = np.argsort(probs)[::-1][: args.top_k]
    for idx in top:
        print(f"{label_map.id_to_label[int(idx)]}\t{float(probs[idx]):.6f}")


if __name__ == "__main__":
    main()
