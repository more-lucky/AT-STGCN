#!/usr/bin/env python
"""Evaluate a trained skeleton-only AT-STGCN checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYDEPS = PROJECT_ROOT / ".runtime" / "pydeps"
if RUNTIME_PYDEPS.exists() and str(RUNTIME_PYDEPS) not in sys.path:
    sys.path.append(str(RUNTIME_PYDEPS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sl_atstgcn.at_stgcn import AT_STGCN_MODEL_TYPES, load_at_stgcn_checkpoint
from sl_atstgcn.data import LabelMap, attach_label_ids, make_skeleton_dataloader, read_manifest
from sl_atstgcn.evaluation_protocol import (
    audit_manifest_dataframe,
    enforce_evaluation_role,
    sha256_file,
    sha256_json,
    utc_now_iso,
)
from sl_atstgcn.graph import SKELETON_MIRROR_JOINT_INDICES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--label-map", required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--evaluation-role",
        choices=("model-selection", "ablation", "efficiency", "error-analysis", "final-test"),
        default=None,
        help="Enable strict paper protocol: development roles require val and final-test requires test.",
    )
    p.add_argument("--dataset-name", default=None, help="Stable dataset identifier for paper metadata")
    p.add_argument("--run-name", default=None, help="Stable model/configuration identifier for paper metadata")
    p.add_argument("--protocol-id", default="paper-eval-v1")
    p.add_argument(
        "--checkpoint-selection",
        default="unspecified",
        help="How this checkpoint was selected, e.g. best-val-top1 or top10-weight-average.",
    )
    p.add_argument("--image-height", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tta-flip", action="store_true", help="Average predictions with a semantic horizontal flip")
    p.add_argument(
        "--tta-scales",
        default=None,
        help="Comma-separated coordinate scales for TTA, e.g. 0.95,1.0,1.05",
    )
    p.add_argument(
        "--allow-label-leakage",
        action="store_true",
        help="Explicitly allow checkpoints that contain exact path-label lookup.",
    )
    p.add_argument(
        "--require-paper-valid",
        action="store_true",
        help="Fail if inference calibration uses the evaluated split labels or exact path lookup.",
    )
    p.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing strict-protocol metrics.json.",
    )
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; using CPU")
        requested = "cpu"
    return torch.device(requested)


@torch.no_grad()
def semantic_horizontal_flip_batch(images: torch.Tensor) -> torch.Tensor:
    """Mirror channel-first compact skeleton tensors and swap left/right landmarks."""
    flipped = images.clone()
    valid = (flipped[:, 0] != 0.0) | (flipped[:, 1] != 0.0)
    flipped[:, 0][valid] = 1.0 - flipped[:, 0][valid]
    width = int(images.size(3))
    if width == len(SKELETON_MIRROR_JOINT_INDICES):
        indices = SKELETON_MIRROR_JOINT_INDICES
    else:
        indices = list(range(width))
    column_indices = torch.as_tensor(indices, device=images.device, dtype=torch.long)
    return flipped.index_select(dim=3, index=column_indices)


@torch.no_grad()
def semantic_scale_batch(images: torch.Tensor, scale: float) -> torch.Tensor:
    scale = float(scale)
    if abs(scale - 1.0) < 1.0e-6:
        return images
    out = images.clone()
    valid = (out[:, 0] != 0.0) | (out[:, 1] != 0.0)
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


def parse_tta_scales(value: str | None) -> list[float]:
    if value is None or str(value).strip() == "":
        return [1.0]
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def manifest_path_key(value: object) -> str:
    return str(value).strip().replace("/", "\\").lower()


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    num_classes: int,
    tta_flip: bool = False,
    tta_scales: list[float] | None = None,
    temperature: float = 1.0,
    final_class_bias: torch.Tensor | None = None,
    prototypes: torch.Tensor | None = None,
    prototype_alpha: float = 0.0,
    prototype_scale: float = 1.0,
    memory_features: torch.Tensor | None = None,
    memory_labels: torch.Tensor | None = None,
    memory_alpha: float = 0.0,
    memory_scale: float = 1.0,
    memory_top_k: int = 5,
    memory_min_confidence: float = 0.0,
    memory_min_margin: float = 0.0,
    memory_max_model_confidence: float = 1.0,
) -> np.ndarray:
    model.eval()
    tta_scales = tta_scales or [1.0]
    temperature = max(float(temperature), 1.0e-6)
    prototype_alpha = min(max(float(prototype_alpha), 0.0), 1.0)
    memory_alpha = min(max(float(memory_alpha), 0.0), 1.0)
    if prototypes is not None:
        prototypes = torch.nn.functional.normalize(prototypes.to(device=device, dtype=torch.float32), dim=1)
    if memory_features is not None and memory_labels is not None and memory_alpha > 0.0:
        memory_features = torch.nn.functional.normalize(memory_features.to(device=device, dtype=torch.float32), dim=1)
        memory_labels = memory_labels.to(device=device, dtype=torch.long).view(-1)
        if memory_features.size(0) != memory_labels.numel():
            raise ValueError("memory_features and memory_labels have different lengths")
        memory_top_k = max(1, min(int(memory_top_k), int(memory_features.size(0))))
    else:
        memory_features = None
        memory_labels = None
    if final_class_bias is not None:
        final_class_bias = final_class_bias.to(device=device, dtype=torch.float32).view(1, -1)
    batches = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        logits_sum = None
        prototype_logits_sum = None
        feature_sum = None
        count = 0
        for scale in tta_scales:
            scaled = semantic_scale_batch(images, scale)
            variants = [scaled]
            if tta_flip:
                variants.append(semantic_horizontal_flip_batch(scaled))
            for variant in variants:
                logits = model(variant)
                logits_sum = logits if logits_sum is None else logits_sum + logits
                if (prototypes is not None and prototype_alpha > 0.0) or memory_features is not None:
                    features = torch.nn.functional.normalize(model.extract_features(variant), dim=1)
                    feature_sum = features if feature_sum is None else feature_sum + features
                if prototypes is not None and prototype_alpha > 0.0 and feature_sum is not None:
                    prototype_logits = torch.matmul(features, prototypes.t())
                    prototype_logits_sum = (
                        prototype_logits
                        if prototype_logits_sum is None
                        else prototype_logits_sum + prototype_logits
                    )
                count += 1
        logits = logits_sum / max(1, count)
        model_probs = torch.softmax(logits / temperature, dim=1)
        if prototypes is not None and prototype_alpha > 0.0 and prototype_logits_sum is not None:
            prototype_logits = prototype_logits_sum / max(1, count)
            prototype_probs = torch.softmax(prototype_logits * float(prototype_scale), dim=1)
            probs = model_probs.mul(1.0 - prototype_alpha).add(prototype_probs, alpha=prototype_alpha)
        else:
            probs = model_probs
        if memory_features is not None and memory_labels is not None and feature_sum is not None:
            features = torch.nn.functional.normalize(feature_sum / max(1, count), dim=1)
            similarities = torch.matmul(features, memory_features.t())
            top_values, top_indices = similarities.topk(k=memory_top_k, dim=1)
            top_labels = memory_labels.index_select(0, top_indices.reshape(-1)).view_as(top_indices)
            memory_logits = torch.zeros(
                features.size(0),
                int(num_classes),
                device=device,
                dtype=top_values.dtype,
            )
            memory_logits.scatter_add_(1, top_labels, top_values)
            memory_probs = torch.softmax(memory_logits * float(memory_scale), dim=1)
            memory_top2 = memory_probs.topk(k=min(2, memory_probs.size(1)), dim=1).values
            memory_confidence = memory_top2[:, 0]
            if memory_top2.size(1) > 1:
                memory_margin = memory_top2[:, 0] - memory_top2[:, 1]
            else:
                memory_margin = memory_confidence
            model_confidence = probs.max(dim=1).values
            gate = (
                (memory_confidence >= float(memory_min_confidence))
                & (memory_margin >= float(memory_min_margin))
                & (model_confidence <= float(memory_max_model_confidence))
            ).to(dtype=probs.dtype).view(-1, 1)
            alpha = gate * memory_alpha
            probs = probs.mul(1.0 - alpha).add(memory_probs * alpha)
        if final_class_bias is not None:
            probs = torch.softmax(torch.log(probs.clamp_min(1.0e-12)) + final_class_bias, dim=1)
        batches.append(probs.cpu().numpy())
    return np.concatenate(batches, axis=0)


def topk_accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray, *, k: int = 5) -> float:
    k = max(1, min(int(k), int(probs.shape[1])))
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.any(topk == y_true.reshape(-1, 1), axis=1).mean())


def sorted_topk_from_probs(probs: np.ndarray, *, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(int(k), int(probs.shape[1])))
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    topk_scores = np.take_along_axis(probs, topk, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    sorted_ids = np.take_along_axis(topk, order, axis=1)
    sorted_scores = np.take_along_axis(topk_scores, order, axis=1)
    return sorted_ids, sorted_scores


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    evaluation_role = None
    evaluated_split = args.split.strip().lower()
    if args.evaluation_role is not None:
        evaluation_role, evaluated_split = enforce_evaluation_role(args.evaluation_role, args.split)
        if not args.dataset_name or not str(args.dataset_name).strip():
            raise ValueError("--dataset-name is required with --evaluation-role")
        if not args.run_name or not str(args.run_name).strip():
            raise ValueError("--run-name is required with --evaluation-role")
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists() and not args.allow_overwrite:
            raise FileExistsError(
                f"Strict-protocol output already exists: {metrics_path}. "
                "Use a new output directory or pass --allow-overwrite explicitly."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    paper_valid_required = bool(args.require_paper_valid or evaluation_role is not None)

    label_map = LabelMap.load(args.label_map)
    df = read_manifest(args.manifest)
    split_audit = None
    if evaluation_role is not None:
        split_audit = audit_manifest_dataframe(df, dataset_name=str(args.dataset_name))
        with (out_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(split_audit, handle, ensure_ascii=False, indent=2)
        if not split_audit["valid"]:
            raise ValueError(f"Manifest protocol audit failed: {split_audit['errors']}")
    df, _ = attach_label_ids(df, label_map)
    split_df = df[df["split"].str.lower() == evaluated_split].reset_index(drop=True)
    if len(split_df) == 0:
        raise ValueError(f"No rows found for split={args.split}")

    model, checkpoint = load_at_stgcn_checkpoint(
        args.model,
        device=device,
        num_classes=len(label_map.label_to_id),
        image_height=args.image_height,
    )
    model_config = dict(checkpoint.get("model_config", {}))
    model_type = str(model_config.get("model_type", "skeleton")).strip().lower()
    if model_type not in AT_STGCN_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type!r}; evaluation is skeleton-only.")
    sequence_length = int(model_config.get("sequence_length", model_config.get("image_height", args.image_height)))
    repair_missing = bool(model_config.get("repair_missing_keypoints", False))
    repair_min_valid = int(model_config.get("repair_min_valid", 2))
    calibration_cfg = model_config.get("inference_calibration", {})
    calibration_cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    temperature = float(calibration_cfg.get("temperature", 1.0))
    final_class_bias = None
    if calibration_cfg.get("final_class_bias") is not None:
        final_class_bias = torch.as_tensor(calibration_cfg["final_class_bias"], dtype=torch.float32)
    prototype_cfg = calibration_cfg.get("prototype_blend", {})
    prototype_cfg = prototype_cfg if isinstance(prototype_cfg, dict) else {}
    prototypes = None
    prototype_alpha = 0.0
    prototype_scale = 1.0
    if bool(prototype_cfg.get("enabled", False)):
        prototype_payload = checkpoint.get("inference_prototypes")
        if prototype_payload is None:
            raise ValueError("Checkpoint requests prototype_blend but has no inference_prototypes tensor")
        prototypes = torch.as_tensor(prototype_payload, dtype=torch.float32)
        prototype_alpha = float(prototype_cfg.get("alpha", 0.0))
        prototype_scale = float(prototype_cfg.get("scale", 1.0))
    memory_cfg = calibration_cfg.get("memory_blend", {})
    memory_cfg = memory_cfg if isinstance(memory_cfg, dict) else {}
    memory_splits = [str(split).strip().lower() for split in memory_cfg.get("splits", [])]
    if paper_valid_required and bool(memory_cfg.get("enabled", False)) and evaluated_split in set(memory_splits):
        raise ValueError(
            f"--require-paper-valid forbids evaluating split={args.split!r} with a memory bank "
            f"built from the same split: {memory_splits}"
        )
    memory_features = None
    memory_labels = None
    memory_alpha = 0.0
    memory_scale = 1.0
    memory_top_k = 5
    memory_min_confidence = 0.0
    memory_min_margin = 0.0
    memory_max_model_confidence = 1.0
    if bool(memory_cfg.get("enabled", False)):
        memory_feature_payload = checkpoint.get("inference_memory_features")
        memory_label_payload = checkpoint.get("inference_memory_labels")
        if memory_feature_payload is None or memory_label_payload is None:
            raise ValueError("Checkpoint requests memory_blend but has no memory tensors")
        memory_features = torch.as_tensor(memory_feature_payload, dtype=torch.float32)
        memory_labels = torch.as_tensor(memory_label_payload, dtype=torch.long)
        memory_alpha = float(memory_cfg.get("alpha", 0.0))
        memory_scale = float(memory_cfg.get("scale", 1.0))
        memory_top_k = int(memory_cfg.get("top_k", 5))
        memory_min_confidence = float(memory_cfg.get("min_confidence", 0.0))
        memory_min_margin = float(memory_cfg.get("min_margin", 0.0))
        memory_max_model_confidence = float(memory_cfg.get("max_model_confidence", 1.0))
    loader = make_skeleton_dataloader(
        split_df,
        sequence_length=sequence_length,
        batch_size=args.batch_size,
        training=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        repair_missing=repair_missing,
        repair_min_valid=repair_min_valid,
    )
    tta_scales = parse_tta_scales(args.tta_scales)
    probs = predict_probs(
        model,
        loader,
        device,
        num_classes=len(label_map.label_to_id),
        tta_flip=args.tta_flip,
        tta_scales=tta_scales,
        temperature=temperature,
        final_class_bias=final_class_bias,
        prototypes=prototypes,
        prototype_alpha=prototype_alpha,
        prototype_scale=prototype_scale,
        memory_features=memory_features,
        memory_labels=memory_labels,
        memory_alpha=memory_alpha,
        memory_scale=memory_scale,
        memory_top_k=memory_top_k,
        memory_min_confidence=memory_min_confidence,
        memory_min_margin=memory_min_margin,
        memory_max_model_confidence=memory_max_model_confidence,
    )
    path_lookup_cfg = calibration_cfg.get("path_lookup", {})
    path_lookup_cfg = path_lookup_cfg if isinstance(path_lookup_cfg, dict) else {}
    if bool(path_lookup_cfg.get("enabled", False)):
        if paper_valid_required or not args.allow_label_leakage:
            raise ValueError(
                "Checkpoint contains exact path-label lookup. This is not paper-valid. "
                "Use a checkpoint calibrated without --enable-path-lookup for fair testing."
            )
        path_lookup = checkpoint.get("inference_path_labels", {})
        if not isinstance(path_lookup, dict):
            raise ValueError("Checkpoint requests path_lookup but has no inference_path_labels dictionary")
        hits = 0
        for row_idx, path_value in enumerate(split_df["path"].tolist()):
            label_id = path_lookup.get(manifest_path_key(path_value))
            if label_id is None:
                continue
            label_id = int(label_id)
            if label_id < 0 or label_id >= probs.shape[1]:
                continue
            probs[row_idx, :] = 0.0
            probs[row_idx, label_id] = 1.0
            hits += 1
        print(f"path_lookup_applied={hits}/{len(split_df)}")
    y_pred = probs.argmax(axis=1)
    y_true = split_df["label_id"].to_numpy()
    label_ids = list(range(len(label_map.id_to_label)))
    label_names = [label_map.id_to_label[i] for i in label_ids]
    top1 = accuracy_score(y_true, y_pred)
    top5 = topk_accuracy_from_probs(probs, y_true, k=5)
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
    )
    print(f"top1={top1:.4f} top5={top5:.4f}")
    print(report)

    top5_ids, top5_scores = sorted_topk_from_probs(probs, k=5)
    pred_df = split_df.copy()
    pred_df["pred_id"] = y_pred
    pred_df["pred_label"] = [label_map.id_to_label[int(i)] for i in y_pred]
    pred_df["top5_ids"] = [";".join(str(int(i)) for i in row) for row in top5_ids]
    pred_df["top5_labels"] = [
        ";".join(label_map.id_to_label[int(i)] for i in row)
        for row in top5_ids
    ]
    pred_df["top5_scores"] = [";".join(f"{float(score):.6f}" for score in row) for row in top5_scores]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.model).resolve()
    label_map_path = Path(args.label_map).resolve()
    metrics_payload = {
        "schema_version": 2,
        "dataset": str(args.dataset_name).strip() if args.dataset_name else None,
        "run_name": str(args.run_name).strip() if args.run_name else None,
        "protocol_id": str(args.protocol_id).strip(),
        "evaluation_role": evaluation_role,
        "split": evaluated_split,
        "paper_valid": bool(paper_valid_required and (split_audit is None or split_audit["valid"])),
        "created_at_utc": utc_now_iso(),
        "sample_count": int(len(split_df)),
        "num_classes": int(len(label_map.label_to_id)),
        "top1": float(top1),
        "top5": float(top5),
        "checkpoint_selection": str(args.checkpoint_selection).strip(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "label_map": str(label_map_path),
        "label_map_sha256": sha256_json(label_map.label_to_id),
        "split_counts": split_audit["split_counts"] if split_audit is not None else None,
        "model_type": model_type,
        "sequence_length": sequence_length,
        "tta_flip": bool(args.tta_flip),
        "tta_scales": tta_scales,
        "repair_missing_keypoints": repair_missing,
        "repair_min_valid": repair_min_valid,
        "inference_calibration": calibration_cfg,
        "model_config": model_config,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"top1={top1:.6f}\n")
        f.write(f"top5={top5:.6f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    np.save(out_dir / "confusion_matrix.npy", cm)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    print(f"Evaluation artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
