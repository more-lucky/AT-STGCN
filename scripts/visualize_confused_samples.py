#!/usr/bin/env python
"""Visualize representative samples from a confused class pair.

The script reads an evaluation ``predictions.csv`` file, finds the most frequent
``true -> predicted`` mistake by default, and creates a grid of sampled frames.
If RGB videos are available through ``source_path``/``video_path`` or an
optional manifest, each sample is shown as an RGB row followed by a skeleton row.
When videos are unavailable, the script still renders the skeleton sequence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYDEPS = PROJECT_ROOT / ".runtime" / "pydeps"
if RUNTIME_PYDEPS.exists() and str(RUNTIME_PYDEPS) not in sys.path:
    sys.path.append(str(RUNTIME_PYDEPS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # OpenCV is optional; skeleton-only visualizations still work without it.
    import cv2
except Exception:  # pragma: no cover - depends on local environment
    cv2 = None

from sl_atstgcn.data import prepare_skeleton_sequence, resolve_portable_path
from sl_atstgcn.graph import JOINT_INDEX, paper_tree_edges


SKELETON_EDGES = [(JOINT_INDEX[a], JOINT_INDEX[b]) for a, b in paper_tree_edges()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, help="Path to evaluation predictions.csv")
    p.add_argument("--output", required=True, help="Output image path, e.g. figures/confused_6_79.png")
    p.add_argument("--manifest", default=None, help="Optional manifest CSV used to recover source video paths")
    p.add_argument("--true-id", type=int, default=None, help="True label id to visualize")
    p.add_argument("--pred-id", type=int, default=None, help="Predicted label id to visualize")
    p.add_argument("--top-pair", action="store_true", help="Use the most frequent confused pair")
    p.add_argument("--samples", type=int, default=3, help="Number of misclassified samples to visualize")
    p.add_argument(
        "--contrast-triplet",
        action="store_true",
        help=(
            "Visualize a contrastive triplet: one correctly predicted true class sample, "
            "one correctly predicted predicted-class sample, and one true->pred mistake."
        ),
    )
    p.add_argument("--frame-indices", default="1,9,17,24,32", help="1-based frame indices in the model sequence")
    p.add_argument("--sequence-length", type=int, default=64, help="Skeleton sequence length used for rendering")
    p.add_argument("--video-root", action="append", default=[], help="Extra roots for resolving source videos")
    p.add_argument("--skeleton-root", action="append", default=[], help="Extra roots for resolving skeleton .npy files")
    p.add_argument(
        "--label-display",
        choices=("name", "id", "numeric", "both"),
        default="name",
        help=(
            "How to display labels in the figure. 'numeric' keeps numeric dataset labels "
            "when available, otherwise falls back to label_id/pred_id."
        ),
    )
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def norm_key(value: object) -> str:
    return str(value).strip().replace("\\", "/").lower()


def first_existing(value: object, roots: Iterable[str | Path] = ()) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    candidates = [Path(resolve_portable_path(text))]
    raw = Path(text)
    candidates.append(raw)
    normalized = Path(text.replace("\\", "/"))
    candidates.append(normalized)
    basename = normalized.name
    suffix_parts = []
    normalized_str = text.replace("\\", "/")
    for marker in ("/data/", "/runs/", "/videos/"):
        idx = normalized_str.lower().rfind(marker)
        if idx >= 0:
            suffix_parts.append(normalized_str[idx + 1 :])
    for root in roots:
        root_path = Path(root)
        if basename:
            candidates.append(root_path / basename)
        for suffix in suffix_parts:
            candidates.append(root_path / suffix)
            if suffix.startswith("data/"):
                candidates.append(root_path / suffix.removeprefix("data/"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_manifest_video_lookup(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    video_col = None
    for col in ("source_path", "video_path", "rgb_path"):
        if col in df.columns:
            video_col = col
            break
    if video_col is None:
        return {}
    lookup: dict[str, str] = {}
    video_only = "keypoints_path" not in df.columns and "path" not in df.columns
    for _, row in df.iterrows():
        keys = []
        for col in ("keypoints_path", "path"):
            if col in df.columns:
                keys.append(norm_key(row[col]))
                keys.append(Path(str(row[col])).name.lower())
        if video_only:
            video_name = Path(str(row[video_col])).name
            video_stem = Path(video_name).stem
            keys.append(video_name.lower())
            keys.append(video_stem.lower())
            if "label" in df.columns:
                keys.append(f"{str(row['label']).strip().lower()}::{video_stem.lower()}")
        for key in keys:
            if key:
                lookup[key] = str(row[video_col])
    return lookup


def video_lookup_keys_from_keypoints(path_value: object, label_value: object | None = None) -> list[str]:
    if path_value is None or pd.isna(path_value):
        return []
    name = Path(str(path_value)).name
    stem = Path(name).stem
    keys = [norm_key(path_value), name.lower(), stem.lower()]
    # Preprocessed skeleton names often append an extraction index, e.g.
    # 1512781787144335-SAIL_040154.npy -> 1512781787144335-SAIL.mp4.
    if "_" in stem:
        video_stem = stem.rsplit("_", 1)[0]
        keys.append(video_stem.lower())
        if label_value is not None and not pd.isna(label_value):
            keys.append(f"{str(label_value).strip().lower()}::{video_stem.lower()}")
    return keys


def select_confusion_pair(df: pd.DataFrame, true_id: int | None, pred_id: int | None) -> tuple[int, int, int]:
    mistakes = df[df["label_id"].astype(int) != df["pred_id"].astype(int)].copy()
    if mistakes.empty:
        raise ValueError("No misclassified rows found in predictions.csv")
    if true_id is not None and pred_id is not None:
        count = int(((mistakes["label_id"].astype(int) == true_id) & (mistakes["pred_id"].astype(int) == pred_id)).sum())
        if count == 0:
            raise ValueError(f"No mistakes found for true {true_id} -> predicted {pred_id}")
        return int(true_id), int(pred_id), count
    grouped = (
        mistakes.groupby(["label_id", "pred_id"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    row = grouped.iloc[0]
    return int(row["label_id"]), int(row["pred_id"]), int(row["size"])


def prediction_confidence(row: pd.Series) -> float:
    if "top5_scores" in row and not pd.isna(row["top5_scores"]):
        first_score = str(row["top5_scores"]).strip().split(";")[0]
        if first_score:
            try:
                return float(first_score)
            except ValueError:
                pass
    for col in ("score", "confidence", "pred_score"):
        if col in row and not pd.isna(row[col]):
            try:
                return float(row[col])
            except ValueError:
                pass
    return 0.0


def select_top_rows(df: pd.DataFrame, mask: pd.Series, count: int, description: str) -> pd.DataFrame:
    rows = df[mask].copy()
    if rows.empty:
        raise ValueError(f"No rows found for {description}")
    rows["_select_confidence"] = rows.apply(prediction_confidence, axis=1)
    rows = rows.sort_values("_select_confidence", ascending=False).head(max(1, int(count)))
    return rows.drop(columns=["_select_confidence"])


def select_visualization_rows(
    predictions: pd.DataFrame,
    true_id: int,
    pred_id: int,
    *,
    samples: int,
    contrast_triplet: bool,
) -> pd.DataFrame:
    label_ids = predictions["label_id"].astype(int)
    pred_ids = predictions["pred_id"].astype(int)
    if not contrast_triplet:
        return predictions[(label_ids == true_id) & (pred_ids == pred_id)].head(max(1, int(samples))).copy()

    true_correct = select_top_rows(
        predictions,
        (label_ids == true_id) & (pred_ids == true_id),
        1,
        f"correct predictions for class {true_id}",
    )
    pred_correct = select_top_rows(
        predictions,
        (label_ids == pred_id) & (pred_ids == pred_id),
        1,
        f"correct predictions for class {pred_id}",
    )
    confused = select_top_rows(
        predictions,
        (label_ids == true_id) & (pred_ids == pred_id),
        1,
        f"mistakes for true {true_id} -> predicted {pred_id}",
    )
    true_correct["_display_group"] = "correct class A"
    pred_correct["_display_group"] = "correct class B"
    confused["_display_group"] = "A -> B error"
    return pd.concat([true_correct, pred_correct, confused], ignore_index=True)


def load_skeleton(path_value: object, sequence_length: int, roots: Iterable[str | Path]) -> np.ndarray | None:
    path = first_existing(path_value, roots)
    if path is None:
        return None
    try:
        arr = np.load(str(path)).astype(np.float32)
        return prepare_skeleton_sequence(arr, sequence_length=sequence_length)
    except Exception as exc:
        print(f"WARNING: failed to load skeleton {path}: {exc}")
        return None


def render_skeleton(frame: np.ndarray, *, size: int = 180) -> np.ndarray:
    if cv2 is not None:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        xy = frame[:, :2].astype(np.float32)
        valid = np.isfinite(xy).all(axis=1) & ((xy[:, 0] != 0.0) | (xy[:, 1] != 0.0))
        pts = np.zeros((xy.shape[0], 2), dtype=np.int32)
        pts[:, 0] = np.clip((xy[:, 0] * (size - 1)).round(), 0, size - 1).astype(np.int32)
        pts[:, 1] = np.clip((xy[:, 1] * (size - 1)).round(), 0, size - 1).astype(np.int32)
        for a, b in SKELETON_EDGES:
            if valid[a] and valid[b]:
                cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (220, 220, 220), 2, lineType=cv2.LINE_AA)
        for idx, point in enumerate(pts):
            if valid[idx]:
                color = (255, 255, 255)
                cv2.circle(canvas, tuple(point), 3, color, -1, lineType=cv2.LINE_AA)
        return canvas

    fig = plt.figure(figsize=(2, 2), dpi=size // 2)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    xy = frame[:, :2]
    valid = np.isfinite(xy).all(axis=1) & ((xy[:, 0] != 0.0) | (xy[:, 1] != 0.0))
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color="white", linewidth=1.5)
    ax.scatter(xy[valid, 0], xy[valid, 1], s=8, c="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis("off")
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def read_video_frames(path_value: object, frame_indices: list[int], sequence_length: int, roots: Iterable[str | Path]) -> list[np.ndarray] | None:
    if cv2 is None:
        return None
    path = first_existing(path_value, roots)
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None
    frames: list[np.ndarray] = []
    for seq_idx in frame_indices:
        rel = 0.0 if sequence_length <= 1 else (max(1, seq_idx) - 1) / float(sequence_length - 1)
        frame_id = int(round(rel * (total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            frames.append(np.zeros((180, 180, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def is_numeric_label(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def label_name(row: pd.Series, id_col: str, label_col: str, *, mode: str = "name") -> str:
    id_text = str(int(row[id_col]))
    has_name = label_col in row and not pd.isna(row[label_col]) and str(row[label_col]).strip() != ""
    name_text = str(row[label_col]).strip() if has_name else id_text
    if mode == "id":
        return id_text
    if mode == "numeric":
        return name_text if is_numeric_label(name_text) else id_text
    if mode == "both":
        return name_text if name_text == id_text else f"{name_text} ({id_text})"
    return name_text


def sample_row_label(sample_idx: int, row: pd.Series, suffix: str, *, label_display: str) -> str:
    true_label = label_name(row, "label_id", "label", mode=label_display)
    pred_label = label_name(row, "pred_id", "pred_label", mode=label_display)
    group = row.get("_display_group", None)
    prefix = str(group) if group is not None and not pd.isna(group) else f"sample {sample_idx}"
    if str(true_label) == str(pred_label):
        status = f"class {true_label}"
    else:
        status = f"true {true_label}\npred {pred_label}"
    return f"{prefix}\n{status}\n{suffix}"


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    required = {"label_id", "pred_id"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{args.predictions} missing columns: {sorted(missing)}")

    video_lookup = read_manifest_video_lookup(args.manifest)
    true_id, pred_id, count = select_confusion_pair(
        predictions,
        true_id=args.true_id,
        pred_id=args.pred_id,
    )
    selected = select_visualization_rows(
        predictions,
        true_id,
        pred_id,
        samples=args.samples,
        contrast_triplet=bool(args.contrast_triplet),
    )
    if selected.empty:
        raise ValueError("No selected samples")

    frame_indices = [int(x.strip()) for x in args.frame_indices.split(",") if x.strip()]
    skeleton_roots = [Path(x) for x in args.skeleton_root]
    video_roots = [Path(x) for x in args.video_root]

    rows: list[tuple[str, list[np.ndarray]]] = []
    pair_rows = predictions[
        (predictions["label_id"].astype(int) == true_id) & (predictions["pred_id"].astype(int) == pred_id)
    ]
    pair_row = pair_rows.iloc[0] if not pair_rows.empty else selected.iloc[-1]
    true_label = label_name(pair_row, "label_id", "label", mode=args.label_display)
    pred_label = label_name(pair_row, "pred_id", "pred_label", mode=args.label_display)

    for sample_idx, (_, row) in enumerate(selected.iterrows(), start=1):
        key_path = row.get("keypoints_path", row.get("path", ""))
        skeleton = load_skeleton(key_path, args.sequence_length, skeleton_roots)
        video_value = row.get("source_path", row.get("video_path", None))
        if video_value is None or pd.isna(video_value):
            for key in video_lookup_keys_from_keypoints(key_path, row.get("label", None)):
                video_value = video_lookup.get(key)
                if video_value:
                    break
        rgb_frames = read_video_frames(video_value, frame_indices, args.sequence_length, video_roots)
        if rgb_frames is not None:
            rows.append((sample_row_label(sample_idx, row, "RGB", label_display=args.label_display), rgb_frames))
        if skeleton is not None:
            skeleton_frames = []
            for idx in frame_indices:
                frame_idx = min(max(idx - 1, 0), skeleton.shape[0] - 1)
                skeleton_frames.append(render_skeleton(skeleton[frame_idx]))
            rows.append((sample_row_label(sample_idx, row, "skeleton", label_display=args.label_display), skeleton_frames))

    if not rows:
        raise RuntimeError("No RGB or skeleton frames could be rendered. Check path roots.")

    fig_w = max(8.0, len(frame_indices) * 2.35)
    fig_h = max(3.0, len(rows) * 1.65 + 0.8)
    fig, axes = plt.subplots(len(rows), len(frame_indices), figsize=(fig_w, fig_h), squeeze=False)
    if args.contrast_triplet:
        title = f"Class contrast: {true_label} correct / {pred_label} correct / {true_label}->{pred_label} error (mistakes={count})"
    else:
        title = f"Confused class pair: true {true_label} -> predicted {pred_label} (count={count})"
    fig.suptitle(title, fontsize=14)
    for r, (row_name, images) in enumerate(rows):
        for c, image in enumerate(images):
            ax = axes[r, c]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"t={frame_indices[c]}", fontsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=8.5, rotation=0, labelpad=62, va="center")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
