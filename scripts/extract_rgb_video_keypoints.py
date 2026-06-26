#!/usr/bin/env python
"""Extract paper-aligned 68-point skeletons from one RGB sign-language video.

Outputs:
  - *_keypoints_raw.npy: extracted valid skeleton sequence, shape (T_valid, 68, 3)
  - *_keypoints_64.npy: fixed-length sequence for the paper model, shape (64, 68, 3)
  - *_frame_indices.npy: original 0-based RGB frame indices kept by MediaPipe
  - *_rgb_video_frames.png/pdf/svg: paper-style RGB video-frame stack
  - *_skeleton_estimation.png/pdf/svg: paper-style extracted skeleton sequence
  - *_metadata.json: extraction settings and output paths

Example:
  python scripts/extract_rgb_video_keypoints.py \
      --video data/demo/sample.mp4 \
      --out-dir outputs/demo_keypoints \
      --sequence-length 64 \
      --show-progress
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.extractor import ExtractionConfig, validate_extraction_backend
from sl_atstgcn.graph import paper_tree_edges
from sl_atstgcn.keypoints import JOINT_INDEX, selected_keypoints_from_holistic_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 68 MediaPipe Holistic keypoints from an RGB video and draw separate paper-style RGB and skeleton figures."
    )
    parser.add_argument("--video", required=True, help="Input RGB video path, e.g. signer0_sample1_color.mp4")
    parser.add_argument("--out-dir", default="outputs/rgb_skeleton", help="Directory for .npy and figure outputs")
    parser.add_argument("--name", default=None, help="Output file prefix. Default: input video stem")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Fixed temporal length saved for the paper model. Use 0 to skip fixed-length output.",
    )
    parser.add_argument("--target-fps", type=float, default=None, help="Optional sampling FPS before keypoint extraction")
    parser.add_argument("--frame-start", type=int, default=None, help="Optional 1-based inclusive start frame")
    parser.add_argument("--frame-end", type=int, default=None, help="Optional 1-based inclusive end frame")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--static-image-mode", action="store_true")
    parser.add_argument("--no-refine-face-landmarks", action="store_true")
    parser.add_argument(
        "--holistic-task-model",
        default=None,
        help="Optional MediaPipe HolisticLandmarker .task model for environments without mp.solutions.holistic.",
    )
    parser.add_argument("--visual-frames", type=int, default=5, help="Number of frames shown in the visualization")
    parser.add_argument("--dpi", type=int, default=240, help="Figure resolution")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--no-figure", action="store_true", help="Skip RGB and skeleton visualization figures")
    parser.add_argument(
        "--image-only-figures",
        action="store_true",
        help="Remove all titles, axis labels, arrows, and captions from exported RGB/skeleton figures.",
    )
    return parser.parse_args()


def frame_bound_to_index(value: int | None, *, default: int | None) -> int | None:
    if value is None or int(value) < 0:
        return default
    # Dataset annotations are usually 1-based and inclusive.
    return max(0, int(value) - 1)


def video_info(video_path: str | Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def iter_video_frames_with_indices(
    video_path: str | Path,
    *,
    target_fps: float | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> Iterable[tuple[int, np.ndarray]]:
    """Yield ``(original_frame_index, RGB frame)`` pairs.

    ``frame_start`` and ``frame_end`` follow common dataset convention:
    1-based and inclusive. Internally saved frame indices are 0-based.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1
    if target_fps is not None and source_fps > target_fps > 0:
        step = max(1, int(round(source_fps / target_fps)))

    start_idx = frame_bound_to_index(frame_start, default=0) or 0
    end_idx = frame_bound_to_index(frame_end, default=None)
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx < start_idx:
                idx += 1
                continue
            if end_idx is not None and idx > end_idx:
                break
            if (idx - start_idx) % step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield idx, frame_rgb
            idx += 1
    finally:
        cap.release()


class _LandmarkList:
    def __init__(self, landmarks) -> None:
        self.landmark = landmarks


class _HolisticResults:
    def __init__(self, result) -> None:
        self.pose_landmarks = _LandmarkList(result.pose_landmarks) if getattr(result, "pose_landmarks", None) else None
        self.face_landmarks = _LandmarkList(result.face_landmarks) if getattr(result, "face_landmarks", None) else None
        self.left_hand_landmarks = (
            _LandmarkList(result.left_hand_landmarks) if getattr(result, "left_hand_landmarks", None) else None
        )
        self.right_hand_landmarks = (
            _LandmarkList(result.right_hand_landmarks) if getattr(result, "right_hand_landmarks", None) else None
        )


def wrap_task_holistic_result(result) -> _HolisticResults:
    return _HolisticResults(result)


def extract_keypoints_with_frame_indices(
    video_path: str | Path,
    cfg: ExtractionConfig,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract selected 68-keypoint skeletons and exact original frame indices."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("mediapipe is required. Install it with: pip install mediapipe==0.10.14") from exc

    validate_extraction_backend(cfg)
    frames = list(
        iter_video_frames_with_indices(
            video_path,
            target_fps=cfg.target_fps,
            frame_start=frame_start,
            frame_end=frame_end,
        )
    )
    iterator = tqdm(frames, desc=f"Extract {Path(video_path).name}") if show_progress else frames

    selected: list[np.ndarray] = []
    original_indices: list[int] = []

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
        with mp.solutions.holistic.Holistic(
            static_image_mode=cfg.static_image_mode,
            model_complexity=cfg.model_complexity,
            refine_face_landmarks=cfg.refine_face_landmarks,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        ) as holistic:
            for frame_index, frame_rgb in iterator:
                results = holistic.process(frame_rgb)
                points = selected_keypoints_from_holistic_results(results)
                if points is None:
                    continue
                selected.append(points)
                original_indices.append(int(frame_index))
    elif cfg.holistic_task_model:
        task_model = Path(cfg.holistic_task_model)
        if not task_model.exists():
            raise FileNotFoundError(f"MediaPipe Holistic task model does not exist: {task_model}")
        vision = mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task_model)),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_landmarks_confidence=cfg.min_detection_confidence,
            min_pose_detection_confidence=cfg.min_detection_confidence,
            min_pose_landmarks_confidence=cfg.min_tracking_confidence,
            min_hand_landmarks_confidence=cfg.min_tracking_confidence,
        )
        source_fps = float(video_info(video_path)["fps"] or cfg.target_fps or 30.0)
        with vision.HolisticLandmarker.create_from_options(options) as holistic:
            for frame_index, frame_rgb in iterator:
                timestamp_ms = int(round(1000.0 * frame_index / source_fps))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
                task_result = holistic.detect_for_video(image, timestamp_ms)
                points = selected_keypoints_from_holistic_results(wrap_task_holistic_result(task_result))
                if points is None:
                    continue
                selected.append(points)
                original_indices.append(int(frame_index))
    else:
        validate_extraction_backend(cfg)

    if not selected:
        return np.empty((0, 68, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(selected).astype(np.float32), np.asarray(original_indices, dtype=np.int64)


def resize_or_pad_skeleton(sequence: np.ndarray, length: int) -> np.ndarray:
    """Resize long sequences by uniform sampling; pad short sequences with zeros."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 68 or arr.shape[2] < 2:
        raise ValueError(f"Expected skeleton shape (T, 68, C>=2), got {arr.shape}")
    arr = arr[:, :, :3]
    target = int(length)
    if target <= 0:
        raise ValueError(f"sequence_length must be positive, got {length}")
    t = int(arr.shape[0])
    if t == target:
        return arr.astype(np.float32, copy=False)
    if t <= 0:
        return np.zeros((target, 68, 3), dtype=np.float32)
    if t > target:
        indices = np.linspace(0, t - 1, target).round().astype(np.int64)
        return arr[indices].astype(np.float32, copy=False)
    out = np.zeros((target, 68, 3), dtype=np.float32)
    out[:t] = arr
    return out


def read_frames_at_indices(video_path: str | Path, indices: Iterable[int]) -> list[np.ndarray]:
    needed = [int(i) for i in indices]
    if not needed:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for idx in needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def choose_visual_positions(valid_count: int, requested: int) -> np.ndarray:
    if valid_count <= 0:
        return np.empty((0,), dtype=np.int64)
    count = max(1, min(int(requested), valid_count))
    return np.linspace(0, valid_count - 1, count).round().astype(np.int64)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "#6B7280", lw: float = 1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_dimension_axes(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    color: str = "#6B7280",
    show_text: bool = True,
) -> None:
    ax.plot([x, x + w], [y - 0.28, y - 0.28], color=color, lw=1.1)
    ax.plot([x, x], [y - 0.38, y - 0.18], color=color, lw=1.1)
    ax.plot([x + w, x + w], [y - 0.38, y - 0.18], color=color, lw=1.1)
    ax.plot([x + w + 0.25, x + w + 0.25], [y, y + h], color=color, lw=1.1)
    ax.plot([x + w + 0.12, x + w + 0.38], [y, y], color=color, lw=1.1)
    ax.plot([x + w + 0.12, x + w + 0.38], [y + h, y + h], color=color, lw=1.1)
    if show_text:
        ax.text(x + w / 2, y - 0.53, "x", ha="center", va="top", fontsize=12, color="#374151")
        ax.text(x + w + 0.42, y + h / 2, "y", ha="left", va="center", fontsize=12, color="#374151")


def draw_video_stack(
    ax,
    frames: list[np.ndarray],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    show_annotations: bool = True,
) -> None:
    visible_frames = frames[-5:] if len(frames) > 5 else frames
    if visible_frames:
        offsets = np.linspace(0.44, 0.0, len(visible_frames))
        for i, (frame, off) in enumerate(zip(visible_frames, offsets)):
            layer_x = x - float(off)
            layer_y = y + float(off)
            ax.imshow(frame, extent=(layer_x, layer_x + w, layer_y, layer_y + h), zorder=2 + i, aspect="auto")
            ax.add_patch(
                Rectangle(
                    (layer_x, layer_y),
                    w,
                    h,
                    fill=False,
                    edgecolor="#111827",
                    linewidth=0.85,
                    zorder=3 + i,
                )
            )
    else:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="#E5E7EB",
                edgecolor="#111827",
                linewidth=0.9,
                zorder=1,
            )
        )
    if show_annotations:
        draw_dimension_axes(ax, x, y, w, h)
        ax.text(x + w / 2, y - 0.95, "Video", ha="center", va="top", fontsize=13, color="#4B5563")


def normalize_points_for_box(points: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    xy = points[:, :2].astype(np.float32, copy=True)
    valid = np.isfinite(xy).all(axis=1) & ((xy[:, 0] != 0.0) | (xy[:, 1] != 0.0))
    if not np.any(valid):
        return np.zeros((points.shape[0], 2), dtype=np.float32)
    min_xy = xy[valid].min(axis=0)
    max_xy = xy[valid].max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = np.maximum(max_xy - min_xy, 1.0e-6)
    scale = min(w * 0.78 / float(span[0]), h * 0.78 / float(span[1]))
    out = np.zeros((points.shape[0], 2), dtype=np.float32)
    out[:, 0] = x + w / 2 + (xy[:, 0] - center[0]) * scale
    out[:, 1] = y + h / 2 - (xy[:, 1] - center[1]) * scale
    return out


def draw_skeleton(ax, points: np.ndarray, x: float, y: float, w: float, h: float, *, alpha: float = 1.0) -> None:
    projected = normalize_points_for_box(points, x, y, w, h)
    body_color = "#2F7D7B"
    hand_color = "#2B6CB0"
    face_color = "#7C3AED"
    edge_color = "#4B8F88"
    for parent_name, child_name in paper_tree_edges():
        a = JOINT_INDEX[parent_name]
        b = JOINT_INDEX[child_name]
        ax.plot(
            [projected[a, 0], projected[b, 0]],
            [projected[a, 1], projected[b, 1]],
            color=edge_color,
            lw=0.95,
            alpha=alpha,
            zorder=12,
        )
    for idx, (px, py) in enumerate(projected):
        if 0 <= idx <= 5:
            color = body_color
            size = 12
        elif 6 <= idx <= 25:
            color = face_color
            size = 8
        else:
            color = hand_color
            size = 7
        ax.scatter([px], [py], s=size, color=color, edgecolor="white", linewidth=0.25, alpha=alpha, zorder=13)


def draw_skeleton_stack(
    ax,
    skeletons: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    show_annotations: bool = True,
) -> None:
    offsets = np.linspace(0.44, 0.0, max(1, skeletons.shape[0]))
    for i, off in enumerate(offsets):
        alpha = 0.28 + 0.70 * (i / max(1, len(offsets) - 1))
        ax.add_patch(
            Rectangle(
                (x - off, y + off),
                w,
                h,
                fill=False,
                edgecolor="#9CA3AF",
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                alpha=alpha,
                zorder=2 + i,
            )
        )
        draw_skeleton(ax, skeletons[i], x - off, y + off, w, h, alpha=alpha)
    if show_annotations:
        draw_dimension_axes(ax, x, y, w, h)
        ax.text(x + w / 2, y - 0.95, "Skeleton estimation", ha="center", va="top", fontsize=13, color="#4B5563")


def select_visual_data(
    video_path: str | Path,
    sequence: np.ndarray,
    frame_indices: np.ndarray,
    *,
    visual_frames: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    positions = choose_visual_positions(int(sequence.shape[0]), int(visual_frames))
    selected_skeletons = sequence[positions] if len(positions) else np.empty((0, 68, 3), dtype=np.float32)
    selected_frame_indices = frame_indices[positions] if len(positions) else np.empty((0,), dtype=np.int64)
    rgb_frames = read_frames_at_indices(video_path, selected_frame_indices)
    return rgb_frames, selected_skeletons, selected_frame_indices


def draw_rgb_video_figure(
    rgb_frames: list[np.ndarray],
    selected_frame_indices: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    out_svg: Path,
    *,
    dpi: int,
    image_only: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.3), dpi=dpi)
    if image_only:
        ax.set_xlim(0.45, 3.62)
        ax.set_ylim(1.25, 3.95)
    else:
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
    ax.axis("off")

    if not image_only:
        ax.text(2.35, 4.55, "RGB video frames", ha="center", va="center", fontsize=15, weight="bold")
    draw_video_stack(ax, rgb_frames, 1.05, 1.55, 2.35, 1.75, show_annotations=not image_only)
    if not image_only:
        add_arrow(ax, (3.90, 3.58), (3.58, 3.05))
        ax.text(3.98, 3.73, "frames", ha="left", va="center", fontsize=13, color="#4B5563")

        frame_text = ", ".join(str(int(x)) for x in selected_frame_indices.tolist())
        if len(frame_text) > 32:
            frame_text = frame_text[:29] + "..."
        ax.text(
            2.5,
            0.25,
            f"Visualized original frame indices: {frame_text}",
            ha="center",
            va="center",
            fontsize=8.7,
            color="#6B7280",
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_skeleton_estimation_figure(
    selected_skeletons: np.ndarray,
    selected_frame_indices: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    out_svg: Path,
    *,
    dpi: int,
    image_only: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.3), dpi=dpi)
    if image_only:
        ax.set_xlim(0.45, 3.62)
        ax.set_ylim(1.25, 3.95)
    else:
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
    ax.axis("off")

    if not image_only:
        ax.text(2.45, 4.55, "68-point skeleton sequence", ha="center", va="center", fontsize=15, weight="bold")
    draw_skeleton_stack(ax, selected_skeletons, 1.05, 1.55, 2.35, 1.75, show_annotations=not image_only)
    if not image_only:
        add_arrow(ax, (4.05, 3.58), (3.72, 3.05))
        ax.text(4.12, 3.73, "frames", ha="left", va="center", fontsize=13, color="#4B5563")

        frame_text = ", ".join(str(int(x)) for x in selected_frame_indices.tolist())
        if len(frame_text) > 32:
            frame_text = frame_text[:29] + "..."
        ax.text(
            2.5,
            0.25,
            f"Skeletons from original frame indices: {frame_text}",
            ha="center",
            va="center",
            fontsize=8.7,
            color="#6B7280",
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.name or video_path.stem

    cfg = ExtractionConfig(
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
        model_complexity=int(args.model_complexity),
        static_image_mode=bool(args.static_image_mode),
        refine_face_landmarks=not bool(args.no_refine_face_landmarks),
        target_fps=args.target_fps,
        holistic_task_model=args.holistic_task_model,
    )

    raw_sequence, frame_indices = extract_keypoints_with_frame_indices(
        video_path,
        cfg,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        show_progress=bool(args.show_progress),
    )
    if raw_sequence.shape[0] == 0:
        raise RuntimeError("No valid pose frames were extracted. Check the video, frame range, and MediaPipe setup.")

    raw_path = out_dir / f"{prefix}_keypoints_raw.npy"
    frame_index_path = out_dir / f"{prefix}_frame_indices.npy"
    np.save(raw_path, raw_sequence.astype(np.float32))
    np.save(frame_index_path, frame_indices.astype(np.int64))

    fixed_path = None
    if int(args.sequence_length) > 0:
        fixed_sequence = resize_or_pad_skeleton(raw_sequence, int(args.sequence_length))
        fixed_path = out_dir / f"{prefix}_keypoints_{int(args.sequence_length)}.npy"
        np.save(fixed_path, fixed_sequence.astype(np.float32))

    rgb_figure_png = None
    rgb_figure_pdf = None
    rgb_figure_svg = None
    skeleton_figure_png = None
    skeleton_figure_pdf = None
    skeleton_figure_svg = None
    if not args.no_figure:
        rgb_frames, selected_skeletons, selected_frame_indices = select_visual_data(
            video_path,
            raw_sequence,
            frame_indices,
            visual_frames=int(args.visual_frames),
        )
        figure_suffix = "_image_only" if bool(args.image_only_figures) else ""
        rgb_figure_png = out_dir / f"{prefix}_rgb_video_frames{figure_suffix}.png"
        rgb_figure_pdf = out_dir / f"{prefix}_rgb_video_frames{figure_suffix}.pdf"
        rgb_figure_svg = out_dir / f"{prefix}_rgb_video_frames{figure_suffix}.svg"
        skeleton_figure_png = out_dir / f"{prefix}_skeleton_estimation{figure_suffix}.png"
        skeleton_figure_pdf = out_dir / f"{prefix}_skeleton_estimation{figure_suffix}.pdf"
        skeleton_figure_svg = out_dir / f"{prefix}_skeleton_estimation{figure_suffix}.svg"
        draw_rgb_video_figure(
            rgb_frames,
            selected_frame_indices,
            rgb_figure_png,
            rgb_figure_pdf,
            rgb_figure_svg,
            dpi=int(args.dpi),
            image_only=bool(args.image_only_figures),
        )
        draw_skeleton_estimation_figure(
            selected_skeletons,
            selected_frame_indices,
            skeleton_figure_png,
            skeleton_figure_pdf,
            skeleton_figure_svg,
            dpi=int(args.dpi),
            image_only=bool(args.image_only_figures),
        )

    metadata = {
        "video": str(video_path.resolve(strict=False)),
        "video_info": video_info(video_path),
        "extraction_config": asdict(cfg),
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "raw_sequence_shape": list(raw_sequence.shape),
        "fixed_sequence_shape": [int(args.sequence_length), 68, 3] if fixed_path else None,
        "image_only_figures": bool(args.image_only_figures),
        "raw_keypoints_path": str(raw_path.resolve(strict=False)),
        "fixed_keypoints_path": str(fixed_path.resolve(strict=False)) if fixed_path else None,
        "frame_indices_path": str(frame_index_path.resolve(strict=False)),
        "rgb_figure_png": str(rgb_figure_png.resolve(strict=False)) if rgb_figure_png else None,
        "rgb_figure_pdf": str(rgb_figure_pdf.resolve(strict=False)) if rgb_figure_pdf else None,
        "rgb_figure_svg": str(rgb_figure_svg.resolve(strict=False)) if rgb_figure_svg else None,
        "skeleton_figure_png": str(skeleton_figure_png.resolve(strict=False)) if skeleton_figure_png else None,
        "skeleton_figure_pdf": str(skeleton_figure_pdf.resolve(strict=False)) if skeleton_figure_pdf else None,
        "skeleton_figure_svg": str(skeleton_figure_svg.resolve(strict=False)) if skeleton_figure_svg else None,
    }
    metadata_path = out_dir / f"{prefix}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Raw keypoints: {raw_path} shape={tuple(raw_sequence.shape)}")
    if fixed_path:
        print(f"Fixed keypoints: {fixed_path} shape=({int(args.sequence_length)}, 68, 3)")
    print(f"Frame indices: {frame_index_path} count={len(frame_indices)}")
    if rgb_figure_png:
        print(f"RGB figure: {rgb_figure_png}")
    if rgb_figure_svg:
        print(f"RGB SVG: {rgb_figure_svg}")
    if skeleton_figure_png:
        print(f"Skeleton figure: {skeleton_figure_png}")
    if skeleton_figure_svg:
        print(f"Skeleton SVG: {skeleton_figure_svg}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
