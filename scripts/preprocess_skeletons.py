#!/usr/bin/env python
"""Generate compact skeleton .npy files for skeleton-only training.

Supported input manifests:
  1. Raw videos: video_path,label,split[,frame_start,frame_end]
  2. Existing arrays: keypoints_path,label,split or path,label,split

Output manifest columns:
  keypoints_path,path,label,split,valid_pose_frames,source_path

Saved arrays are shaped (T, 68, 3). By default T is the extracted valid
sequence length; pass --sequence-length to save fixed-length arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.data import prepare_skeleton_sequence
from sl_atstgcn.extractor import ExtractionConfig, extract_selected_keypoint_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, help="CSV manifest to read")
    parser.add_argument("--output-dir", required=True, help="Directory where skeleton .npy files are written")
    parser.add_argument("--output-manifest", required=True, help="CSV manifest to write")
    parser.add_argument(
        "--source",
        choices=["auto", "video", "array"],
        default="auto",
        help="auto uses video_path when present, otherwise keypoints_path/path arrays",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Optional fixed temporal length to save; omit to save raw extracted length",
    )
    parser.add_argument("--target-fps", type=float, default=None, help="Optional video sampling fps, e.g. 25")
    parser.add_argument("--mediapipe-config", type=str, default=None, help="Optional YAML ExtractionConfig")
    parser.add_argument("--min-valid-pose-frames", type=int, default=1)
    parser.add_argument(
        "--fallback-full-video-on-short",
        action="store_true",
        help="Retry full video if annotated segment has too few valid pose frames",
    )
    parser.add_argument(
        "--skip-below-min-valid",
        action="store_true",
        help="Skip samples whose final sequence is shorter than --min-valid-pose-frames",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files that already exist")
    parser.add_argument("--relative-to", default=None, help="Write output paths relative to this directory")
    parser.add_argument("--show-frame-progress", action="store_true")
    return parser.parse_args()


def load_extraction_config(path: str | None, target_fps: float | None) -> ExtractionConfig:
    params = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
    params["target_fps"] = target_fps
    return ExtractionConfig(**params)


def optional_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    return int(float(text))


def display_path(path: Path, relative_to: Path | None) -> str:
    resolved = path.resolve(strict=False)
    if relative_to is None:
        return str(resolved)
    base = relative_to.resolve(strict=False)
    try:
        return Path(os.path.relpath(resolved, base)).as_posix()
    except ValueError:
        return str(resolved)


def safe_stem(source: str, index: int) -> str:
    path = Path(str(source))
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{index:06d}_{digest}"


def resolve_source_mode(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    if "video_path" in df.columns:
        return "video"
    if "keypoints_path" in df.columns or "path" in df.columns:
        return "array"
    raise ValueError("Manifest needs video_path, keypoints_path, or path column")


def load_array_sequence(row, source_column: str, sequence_length: int | None) -> tuple[np.ndarray, str, int]:
    source_path = str(row[source_column])
    arr = np.load(source_path).astype(np.float32)
    raw_length = int(row.get("valid_pose_frames", arr.shape[0]))
    target_length = int(sequence_length) if sequence_length is not None else int(arr.shape[0])
    sequence = prepare_skeleton_sequence(arr, sequence_length=target_length)
    return sequence, source_path, raw_length


def extract_video_sequence(
    row,
    *,
    cfg: ExtractionConfig,
    sequence_length: int | None,
    min_valid_pose_frames: int,
    fallback_full_video_on_short: bool,
    show_frame_progress: bool,
) -> tuple[np.ndarray, str, bool, int]:
    video_path = str(row["video_path"])
    frame_start = optional_int(getattr(row, "frame_start", None))
    frame_end = optional_int(getattr(row, "frame_end", None))
    seq = extract_selected_keypoint_sequence(
        video_path,
        cfg,
        frame_start=frame_start,
        frame_end=frame_end,
        show_progress=show_frame_progress,
    )
    fallback_used = False
    if (
        fallback_full_video_on_short
        and (frame_start is not None or frame_end is not None)
        and int(seq.shape[0]) < int(min_valid_pose_frames)
    ):
        full_seq = extract_selected_keypoint_sequence(
            video_path,
            cfg,
            frame_start=None,
            frame_end=None,
            show_progress=show_frame_progress,
        )
        if int(full_seq.shape[0]) > int(seq.shape[0]):
            seq = full_seq
            fallback_used = True
    raw_length = int(seq.shape[0])
    if sequence_length is not None:
        seq = prepare_skeleton_sequence(seq, sequence_length=int(sequence_length))
    else:
        seq = seq.astype(np.float32, copy=False)
    return seq, video_path, fallback_used, raw_length


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_manifest)
    required = {"label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")
    source_mode = resolve_source_mode(df, str(args.source))
    if source_mode == "video" and "video_path" not in df.columns:
        raise ValueError("source=video requires a video_path column")
    if source_mode == "array" and "keypoints_path" not in df.columns and "path" not in df.columns:
        raise ValueError("source=array requires keypoints_path or path column")

    out_dir = Path(args.output_dir)
    out_manifest = Path(args.output_manifest)
    relative_to = Path(args.relative_to) if args.relative_to else None
    extraction_cfg = load_extraction_config(args.mediapipe_config, args.target_fps)
    array_source_column = "keypoints_path" if "keypoints_path" in df.columns else "path"

    rows: list[dict[str, object]] = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generate skeletons"):
        row_obj = row
        label = str(row_obj["label"])
        split = str(row_obj["split"])
        if source_mode == "video":
            source_value = str(row_obj["video_path"])
        else:
            source_value = str(row_obj[array_source_column])
        out_path = out_dir / split / label / f"{safe_stem(source_value, int(index))}.npy"

        fallback_used = False
        if out_path.exists() and not args.overwrite:
            sequence = np.load(out_path, mmap_mode="r")
            valid_pose_frames = int(sequence.shape[0])
        else:
            if source_mode == "video":
                sequence, source_value, fallback_used, valid_pose_frames = extract_video_sequence(
                    row_obj,
                    cfg=extraction_cfg,
                    sequence_length=args.sequence_length,
                    min_valid_pose_frames=int(args.min_valid_pose_frames),
                    fallback_full_video_on_short=bool(args.fallback_full_video_on_short),
                    show_frame_progress=bool(args.show_frame_progress),
                )
            else:
                sequence, source_value, valid_pose_frames = load_array_sequence(
                    row_obj,
                    array_source_column,
                    args.sequence_length,
                )
            if args.skip_below_min_valid and valid_pose_frames < int(args.min_valid_pose_frames):
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, np.asarray(sequence, dtype=np.float32))

        out_row: dict[str, object] = {
            "keypoints_path": display_path(out_path, relative_to),
            "path": display_path(out_path, relative_to),
            "label": label,
            "split": split,
            "valid_pose_frames": valid_pose_frames,
            "source_path": source_value,
        }
        if "video_id" in df.columns:
            out_row["video_id"] = row_obj["video_id"]
        if "frame_start" in df.columns:
            frame_start = optional_int(row_obj.get("frame_start"))
            if frame_start is not None:
                out_row["frame_start"] = int(frame_start)
        if "frame_end" in df.columns:
            frame_end = optional_int(row_obj.get("frame_end"))
            if frame_end is not None:
                out_row["frame_end"] = int(frame_end)
        if fallback_used:
            out_row["fallback_full_video"] = 1
        rows.append(out_row)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(f"Wrote {len(rows)} skeleton rows to {out_manifest}")
    print(f"Skeleton files written under {out_dir}")


if __name__ == "__main__":
    main()
