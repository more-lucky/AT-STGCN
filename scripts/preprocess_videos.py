#!/usr/bin/env python
"""Preprocess raw sign-language videos into compact skeleton .npy files.

Input manifest CSV columns:
  video_path,label,split
Optional input columns:
  frame_start,frame_end

Output skeleton manifest columns:
  keypoints_path,path,label,split,valid_pose_frames
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.extractor import ExtractionConfig, extract_selected_keypoint_sequence, validate_extraction_backend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-manifest", required=True, help="CSV with video_path,label,split")
    p.add_argument("--output-dir", required=True, help="Directory for .npy output files")
    p.add_argument("--output-manifest", required=True, help="CSV to write preprocessed manifest")
    p.add_argument("--target-fps", type=float, default=None, help="Optional sampling fps, e.g. 25 for WLASL")
    p.add_argument("--mediapipe-config", type=str, default=None, help="Optional YAML extraction config")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel preprocessing worker processes. Use 0 for serial execution.",
    )
    p.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV threads per worker. Keep this low when --num-workers > 1.",
    )
    p.add_argument("--skip-existing", action="store_true", help="Reuse existing .npy files for resume/restart runs")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip failed videos and keep preprocessing the remaining rows.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="Task chunksize for multiprocessing.",
    )
    p.add_argument(
        "--min-valid-pose-frames",
        type=int,
        default=1,
        help="Minimum usable pose frames before fallback/skip logic is triggered",
    )
    p.add_argument(
        "--fallback-full-video-on-short",
        action="store_true",
        help="For clipped manifests, retry the full video when the annotated segment yields too few pose frames",
    )
    p.add_argument(
        "--skip-below-min-valid",
        action="store_true",
        help="Skip samples whose final extracted sequence is shorter than --min-valid-pose-frames",
    )
    p.add_argument("--show-frame-progress", action="store_true")
    return p.parse_args()


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


def _safe_existing_length(path: Path) -> int:
    try:
        arr = np.load(path)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    except Exception:
        return 0
    return 0


def _build_output_row(
    *,
    out_path: Path,
    label: str,
    split: str,
    valid_pose_frames: int,
    frame_start: int | None,
    frame_end: int | None,
    fallback_used: bool = False,
) -> dict:
    out_row = {
        "label": label,
        "split": split,
        "valid_pose_frames": int(valid_pose_frames),
        "keypoints_path": str(out_path),
        "path": str(out_path),
    }
    if frame_start is not None:
        out_row["frame_start"] = int(frame_start)
    if frame_end is not None:
        out_row["frame_end"] = int(frame_end)
    if fallback_used:
        out_row["fallback_full_video"] = 1
    return out_row


def _process_video_task(task: dict) -> dict:
    if int(task["opencv_threads"]) >= 0:
        try:
            import cv2

            cv2.setNumThreads(int(task["opencv_threads"]))
        except Exception:
            pass
    try:
        video_path = Path(task["video_path"])
        out_path = Path(task["out_path"])
        label = str(task["label"])
        split = str(task["split"])
        frame_start = task["frame_start"]
        frame_end = task["frame_end"]
        min_valid_pose_frames = int(task["min_valid_pose_frames"])
        if bool(task["skip_existing"]) and out_path.exists():
            valid_pose_frames = _safe_existing_length(out_path)
            return {
                "index": int(task["index"]),
                "row": _build_output_row(
                    out_path=out_path,
                    label=label,
                    split=split,
                    valid_pose_frames=valid_pose_frames,
                    frame_start=frame_start,
                    frame_end=frame_end,
                ),
                "error": None,
            }
        cfg = ExtractionConfig(**task["config"])
        seq = extract_selected_keypoint_sequence(
            video_path,
            cfg,
            frame_start=frame_start,
            frame_end=frame_end,
            show_progress=bool(task["show_frame_progress"]),
        )
        fallback_used = False
        if (
            bool(task["fallback_full_video_on_short"])
            and (frame_start is not None or frame_end is not None)
            and int(seq.shape[0]) < min_valid_pose_frames
        ):
            full_seq = extract_selected_keypoint_sequence(
                video_path,
                cfg,
                frame_start=None,
                frame_end=None,
                show_progress=bool(task["show_frame_progress"]),
            )
            if int(full_seq.shape[0]) > int(seq.shape[0]):
                seq = full_seq
                fallback_used = True
        if bool(task["skip_below_min_valid"]) and int(seq.shape[0]) < min_valid_pose_frames:
            return {"index": int(task["index"]), "row": None, "error": None}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, seq.astype(np.float32, copy=False))
        return {
            "index": int(task["index"]),
            "row": _build_output_row(
                out_path=out_path,
                label=label,
                split=split,
                valid_pose_frames=int(seq.shape[0]),
                frame_start=frame_start,
                frame_end=frame_end,
                fallback_used=fallback_used,
            ),
            "error": None,
        }
    except Exception as exc:
        return {"index": int(task["index"]), "row": None, "error": f"{task['video_path']}: {exc}"}


def _write_manifest(rows: list[dict], output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_manifest, index=False)


def main() -> None:
    args = parse_args()
    in_df = pd.read_csv(args.input_manifest)
    required = {"video_path", "label", "split"}
    missing = required - set(in_df.columns)
    if missing:
        raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")

    out_dir = Path(args.output_dir)
    cfg = load_extraction_config(args.mediapipe_config, args.target_fps)
    validate_extraction_backend(cfg)
    config_payload = {
        "min_detection_confidence": cfg.min_detection_confidence,
        "min_tracking_confidence": cfg.min_tracking_confidence,
        "model_complexity": cfg.model_complexity,
        "static_image_mode": cfg.static_image_mode,
        "refine_face_landmarks": cfg.refine_face_landmarks,
        "target_fps": cfg.target_fps,
        "holistic_task_model": cfg.holistic_task_model,
    }
    tasks = []
    for i, row in in_df.iterrows():
        video_path = Path(row["video_path"])
        label = str(row["label"])
        split = str(row["split"])
        frame_start = optional_int(row["frame_start"]) if "frame_start" in in_df.columns else None
        frame_end = optional_int(row["frame_end"]) if "frame_end" in in_df.columns else None
        stem = f"{video_path.stem}_{i:06d}.npy"
        out_path = out_dir / split / label / stem
        tasks.append(
            {
                "index": int(i),
                "video_path": str(video_path),
                "label": label,
                "split": split,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "out_path": str(out_path),
                "config": config_payload,
                "fallback_full_video_on_short": bool(args.fallback_full_video_on_short),
                "skip_below_min_valid": bool(args.skip_below_min_valid),
                "min_valid_pose_frames": int(args.min_valid_pose_frames),
                "show_frame_progress": bool(args.show_frame_progress),
                "skip_existing": bool(args.skip_existing),
                "opencv_threads": int(args.opencv_threads),
            }
        )

    out_manifest = Path(args.output_manifest)
    rows_by_index: dict[int, dict] = {}
    if int(args.num_workers) <= 1:
        iterator = (_process_video_task(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Preprocess videos"):
            if result["error"] is not None:
                if bool(args.continue_on_error):
                    print(f"WARNING: {result['error']}")
                    continue
                raise RuntimeError(result["error"])
            if result["row"] is not None:
                rows_by_index[int(result["index"])] = result["row"]
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            iterator = executor.map(_process_video_task, tasks, chunksize=max(1, int(args.chunksize)))
            for result in tqdm(iterator, total=len(tasks), desc=f"Preprocess videos ({args.num_workers} workers)"):
                if result["error"] is not None:
                    if bool(args.continue_on_error):
                        print(f"WARNING: {result['error']}")
                        continue
                    raise RuntimeError(result["error"])
                if result["row"] is not None:
                    rows_by_index[int(result["index"])] = result["row"]
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    _write_manifest(rows, out_manifest)
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
