"""MediaPipe-based skeleton extraction from sign-language videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np
from tqdm import tqdm

from .keypoints import selected_keypoints_from_holistic_results


@dataclass(frozen=True)
class ExtractionConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    static_image_mode: bool = False
    refine_face_landmarks: bool = True
    target_fps: float | None = None
    holistic_task_model: str | None = None


def validate_extraction_backend(config: ExtractionConfig = ExtractionConfig()) -> None:
    """Fail early when MediaPipe cannot provide a usable holistic extractor."""
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mediapipe is required for video preprocessing") from exc

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
        return
    if config.holistic_task_model:
        task_model = Path(config.holistic_task_model)
        if not task_model.exists():
            raise FileNotFoundError(f"MediaPipe Holistic task model does not exist: {task_model}")
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and hasattr(mp.tasks.vision, "HolisticLandmarker"):
            return
    version = getattr(mp, "__version__", "unknown")
    raise RuntimeError(
        "The installed mediapipe package cannot run this preprocessing pipeline. "
        "It does not provide the legacy `mp.solutions.holistic` API "
        f"(version={version}). Use a Python 3.10/3.11 environment with a legacy "
        "MediaPipe build that includes solutions, for example `pip install "
        "mediapipe==0.10.14`, or provide a MediaPipe HolisticLandmarker .task "
        "model via --mediapipe-config with `holistic_task_model: path/to/model.task`."
    )


def _frame_bound_to_index(value: int | None, *, default: int | None) -> int | None:
    if value is None or int(value) < 0:
        return default
    # WLASL frame_start/frame_end annotations are 1-based and inclusive.
    return max(0, int(value) - 1)


def iter_video_frames(
    video_path: str | Path,
    target_fps: float | None = None,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield RGB frames from a video.

    If target_fps is set, frames are sampled approximately to that fps.
    If frame_start/frame_end are set, they are treated as 1-based inclusive
    annotation bounds, matching WLASL metadata.
    """
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1
    if target_fps is not None and source_fps > target_fps > 0:
        step = max(1, int(round(source_fps / target_fps)))
    start_idx = _frame_bound_to_index(frame_start, default=0) or 0
    end_idx = _frame_bound_to_index(frame_end, default=None)
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
                yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


def extract_selected_keypoint_sequence(
    video_path: str | Path,
    config: ExtractionConfig = ExtractionConfig(),
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Extract selected 68-keypoint sequence with MediaPipe Holistic.

    Frames where MediaPipe cannot estimate body pose are discarded, following
    the paper. Missing hands and face are handled in keypoints.py.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mediapipe is required for video preprocessing") from exc
    validate_extraction_backend(config)

    frames = list(
        iter_video_frames(
            video_path,
            config.target_fps,
            frame_start=frame_start,
            frame_end=frame_end,
        )
    )
    iterator = tqdm(frames, desc=f"Extract {Path(video_path).name}") if show_progress else frames
    selected: List[np.ndarray] = []

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
        with mp.solutions.holistic.Holistic(
            static_image_mode=config.static_image_mode,
            model_complexity=config.model_complexity,
            refine_face_landmarks=config.refine_face_landmarks,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as holistic:
            for frame_rgb in iterator:
                results = holistic.process(frame_rgb)
                points = selected_keypoints_from_holistic_results(results)
                if points is None:
                    continue
                selected.append(points)
    elif config.holistic_task_model:
        task_model = Path(config.holistic_task_model)
        if not task_model.exists():
            raise FileNotFoundError(f"MediaPipe Holistic task model does not exist: {task_model}")
        vision = mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task_model)),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_landmarks_confidence=config.min_detection_confidence,
            min_pose_detection_confidence=config.min_detection_confidence,
            min_pose_landmarks_confidence=config.min_tracking_confidence,
            min_hand_landmarks_confidence=config.min_tracking_confidence,
        )
        timestamp_step_ms = int(round(1000.0 / float(config.target_fps or 30.0)))
        with vision.HolisticLandmarker.create_from_options(options) as holistic:
            for frame_index, frame_rgb in enumerate(iterator):
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
                task_result = holistic.detect_for_video(image, frame_index * timestamp_step_ms)
                results = _wrap_task_holistic_result(task_result)
                points = selected_keypoints_from_holistic_results(results)
                if points is None:
                    continue
                selected.append(points)
    else:
        validate_extraction_backend(config)

    if len(selected) == 0:
        return np.empty((0, 68, 3), dtype=np.float32)
    return np.stack(selected).astype(np.float32)


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


def _wrap_task_holistic_result(result) -> _HolisticResults:
    return _HolisticResults(result)
