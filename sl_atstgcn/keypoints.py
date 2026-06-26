"""Keypoint definitions used by the paper-aligned skeleton graph.

The paper selects 68 MediaPipe Holistic keypoints:
  - 6 body keypoints: nose, right/left shoulders, right/left elbows, and a
    computed middle-chest point (-1) at the midpoint of the shoulders.
  - 20 face keypoints.
  - 21 left-hand keypoints and 21 right-hand keypoints.

The labels below keep the paper's names and the MediaPipe indices shown in
Figure 1. MediaPipe official anatomical left/right naming may differ depending
on the camera convention; the code uses the numeric indices from the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class JointSpec:
    name: str
    source: str  # one of: pose, face, left_hand, right_hand, computed
    index: int | None


# Paper body keypoints, with -1 represented as a computed joint.
BODY_JOINTS: List[JointSpec] = [
    JointSpec("nose", "pose", 0),
    JointSpec("right_shoulder", "pose", 11),
    JointSpec("left_shoulder", "pose", 12),
    JointSpec("right_elbow", "pose", 13),
    JointSpec("left_elbow", "pose", 14),
    JointSpec("middle_chest", "computed", None),
]

FACE_JOINTS: List[JointSpec] = [
    # right eyebrow
    JointSpec("right_eyebrow_46", "face", 46),
    JointSpec("right_eyebrow_52", "face", 52),
    JointSpec("right_eyebrow_53", "face", 53),
    JointSpec("right_eyebrow_65", "face", 65),
    # left eyebrow
    JointSpec("left_eyebrow_295", "face", 295),
    JointSpec("left_eyebrow_283", "face", 283),
    JointSpec("left_eyebrow_282", "face", 282),
    JointSpec("left_eyebrow_276", "face", 276),
    # right eye
    JointSpec("right_eye_7", "face", 7),
    JointSpec("right_eye_159", "face", 159),
    JointSpec("right_eye_155", "face", 155),
    JointSpec("right_eye_145", "face", 145),
    # left eye
    JointSpec("left_eye_382", "face", 382),
    JointSpec("left_eye_386", "face", 386),
    JointSpec("left_eye_249", "face", 249),
    JointSpec("left_eye_374", "face", 374),
    # mouth
    JointSpec("mouth_324", "face", 324),
    JointSpec("mouth_13", "face", 13),
    JointSpec("mouth_78", "face", 78),
    JointSpec("mouth_14", "face", 14),
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

LEFT_HAND_JOINTS = [JointSpec(f"left_hand_{name}", "left_hand", i) for i, name in enumerate(HAND_NAMES)]
RIGHT_HAND_JOINTS = [JointSpec(f"right_hand_{name}", "right_hand", i) for i, name in enumerate(HAND_NAMES)]

ALL_JOINTS: List[JointSpec] = BODY_JOINTS + FACE_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
JOINT_INDEX: Dict[str, int] = {joint.name: i for i, joint in enumerate(ALL_JOINTS)}

MIDDLE_CHEST_INDEX = JOINT_INDEX["middle_chest"]
NOSE_INDEX = JOINT_INDEX["nose"]
RIGHT_SHOULDER_INDEX = JOINT_INDEX["right_shoulder"]
LEFT_SHOULDER_INDEX = JOINT_INDEX["left_shoulder"]
RIGHT_ELBOW_INDEX = JOINT_INDEX["right_elbow"]
LEFT_ELBOW_INDEX = JOINT_INDEX["left_elbow"]
LEFT_HAND_WRIST_INDEX = JOINT_INDEX["left_hand_wrist"]
RIGHT_HAND_WRIST_INDEX = JOINT_INDEX["right_hand_wrist"]


def num_selected_joints() -> int:
    """Return the number of selected joints before DFS backtracking expansion."""
    return len(ALL_JOINTS)  # 68


def _landmarks_to_array(landmarks, *, expected: int | None = None) -> np.ndarray | None:
    """Convert a MediaPipe NormalizedLandmarkList to a float32 array.

    Returns None when the landmark list is missing.
    """
    if landmarks is None:
        return None
    lm = landmarks.landmark
    if expected is not None and len(lm) < expected:
        return None
    return np.asarray([[p.x, p.y, p.z] for p in lm], dtype=np.float32)


def selected_keypoints_from_holistic_results(results) -> np.ndarray | None:
    """Extract the paper's 68 selected keypoints from MediaPipe Holistic results.

    Policy aligned to the paper:
      - discard the frame if body pose is not estimated;
      - when a hand is missing, replace its 21 points with the wrist coordinates;
      - when the face is missing, replace face points with the nose coordinates;
      - compute middle_chest as the midpoint between both shoulders;
      - set z to 0 later in preprocessing when MediaPipe z is unreliable.

    Returns:
        Array with shape (68, 3), or None when pose is missing.
    """
    pose = _landmarks_to_array(results.pose_landmarks, expected=33)
    if pose is None:
        return None

    face = _landmarks_to_array(results.face_landmarks, expected=468)
    left_hand = _landmarks_to_array(results.left_hand_landmarks, expected=21)
    right_hand = _landmarks_to_array(results.right_hand_landmarks, expected=21)

    # Paper-indexed fallback. The numeric wrist indices match the MediaPipe pose
    # wrist locations neighboring elbows 13 and 14.
    # If your dataset uses a mirrored camera convention, adjust these in config.
    left_hand_fallback = pose[16].copy()
    right_hand_fallback = pose[15].copy()
    nose_fallback = pose[0].copy()
    middle_chest = (pose[11] + pose[12]) / 2.0

    selected: List[np.ndarray] = []
    for spec in ALL_JOINTS:
        if spec.source == "pose":
            selected.append(pose[int(spec.index)].copy())
        elif spec.source == "computed":
            selected.append(middle_chest.copy())
        elif spec.source == "face":
            if face is None:
                selected.append(nose_fallback.copy())
            else:
                selected.append(face[int(spec.index)].copy())
        elif spec.source == "left_hand":
            if left_hand is None:
                selected.append(left_hand_fallback.copy())
            else:
                selected.append(left_hand[int(spec.index)].copy())
        elif spec.source == "right_hand":
            if right_hand is None:
                selected.append(right_hand_fallback.copy())
            else:
                selected.append(right_hand[int(spec.index)].copy())
        else:  # pragma: no cover
            raise ValueError(f"Unknown landmark source: {spec.source}")

    arr = np.stack(selected).astype(np.float32)
    return arr
