from sl_atstgcn.graph import (
    DFS_MIRROR_COLUMN_INDICES,
    DFS_UNIQUE_COLUMN_INDICES,
    DFS_WALK_ORDER,
    DFS_WALK_WIDTH,
    SKELETON_PARENT_JOINT_INDICES,
    mirrored_joint_indices,
)
from sl_atstgcn.keypoints import JOINT_INDEX
from sl_atstgcn.keypoints import num_selected_joints


def test_dfs_width_matches_paper():
    assert num_selected_joints() == 68
    assert DFS_WALK_WIDTH == 135
    assert len(DFS_WALK_ORDER) == 135
    assert len(set(DFS_WALK_ORDER)) == 68


def test_mirror_column_map_preserves_dfs_shape_and_pairs():
    assert len(DFS_MIRROR_COLUMN_INDICES) == DFS_WALK_WIDTH
    assert sorted(DFS_MIRROR_COLUMN_INDICES) == list(range(DFS_WALK_WIDTH))

    mirror = mirrored_joint_indices()
    assert mirror[JOINT_INDEX["right_hand_wrist"]] == JOINT_INDEX["left_hand_wrist"]
    assert mirror[JOINT_INDEX["left_elbow"]] == JOINT_INDEX["right_elbow"]
    assert mirror[mirror[JOINT_INDEX["right_eye_159"]]] == JOINT_INDEX["right_eye_159"]


def test_unique_columns_and_parent_indices_cover_compact_skeleton():
    assert len(DFS_UNIQUE_COLUMN_INDICES) == num_selected_joints()
    assert sorted(DFS_WALK_ORDER[col] for col in DFS_UNIQUE_COLUMN_INDICES) == list(range(num_selected_joints()))
    assert len(SKELETON_PARENT_JOINT_INDICES) == num_selected_joints()
    assert SKELETON_PARENT_JOINT_INDICES[JOINT_INDEX["middle_chest"]] == JOINT_INDEX["middle_chest"]
    assert SKELETON_PARENT_JOINT_INDICES[JOINT_INDEX["left_hand_thumb_tip"]] == JOINT_INDEX["left_hand_thumb_ip"]
