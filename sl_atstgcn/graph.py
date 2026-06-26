"""Paper skeleton graph and optional DFS traversal helpers.

The active project path consumes compact 68-joint skeleton sequences directly.
DFS traversal helpers remain as metadata for reproducing deterministic
paper-figure ordering and topology tests.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .keypoints import JOINT_INDEX, num_selected_joints

Edge = Tuple[str, str]


def _hand_edges(prefix: str) -> List[Edge]:
    """Return anatomical tree edges for one 21-keypoint hand."""
    wrist = f"{prefix}_hand_wrist"
    return [
        (wrist, f"{prefix}_hand_thumb_cmc"),
        (f"{prefix}_hand_thumb_cmc", f"{prefix}_hand_thumb_mcp"),
        (f"{prefix}_hand_thumb_mcp", f"{prefix}_hand_thumb_ip"),
        (f"{prefix}_hand_thumb_ip", f"{prefix}_hand_thumb_tip"),
        (wrist, f"{prefix}_hand_index_finger_mcp"),
        (f"{prefix}_hand_index_finger_mcp", f"{prefix}_hand_index_finger_pip"),
        (f"{prefix}_hand_index_finger_pip", f"{prefix}_hand_index_finger_dip"),
        (f"{prefix}_hand_index_finger_dip", f"{prefix}_hand_index_finger_tip"),
        (wrist, f"{prefix}_hand_middle_finger_mcp"),
        (f"{prefix}_hand_middle_finger_mcp", f"{prefix}_hand_middle_finger_pip"),
        (f"{prefix}_hand_middle_finger_pip", f"{prefix}_hand_middle_finger_dip"),
        (f"{prefix}_hand_middle_finger_dip", f"{prefix}_hand_middle_finger_tip"),
        (wrist, f"{prefix}_hand_ring_finger_mcp"),
        (f"{prefix}_hand_ring_finger_mcp", f"{prefix}_hand_ring_finger_pip"),
        (f"{prefix}_hand_ring_finger_pip", f"{prefix}_hand_ring_finger_dip"),
        (f"{prefix}_hand_ring_finger_dip", f"{prefix}_hand_ring_finger_tip"),
        (wrist, f"{prefix}_hand_pinky_mcp"),
        (f"{prefix}_hand_pinky_mcp", f"{prefix}_hand_pinky_pip"),
        (f"{prefix}_hand_pinky_pip", f"{prefix}_hand_pinky_dip"),
        (f"{prefix}_hand_pinky_dip", f"{prefix}_hand_pinky_tip"),
    ]


def paper_tree_edges() -> List[Edge]:
    """Edges of the paper-aligned skeleton tree.

    The tree follows Figure 1's body/face/hands topology and is rooted at the
    middle chest. Child order is deterministic so graph-derived metadata stays
    reproducible.
    """
    edges: List[Edge] = [
        # Torso and arms.
        ("middle_chest", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_hand_wrist"),
        ("middle_chest", "nose"),
        ("middle_chest", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_hand_wrist"),
        # Face branches from nose.
        ("nose", "right_eyebrow_46"),
        ("right_eyebrow_46", "right_eyebrow_52"),
        ("right_eyebrow_52", "right_eyebrow_53"),
        ("right_eyebrow_53", "right_eyebrow_65"),
        ("nose", "right_eye_7"),
        ("right_eye_7", "right_eye_159"),
        ("right_eye_159", "right_eye_155"),
        ("right_eye_155", "right_eye_145"),
        ("nose", "mouth_324"),
        ("mouth_324", "mouth_13"),
        ("mouth_13", "mouth_78"),
        ("mouth_78", "mouth_14"),
        ("nose", "left_eye_382"),
        ("left_eye_382", "left_eye_386"),
        ("left_eye_386", "left_eye_249"),
        ("left_eye_249", "left_eye_374"),
        ("nose", "left_eyebrow_295"),
        ("left_eyebrow_295", "left_eyebrow_283"),
        ("left_eyebrow_283", "left_eyebrow_282"),
        ("left_eyebrow_282", "left_eyebrow_276"),
    ]
    edges.extend(_hand_edges("right"))
    edges.extend(_hand_edges("left"))
    return edges


def build_adjacency(edges: Iterable[Edge]) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for parent_name, child_name in edges:
        parent = JOINT_INDEX[parent_name]
        child = JOINT_INDEX[child_name]
        adjacency[parent].append(child)
    return dict(adjacency)


def parent_joint_indices(root_name: str = "middle_chest") -> List[int]:
    """Return each selected joint's parent index in the paper tree.

    The returned list is ordered by the compact 68-joint index. The root uses
    itself as parent so bone features for that joint become zero after
    subtraction.
    """
    parents = list(range(num_selected_joints()))
    root = JOINT_INDEX[root_name]
    parents[root] = root
    for parent_name, child_name in paper_tree_edges():
        parents[JOINT_INDEX[child_name]] = JOINT_INDEX[parent_name]
    return parents


def first_dfs_column_indices(order: List[int] | None = None) -> List[int]:
    """Map every compact joint index to its first DFS-walk column."""
    resolved_order = DFS_WALK_ORDER if order is None else list(order)
    first_columns = [-1] * num_selected_joints()
    for col, joint in enumerate(resolved_order):
        joint = int(joint)
        if first_columns[joint] < 0:
            first_columns[joint] = int(col)
    if any(col < 0 for col in first_columns):  # pragma: no cover - invalid graph
        missing = [i for i, col in enumerate(first_columns) if col < 0]
        raise RuntimeError(f"DFS walk does not include selected joints: {missing}")
    return first_columns


def mirrored_joint_name(name: str) -> str:
    """Return the left/right semantic counterpart for horizontal flips."""
    explicit = {
        "right_shoulder": "left_shoulder",
        "left_shoulder": "right_shoulder",
        "right_elbow": "left_elbow",
        "left_elbow": "right_elbow",
        "right_eyebrow_46": "left_eyebrow_295",
        "left_eyebrow_295": "right_eyebrow_46",
        "right_eyebrow_52": "left_eyebrow_283",
        "left_eyebrow_283": "right_eyebrow_52",
        "right_eyebrow_53": "left_eyebrow_282",
        "left_eyebrow_282": "right_eyebrow_53",
        "right_eyebrow_65": "left_eyebrow_276",
        "left_eyebrow_276": "right_eyebrow_65",
        "right_eye_7": "left_eye_382",
        "left_eye_382": "right_eye_7",
        "right_eye_159": "left_eye_386",
        "left_eye_386": "right_eye_159",
        "right_eye_155": "left_eye_249",
        "left_eye_249": "right_eye_155",
        "right_eye_145": "left_eye_374",
        "left_eye_374": "right_eye_145",
        "mouth_324": "mouth_78",
        "mouth_78": "mouth_324",
    }
    if name in explicit:
        return explicit[name]
    if name.startswith("right_hand_"):
        return "left_hand_" + name[len("right_hand_") :]
    if name.startswith("left_hand_"):
        return "right_hand_" + name[len("left_hand_") :]
    return name


def mirrored_joint_indices() -> Dict[int, int]:
    """Map every selected joint index to its mirrored semantic joint index."""
    index_to_name = {idx: name for name, idx in JOINT_INDEX.items()}
    return {idx: JOINT_INDEX[mirrored_joint_name(name)] for idx, name in index_to_name.items()}


def mirrored_dfs_column_indices(order: List[int] | None = None) -> List[int]:
    """Return source column indices that horizontally mirror a DFS-walk layout.

    DFS columns are semantic positions, not image-space x locations. A faithful
    horizontal flip must invert x coordinates and also exchange the left/right
    semantic columns while preserving repeated DFS backtracking occurrences.
    """
    resolved_order = DFS_WALK_ORDER if order is None else list(order)
    mirror_by_joint = mirrored_joint_indices()
    positions_by_joint: Dict[int, List[int]] = defaultdict(list)
    for col, joint in enumerate(resolved_order):
        positions_by_joint[int(joint)].append(col)

    seen_by_joint: Dict[int, int] = defaultdict(int)
    column_indices: List[int] = []
    for joint in resolved_order:
        joint = int(joint)
        mirrored_joint = mirror_by_joint[joint]
        occurrence = seen_by_joint[joint]
        seen_by_joint[joint] += 1
        mirrored_positions = positions_by_joint[mirrored_joint]
        if occurrence >= len(mirrored_positions):  # pragma: no cover - invalid mirror map
            raise RuntimeError(f"Mirror occurrence mismatch for joint index {joint}")
        column_indices.append(mirrored_positions[occurrence])
    return column_indices


def dfs_walk_order(root_name: str = "middle_chest") -> List[int]:
    """Return DFS walk order with backtracking nodes included.

    Each edge is walked down and back up; therefore non-root internal nodes
    appear multiple times and the resulting walk has 135 columns.
    """
    adjacency = build_adjacency(paper_tree_edges())
    root = JOINT_INDEX[root_name]
    order: List[int] = []

    def visit(node: int) -> None:
        order.append(node)
        for child in adjacency.get(node, []):
            visit(child)
            order.append(node)

    visit(root)
    expected = 2 * num_selected_joints() - 1
    if len(order) != expected:
        raise RuntimeError(f"DFS walk length {len(order)} != expected {expected}")
    if len(set(order)) != num_selected_joints():
        raise RuntimeError("DFS walk does not cover all selected joints")
    return order


DFS_WALK_ORDER = dfs_walk_order()
DFS_WALK_WIDTH = len(DFS_WALK_ORDER)
DFS_MIRROR_COLUMN_INDICES = mirrored_dfs_column_indices(DFS_WALK_ORDER)
DFS_UNIQUE_COLUMN_INDICES = first_dfs_column_indices(DFS_WALK_ORDER)
SKELETON_PARENT_JOINT_INDICES = parent_joint_indices()
SKELETON_MIRROR_JOINT_INDICES = [
    mirrored_joint_indices()[idx] for idx in range(num_selected_joints())
]
