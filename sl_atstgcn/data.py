"""Dataset utilities for manifests and PyTorch input pipelines."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augment import AugmentConfig
from .graph import SKELETON_MIRROR_JOINT_INDICES
from .keypoints import num_selected_joints


REQUIRED_MANIFEST_COLUMNS = {"label", "split"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_PATH_MARKERS = ("data", "runs")
NUM_SELECTED_JOINTS = num_selected_joints()


@dataclass(frozen=True)
class LabelMap:
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> "LabelMap":
        unique = sorted({str(x) for x in labels})
        label_to_id = {label: i for i, label in enumerate(unique)}
        return cls(label_to_id=label_to_id, id_to_label={i: label for label, i in label_to_id.items()})

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.label_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        import json

        with Path(path).open("r", encoding="utf-8") as f:
            label_to_id = {str(k): int(v) for k, v in json.load(f).items()}
        return cls(label_to_id=label_to_id, id_to_label={v: k for k, v in label_to_id.items()})


def _split_portable_suffix(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    lowered = normalized.lower()
    for marker in PORTABLE_PATH_MARKERS:
        token = f"/{marker}/"
        index = lowered.rfind(token)
        if index >= 0:
            return normalized[index + 1 :]
    repo_name = PROJECT_ROOT.name.lower()
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    lowered_parts = [part.lower() for part in parts]
    if repo_name in lowered_parts:
        repo_index = lowered_parts.index(repo_name)
        return "/".join(parts[repo_index + 1 :])
    if len(normalized) >= 2 and normalized[1] == ":":
        return normalized[2:].lstrip("/")
    return normalized.lstrip("/")


def _manifest_search_roots(manifest_path: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    env_values = [
        os.environ.get("SL_ATSTGCN_DATA_ROOT", ""),
        os.environ.get("SL_SKELETON_DATA_ROOT", ""),
        os.environ.get("SL_TSSI_DATA_ROOT", ""),
    ]
    for env_value in env_values:
        for item in env_value.split(os.pathsep):
            item = item.strip()
            if item:
                roots.append(Path(item))
    if manifest_path is not None:
        roots.append(manifest_path.resolve().parent)
    roots.extend([PROJECT_ROOT, Path.cwd()])
    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def resolve_portable_path(value: object, *, search_roots: Sequence[str | Path] | None = None) -> str:
    """Resolve manifest paths after moving CSVs between Windows/Linux machines."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return text
    direct = Path(text)
    if direct.exists():
        return text
    normalized = text.replace("\\", "/")
    normalized_path = Path(normalized)
    if normalized_path.exists():
        return str(normalized_path)
    suffix = _split_portable_suffix(normalized)
    roots = [Path(root) for root in (search_roots or _manifest_search_roots())]
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / suffix)
        if suffix.startswith("data/"):
            candidates.append(root / suffix.removeprefix("data/"))
    basename = Path(normalized).name
    if basename:
        candidates.extend(root / basename for root in roots)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return text


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    if "path" not in df.columns and "keypoints_path" not in df.columns:
        raise ValueError(f"Manifest {path} must contain either 'path' or 'keypoints_path'")
    df = df.copy()
    search_roots = _manifest_search_roots(manifest_path)
    for column in ("path", "keypoints_path", "video_path"):
        if column in df.columns:
            df[column] = df[column].map(lambda value: resolve_portable_path(value, search_roots=search_roots))
    if "path" not in df.columns:
        df["path"] = df["keypoints_path"]
    df["label"] = df["label"].astype(str)
    df["split"] = df["split"].astype(str)
    return df


def attach_label_ids(df: pd.DataFrame, label_map: LabelMap | None = None) -> Tuple[pd.DataFrame, LabelMap]:
    if label_map is None:
        label_map = LabelMap.from_labels(df["label"])
    out = df.copy()
    out["label_id"] = out["label"].map(label_map.label_to_id)
    if out["label_id"].isna().any():
        bad = out[out["label_id"].isna()]["label"].unique().tolist()
        raise ValueError(f"Labels missing from label map: {bad}")
    out["label_id"] = out["label_id"].astype(int)
    return out, label_map


def _load_npy(path: str | Path) -> np.ndarray:
    return np.load(str(path)).astype(np.float32)


def _resize_or_pad_skeleton(sequence: np.ndarray, *, length: int) -> np.ndarray:
    if sequence.ndim != 3 or sequence.shape[1] != NUM_SELECTED_JOINTS:
        raise ValueError(f"Expected skeleton shape (T, 68, C), got {sequence.shape}")
    sequence = sequence.astype(np.float32, copy=False)
    target = int(length)
    if target <= 0:
        raise ValueError(f"sequence_length must be positive, got {length}")
    t = int(sequence.shape[0])
    if t == target:
        return sequence.astype(np.float32, copy=False)
    if t <= 0:
        return np.zeros((target, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    if t > target:
        indices = np.linspace(0, t - 1, target).round().astype(np.int64)
        return sequence[indices].astype(np.float32, copy=False)
    out = np.zeros((target, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    out[:t] = sequence
    return out


def _infer_valid_skeleton_length(sequence: np.ndarray) -> int:
    valid_rows = np.any((sequence[:, :, 0] != 0.0) | (sequence[:, :, 1] != 0.0), axis=1)
    valid_indices = np.flatnonzero(valid_rows)
    if len(valid_indices) == 0:
        return int(sequence.shape[0])
    return int(valid_indices[-1]) + 1


def _resample_skeleton(sequence: np.ndarray, *, length: int) -> np.ndarray:
    target = int(max(1, length))
    if sequence.shape[0] == target:
        return sequence.astype(np.float32, copy=True)
    source_t = int(sequence.shape[0])
    if source_t <= 0:
        return np.zeros((target, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    source_x = np.arange(source_t, dtype=np.float32)
    target_x = np.linspace(0.0, float(source_t - 1), target, dtype=np.float32)
    out = np.zeros((target, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    for joint in range(sequence.shape[1]):
        for channel in range(sequence.shape[2]):
            out[:, joint, channel] = np.interp(target_x, source_x, sequence[:, joint, channel]).astype(np.float32)
    return out


def prepare_skeleton_sequence(sequence: np.ndarray, *, sequence_length: int) -> np.ndarray:
    """Convert a raw skeleton array to compact ``(T, 68, 3)`` format."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D skeleton array, got {arr.shape}")
    if arr.shape[1] == NUM_SELECTED_JOINTS and arr.shape[2] >= 2:
        compact = arr[:, :, : min(3, arr.shape[2])]
    elif arr.shape[0] in {2, 3} and arr.shape[2] == NUM_SELECTED_JOINTS:
        compact = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(
            f"Expected skeleton (T,68,C) or channel-first (C,T,68), got {arr.shape}"
        )
    if compact.shape[2] == 2:
        zeros = np.zeros((*compact.shape[:2], 1), dtype=np.float32)
        compact = np.concatenate([compact, zeros], axis=2)
    return _resize_or_pad_skeleton(compact[:, :, :3], length=int(sequence_length))


def load_skeleton_sequence(path: str | Path, *, sequence_length: int) -> np.ndarray:
    """Load a compact ``(T, 68, 3)`` skeleton sequence.

    The current paper path expects already extracted 68-joint keypoint
    sequences and consumes joints directly instead of treating them as images.
    """
    return prepare_skeleton_sequence(_load_npy(path), sequence_length=sequence_length)


def repair_missing_skeleton_keypoints(sequence: np.ndarray, *, min_valid: int = 2) -> np.ndarray:
    if sequence.ndim != 3 or sequence.shape[1] != NUM_SELECTED_JOINTS:
        raise ValueError(f"Expected skeleton shape (T, 68, C), got {sequence.shape}")
    out = sequence.astype(np.float32, copy=True)
    valid_t = out.shape[0]
    frame_index = np.arange(valid_t, dtype=np.float32)
    valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    min_valid = max(1, int(min_valid))
    for joint in range(out.shape[1]):
        mask = valid[:, joint]
        if int(mask.sum()) < min_valid or bool(mask.all()):
            continue
        source_x = frame_index[mask]
        for channel in range(min(2, out.shape[2])):
            source_y = out[:, joint, channel][mask]
            out[:, joint, channel] = np.interp(frame_index, source_x, source_y).astype(np.float32)
    return out


def augment_skeleton_sequence(sequence: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    out = sequence.astype(np.float32, copy=True)
    valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    if bool(cfg.flip) and rng.random() < float(cfg.flip_prob):
        out[:, :, 0] = np.where(valid, 1.0 - out[:, :, 0], out[:, :, 0])
        if bool(cfg.swap_sides_on_flip):
            out = out[:, SKELETON_MIRROR_JOINT_INDICES, :]
            valid = valid[:, SKELETON_MIRROR_JOINT_INDICES]
    if bool(cfg.scale):
        scale = float(rng.uniform(float(cfg.scale_min), float(cfg.scale_max)))
        valid_f = valid.astype(np.float32)
        denom = np.clip(valid_f.sum(), 1.0, None)
        center = (out[:, :, :2] * valid_f[:, :, None]).sum(axis=(0, 1)) / denom
        out[:, :, :2] = np.where(
            valid[:, :, None],
            np.clip((out[:, :, :2] - center) * scale + center, 0.0, 1.0),
            out[:, :, :2],
        )
    if bool(cfg.shift):
        shift = rng.uniform(-float(cfg.shift_max), float(cfg.shift_max), size=(2,)).astype(np.float32)
        out[:, :, :2] = np.where(valid[:, :, None], np.clip(out[:, :, :2] + shift, 0.0, 1.0), out[:, :, :2])
    if bool(cfg.noise) and float(cfg.noise_std) > 0.0:
        noise = rng.normal(0.0, float(cfg.noise_std), size=out[:, :, :2].shape).astype(np.float32)
        out[:, :, :2] = np.where(valid[:, :, None], np.clip(out[:, :, :2] + noise, 0.0, 1.0), out[:, :, :2])
    if bool(cfg.rotate) and float(cfg.rotate_degrees) > 0.0 and np.any(valid):
        angle = float(rng.uniform(-float(cfg.rotate_degrees), float(cfg.rotate_degrees)))
        radians = np.deg2rad(angle)
        cos_v = float(np.cos(radians))
        sin_v = float(np.sin(radians))
        valid_f = valid.astype(np.float32)
        denom = np.clip(valid_f.sum(), 1.0, None)
        center = (out[:, :, :2] * valid_f[:, :, None]).sum(axis=(0, 1)) / denom
        coords = out[:, :, :2] - center
        rotated = np.empty_like(coords)
        rotated[:, :, 0] = coords[:, :, 0] * cos_v - coords[:, :, 1] * sin_v
        rotated[:, :, 1] = coords[:, :, 0] * sin_v + coords[:, :, 1] * cos_v
        out[:, :, :2] = np.where(valid[:, :, None], np.clip(rotated + center, 0.0, 1.0), out[:, :, :2])
    if bool(cfg.speed):
        valid_t = _infer_valid_skeleton_length(out)
        new_t = int(rng.integers(int(cfg.speed_min_frames), int(cfg.speed_max_frames) + 1))
        resized = _resample_skeleton(out[:valid_t], length=max(1, new_t))
        out = _resize_or_pad_skeleton(resized, length=out.shape[0])
        valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    elif bool(cfg.temporal_crop):
        valid_t = _infer_valid_skeleton_length(out)
        if valid_t > 2:
            ratio = float(rng.uniform(float(cfg.temporal_crop_min_ratio), float(cfg.temporal_crop_max_ratio)))
            crop_t = max(2, min(valid_t, int(round(valid_t * ratio))))
            if crop_t < valid_t:
                start = int(rng.integers(0, valid_t - crop_t + 1))
                out = _resize_or_pad_skeleton(out[start : start + crop_t], length=out.shape[0])
                valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    if bool(cfg.temporal_crop) and bool(cfg.speed):
        valid_t = _infer_valid_skeleton_length(out)
        if valid_t > 2:
            ratio = float(rng.uniform(float(cfg.temporal_crop_min_ratio), float(cfg.temporal_crop_max_ratio)))
            crop_t = max(2, min(valid_t, int(round(valid_t * ratio))))
            if crop_t < valid_t:
                start = int(rng.integers(0, valid_t - crop_t + 1))
                out = _resize_or_pad_skeleton(out[start : start + crop_t], length=out.shape[0])
                valid = (out[:, :, 0] != 0.0) | (out[:, :, 1] != 0.0)
    if bool(cfg.time_mask) and rng.random() < float(cfg.time_mask_prob):
        valid_t = _infer_valid_skeleton_length(out)
        width = max(1, int(round(valid_t * float(cfg.time_mask_max_ratio))))
        start = int(rng.integers(0, max(1, valid_t - width + 1)))
        out[start : start + width] = 0.0
    if bool(cfg.joint_mask) and rng.random() < float(cfg.joint_mask_prob):
        width = max(1, min(int(cfg.joint_mask_max_width), out.shape[1]))
        start = int(rng.integers(0, max(1, out.shape[1] - width + 1)))
        out[:, start : start + width] = 0.0
    return out.astype(np.float32, copy=False)


class SkeletonSequenceDataset(Dataset):
    """Dataset that feeds compact 68-joint skeleton sequences directly."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        sequence_length: int,
        training: bool,
        augment_config: AugmentConfig | None = None,
        seed: int = 42,
        repair_missing: bool = False,
        repair_min_valid: int = 2,
        soft_targets: np.ndarray | None = None,
    ) -> None:
        path_column = "keypoints_path" if "keypoints_path" in df.columns else "path"
        self.paths = df[path_column].astype(str).tolist()
        self.labels = df["label_id"].astype(np.int64).to_numpy()
        self.sequence_length = int(sequence_length)
        self.training = bool(training)
        self.repair_missing = bool(repair_missing)
        self.repair_min_valid = int(repair_min_valid)
        self.augment_config = augment_config if augment_config is not None and augment_config.enabled else None
        self.seed = int(seed)
        self.soft_targets = None
        if soft_targets is not None:
            soft_targets = np.asarray(soft_targets, dtype=np.float32)
            if soft_targets.ndim != 2:
                raise ValueError(f"soft_targets must be a 2D array, got shape {soft_targets.shape}")
            if len(soft_targets) != len(self.labels):
                raise ValueError(f"soft_targets rows {len(soft_targets)} != dataset rows {len(self.labels)}")
            self.soft_targets = np.ascontiguousarray(soft_targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        sequence = load_skeleton_sequence(self.paths[index], sequence_length=self.sequence_length)
        if self.repair_missing:
            sequence = repair_missing_skeleton_keypoints(sequence, min_valid=self.repair_min_valid)
        if self.training and self.augment_config is not None:
            worker_seed = int(torch.initial_seed() % (2**32 - 1))
            draw_seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
            rng_seed = (worker_seed ^ draw_seed ^ (self.seed + int(index) * 9973)) % (2**32 - 1)
            rng = np.random.default_rng(rng_seed)
            sequence = augment_skeleton_sequence(sequence, self.augment_config, rng)
        sequence = np.ascontiguousarray(sequence.transpose(2, 0, 1), dtype=np.float32)
        label = int(self.labels[index])
        sequence_tensor = torch.from_numpy(sequence)
        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.soft_targets is None:
            return sequence_tensor, label_tensor
        return sequence_tensor, label_tensor, torch.from_numpy(self.soft_targets[index])


def make_skeleton_dataloader(
    df: pd.DataFrame,
    *,
    sequence_length: int,
    batch_size: int,
    training: bool,
    augment_config: AugmentConfig | None = None,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    class_balanced: bool = False,
    repair_missing: bool = False,
    repair_min_valid: int = 2,
    soft_targets: np.ndarray | None = None,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    """Create a DataLoader for compact skeleton sequences shaped ``(C, T, 68)``."""
    dataset = SkeletonSequenceDataset(
        df,
        sequence_length=sequence_length,
        training=training,
        augment_config=augment_config,
        seed=seed,
        repair_missing=repair_missing,
        repair_min_valid=repair_min_valid,
        soft_targets=soft_targets,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = None
    shuffle = bool(training)
    if training and bool(class_balanced):
        counts = np.bincount(dataset.labels)
        counts = np.maximum(counts, 1)
        sample_weights = 1.0 / counts[dataset.labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator if training and sampler is None else None,
    )


def compute_mean_sequence_length_from_manifest(df: pd.DataFrame) -> int:
    """Compute the rounded mean temporal length of saved skeleton sequences."""
    lengths: List[int] = []
    for p in df["path"].astype(str):
        lengths.append(int(np.load(p, mmap_mode="r").shape[0]))
    if not lengths:
        raise ValueError("Cannot compute mean sequence length from an empty manifest")
    return int(round(float(np.mean(lengths))))
