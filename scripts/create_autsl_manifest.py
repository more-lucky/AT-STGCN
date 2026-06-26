#!/usr/bin/env python
"""Create a project-compatible video manifest from an AUTSL directory.

The expected AUTSL video layout is:

  autsl_root/
    train/
      signer0_sample1_color.mp4
      signer0_sample1_depth.mp4
      ...
    val/
      ...
    test/
      ...

AUTSL label files contain rows in the form:

  signerX_sampleY,label

The output CSV is the raw-video manifest consumed by scripts/preprocess_videos.py:

  video_path,label,split
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
LABEL_EXTENSIONS = {".csv", ".txt"}
MODALITY_SUFFIXES = {"color": "_color", "depth": "_depth"}


@dataclass(frozen=True)
class ManifestRow:
    video_path: Path
    label: str
    split: str
    sample_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create video_path,label,split CSV rows from AUTSL train/val/test directories."
    )
    p.add_argument("--autsl-root", required=True, help="Directory containing AUTSL train/val/test folders")
    p.add_argument("--output", default="data/autsl_videos_manifest.csv", help="Output CSV path")
    p.add_argument(
        "--modality",
        choices=sorted(MODALITY_SUFFIXES),
        default="color",
        help="Video modality to include; use color for this RGB skeleton pipeline",
    )
    p.add_argument("--train-labels", default=None, help="Optional explicit train label CSV/TXT")
    p.add_argument("--val-labels", default=None, help="Optional explicit validation label CSV/TXT")
    p.add_argument("--test-labels", default=None, help="Optional explicit test label CSV/TXT")
    p.add_argument(
        "--include-splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Splits to include; aliases like validation are accepted",
    )
    p.add_argument(
        "--relative-to",
        default=None,
        help="Store video paths relative to this directory. Defaults to absolute paths.",
    )
    p.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Include rows with missing labels as empty strings. Not suitable for training.",
    )
    p.add_argument(
        "--require-all-labels",
        action="store_true",
        help="Fail if any included video is missing a label.",
    )
    return p.parse_args()


def normalize_split(value: str) -> str:
    key = value.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if key in SPLIT_ALIASES:
        return SPLIT_ALIASES[key]
    raise ValueError(f"Unknown split name: {value!r}")


def split_tokens(name: str) -> list[str]:
    return [x for x in re.split(r"[^a-z0-9]+", name.lower()) if x]


def detect_split(path: Path, root: Path) -> str | None:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]

    for part in parts:
        tokens = split_tokens(part)
        for token in tokens:
            if token in SPLIT_ALIASES:
                return SPLIT_ALIASES[token]

        compact = re.sub(r"[^a-z0-9]+", "", part.lower())
        if compact.startswith("train"):
            return "train"
        if compact.startswith(("val", "valid", "validation")):
            return "val"
        if compact.startswith("test"):
            return "test"
    return None


def normalize_sample_id(value: str | Path) -> str:
    stem = Path(str(value).strip()).stem
    for suffix in MODALITY_SUFFIXES.values():
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_label_line(line: str) -> list[str]:
    if "," in line:
        return [cell.strip() for cell in next(csv.reader([line]))]
    if ";" in line:
        return [cell.strip() for cell in next(csv.reader([line], delimiter=";"))]
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t")]
    return [cell.strip() for cell in line.split()]


def is_header_row(cells: list[str]) -> bool:
    if len(cells) < 2:
        return False
    first = cells[0].strip().lower()
    second = cells[1].strip().lower()
    first_names = {"id", "sample", "sample_id", "video", "video_id", "filename", "name"}
    return first in first_names or second in {"label", "class", "class_id", "gloss"}


def read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    first_data_row = True
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        cells = parse_label_line(line)
        if len(cells) < 2:
            raise ValueError(f"Invalid label row in {path} line {line_no}: {raw.rstrip()!r}")
        if first_data_row and is_header_row(cells):
            first_data_row = False
            continue
        first_data_row = False
        sample_id = normalize_sample_id(cells[0])
        label = str(cells[1]).strip()
        if not sample_id or not label:
            raise ValueError(f"Invalid empty sample/label in {path} line {line_no}")
        if sample_id in labels and labels[sample_id] != label:
            raise ValueError(
                f"Conflicting labels for {sample_id!r} in {path}: {labels[sample_id]!r} vs {label!r}"
            )
        labels[sample_id] = label
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def label_file_score(path: Path, split: str) -> int:
    if path.suffix.lower() not in LABEL_EXTENSIONS:
        return -1000

    name = path.stem.lower()
    compact = re.sub(r"[^a-z0-9]+", "", name)
    if "class" in compact and "label" not in compact:
        return -1000
    if "label" not in compact and "groundtruth" not in compact and "gt" != compact:
        return -1000

    aliases = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation"),
        "test": ("test", "testing"),
    }[split]
    score = 0
    if any(alias in split_tokens(name) for alias in aliases):
        score += 100
    if any(compact.startswith(alias) or compact.endswith(alias) for alias in aliases):
        score += 25
    if "label" in compact:
        score += 10
    if "groundtruth" in compact or compact.startswith("gt"):
        score += 5
    return score


def find_label_file(root: Path, split: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        score = label_file_score(path, split)
        if score > 0:
            candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], len(x[1].parts), str(x[1]).lower()))
    return candidates[0][1]


def iter_videos(root: Path, modality: str) -> Iterable[Path]:
    suffix = MODALITY_SUFFIXES[modality]
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if normalize_sample_id(path) == path.stem:
            continue
        if not path.stem.lower().endswith(suffix):
            continue
        yield path


def display_path(path: Path, relative_to: Path | None) -> str:
    resolved = path.resolve(strict=False)
    if relative_to is None:
        return resolved.as_posix()
    base = relative_to.resolve(strict=False)
    try:
        return Path(os.path.relpath(resolved, base)).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_manifest(
    root: Path,
    *,
    modality: str,
    labels_by_split: dict[str, dict[str, str]],
    include_splits: set[str],
    allow_unlabeled: bool,
    require_all_labels: bool,
) -> tuple[list[ManifestRow], dict[str, int]]:
    rows: list[ManifestRow] = []
    stats = {
        "videos_seen": 0,
        "unknown_split": 0,
        "excluded_split": 0,
        "missing_label": 0,
    }

    for video_path in iter_videos(root, modality):
        stats["videos_seen"] += 1
        split = detect_split(video_path, root)
        if split is None:
            stats["unknown_split"] += 1
            continue
        if split not in include_splits:
            stats["excluded_split"] += 1
            continue

        sample_id = normalize_sample_id(video_path)
        label = labels_by_split.get(split, {}).get(sample_id)
        if label is None:
            stats["missing_label"] += 1
            if require_all_labels:
                raise ValueError(f"Missing {split} label for video {video_path}")
            if not allow_unlabeled:
                continue
            label = ""

        rows.append(ManifestRow(video_path=video_path, label=label, split=split, sample_id=sample_id))

    rows.sort(key=lambda row: (SPLIT_ORDER[row.split], row.sample_id, str(row.video_path).lower()))
    return rows, stats


def write_manifest(rows: list[ManifestRow], output: Path, relative_to: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_path": display_path(row.video_path, relative_to),
                    "label": row.label,
                    "split": row.split,
                }
            )


def main() -> None:
    args = parse_args()
    root = Path(args.autsl_root)
    if not root.exists():
        raise FileNotFoundError(f"AUTSL root does not exist: {root}")

    include_splits = {normalize_split(x) for x in args.include_splits}
    explicit_labels = {
        "train": args.train_labels,
        "val": args.val_labels,
        "test": args.test_labels,
    }

    labels_by_split: dict[str, dict[str, str]] = {}
    label_paths: dict[str, Path] = {}
    for split in include_splits:
        label_path = Path(explicit_labels[split]) if explicit_labels[split] else find_label_file(root, split)
        if label_path is None:
            print(f"WARNING: no label file found for split={split}; labeled rows from this split will be skipped")
            labels_by_split[split] = {}
            continue
        labels_by_split[split] = read_labels(label_path)
        label_paths[split] = label_path

    rows, stats = build_manifest(
        root,
        modality=args.modality,
        labels_by_split=labels_by_split,
        include_splits=include_splits,
        allow_unlabeled=args.allow_unlabeled,
        require_all_labels=args.require_all_labels,
    )
    if not rows:
        raise ValueError(
            "No manifest rows were generated. Check --autsl-root, split folder names, modality, and label files."
        )

    output = Path(args.output)
    relative_to = Path(args.relative_to) if args.relative_to else None
    write_manifest(rows, output, relative_to)

    print(f"Wrote {len(rows)} rows to {output}")
    for split in sorted(label_paths, key=lambda x: SPLIT_ORDER[x]):
        print(f"Loaded {len(labels_by_split[split])} {split} labels from {label_paths[split]}")
    print(
        "Scanned {videos_seen} videos; skipped {unknown_split} with unknown split, "
        "{excluded_split} from excluded splits, {missing_label} without labels".format(**stats)
    )
    if args.allow_unlabeled:
        print("WARNING: rows with empty labels are not suitable for scripts/train.py")


if __name__ == "__main__":
    main()
