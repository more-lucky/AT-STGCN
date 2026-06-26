#!/usr/bin/env python
"""Create a project-compatible video manifest from ASL_Citizen split CSVs.

Expected dataset layout:

  ASL_Citizen/
    splits/
      train.csv
      val.csv
      test.csv
    videos/
      15890366051589533-APPLE.mp4
      ...

The output CSV is consumed by scripts/preprocess_videos.py:

  video_path,label,split,participant_id,asl_lex_code
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class ManifestRow:
    video_path: Path
    label: str
    split: str
    participant_id: str
    asl_lex_code: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an ASL_Citizen video manifest.")
    parser.add_argument(
        "--asl-root",
        default=r"E:\dataset\DynamicSignDataset\ASL_Citizen",
        help="ASL_Citizen root directory",
    )
    parser.add_argument("--splits-dir", default=None, help="Directory containing train.csv/val.csv/test.csv")
    parser.add_argument("--videos-dir", default=None, help="Directory containing ASL_Citizen video files")
    parser.add_argument("--output", default="data/asl_citizen_videos_manifest.csv", help="Output CSV path")
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Splits to include",
    )
    parser.add_argument(
        "--label-column",
        choices=("Gloss", "ASL-LEX Code"),
        default="Gloss",
        help="Column to use as the class label",
    )
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Store video paths relative to this directory when possible.",
    )
    parser.add_argument(
        "--require-all-videos",
        action="store_true",
        help="Fail if any split row has no matching local video file.",
    )
    return parser.parse_args()


def normalize_split(value: str) -> str:
    key = str(value).strip().lower()
    if key in SPLIT_ALIASES:
        return SPLIT_ALIASES[key]
    raise ValueError(f"Unknown split name: {value!r}")


def build_video_index(videos_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in videos_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            index.setdefault(path.name, path)
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem, path)
            index.setdefault(path.stem.lower(), path)
    return index


def resolve_video(row_value: object, videos_dir: Path, video_index: dict[str, Path]) -> Path | None:
    text = str(row_value).strip()
    if text == "":
        return None
    direct = videos_dir / text
    if direct.exists():
        return direct
    if text in video_index:
        return video_index[text]
    lower = text.lower()
    if lower in video_index:
        return video_index[lower]
    stem = Path(text).stem
    if stem in video_index:
        return video_index[stem]
    if stem.lower() in video_index:
        return video_index[stem.lower()]
    return None


def display_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        try:
            return os.path.relpath(str(path), start=str(relative_to))
        except ValueError:
            return str(path)


def build_manifest(
    *,
    splits_dir: Path,
    videos_dir: Path,
    include_splits: list[str],
    label_column: str,
    require_all_videos: bool,
) -> tuple[list[ManifestRow], dict[str, int]]:
    video_index = build_video_index(videos_dir)
    rows: list[ManifestRow] = []
    stats = {"missing_videos": 0, "rows": 0}
    required_columns = {"Participant ID", "Video file", "Gloss", "ASL-LEX Code"}

    for split_name in include_splits:
        split = normalize_split(split_name)
        split_csv = splits_dir / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"ASL_Citizen split file does not exist: {split_csv}")
        df = pd.read_csv(split_csv)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{split_csv} is missing columns: {sorted(missing)}")
        for record in df.to_dict("records"):
            stats["rows"] += 1
            video_file = record["Video file"]
            video_path = resolve_video(video_file, videos_dir, video_index)
            if video_path is None:
                stats["missing_videos"] += 1
                if require_all_videos:
                    raise FileNotFoundError(f"No video file found for {video_file!r}")
                continue
            label = record[label_column]
            rows.append(
                ManifestRow(
                    video_path=video_path,
                    label=str(label).strip(),
                    split=split,
                    participant_id=str(record["Participant ID"]).strip(),
                    asl_lex_code=str(record["ASL-LEX Code"]).strip(),
                )
            )
    return rows, stats


def write_manifest(rows: list[ManifestRow], output: Path, relative_to: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "label", "split", "participant_id", "asl_lex_code"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_path": display_path(row.video_path, relative_to),
                    "label": row.label,
                    "split": row.split,
                    "participant_id": row.participant_id,
                    "asl_lex_code": row.asl_lex_code,
                }
            )


def main() -> None:
    args = parse_args()
    asl_root = Path(args.asl_root)
    if not asl_root.exists():
        raise FileNotFoundError(f"ASL_Citizen root does not exist: {asl_root}")
    splits_dir = Path(args.splits_dir) if args.splits_dir else asl_root / "splits"
    videos_dir = Path(args.videos_dir) if args.videos_dir else asl_root / "videos"
    if not splits_dir.exists():
        raise FileNotFoundError(f"ASL_Citizen splits dir does not exist: {splits_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"ASL_Citizen videos dir does not exist: {videos_dir}")
    rows, stats = build_manifest(
        splits_dir=splits_dir,
        videos_dir=videos_dir,
        include_splits=list(args.include_splits),
        label_column=str(args.label_column),
        require_all_videos=bool(args.require_all_videos),
    )
    if not rows:
        raise ValueError("No ASL_Citizen manifest rows were generated.")
    relative_to = Path(args.relative_to) if args.relative_to else None
    write_manifest(rows, Path(args.output), relative_to)
    labels = sorted({row.label for row in rows})
    split_counts = {split: sum(1 for row in rows if row.split == split) for split in ("train", "val", "test")}
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Classes: {len(labels)}")
    print(f"Split counts: {split_counts}")
    if stats["missing_videos"]:
        print(f"WARNING: skipped {stats['missing_videos']} rows with missing videos")


if __name__ == "__main__":
    main()
