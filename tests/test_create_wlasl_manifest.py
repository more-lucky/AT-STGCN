import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_wlasl_manifest.py"
SPEC = importlib.util.spec_from_file_location("create_wlasl_manifest", SCRIPT_PATH)
wlasl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = wlasl
SPEC.loader.exec_module(wlasl)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_builds_wlasl100_manifest_from_official_metadata(tmp_path):
    root = tmp_path / "WLASL"
    videos = root / "videos"
    touch(videos / "00001.mp4")
    touch(videos / "00002.mp4")
    touch(videos / "00003.mp4")
    touch(videos / "00004.mp4")
    metadata = [
        {
            "gloss": "book",
            "instances": [
                {"video_id": "00001", "split": "train", "frame_start": 1, "frame_end": -1},
                {"video_id": "00002", "split": "val", "frame_start": 3, "frame_end": 12},
            ],
        },
        {
            "gloss": "drink",
            "instances": [
                {"video_id": "00003", "split": "test", "frame_start": 2, "frame_end": 7},
            ],
        },
        {
            "gloss": "ignored",
            "instances": [
                {"video_id": "00004", "split": "train", "frame_start": 1, "frame_end": -1},
            ],
        },
    ]
    metadata_path = root / "start_kit" / "WLASL_v0.3.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    records = wlasl.read_records(
        metadata_path,
        subset_size=2,
        gloss_list=None,
        label_mode="gloss",
        include_splits={"train", "val", "test"},
    )
    rows, stats = wlasl.build_manifest(records, wlasl.index_videos(videos), require_all_videos=True)
    output = tmp_path / "manifest.csv"
    wlasl.write_manifest(rows, output, root)

    assert stats == {"records_seen": 3, "missing_video": 0}
    assert [(row.video_id, row.label, row.split, row.frame_start, row.frame_end) for row in rows] == [
        ("00001", "book", "train", 1, -1),
        ("00002", "book", "val", 3, 12),
        ("00003", "drink", "test", 2, 7),
    ]
    assert read_csv(output) == [
        {
            "video_path": "videos/00001.mp4",
            "label": "book",
            "split": "train",
            "video_id": "00001",
            "frame_start": "1",
            "frame_end": "-1",
        },
        {
            "video_path": "videos/00002.mp4",
            "label": "book",
            "split": "val",
            "video_id": "00002",
            "frame_start": "3",
            "frame_end": "12",
        },
        {
            "video_path": "videos/00003.mp4",
            "label": "drink",
            "split": "test",
            "video_id": "00003",
            "frame_start": "2",
            "frame_end": "7",
        },
    ]


def test_nslt_metadata_uses_class_indices_without_glosses(tmp_path):
    videos = tmp_path / "videos"
    touch(videos / "abc.mp4")
    data = {"abc": {"subset": "train", "action": [5, 4, 9]}}
    metadata_path = tmp_path / "nslt_100.json"
    metadata_path.write_text(json.dumps(data), encoding="utf-8")

    records = wlasl.read_records(
        metadata_path,
        subset_size=100,
        gloss_list=None,
        label_mode="gloss",
        include_splits={"train"},
    )
    rows, stats = wlasl.build_manifest(records, wlasl.index_videos(videos), require_all_videos=False)

    assert stats == {"records_seen": 1, "missing_video": 0}
    assert [(row.video_id, row.label, row.split, row.frame_start, row.frame_end) for row in rows] == [
        ("abc", "5", "train", 4, 9)
    ]


def test_missing_videos_are_skipped_by_default(tmp_path):
    records = [
        wlasl.WLASLRecord(video_id="missing", label="book", split="train", class_index=0),
    ]

    rows, stats = wlasl.build_manifest(records, {}, require_all_videos=False)

    assert rows == []
    assert stats == {"records_seen": 1, "missing_video": 1}
