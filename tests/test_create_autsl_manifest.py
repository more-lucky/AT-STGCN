import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_autsl_manifest.py"
SPEC = importlib.util.spec_from_file_location("create_autsl_manifest", SCRIPT_PATH)
autsl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = autsl
SPEC.loader.exec_module(autsl)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_builds_color_manifest_from_official_style_layout(tmp_path):
    root = tmp_path / "AUTSL"
    touch(root / "train" / "signer0_sample1_color.mp4")
    touch(root / "train" / "signer0_sample1_depth.mp4")
    touch(root / "validation" / "signer1_sample2_color.mp4")
    touch(root / "test" / "signer2_sample3_color.mp4")
    (root / "train_labels.csv").write_text("signer0_sample1,4\n", encoding="utf-8")
    (root / "validation_labels.csv").write_text("sample_id,label\nsigner1_sample2_color.mp4,5\n", encoding="utf-8")
    (root / "test_labels.txt").write_text("signer2_sample3 6\n", encoding="utf-8")

    labels = {
        split: autsl.read_labels(autsl.find_label_file(root, split))
        for split in ("train", "val", "test")
    }
    rows, stats = autsl.build_manifest(
        root,
        modality="color",
        labels_by_split=labels,
        include_splits={"train", "val", "test"},
        allow_unlabeled=False,
        require_all_labels=True,
    )
    output = tmp_path / "manifest.csv"
    autsl.write_manifest(rows, output, root)

    assert stats["videos_seen"] == 3
    assert [(row.sample_id, row.label, row.split) for row in rows] == [
        ("signer0_sample1", "4", "train"),
        ("signer1_sample2", "5", "val"),
        ("signer2_sample3", "6", "test"),
    ]
    assert read_csv(output) == [
        {"video_path": "train/signer0_sample1_color.mp4", "label": "4", "split": "train"},
        {"video_path": "validation/signer1_sample2_color.mp4", "label": "5", "split": "val"},
        {"video_path": "test/signer2_sample3_color.mp4", "label": "6", "split": "test"},
    ]


def test_skips_unlabeled_videos_by_default(tmp_path):
    root = tmp_path / "AUTSL"
    touch(root / "train" / "signer0_sample1_color.mp4")
    touch(root / "train" / "signer0_sample2_color.mp4")

    rows, stats = autsl.build_manifest(
        root,
        modality="color",
        labels_by_split={"train": {"signer0_sample1": "4"}},
        include_splits={"train"},
        allow_unlabeled=False,
        require_all_labels=False,
    )

    assert stats["videos_seen"] == 2
    assert stats["missing_label"] == 1
    assert [(row.sample_id, row.label) for row in rows] == [("signer0_sample1", "4")]
