import pandas as pd
import pytest

from sl_atstgcn.evaluation_protocol import (
    audit_manifest_dataframe,
    enforce_evaluation_role,
    validate_metrics_role,
)


def clean_manifest() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for label in ("A", "B"):
            rows.append(
                {
                    "label": label,
                    "split": split,
                    "path": f"data/example/{split}/{label}/{split}_{label}.npy",
                }
            )
    return pd.DataFrame(rows)


def test_manifest_audit_accepts_disjoint_paper_splits():
    report = audit_manifest_dataframe(clean_manifest(), dataset_name="example")

    assert report["valid"] is True
    assert report["split_counts"] == {"train": 2, "val": 2, "test": 2}
    assert report["class_counts"] == {"train": 2, "val": 2, "test": 2}
    assert report["cross_split_duplicate_count"] == 0


def test_manifest_audit_rejects_cross_split_sample_overlap():
    df = clean_manifest()
    df.loc[df["split"] == "val", "path"] = df.loc[df["split"] == "train", "path"].to_numpy()

    report = audit_manifest_dataframe(df, dataset_name="example")

    assert report["valid"] is False
    assert report["cross_split_duplicate_count"] == 2
    assert any("shared across splits" in error for error in report["errors"])


def test_manifest_audit_rejects_eval_label_absent_from_train():
    df = clean_manifest()
    df.loc[df["split"] == "test", "label"] = "UNSEEN"

    report = audit_manifest_dataframe(df, dataset_name="example")

    assert report["valid"] is False
    assert any("absent from train" in error for error in report["errors"])


def test_evaluation_roles_lock_development_and_test_splits():
    assert enforce_evaluation_role("ablation", "val") == ("ablation", "val")
    assert enforce_evaluation_role("final-test", "test") == ("final-test", "test")

    with pytest.raises(ValueError, match="requires split='test'"):
        enforce_evaluation_role("final-test", "val")
    with pytest.raises(ValueError, match="requires split='val'"):
        enforce_evaluation_role("model-selection", "test")


def test_metrics_role_requires_paper_valid_metadata():
    payload = {
        "evaluation_role": "final-test",
        "split": "test",
        "paper_valid": True,
    }
    validate_metrics_role(payload, expected="final-test")

    payload["paper_valid"] = False
    with pytest.raises(ValueError, match="not marked paper_valid"):
        validate_metrics_role(payload)
