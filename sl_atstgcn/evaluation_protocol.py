"""Auditable dataset-split and paper-evaluation protocol helpers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PAPER_SPLITS = ("train", "val", "test")
EVALUATION_ROLE_SPLITS = {
    "model-selection": "val",
    "ablation": "val",
    "efficiency": "val",
    "error-analysis": "val",
    "final-test": "test",
}
IDENTITY_COLUMNS = ("source_path", "video_path", "keypoints_path", "path")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_split(value: object) -> str:
    return str(value).strip().lower()


def enforce_evaluation_role(role: str, split: str) -> tuple[str, str]:
    normalized_role = str(role).strip().lower()
    normalized_split = normalize_split(split)
    if normalized_role not in EVALUATION_ROLE_SPLITS:
        raise ValueError(
            f"Unknown evaluation role {role!r}; expected one of {sorted(EVALUATION_ROLE_SPLITS)}"
        )
    expected_split = EVALUATION_ROLE_SPLITS[normalized_role]
    if normalized_split != expected_split:
        raise ValueError(
            f"evaluation_role={normalized_role!r} requires split={expected_split!r}, "
            f"got split={normalized_split!r}"
        )
    return normalized_role, normalized_split


def _portable_identity(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().replace("\\", "/").lower()
    while "//" in text:
        text = text.replace("//", "/")
    for marker in ("/data/", "/runs/"):
        index = text.rfind(marker)
        if index >= 0:
            return text[index + 1 :]
    if len(text) >= 2 and text[1] == ":":
        text = text[2:]
    return text.lstrip("/")


def _sample_identities(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    available = [column for column in IDENTITY_COLUMNS if column in df.columns]
    if not available:
        return pd.Series([""] * len(df), index=df.index, dtype="object"), []

    identities = pd.Series([""] * len(df), index=df.index, dtype="object")
    for column in available:
        values = df[column].map(_portable_identity)
        missing = identities.eq("") & values.ne("")
        identities.loc[missing] = values.loc[missing]
    return identities, available


def audit_manifest_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    required_splits: Iterable[str] = PAPER_SPLITS,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Audit a manifest without reading sample payloads.

    Errors indicate a protocol violation. Warnings are reproducibility risks that
    should be inspected but do not necessarily invalidate the split.
    """

    errors: list[str] = []
    warnings: list[str] = []
    required_columns = {"label", "split"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        return {
            "schema_version": 1,
            "dataset": str(dataset_name),
            "created_at_utc": utc_now_iso(),
            "valid": False,
            "errors": [f"Missing manifest columns: {missing_columns}"],
            "warnings": [],
            "split_counts": {},
            "class_counts": {},
            "identity_columns": [],
            "cross_split_duplicate_examples": [],
        }

    work = df.copy()
    work["split"] = work["split"].map(normalize_split)
    work["label"] = work["label"].astype(str)
    required = tuple(normalize_split(split) for split in required_splits)
    observed = sorted(split for split in work["split"].unique().tolist() if split)
    missing_splits = sorted(set(required) - set(observed))
    unknown_splits = sorted(set(observed) - set(required))
    if missing_splits:
        errors.append(f"Missing required splits: {missing_splits}")
    if unknown_splits:
        errors.append(f"Unexpected split names: {unknown_splits}")

    split_counts = {
        split: int((work["split"] == split).sum())
        for split in required
    }
    class_counts = {
        split: int(work.loc[work["split"] == split, "label"].nunique())
        for split in required
    }
    train_labels = set(work.loc[work["split"] == "train", "label"].tolist())
    for split in ("val", "test"):
        split_labels = set(work.loc[work["split"] == split, "label"].tolist())
        unseen = sorted(split_labels - train_labels)
        if unseen:
            errors.append(
                f"split={split!r} contains {len(unseen)} labels absent from train; "
                f"examples={unseen[:max_examples]}"
            )

    identities, identity_columns = _sample_identities(work)
    if not identity_columns:
        errors.append(f"No sample identity column found; expected one of {list(IDENTITY_COLUMNS)}")
    missing_identity_count = int(identities.eq("").sum())
    if missing_identity_count:
        errors.append(f"{missing_identity_count} rows have no usable sample identity")

    identity_frame = pd.DataFrame(
        {"sample_id": identities, "split": work["split"]},
        index=work.index,
    )
    identity_frame = identity_frame[identity_frame["sample_id"].ne("")]
    cross_split = (
        identity_frame.groupby("sample_id", sort=False)["split"]
        .agg(lambda values: sorted(set(values)))
    )
    cross_split = cross_split[cross_split.map(len) > 1]
    duplicate_examples = [
        {"sample_id": str(sample_id), "splits": list(splits)}
        for sample_id, splits in cross_split.head(max_examples).items()
    ]
    if len(cross_split):
        errors.append(
            f"Detected {len(cross_split)} sample identities shared across splits; "
            f"examples={duplicate_examples}"
        )

    within_split_duplicates = int(identity_frame.duplicated(["sample_id", "split"]).sum())
    if within_split_duplicates:
        warnings.append(f"Detected {within_split_duplicates} duplicate rows within the same split")

    return {
        "schema_version": 1,
        "dataset": str(dataset_name),
        "created_at_utc": utc_now_iso(),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "row_count": int(len(work)),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "identity_columns": identity_columns,
        "missing_identity_count": missing_identity_count,
        "cross_split_duplicate_count": int(len(cross_split)),
        "cross_split_duplicate_examples": duplicate_examples,
        "within_split_duplicate_count": within_split_duplicates,
    }


def validate_metrics_role(payload: dict[str, Any], *, expected: str | None = None) -> None:
    role = str(payload.get("evaluation_role", "")).strip().lower()
    split = str(payload.get("split", "")).strip().lower()
    if not role or not split:
        raise ValueError("Metrics are missing evaluation_role or split metadata")
    enforce_evaluation_role(role, split)
    if expected is not None and role != str(expected).strip().lower():
        raise ValueError(f"Expected evaluation_role={expected!r}, got {role!r}")
    if not bool(payload.get("paper_valid", False)):
        raise ValueError("Metrics are not marked paper_valid")
