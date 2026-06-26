#!/usr/bin/env python
"""Validate and summarize paper metrics without mixing val and test results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.evaluation_protocol import validate_metrics_role


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main",
        action="append",
        default=[],
        metavar="DATASET=METRICS_JSON",
        help="Official-test main result. Repeat once per dataset.",
    )
    parser.add_argument(
        "--development",
        action="append",
        default=[],
        metavar="DATASET=METRICS_JSON",
        help="Validation-only ablation/efficiency/error-analysis result.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected DATASET=METRICS_JSON, got {value!r}")
    dataset, path = value.split("=", 1)
    dataset = dataset.strip()
    if not dataset or not path.strip():
        raise ValueError(f"Expected DATASET=METRICS_JSON, got {value!r}")
    return dataset, Path(path.strip()).resolve()


def load_result(value: str, *, group: str) -> dict[str, Any]:
    assigned_dataset, path = parse_assignment(value)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_metrics_role(payload)

    payload_dataset = str(payload.get("dataset", "")).strip()
    if payload_dataset.lower() != assigned_dataset.lower():
        raise ValueError(
            f"Dataset mismatch for {path}: assignment={assigned_dataset!r}, "
            f"metrics={payload_dataset!r}"
        )
    role = str(payload["evaluation_role"]).strip().lower()
    if group == "main" and role != "final-test":
        raise ValueError(f"Main result must use evaluation_role='final-test': {path}")
    if group == "development" and role == "final-test":
        raise ValueError(f"Development result cannot use evaluation_role='final-test': {path}")

    return {
        "group": group,
        "dataset": payload_dataset,
        "run_name": str(payload.get("run_name", "")),
        "evaluation_role": role,
        "split": str(payload["split"]).strip().lower(),
        "top1": float(payload["top1"]),
        "top5": float(payload["top5"]),
        "sample_count": int(payload["sample_count"]),
        "num_classes": int(payload["num_classes"]),
        "tta_flip": bool(payload.get("tta_flip", False)),
        "tta_scales": payload.get("tta_scales", [1.0]),
        "checkpoint_selection": str(payload.get("checkpoint_selection", "")),
        "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")),
        "manifest_sha256": str(payload.get("manifest_sha256", "")),
        "label_map_sha256": str(payload.get("label_map_sha256", "")),
        "protocol_id": str(payload.get("protocol_id", "")),
        "metrics_path": str(path),
    }


def validate_dataset_consistency(rows: list[dict[str, Any]]) -> None:
    seen_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            row["dataset"].lower(),
            row["run_name"].lower(),
            row["evaluation_role"],
        )
        if key in seen_keys:
            raise ValueError(f"Duplicate dataset/run/role result: {key}")
        seen_keys.add(key)

    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for field in ("manifest_sha256", "label_map_sha256"):
            values = {row[field] for row in dataset_rows if row[field]}
            if len(values) > 1:
                raise ValueError(f"Dataset {dataset!r} mixes different {field} values")


def main() -> None:
    args = parse_args()
    if not args.main:
        raise ValueError("At least one --main DATASET=METRICS_JSON result is required")

    rows = [load_result(value, group="main") for value in args.main]
    rows.extend(load_result(value, group="development") for value in args.development)
    validate_dataset_consistency(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paper_results.json"
    csv_path = output_dir / "paper_results.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["tta_scales"] = ";".join(str(value) for value in row["tta_scales"])
            writer.writerow(csv_row)

    for row in rows:
        print(
            f"{row['group']:11s} {row['dataset']:16s} {row['run_name']:24s} "
            f"split={row['split']:4s} top1={row['top1'] * 100:.2f} "
            f"top5={row['top5'] * 100:.2f}"
        )
    print(f"Paper result protocol verified: {csv_path}")


if __name__ == "__main__":
    main()
