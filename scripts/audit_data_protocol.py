#!/usr/bin/env python
"""Audit train/validation/test manifests for paper evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.evaluation_protocol import audit_manifest_dataframe, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Stable dataset name used in paper metadata")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Output JSON audit report")
    parser.add_argument(
        "--required-splits",
        default="train,val,test",
        help="Comma-separated required split names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    required_splits = [item.strip() for item in args.required_splits.split(",") if item.strip()]
    df = pd.read_csv(manifest_path)
    report = audit_manifest_dataframe(
        df,
        dataset_name=args.dataset,
        required_splits=required_splits,
    )
    report["manifest"] = str(manifest_path)
    report["manifest_sha256"] = sha256_file(manifest_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"dataset={report['dataset']}")
    print(f"split_counts={report['split_counts']}")
    print(f"class_counts={report['class_counts']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    if not report["valid"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(2)
    print(f"Protocol audit passed: {output_path}")


if __name__ == "__main__":
    main()
