#!/usr/bin/env python
"""Rewrite moved manifest paths so Windows/Linux training machines can share CSVs."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sl_atstgcn.data import resolve_portable_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Additional root to search, e.g. /root/autodl-tmp/test or /root/autodl-tmp/test/data",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    if args.in_place:
        output_manifest = input_manifest
    elif args.output_manifest:
        output_manifest = Path(args.output_manifest)
    else:
        stem = input_manifest.stem + "_portable"
        output_manifest = input_manifest.with_name(stem + input_manifest.suffix)

    env_values = [
        os.environ.get("SL_ATSTGCN_DATA_ROOT", ""),
        os.environ.get("SL_SKELETON_DATA_ROOT", ""),
        os.environ.get("SL_TSSI_DATA_ROOT", ""),
    ]
    env_roots = [
        item
        for env_value in env_values
        for item in env_value.split(os.pathsep)
        if item.strip()
    ]
    search_roots = [Path(root) for root in args.root + env_roots]
    search_roots.extend([input_manifest.resolve().parent, PROJECT_ROOT, Path.cwd()])

    df = pd.read_csv(input_manifest)
    for column in ("path", "keypoints_path", "video_path"):
        if column in df.columns:
            df[column] = df[column].map(lambda value: resolve_portable_path(value, search_roots=search_roots))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest, index=False)
    print(f"Wrote {output_manifest}")


if __name__ == "__main__":
    main()
