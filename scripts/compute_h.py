#!/usr/bin/env python
"""Compute H as the mean training-set sequence length before resizing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV with keypoints_path or path column")
    p.add_argument("--split", default="train")
    args = p.parse_args()
    df = pd.read_csv(args.manifest)
    df = df[df["split"].astype(str) == args.split]
    if "valid_pose_frames" in df.columns:
        lengths = df["valid_pose_frames"].astype(int).tolist()
    elif "original_frames" in df.columns:
        lengths = df["original_frames"].astype(int).tolist()
    else:
        col = "keypoints_path" if "keypoints_path" in df.columns else "path"
        lengths = [int(np.load(path, mmap_mode="r").shape[0]) for path in df[col].astype(str)]
    print(int(round(float(np.mean(lengths)))))


if __name__ == "__main__":
    main()
