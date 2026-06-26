#!/usr/bin/env python
"""Run and summarize repeated-seed experiments for the paper model variants."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS: dict[str, dict[str, str]] = {
    "autsl": {
        "base_config": "configs1/autsl_skeleton.yaml",
        "output_root": "runs/repeated_seeds/autsl",
    },
    "asl_citizen": {
        "base_config": "configs1/asl_citizen_skeleton.yaml",
        "output_root": "runs/repeated_seeds/asl_citizen",
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "main": {
        "display_name": "Main model",
        "updates": {},
    },
    "linear": {
        "display_name": "Linear instead of ArcFace",
        "updates": {
            "classifier_type": "linear",
            "classifier_margin": 0.0,
        },
    },
    "no_absolute_xy": {
        "display_name": "Without absolute xy",
        "updates": {
            "skeleton_include_absolute_xy": False,
        },
    },
}

# Two-sided 95% Student-t critical values indexed by degrees of freedom.
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv_strings(value)]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    temporary.replace(path)


def generate_configs(
    *,
    datasets: list[str],
    variants: list[str],
    seeds: list[int],
    config_root: Path,
) -> list[Path]:
    generated: list[Path] = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset!r}; choices: {', '.join(DATASETS)}")
        info = DATASETS[dataset]
        base_path = PROJECT_ROOT / info["base_config"]
        base_cfg = load_yaml(base_path)
        for variant in variants:
            if variant not in VARIANTS:
                raise ValueError(f"Unknown variant {variant!r}; choices: {', '.join(VARIANTS)}")
            variant_info = VARIANTS[variant]
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)
                cfg.update(variant_info["updates"])
                cfg["seed"] = int(seed)
                cfg["output_dir"] = f"{info['output_root']}/{variant}/seed_{seed}"
                cfg["repeated_seed_experiment"] = {
                    "dataset": dataset,
                    "variant": variant,
                    "display_name": variant_info["display_name"],
                    "seed": int(seed),
                    "base_config": info["base_config"],
                }
                target = config_root / dataset / variant / f"seed_{seed}.yaml"
                write_yaml(target, cfg)
                generated.append(target)
    return generated


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def t_critical_95(n: int) -> float:
    if n < 2:
        return 0.0
    return T_CRITICAL_975.get(n - 1, 1.96)


def mean_std_ci95(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, 0.0, mean, mean
    std = statistics.stdev(values)
    margin = t_critical_95(len(values)) * std / math.sqrt(len(values))
    return mean, std, mean - margin, mean + margin


def tta_arguments(cfg: dict[str, Any]) -> list[str]:
    tta = cfg.get("eval_tta", {})
    if not isinstance(tta, dict):
        return []
    args: list[str] = []
    if bool(tta.get("flip", False)):
        args.append("--tta-flip")
    scales = tta.get("scales", None)
    if isinstance(scales, (list, tuple)) and scales:
        args.extend(["--tta-scales", ",".join(str(float(scale)) for scale in scales)])
    return args


def train_one(config_path: Path, *, python: str, force: bool, dry_run: bool) -> None:
    cfg = load_yaml(config_path)
    output_dir = PROJECT_ROOT / str(cfg["output_dir"])
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if best_path.exists() and last_path.exists() and not force:
        print(f"==> Skip train: {last_path.relative_to(PROJECT_ROOT)} confirms completion", flush=True)
        return
    print(f"==> Train {config_path.relative_to(PROJECT_ROOT)}", flush=True)
    run_command(
        [python, "scripts/train.py", "--config", config_path.relative_to(PROJECT_ROOT).as_posix()],
        dry_run=dry_run,
    )


def evaluate_one(
    config_path: Path,
    *,
    python: str,
    split: str,
    batch_size: int | None,
    num_workers: int,
    force: bool,
    require_paper_valid: bool,
    dry_run: bool,
) -> None:
    cfg = load_yaml(config_path)
    output_dir = PROJECT_ROOT / str(cfg["output_dir"])
    checkpoint = output_dir / "best.pt"
    label_map = output_dir / "label_map.json"
    eval_dir = output_dir / f"{split}_eval"
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"==> Skip eval: {metrics_path.relative_to(PROJECT_ROOT)} exists", flush=True)
        return
    if not checkpoint.exists():
        print(f"WARNING: missing checkpoint: {checkpoint.relative_to(PROJECT_ROOT)}", flush=True)
        return
    resolved_batch_size = int(batch_size or cfg.get("batch_size", 64))
    command = [
        python,
        "scripts/evaluate.py",
        "--model",
        checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "--manifest",
        str(cfg["manifest"]),
        "--label-map",
        label_map.relative_to(PROJECT_ROOT).as_posix(),
        "--split",
        split,
        "--image-height",
        str(int(cfg.get("image_height", 64))),
        "--batch-size",
        str(resolved_batch_size),
        "--num-workers",
        str(int(num_workers)),
        "--output-dir",
        eval_dir.relative_to(PROJECT_ROOT).as_posix(),
    ]
    command.extend(tta_arguments(cfg))
    if require_paper_valid:
        command.append("--require-paper-valid")
    print(f"==> Evaluate {config_path.relative_to(PROJECT_ROOT)} split={split}", flush=True)
    run_command(command, dry_run=dry_run)


def summarize(
    *,
    datasets: list[str],
    variants: list[str],
    seeds: list[int],
    split: str,
    output_root: Path,
) -> None:
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_output = Path(DATASETS[dataset]["output_root"])
        for variant in variants:
            top1_values: list[float] = []
            top5_values: list[float] = []
            found_seeds: list[int] = []
            for seed in seeds:
                metrics_path = (
                    PROJECT_ROOT
                    / dataset_output
                    / variant
                    / f"seed_{seed}"
                    / f"{split}_eval"
                    / "metrics.json"
                )
                if not metrics_path.exists():
                    continue
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                top1 = float(metrics["top1"]) * 100.0
                top5 = float(metrics["top5"]) * 100.0
                found_seeds.append(seed)
                top1_values.append(top1)
                top5_values.append(top5)
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "split": split,
                        "top1_pct": top1,
                        "top5_pct": top5,
                        "metrics_path": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                    }
                )
            if not top1_values:
                continue
            top1_mean, top1_std, top1_low, top1_high = mean_std_ci95(top1_values)
            top5_mean, top5_std, top5_low, top5_high = mean_std_ci95(top5_values)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "display_name": VARIANTS[variant]["display_name"],
                    "split": split,
                    "n": len(found_seeds),
                    "seeds": ";".join(str(seed) for seed in found_seeds),
                    "top1_mean_pct": top1_mean,
                    "top1_std_pct": top1_std,
                    "top1_ci95_low_pct": top1_low,
                    "top1_ci95_high_pct": top1_high,
                    "top5_mean_pct": top5_mean,
                    "top5_std_pct": top5_std,
                    "top5_ci95_low_pct": top5_low,
                    "top5_ci95_high_pct": top5_high,
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / f"multiseed_raw_{split}.csv"
    summary_path = output_root / f"multiseed_summary_{split}.csv"
    markdown_path = output_root / f"multiseed_summary_{split}.md"
    tex_path = output_root / f"multiseed_summary_{split}.tex"

    if raw_rows:
        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        with markdown_path.open("w", encoding="utf-8") as handle:
            handle.write("| Dataset | Variant | n | Top-1 mean+-std | Top-1 95% CI | Top-5 mean+-std | Top-5 95% CI |\n")
            handle.write("|---|---|---:|---:|---:|---:|---:|\n")
            for row in summary_rows:
                handle.write(
                    f"| {row['dataset']} | {row['display_name']} | {row['n']} | "
                    f"{row['top1_mean_pct']:.2f} +- {row['top1_std_pct']:.2f} | "
                    f"[{row['top1_ci95_low_pct']:.2f}, {row['top1_ci95_high_pct']:.2f}] | "
                    f"{row['top5_mean_pct']:.2f} +- {row['top5_std_pct']:.2f} | "
                    f"[{row['top5_ci95_low_pct']:.2f}, {row['top5_ci95_high_pct']:.2f}] |\n"
                )

        with tex_path.open("w", encoding="utf-8") as handle:
            handle.write("% Auto-generated by scripts/run_repeated_seed_experiments.py\n")
            for row in summary_rows:
                handle.write(
                    f"{row['dataset']} & {row['display_name']} & {row['n']} & "
                    f"{row['top1_mean_pct']:.2f} $\\pm$ {row['top1_std_pct']:.2f} & "
                    f"[{row['top1_ci95_low_pct']:.2f}, {row['top1_ci95_high_pct']:.2f}] & "
                    f"{row['top5_mean_pct']:.2f} $\\pm$ {row['top5_std_pct']:.2f} & "
                    f"[{row['top5_ci95_low_pct']:.2f}, {row['top5_ci95_high_pct']:.2f}] \\\\\n"
                )

    for row in summary_rows:
        print(
            f"{row['dataset']:12s} {row['variant']:15s} n={row['n']} "
            f"Top-1={row['top1_mean_pct']:.2f}+-{row['top1_std_pct']:.2f} "
            f"95%CI=[{row['top1_ci95_low_pct']:.2f},{row['top1_ci95_high_pct']:.2f}] "
            f"Top-5={row['top5_mean_pct']:.2f}+-{row['top5_std_pct']:.2f} "
            f"95%CI=[{row['top5_ci95_low_pct']:.2f},{row['top5_ci95_high_pct']:.2f}]"
        )
    print(f"Summary files written under {output_root.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "train", "evaluate", "summarize", "all"], default="all")
    parser.add_argument("--datasets", default="autsl,asl_citizen")
    parser.add_argument("--variants", default="main,linear,no_absolute_xy")
    parser.add_argument("--seeds", default="42,3407,2026")
    parser.add_argument("--split", default="val")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config-root", default="configs1/repeated_seeds")
    parser.add_argument("--result-root", default="runs/repeated_seeds")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-paper-valid", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv_strings(args.datasets)
    variants = parse_csv_strings(args.variants)
    seeds = parse_csv_ints(args.seeds)
    if len(set(seeds)) < 3:
        raise ValueError("At least three distinct seeds are required")

    config_paths = generate_configs(
        datasets=datasets,
        variants=variants,
        seeds=seeds,
        config_root=PROJECT_ROOT / args.config_root,
    )
    if args.stage in {"train", "all"}:
        for config_path in config_paths:
            train_one(config_path, python=args.python, force=args.force, dry_run=args.dry_run)
    if args.stage in {"evaluate", "all"}:
        for config_path in config_paths:
            evaluate_one(
                config_path,
                python=args.python,
                split=args.split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                force=args.force,
                require_paper_valid=args.require_paper_valid,
                dry_run=args.dry_run,
            )
    if args.stage in {"summarize", "all"} and not args.dry_run:
        summarize(
            datasets=datasets,
            variants=variants,
            seeds=seeds,
            split=args.split,
            output_root=PROJECT_ROOT / args.result_root,
        )


if __name__ == "__main__":
    main()
