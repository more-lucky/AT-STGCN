#!/usr/bin/env python
"""Run the experiment suites used by the current skeleton-only paper code."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "autsl": {
        "label": "AUTSL",
        "main": "configs1/autsl_skeleton.yaml",
        "sam27": "configs1/sam27_autsl.yaml",
    },
    "asl_citizen": {
        "label": "ASL_Citizen",
        "main": "configs1/asl_citizen_skeleton.yaml",
        "sam27": "configs1/sam27_asl_citizen.yaml",
    },
}
SUITES = {
    "main",
    "ablation",
    "independent",
    "temporal",
    "sensitivity",
    "complexity",
    "sam27",
    "repeated_seed",
}
SUITE_ALIASES = {
    "reverse": "temporal",
    "expand": "sensitivity",
    "nature": "complexity",
    "seed": "repeated_seed",
    "seeds": "repeated_seed",
    "multiseed": "repeated_seed",
    "repeated_seeds": "repeated_seed",
}
REPEATED_SEED_VARIANTS = ("main", "linear", "no_absolute_xy")
REPEATED_SEEDS = (42, 3407, 2026)
STANDARD_TRAINER = "scripts/train.py"


@dataclass(frozen=True)
class Experiment:
    dataset: str
    suite: str
    name: str
    config: Path
    trainer: str
    standard_checkpoint: bool = True


def parse_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="autsl,asl_citizen",
        help="Comma-separated dataset ids: autsl, asl_citizen, or all.",
    )
    parser.add_argument(
        "--suite",
        default="main,ablation,independent,temporal,sensitivity,complexity,sam27,repeated_seed",
        help=(
            "Comma-separated paper suites: main, ablation, independent, temporal, "
            "sensitivity, complexity, sam27, repeated_seed, or all. "
            "Aliases: reverse=temporal, expand=sensitivity, nature=complexity, "
            "seed/seeds/multiseed/repeated_seeds=repeated_seed."
        ),
    )
    parser.add_argument("--stage", choices=("train", "eval", "all", "audit"), default="train")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--split", default="val", help="Evaluation split for --stage eval/all.")
    parser.add_argument("--protocol-id", default="paper-eval-v1")
    parser.add_argument("--evaluation-role", default=None, help="Override strict evaluation role.")
    parser.add_argument("--no-strict-protocol", action="store_true", help="Do not pass --evaluation-role metadata.")
    parser.add_argument("--eval-output-root", default=None, help="Optional root for evaluation artifacts.")
    parser.add_argument("--device", default=None, help="Device override for evaluation commands, e.g. cpu or cuda.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most this many selected experiments.")
    parser.add_argument("--only", default=None, help="Substring filter applied to dataset/suite/name/config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing configs/checkpoints.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training/eval outputs that already exist.")
    parser.add_argument("--allow-overwrite", action="store_true", help="Pass --allow-overwrite to evaluation.")
    return parser.parse_args()


def resolve_items(raw: str, valid: set[str], label: str) -> list[str]:
    items = parse_csv(raw)
    if not items or "all" in items:
        return sorted(valid)
    unknown = sorted(set(items) - valid)
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Valid values: {sorted(valid)}")
    return items


def resolve_suites(raw: str) -> list[str]:
    items = parse_csv(raw)
    if not items or "all" in items:
        return sorted(SUITES)
    mapped = [SUITE_ALIASES.get(item, item) for item in items]
    unknown = sorted(set(mapped) - SUITES)
    if unknown:
        valid = sorted(SUITES | set(SUITE_ALIASES))
        raise ValueError(f"Unknown suites: {unknown}. Valid values: {valid}")
    selected: list[str] = []
    for item in mapped:
        if item not in selected:
            selected.append(item)
    return selected


def parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [parse_scalar(item) for item in inner.split(",") if item.strip()]
    quoted = (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    )
    if quoted:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_config(path: Path) -> dict:
    config: dict = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line or raw_line[:1].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = parse_scalar(value)
    return config


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError:
        return load_simple_config(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def sorted_yaml(root: Path, *, exclude_contains: tuple[str, ...] = ()) -> list[Path]:
    if not root.exists():
        return []
    paths = sorted(path for path in root.glob("*.yaml") if path.is_file())
    if exclude_contains:
        paths = [path for path in paths if not any(token in path.stem for token in exclude_contains)]
    return paths


def repeated_seed_yaml(root: Path, variant: str) -> list[Path]:
    variant_root = root / variant
    declared = [variant_root / f"seed_{seed}.yaml" for seed in REPEATED_SEEDS]
    declared_set = set(declared)
    extras = [path for path in sorted_yaml(variant_root) if path not in declared_set]
    return declared + extras


def build_experiments(datasets: list[str], suites: list[str], *, allow_missing: bool) -> list[Experiment]:
    experiments: list[Experiment] = []
    for dataset in datasets:
        spec = DATASETS[dataset]
        if "main" in suites:
            experiments.append(
                Experiment(dataset, "main", "main", PROJECT_ROOT / spec["main"], STANDARD_TRAINER)
            )
        if "ablation" in suites:
            base_root = PROJECT_ROOT / "configs1" / "ablations" / dataset
            paths = [
                base_root / "01_base_stgcn.yaml",
                base_root / "02_multi_source.yaml",
                base_root / "03_adaptive_topology.yaml",
            ]
            for path in paths:
                experiments.append(Experiment(dataset, "ablation", path.stem, path, STANDARD_TRAINER))
        if "independent" in suites:
            root = PROJECT_ROOT / "configs1" / "independent" / dataset
            for path in sorted_yaml(root):
                experiments.append(Experiment(dataset, "independent", path.stem, path, STANDARD_TRAINER))
        if "temporal" in suites:
            path = PROJECT_ROOT / "configs1" / "ablations" / f"{dataset}_reverse" / "01_temporal_enhancement.yaml"
            experiments.append(Experiment(dataset, "temporal", path.stem, path, STANDARD_TRAINER))
        if "sensitivity" in suites:
            root = PROJECT_ROOT / "configs1" / "expand" / dataset
            for path in sorted_yaml(root):
                experiments.append(Experiment(dataset, "sensitivity", path.stem, path, STANDARD_TRAINER))
        if "complexity" in suites:
            root = PROJECT_ROOT / "configs1" / "nature_ablation" / dataset
            for path in [root / "01_feature_gate.yaml", root / "02_dual_graph.yaml"]:
                experiments.append(
                    Experiment(
                        dataset,
                        "complexity",
                        path.stem,
                        path,
                        "scripts/train_nature_ablation.py",
                        standard_checkpoint=False,
                    )
                )
        if "sam27" in suites:
            experiments.append(
                Experiment(
                    dataset,
                    "sam27",
                    "sam27",
                    PROJECT_ROOT / spec["sam27"],
                    "scripts/train_sam27.py",
                    standard_checkpoint=False,
                )
            )
        if "repeated_seed" in suites:
            root = PROJECT_ROOT / "configs1" / "repeated_seeds" / dataset
            for variant in REPEATED_SEED_VARIANTS:
                for path in repeated_seed_yaml(root, variant):
                    experiments.append(
                        Experiment(
                            dataset,
                            "repeated_seed",
                            f"{variant}_{path.stem}",
                            path,
                            STANDARD_TRAINER,
                        )
                    )
    if allow_missing:
        return [exp for exp in experiments if exp.config.exists()]
    missing = [str(exp.config.relative_to(PROJECT_ROOT)) for exp in experiments if not exp.config.exists()]
    if missing:
        raise FileNotFoundError(f"Missing experiment configs: {missing}")
    return experiments


def filter_experiments(experiments: list[Experiment], only: str | None, limit: int) -> list[Experiment]:
    selected = experiments
    if only:
        needle = str(only).lower()
        selected = [
            exp
            for exp in selected
            if needle
            in " ".join(
                [
                    exp.dataset,
                    exp.suite,
                    exp.name,
                    str(exp.config.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                ]
            ).lower()
        ]
    if limit > 0:
        selected = selected[: int(limit)]
    return selected


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(command_text(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def output_dir_from_config(config: dict) -> Path:
    output_dir = Path(str(config["output_dir"]))
    return output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir


def train_command(exp: Experiment, python: str) -> list[str]:
    return [python, exp.trainer, "--config", str(exp.config.relative_to(PROJECT_ROOT))]


def audit_command(exp: Experiment, config: dict, python: str, output_root: Path | None) -> list[str]:
    manifest = str(config["manifest"])
    dataset_label = DATASETS[exp.dataset]["label"]
    root = output_root or (PROJECT_ROOT / "runs" / "paper_evaluation")
    output = root / dataset_label / f"{exp.suite}_{exp.name}_split_audit.json"
    return [
        python,
        "scripts/audit_data_protocol.py",
        "--dataset",
        dataset_label,
        "--manifest",
        manifest,
        "--output",
        str(output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output),
    ]


def eval_role(exp: Experiment, split: str, override: str | None) -> str:
    if override:
        return str(override)
    if str(split).strip().lower() == "test":
        return "final-test"
    return "model-selection" if exp.suite == "main" else "ablation"


def eval_output_dir(exp: Experiment, config: dict, split: str, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root / DATASETS[exp.dataset]["label"] / exp.suite / exp.name / f"{split}_eval"
    return output_dir_from_config(config) / f"{split}_eval"


def eval_command(exp: Experiment, config: dict, args: argparse.Namespace) -> list[str] | None:
    if not exp.standard_checkpoint:
        print(f"SKIP eval for {exp.dataset}/{exp.suite}/{exp.name}: custom trainer checkpoint")
        return None
    out_dir = output_dir_from_config(config)
    checkpoint = out_dir / "best.pt"
    label_map = out_dir / "label_map.json"
    if not checkpoint.exists() or not label_map.exists():
        if args.allow_missing:
            print(f"SKIP eval for {exp.dataset}/{exp.suite}/{exp.name}: missing checkpoint or label_map")
            return None
        raise FileNotFoundError(f"Missing checkpoint or label_map under {out_dir}")
    split = str(args.split).strip().lower()
    output_root = Path(args.eval_output_root) if args.eval_output_root else None
    if output_root is not None and not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    eval_dir = eval_output_dir(exp, config, split, output_root)
    metrics_path = eval_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        print(f"SKIP eval for {exp.dataset}/{exp.suite}/{exp.name}: {metrics_path} exists")
        return None
    command = [
        args.python,
        "scripts/evaluate.py",
        "--model",
        str(checkpoint.relative_to(PROJECT_ROOT) if checkpoint.is_relative_to(PROJECT_ROOT) else checkpoint),
        "--manifest",
        str(config["manifest"]),
        "--label-map",
        str(label_map.relative_to(PROJECT_ROOT) if label_map.is_relative_to(PROJECT_ROOT) else label_map),
        "--split",
        split,
        "--dataset-name",
        DATASETS[exp.dataset]["label"],
        "--run-name",
        f"{exp.suite}_{exp.name}",
        "--protocol-id",
        str(args.protocol_id),
        "--checkpoint-selection",
        "best-val-top1",
        "--image-height",
        str(int(config.get("sequence_length", config.get("image_height", 64)))),
        "--batch-size",
        str(int(config.get("batch_size", 64))),
        "--output-dir",
        str(eval_dir.relative_to(PROJECT_ROOT) if eval_dir.is_relative_to(PROJECT_ROOT) else eval_dir),
    ]
    if not args.no_strict_protocol:
        command.extend(["--evaluation-role", eval_role(exp, split, args.evaluation_role)])
    if args.device:
        command.extend(["--device", str(args.device)])
    if args.allow_overwrite:
        command.append("--allow-overwrite")
    eval_tta = config.get("eval_tta", {})
    if isinstance(eval_tta, dict):
        if bool(eval_tta.get("flip", False)):
            command.append("--tta-flip")
        scales = eval_tta.get("scales")
        if isinstance(scales, (list, tuple)) and scales:
            command.extend(["--tta-scales", ",".join(str(float(scale)) for scale in scales)])
    return command


def main() -> None:
    args = parse_args()
    datasets = resolve_items(args.datasets, set(DATASETS), "datasets")
    suites = resolve_suites(args.suite)
    experiments = build_experiments(datasets, suites, allow_missing=bool(args.allow_missing))
    experiments = filter_experiments(experiments, args.only, int(args.limit))
    if not experiments:
        raise ValueError("No experiments selected")

    output_root = Path(args.eval_output_root) if args.eval_output_root else None
    if output_root is not None and not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    for exp in experiments:
        config = load_config(exp.config)
        output_dir = output_dir_from_config(config) if "output_dir" in config else None
        if args.stage in {"train", "all"} and output_dir is not None:
            best_path = output_dir / "best.pt"
            if args.skip_existing and best_path.exists():
                print(f"SKIP train for {exp.dataset}/{exp.suite}/{exp.name}: {best_path} exists")
            else:
                run_command(train_command(exp, args.python), dry_run=bool(args.dry_run))
        if args.stage == "audit":
            run_command(audit_command(exp, config, args.python, output_root), dry_run=bool(args.dry_run))
        if args.stage in {"eval", "all"}:
            command = eval_command(exp, config, args)
            if command is not None:
                run_command(command, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
