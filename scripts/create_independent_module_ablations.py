"""Generate controlled, relation-free module ablations for AUTSL and ASL Citizen."""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DATASETS = {
    "autsl": {
        "base_config": Path("configs/ablations/autsl/01_base_stgcn.yaml"),
        "config_dir": Path("configs/ablations/autsl_independent"),
        "run_root": Path("runs/ablations_independent/autsl"),
        "table": Path("paper/tables/ablation_independent_autsl.tex"),
    },
    "asl_citizen": {
        "base_config": Path("configs/ablations/asl_citizen/01_base_stgcn.yaml"),
        "config_dir": Path("configs/ablations/asl_citizen_independent"),
        "run_root": Path("runs/ablations_independent/asl_citizen"),
        "table": Path("paper/tables/ablation_independent_asl_citizen.tex"),
    },
}


VARIANTS = [
    ("base", "Base", False, False, False, False),
    ("ms_only", "MS only", True, False, False, False),
    ("at_only", "AT only", False, True, False, False),
    ("mt_only", "MT only", False, False, True, False),
    ("pp_only", "PP only", False, False, False, True),
    ("ms_at", "MS + AT", True, True, False, False),
    ("ms_pp", "MS + PP", True, False, False, True),
    ("at_pp", "AT + PP", False, True, False, True),
    ("ms_at_pp", "MS + AT + PP", True, True, False, True),
]


PLAN_FIELDS = [
    "dataset",
    "order",
    "variant",
    "setting",
    "config",
    "output_dir",
    "multi_source",
    "adaptive_topology",
    "relation_graph",
    "multi_scale_temporal",
    "part_pooling",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def posix(path: Path) -> str:
    return path.as_posix()


def mark(value: bool) -> str:
    return "1" if value else "0"


def apply_module_switches(
    config: dict[str, Any],
    *,
    multi_source: bool,
    adaptive_topology: bool,
    multi_scale_temporal: bool,
    part_pooling: bool,
) -> None:
    # Multi-source descriptors are toggled as one paper-level component.
    config["skeleton_use_bone_features"] = multi_source
    config["skeleton_use_motion_features"] = multi_source
    config["skeleton_include_absolute_xy"] = multi_source
    config["skeleton_include_validity"] = multi_source
    config["skeleton_include_temporal_position"] = multi_source
    config["skeleton_include_root_motion"] = multi_source
    config["skeleton_include_acceleration"] = multi_source

    # AT contains both learnable adjacency and multiplicative edge importance.
    config["skeleton_adaptive_graph"] = adaptive_topology
    config["skeleton_edge_importance"] = adaptive_topology

    # Relation graph is excluded from every experiment in this suite.
    config["skeleton_relation_graph"] = False
    config["skeleton_temporal_dilations"] = [1, 2, 3] if multi_scale_temporal else [1]
    config["skeleton_part_pooling"] = part_pooling


def write_plan(config_dir: Path, rows: list[dict[str, str]]) -> Path:
    path = config_dir / "ablation_plan.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def bash_script(
    *,
    rows: list[dict[str, str]],
    plan_path: Path,
    manifest: str,
    image_height: int,
    batch_size: int,
    tta_flip: bool,
    tta_scales: list[Any],
    summary_csv: Path,
    table_path: Path,
) -> str:
    scale_text = ",".join(str(value) for value in tta_scales)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'PYTHON_EXE="${ABLATION_PYTHON:-python}"',
        'ABLATION_RUNS="${ABLATION_RUNS:-all}"',
        'FORCE_TRAIN="${FORCE_TRAIN:-0}"',
        'FORCE_EVAL="${FORCE_EVAL:-0}"',
        "",
        "run_python() {",
        '  "${PYTHON_EXE}" "$@"',
        "}",
        "",
        "selected() {",
        '  local variant="$1"',
        '  [[ "${ABLATION_RUNS}" == "all" || ",${ABLATION_RUNS}," == *",${variant},"* ]]',
        "}",
        "",
        "run_ablation() {",
        '  local variant="$1"',
        '  local config_path="$2"',
        '  local output_dir="$3"',
        '  if ! selected "${variant}"; then',
        '    echo "[skip filter] ${variant}"',
        "    return",
        "  fi",
        '  if [[ "${FORCE_TRAIN}" == "1" || ! -f "${output_dir}/best.pt" ]]; then',
        '    run_python scripts/train.py --config "${config_path}"',
        "  else",
        '    echo "[reuse checkpoint] ${output_dir}/best.pt"',
        "  fi",
        '  [[ -f "${output_dir}/best.pt" ]] || { echo "Missing checkpoint: ${output_dir}/best.pt" >&2; exit 1; }',
        '  [[ -f "${output_dir}/label_map.json" ]] || { echo "Missing label map: ${output_dir}/label_map.json" >&2; exit 1; }',
        '  if [[ "${FORCE_EVAL}" == "1" || ! -f "${output_dir}/val_eval/metrics.json" ]]; then',
        "    local eval_args=(",
        "      scripts/evaluate.py",
        '      --model "${output_dir}/best.pt"',
        f'      --manifest "{manifest}"',
        '      --label-map "${output_dir}/label_map.json"',
        "      --split val",
        f"      --image-height {image_height}",
        f"      --batch-size {batch_size}",
        '      --output-dir "${output_dir}/val_eval"',
        "      --num-workers 0",
        "    )",
    ]
    if tta_flip:
        lines.append("    eval_args+=(--tta-flip)")
    if scale_text:
        lines.append(f'    eval_args+=(--tta-scales "{scale_text}")')
    lines.extend(
        [
            '    run_python "${eval_args[@]}"',
            "  else",
            '    echo "[reuse metrics] ${output_dir}/val_eval/metrics.json"',
            "  fi",
            "}",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f'run_ablation "{row["variant"]}" "{row["config"]}" "{row["output_dir"]}"'
        )
    lines.extend(
        [
            "",
            "run_python scripts/summarize_ablation_results.py \\",
            f'  --plan "{posix(plan_path)}" \\',
            f'  --output-csv "{posix(summary_csv)}" \\',
            f'  --output-tex "{posix(table_path)}"',
            "",
        ]
    )
    return "\n".join(lines)


def powershell_script(
    *,
    rows: list[dict[str, str]],
    plan_path: Path,
    manifest: str,
    image_height: int,
    batch_size: int,
    tta_flip: bool,
    tta_scales: list[Any],
    summary_csv: Path,
    table_path: Path,
) -> str:
    scale_text = ",".join(str(value) for value in tta_scales)
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$PythonExe = if ($env:ABLATION_PYTHON) { $env:ABLATION_PYTHON } else { 'python' }",
        "$RunFilter = if ($env:ABLATION_RUNS) { $env:ABLATION_RUNS } else { 'all' }",
        "$ForceTrain = $env:FORCE_TRAIN -eq '1'",
        "$ForceEval = $env:FORCE_EVAL -eq '1'",
        "",
        "function Invoke-Python {",
        "    param([string[]]$Arguments)",
        "    & $PythonExe @Arguments",
        "    if ($LASTEXITCODE -ne 0) { throw \"Python failed: $($Arguments -join ' ')\" }",
        "}",
        "",
        "function Test-Selected {",
        "    param([string]$Variant)",
        "    return $RunFilter -eq 'all' -or $Variant -in ($RunFilter -split ',')",
        "}",
        "",
        "function Invoke-Ablation {",
        "    param([string]$Variant, [string]$ConfigPath, [string]$OutputDir)",
        "    if (-not (Test-Selected $Variant)) { Write-Host \"[skip filter] $Variant\"; return }",
        "    if ($ForceTrain -or -not (Test-Path -LiteralPath \"$OutputDir/best.pt\")) {",
        "        Invoke-Python @('scripts/train.py', '--config', $ConfigPath)",
        "    } else { Write-Host \"[reuse checkpoint] $OutputDir/best.pt\" }",
        "    if (-not (Test-Path -LiteralPath \"$OutputDir/best.pt\")) { throw \"Missing checkpoint: $OutputDir/best.pt\" }",
        "    if (-not (Test-Path -LiteralPath \"$OutputDir/label_map.json\")) { throw \"Missing label map: $OutputDir/label_map.json\" }",
        "    if ($ForceEval -or -not (Test-Path -LiteralPath \"$OutputDir/val_eval/metrics.json\")) {",
        "        $EvalArgs = @(",
        "            'scripts/evaluate.py', '--model', \"$OutputDir/best.pt\",",
        f"            '--manifest', '{manifest}', '--label-map', \"$OutputDir/label_map.json\",",
        f"            '--split', 'val', '--image-height', '{image_height}', '--batch-size', '{batch_size}',",
        "            '--output-dir', \"$OutputDir/val_eval\", '--num-workers', '0'",
        "        )",
    ]
    if tta_flip:
        lines.append("        $EvalArgs += '--tta-flip'")
    if scale_text:
        lines.append(f"        $EvalArgs += @('--tta-scales', '{scale_text}')")
    lines.extend(
        [
            "        Invoke-Python $EvalArgs",
            "    } else { Write-Host \"[reuse metrics] $OutputDir/val_eval/metrics.json\" }",
            "}",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"Invoke-Ablation '{row['variant']}' '{row['config']}' '{row['output_dir']}'"
        )
    lines.extend(
        [
            "",
            "Invoke-Python @(",
            "    'scripts/summarize_ablation_results.py',",
            f"    '--plan', '{posix(plan_path)}',",
            f"    '--output-csv', '{posix(summary_csv)}',",
            f"    '--output-tex', '{posix(table_path)}'",
            ")",
            "",
        ]
    )
    return "\n".join(lines)


def generate_dataset(dataset: str, spec: dict[str, Path]) -> None:
    base_config = load_yaml(spec["base_config"])
    config_dir = spec["config_dir"]
    run_root = spec["run_root"]
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for index, (name, setting, ms, at, mt, pp) in enumerate(VARIANTS, start=1):
        config = deepcopy(base_config)
        apply_module_switches(
            config,
            multi_source=ms,
            adaptive_topology=at,
            multi_scale_temporal=mt,
            part_pooling=pp,
        )
        output_dir = run_root / f"{index:02d}_{name}"
        config_path = config_dir / f"{index:02d}_{name}.yaml"
        config["output_dir"] = posix(output_dir)
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        rows.append(
            {
                "dataset": dataset,
                "order": str(index),
                "variant": name,
                "setting": setting,
                "config": posix(config_path),
                "output_dir": posix(output_dir),
                "multi_source": mark(ms),
                "adaptive_topology": mark(at),
                "relation_graph": "0",
                "multi_scale_temporal": mark(mt),
                "part_pooling": mark(pp),
            }
        )

    plan_path = write_plan(config_dir, rows)
    eval_tta = base_config.get("eval_tta", {})
    if not isinstance(eval_tta, dict):
        eval_tta = {}
    common = {
        "rows": rows,
        "plan_path": plan_path,
        "manifest": str(base_config["manifest"]),
        "image_height": int(base_config.get("image_height", 64)),
        "batch_size": int(base_config.get("batch_size", 64)),
        "tta_flip": bool(eval_tta.get("flip", False)),
        "tta_scales": list(eval_tta.get("scales", [])),
        "summary_csv": run_root / "ablation_summary.csv",
        "table_path": spec["table"],
    }
    (config_dir / "run_ablation.sh").write_text(
        bash_script(**common), encoding="utf-8", newline="\n"
    )
    (config_dir / "run_ablation.ps1").write_text(
        powershell_script(**common), encoding="utf-8"
    )
    print(f"Generated {len(rows)} {dataset} configs in {config_dir}")


def main() -> None:
    for dataset, spec in DATASETS.items():
        generate_dataset(dataset, spec)


if __name__ == "__main__":
    main()
