#!/usr/bin/env python
"""Plot t-SNE feature visualizations for ablation checkpoints.

The script loads each ablation checkpoint, extracts penultimate features on a
chosen manifest split, and writes one t-SNE figure per run. It can also select
easy classes from evaluation predictions and place all ablation variants in a
single grid figure for side-by-side comparison.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYDEPS = PROJECT_ROOT / ".runtime" / "pydeps"
if RUNTIME_PYDEPS.exists() and str(RUNTIME_PYDEPS) not in sys.path:
    sys.path.append(str(RUNTIME_PYDEPS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sl_atstgcn.at_stgcn import AT_STGCN_MODEL_TYPES, load_at_stgcn_checkpoint
from sl_atstgcn.data import LabelMap, attach_label_ids, make_skeleton_dataloader, read_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation-root", action="append", default=[], help="Directory containing ablation run folders")
    p.add_argument("--run-dir", action="append", default=[], help="Specific run directory; can be passed multiple times")
    p.add_argument("--checkpoint-name", default="best.pt", help="Checkpoint filename inside each run directory")
    p.add_argument("--manifest", required=True, help="Manifest CSV used for feature extraction")
    p.add_argument("--label-map", default=None, help="Optional shared label_map.json; otherwise use each run's label map")
    p.add_argument("--split", default="val", help="Manifest split to visualize")
    p.add_argument("--image-height", type=int, default=64, help="Fallback sequence length for old checkpoints without metadata")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=2000, help="Balanced sample cap; <=0 uses all rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=35.0)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument(
        "--focus-labels",
        default=None,
        help="Comma-separated label names or ids to visualize; useful for ASL Citizen or confused classes",
    )
    p.add_argument(
        "--auto-easy-classes",
        type=int,
        default=0,
        help="Automatically select this many easiest classes from an evaluation predictions.csv",
    )
    p.add_argument(
        "--easy-predictions",
        default=None,
        help="predictions.csv used by --auto-easy-classes; defaults to 06_full_model/val_eval/predictions.csv when available",
    )
    p.add_argument("--easy-min-support", type=int, default=3, help="Minimum samples required for an automatically selected class")
    p.add_argument(
        "--combined-output",
        default=None,
        help="Optional filename for a combined t-SNE grid containing all selected ablation runs",
    )
    p.add_argument("--combined-cols", type=int, default=3, help="Number of columns in the combined t-SNE grid")
    p.add_argument("--point-size", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=0.78)
    p.add_argument("--legend-max-classes", type=int, default=20)
    p.add_argument("--save-csv", action="store_true", help="Also save per-sample 2D coordinates")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; using CPU")
        requested = "cpu"
    return torch.device(requested)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "run"


def discover_run_dirs(ablation_roots: Iterable[str], run_dirs: Iterable[str]) -> list[Path]:
    runs: list[Path] = []
    for value in run_dirs:
        path = Path(value)
        if path.is_dir():
            runs.append(path)
    for value in ablation_roots:
        root = Path(value)
        if not root.is_dir():
            print(f"WARNING: ablation root not found: {root}")
            continue
        runs.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    unique: list[Path] = []
    seen: set[str] = set()
    for run in runs:
        key = str(run.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(run)
    return unique


def load_label_map(run_dir: Path, explicit_path: str | None) -> LabelMap:
    path = Path(explicit_path) if explicit_path else run_dir / "label_map.json"
    if not path.exists():
        raise FileNotFoundError(f"label map not found: {path}")
    return LabelMap.load(path)


def parse_focus_labels(value: str | None) -> set[str]:
    if value is None or str(value).strip() == "":
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def filter_focus_labels(df: pd.DataFrame, focus_labels: set[str]) -> pd.DataFrame:
    if not focus_labels:
        return df
    label_id_text = df["label_id"].astype(str)
    label_text = df["label"].astype(str)
    mask = label_id_text.isin(focus_labels) | label_text.isin(focus_labels)
    return df[mask].reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, *, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    groups: list[list[int]] = []
    for _, group in df.groupby("label_id", sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        groups.append(indices.tolist())
    rng.shuffle(groups)
    selected: list[int] = []
    active = True
    while active and len(selected) < max_samples:
        active = False
        for group in groups:
            if group and len(selected) < max_samples:
                selected.append(group.pop())
                active = True
    sampled = df.loc[selected].sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return sampled


def resolve_easy_predictions_path(run_dirs: list[Path], explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"easy predictions file not found: {path}")
        return path

    preferred = [run for run in run_dirs if run.name == "06_full_model"]
    candidates = preferred + [run for run in reversed(run_dirs) if run.name != "06_full_model"]
    for run in candidates:
        path = run / "val_eval" / "predictions.csv"
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "Cannot auto-select easy classes because no val_eval/predictions.csv was found. "
        "Run evaluation first or pass --easy-predictions."
    )


def select_easy_labels_from_predictions(
    predictions_path: Path,
    *,
    count: int,
    min_support: int,
) -> list[str]:
    df = pd.read_csv(predictions_path)
    required = {"label_id", "pred_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{predictions_path} is missing required columns: {missing}")
    if "label" not in df.columns:
        df["label"] = df["label_id"].astype(str)

    rows: list[dict[str, object]] = []
    for label_id, group in df.groupby("label_id", sort=True):
        support = int(len(group))
        if support < int(min_support):
            continue
        correct = int((group["pred_id"].astype(str) == group["label_id"].astype(str)).sum())
        accuracy = correct / max(1, support)
        label_name = str(group["label"].iloc[0])
        rows.append(
            {
                "label_id": int(label_id),
                "label": label_name,
                "support": support,
                "correct": correct,
                "accuracy": float(accuracy),
            }
        )
    if not rows:
        raise ValueError(f"No class in {predictions_path} satisfies min_support={min_support}")

    ranked = sorted(rows, key=lambda item: (-float(item["accuracy"]), -int(item["support"]), int(item["label_id"])))
    selected = ranked[: max(1, int(count))]
    return [str(item["label_id"]) for item in selected]


def write_easy_classes_csv(path: Path, predictions_path: Path, selected_ids: set[str]) -> None:
    df = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    for label_id, group in df.groupby("label_id", sort=True):
        label_id_text = str(label_id)
        if label_id_text not in selected_ids:
            continue
        support = int(len(group))
        correct = int((group["pred_id"].astype(str) == group["label_id"].astype(str)).sum())
        rows.append(
            {
                "label_id": int(label_id),
                "label": str(group["label"].iloc[0]) if "label" in group.columns else label_id_text,
                "support": support,
                "correct": correct,
                "accuracy": correct / max(1, support),
                "source_predictions": str(predictions_path),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["accuracy", "support"], ascending=[False, False]).to_csv(path, index=False)


def make_loader(
    split_df: pd.DataFrame,
    *,
    model_config: dict,
    image_height: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    model_type = str(model_config.get("model_type", "skeleton")).strip().lower()
    if model_type not in AT_STGCN_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type!r}; t-SNE visualization is skeleton-only.")
    repair_missing = bool(model_config.get("repair_missing_keypoints", False))
    repair_min_valid = int(model_config.get("repair_min_valid", 2))
    sequence_length = int(model_config.get("sequence_length", model_config.get("image_height", image_height)))
    return make_skeleton_dataloader(
        split_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        training=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        repair_missing=repair_missing,
        repair_min_valid=repair_min_valid,
    )


@torch.no_grad()
def collect_features(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        features = model.extract_features(images)
        features = torch.nn.functional.normalize(features.float(), dim=1)
        feature_batches.append(features.cpu().numpy())
        label_batches.append(labels.cpu().numpy())
    if not feature_batches:
        raise RuntimeError("No features were extracted")
    return np.concatenate(feature_batches, axis=0), np.concatenate(label_batches, axis=0).astype(np.int64)


def compute_tsne(features: np.ndarray, *, seed: int, pca_dim: int, perplexity: float) -> np.ndarray:
    n_samples = int(features.shape[0])
    if n_samples < 3:
        raise ValueError("t-SNE needs at least 3 samples")
    features = np.asarray(features, dtype=np.float32)
    pca_components = min(int(pca_dim), features.shape[1], n_samples - 1)
    if pca_components >= 2:
        features = PCA(n_components=pca_components, random_state=int(seed)).fit_transform(features)
    resolved_perplexity = min(float(perplexity), max(2.0, (n_samples - 1) / 3.0))
    resolved_perplexity = min(resolved_perplexity, float(n_samples - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=resolved_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        metric="euclidean",
    )
    return tsne.fit_transform(features)


def plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    label_map: LabelMap,
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
    legend_max_classes: int,
) -> None:
    unique = np.array(sorted(np.unique(labels).tolist()), dtype=np.int64)
    class_to_color = {int(label): index for index, label in enumerate(unique)}
    colors = np.array([class_to_color[int(label)] for label in labels], dtype=np.int64)
    cmap_name = "tab20" if len(unique) <= 20 else "turbo"
    cmap = plt.get_cmap(cmap_name, max(1, len(unique)))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap=cmap,
        s=float(point_size),
        alpha=float(alpha),
        linewidths=0,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)

    if len(unique) <= int(legend_max_classes):
        handles = []
        legend_labels = []
        for idx, label_id in enumerate(unique):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=5,
                    markerfacecolor=cmap(idx),
                    markeredgecolor="none",
                )
            )
            legend_labels.append(str(label_map.id_to_label.get(int(label_id), int(label_id))))
        ax.legend(handles, legend_labels, loc="best", fontsize=7, frameon=True, ncol=1)
    else:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("class color index")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_embeddings(
    plot_rows: list[dict[str, object]],
    *,
    label_map: LabelMap,
    output_path: Path,
    point_size: float,
    alpha: float,
    legend_max_classes: int,
    cols: int,
) -> None:
    if not plot_rows:
        return
    unique = sorted({int(label) for row in plot_rows for label in np.unique(row["labels"]).tolist()})
    class_to_color = {int(label): index for index, label in enumerate(unique)}
    cmap_name = "tab10" if len(unique) <= 10 else "tab20" if len(unique) <= 20 else "turbo"
    cmap = plt.get_cmap(cmap_name, max(1, len(unique)))

    cols = max(1, int(cols))
    rows = int(np.ceil(len(plot_rows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.35), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for index, row in enumerate(plot_rows):
        ax = axes[index // cols][index % cols]
        ax.axis("on")
        embedding = np.asarray(row["embedding"])
        labels = np.asarray(row["labels"])
        colors = np.array([class_to_color[int(label)] for label in labels], dtype=np.int64)
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            cmap=cmap,
            s=float(point_size),
            alpha=float(alpha),
            linewidths=0,
            vmin=0,
            vmax=max(1, len(unique) - 1),
        )
        ax.set_title(str(row["title"]), fontsize=10)
        ax.set_xlabel("t-SNE-1", fontsize=8)
        ax.set_ylabel("t-SNE-2", fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)
        ax.tick_params(axis="both", labelsize=7)

    if len(unique) <= int(legend_max_classes):
        handles = []
        legend_labels = []
        for idx, label_id in enumerate(unique):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=5,
                    markerfacecolor=cmap(idx),
                    markeredgecolor="none",
                )
            )
            legend_labels.append(str(label_map.id_to_label.get(int(label_id), int(label_id))))
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=8,
            frameon=True,
            ncol=min(5, max(1, len(unique))),
        )
        bottom = 0.12
    else:
        bottom = 0.04

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_embedding_csv(path: Path, embedding: np.ndarray, labels: np.ndarray, label_map: LabelMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label_id", "label"])
        for (x, y), label_id in zip(embedding, labels):
            writer.writerow([f"{float(x):.8f}", f"{float(y):.8f}", int(label_id), label_map.id_to_label.get(int(label_id), "")])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = discover_run_dirs(args.ablation_root, args.run_dir)
    if not run_dirs:
        raise ValueError("No ablation run directories found. Use --ablation-root or --run-dir.")

    base_manifest = read_manifest(args.manifest)
    focus_labels = parse_focus_labels(args.focus_labels)
    easy_predictions_path: Path | None = None
    if int(args.auto_easy_classes) > 0:
        easy_predictions_path = resolve_easy_predictions_path(run_dirs, args.easy_predictions)
        easy_labels = select_easy_labels_from_predictions(
            easy_predictions_path,
            count=int(args.auto_easy_classes),
            min_support=int(args.easy_min_support),
        )
        focus_labels = set(easy_labels)
        write_easy_classes_csv(output_dir / "easy_classes.csv", easy_predictions_path, focus_labels)
        print(f"Selected easy classes from {easy_predictions_path}: {','.join(easy_labels)}")
        print(f"Wrote {output_dir / 'easy_classes.csv'}")
    summary_rows: list[dict[str, object]] = []
    combined_rows: list[dict[str, object]] = []
    combined_label_map: LabelMap | None = None

    for run_dir in run_dirs:
        checkpoint_path = run_dir / args.checkpoint_name
        if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
            print(f"WARNING: skip {run_dir.name}, checkpoint not found or empty: {checkpoint_path}")
            continue
        print(f"==> {run_dir.name}: loading {checkpoint_path}")
        label_map = load_label_map(run_dir, args.label_map)
        manifest, _ = attach_label_ids(base_manifest, label_map)
        split_df = manifest[manifest["split"].str.lower() == str(args.split).lower()].reset_index(drop=True)
        split_df = filter_focus_labels(split_df, focus_labels)
        split_df = balanced_sample(split_df, max_samples=int(args.max_samples), seed=int(args.seed))
        if len(split_df) < 3:
            print(f"WARNING: skip {run_dir.name}, too few rows after filtering: {len(split_df)}")
            continue

        model, checkpoint = load_at_stgcn_checkpoint(
            checkpoint_path,
            device=device,
            num_classes=len(label_map.label_to_id),
            image_height=int(args.image_height),
        )
        model_config = dict(checkpoint.get("model_config", {}))
        loader = make_loader(
            split_df,
            model_config=model_config,
            image_height=int(args.image_height),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        features, labels = collect_features(model, loader, device)
        embedding = compute_tsne(
            features,
            seed=int(args.seed),
            pca_dim=int(args.pca_dim),
            perplexity=float(args.perplexity),
        )
        run_name = safe_name(run_dir.name)
        output_path = output_dir / f"{run_name}_tsne.png"
        title = f"{run_dir.name} ({args.split}, n={len(labels)}, classes={len(np.unique(labels))})"
        plot_embedding(
            embedding,
            labels,
            label_map=label_map,
            title=title,
            output_path=output_path,
            point_size=float(args.point_size),
            alpha=float(args.alpha),
            legend_max_classes=int(args.legend_max_classes),
        )
        if args.save_csv:
            write_embedding_csv(output_dir / f"{run_name}_tsne.csv", embedding, labels, label_map)
        combined_rows.append(
            {
                "title": f"{run_dir.name}\nn={len(labels)}, classes={len(np.unique(labels))}",
                "embedding": embedding,
                "labels": labels,
            }
        )
        if combined_label_map is None:
            combined_label_map = label_map
        summary_rows.append(
            {
                "run": run_dir.name,
                "checkpoint": str(checkpoint_path),
                "samples": int(len(labels)),
                "classes": int(len(np.unique(labels))),
                "figure": str(output_path),
                "easy_predictions": "" if easy_predictions_path is None else str(easy_predictions_path),
            }
        )
        print(f"Wrote {output_path}")

    if args.combined_output and combined_rows:
        combined_path = output_dir / str(args.combined_output)
        plot_combined_embeddings(
            combined_rows,
            label_map=combined_label_map or label_map,
            output_path=combined_path,
            point_size=float(args.point_size),
            alpha=float(args.alpha),
            legend_max_classes=int(args.legend_max_classes),
            cols=int(args.combined_cols),
        )
        print(f"Wrote {combined_path}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "tsne_summary.csv", index=False)
        print(f"Wrote {output_dir / 'tsne_summary.csv'}")


if __name__ == "__main__":
    main()
