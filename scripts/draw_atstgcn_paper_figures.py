from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("paper/figures")


def box(ax, xy, w, h, text, fc="#ffffff", ec="#333333", lw=1.4, fs=10, style="round", alpha=1.0):
    if style == "round":
        patch = FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
        )
    else:
        patch = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def arrow(ax, p1, p2, color="#111111", lw=1.6, dashed=False, rad=0.0):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=(0, (5, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def draw_feature_group(ax, x, y, title, labels):
    box(ax, (x, y), 1.95, 1.32, "", fc="#e9f6df", ec="#5c9b5c", lw=1.2)
    ax.text(x + 0.98, y + 1.13, title, ha="center", va="center", fontsize=9, weight="bold")
    for i, label in enumerate(labels):
        yy = y + 0.80 - 0.34 * i
        box(ax, (x + 0.18, yy), 0.78, 0.22, label, fc="#ffffff", ec="#333333", lw=0.9, fs=7)
        for j, c in enumerate(["#7be35f", "#9bf86d", "#ffdf7f", "#ffeab2"]):
            ax.add_patch(Rectangle((x + 1.05 + 0.08 * j, yy + 0.02), 0.045, 0.18, facecolor=c, edgecolor="none"))


def draw_framework():
    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=160)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Input and skeleton.
    box(ax, (0.35, 5.45), 1.55, 1.35, "RGB video\nframes", fc="#f2f2f2", ec="#333333", fs=10)
    box(ax, (0.35, 2.65), 1.55, 1.35, "68-keypoint\nskeleton\nT=64, V=68", fc="#e8f1ff", ec="#4c78a8", fs=10)
    arrow(ax, (1.12, 5.45), (1.12, 4.05))
    ax.text(1.35, 4.75, "MediaPipe Holistic\nrepair + normalize", fontsize=8.5, va="center")

    draw_feature_group(ax, 3.0, 5.65, "Spatial descriptors", ["Joint J", "Bone B", "Position P"])
    draw_feature_group(ax, 3.0, 3.75, "Temporal descriptors", ["Motion dJ", "Bone motion dB", "Accel d2J"])
    draw_feature_group(ax, 3.0, 1.85, "Auxiliary descriptors", ["Root motion", "Time pos", "Validity mask"])
    arrow(ax, (1.9, 3.32), (3.0, 4.35))
    arrow(ax, (4.95, 6.25), (5.65, 6.25))
    arrow(ax, (4.95, 4.35), (5.65, 4.35))
    arrow(ax, (4.95, 2.45), (5.65, 2.45))

    box(ax, (5.65, 2.0), 0.62, 4.7, "Concat", fc="#c9fb91", ec="#7ab65b", fs=10)
    box(ax, (6.75, 3.75), 0.92, 0.75, "X", fc="#dcecff", ec="#75a3d6", fs=13, style="rect")
    arrow(ax, (6.27, 4.35), (6.75, 4.12))

    box(ax, (8.15, 3.55), 1.65, 1.15, "AT-STGCN\ncore blocks xL", fc="#ffe6a6", ec="#cc9c36", fs=11)
    arrow(ax, (7.67, 4.12), (8.15, 4.12))
    box(ax, (10.35, 3.75), 0.92, 0.75, "Z", fc="#dcecff", ec="#75a3d6", fs=13, style="rect")
    arrow(ax, (9.8, 4.12), (10.35, 4.12))

    box(ax, (11.85, 4.55), 1.15, 0.42, "STC attention", fc="#faf5ff", ec="#8d6aa8", fs=8.5)
    box(ax, (11.55, 2.65), 1.72, 1.35, "Global + part-aware\npooling\n\nGlobal | Body/Pose\nFace/Head | L/R hands", fc="#fbf0ff", ec="#8d6aa8", fs=8.2)
    arrow(ax, (11.27, 4.12), (11.85, 4.76))
    arrow(ax, (12.43, 4.55), (12.43, 4.0))
    arrow(ax, (12.43, 2.65), (13.55, 3.1))
    box(ax, (13.55, 3.05), 0.95, 0.55, "Concat\nLN + Dropout", fc="#ffffff", ec="#8d6aa8", fs=8.2)
    arrow(ax, (14.5, 3.32), (14.95, 3.32))
    box(ax, (14.95, 3.05), 0.48, 0.55, "ArcFace", fc="#ffffff", ec="#8d6aa8", fs=8.2)

    # Topology note.
    box(ax, (7.15, 0.48), 5.65, 1.38, "", fc="#fff9d8", ec="#d2a84b", lw=1.3)
    ax.text(9.95, 1.55, "Adaptive topology inside each core block", ha="center", va="center", fontsize=9.5, weight="bold")
    for x, text, fc in [(7.45, "A_k\nfixed\nmulti-hop", "#dcecff"), (8.55, "P_k\nlearnable\nadjacency", "#ffdede"), (9.65, "E_k\nedge\nimportance", "#ffe6bf")]:
        box(ax, (x, 0.78), 0.85, 0.5, text, fc=fc, ec="#666666", fs=7.5)
    box(ax, (10.72, 0.73), 1.82, 0.62, "Ahat_k core\n= (A_k + alpha tanh(P_k)) * E_k", fc="#ffffff", ec="#d2a84b", lw=0.9, fs=7.5)
    arrow(ax, (9.95, 1.8), (8.95, 3.55), color="#d2a84b", dashed=True)

    box(ax, (12.95, 0.55), 2.1, 1.25, "Candidate branches\n(ablation only)\nR_k(X), rho>0\nmulti-scale TCN", fc="#f6f6f6", ec="#999999", lw=1.2, fs=8.2)
    arrow(ax, (12.95, 1.18), (9.8, 3.55), color="#777777", dashed=True, rad=0.15)

    # Legend.
    ax.plot([0.5, 1.0], [0.65, 0.65], color="#111111", lw=1.7)
    arrow(ax, (1.0, 0.65), (1.45, 0.65))
    ax.text(1.55, 0.65, "main flow", va="center", fontsize=8)
    ax.plot([2.35, 2.85], [0.65, 0.65], color="#777777", lw=1.5, linestyle=(0, (5, 4)))
    ax.text(2.95, 0.65, "optional / ablation link", va="center", fontsize=8)
    ax.text(4.55, 0.65, "stable core: multi-source features + adaptive topology + part pooling", va="center", fontsize=8)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "fig1_framework_atstgcn.png", bbox_inches="tight")
    plt.close(fig)


def draw_block():
    fig, ax = plt.subplots(figsize=(15.5, 7.2), dpi=160)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    box(ax, (0.35, 3.1), 0.85, 0.62, "X", fc="#dcecff", ec="#75a3d6", fs=13, style="rect")
    arrow(ax, (1.2, 3.41), (2.1, 3.41))

    box(ax, (2.1, 2.25), 3.15, 2.35, "", fc="#fff8dd", ec="#d2a84b", lw=1.3)
    ax.text(3.68, 4.35, "Stable adaptive topology support", ha="center", va="center", fontsize=10, weight="bold")
    box(ax, (2.35, 3.55), 0.72, 0.45, "A_k", fc="#dcecff", ec="#4c78a8", fs=10)
    box(ax, (3.25, 3.55), 0.72, 0.45, "P_k", fc="#ffdede", ec="#be5c5c", fs=10)
    box(ax, (4.15, 3.55), 0.72, 0.45, "E_k", fc="#ffe6bf", ec="#c7892f", fs=10)
    ax.text(3.68, 3.0, "Ahat_core = (A_k + alpha tanh(P_k)) * E_k", ha="center", va="center", fontsize=8.5)
    box(ax, (3.25, 2.42), 0.85, 0.42, "Ahat_k", fc="#fff0b3", ec="#cfa13c", fs=9)

    box(ax, (2.15, 5.2), 3.05, 1.05, "Optional dynamic relation branch\n(ablation candidate)\nR_k(X), enabled only when rho > 0", fc="#f6fff0", ec="#6caa56", lw=1.1, fs=8.2)
    arrow(ax, (1.0, 3.72), (2.15, 5.72), color="#6caa56", dashed=True, rad=0.22)
    arrow(ax, (3.68, 5.2), (4.0, 2.84), color="#6caa56", dashed=True)
    ax.text(4.45, 2.82, "+ rho R_k(X)", fontsize=8, color="#3f7e37", va="center")

    arrow(ax, (5.25, 3.41), (6.0, 3.41))
    box(ax, (6.0, 3.0), 1.35, 0.82, "K-hop\nGraph Conv", fc="#ffffff", ec="#4c78a8", fs=10)
    arrow(ax, (7.35, 3.41), (8.25, 3.41))
    box(ax, (8.25, 3.1), 0.85, 0.62, "Z_s", fc="#dcecff", ec="#75a3d6", fs=12, style="rect")

    box(ax, (9.75, 3.0), 1.55, 0.82, "Residual\nTemporal Conv\n(core)", fc="#e9f2ff", ec="#4c78a8", fs=9.5)
    arrow(ax, (9.1, 3.41), (9.75, 3.41))
    arrow(ax, (11.3, 3.41), (12.1, 3.41))
    ax.text(11.7, 3.65, "+", fontsize=18, ha="center", va="center")

    box(ax, (9.15, 4.75), 2.75, 1.25, "Optional multi-scale temporal conv\n(ablation candidate)\nTCN k=7, d=1 | d=2 | d=3\nconcat + 1x1 fuse", fc="#f2f6ff", ec="#6e8fc2", lw=1.1, fs=8.2)
    arrow(ax, (8.95, 3.72), (9.3, 4.75), color="#6e8fc2", dashed=True, rad=0.15)
    arrow(ax, (11.55, 4.75), (11.85, 3.55), color="#6e8fc2", dashed=True)

    box(ax, (0.75, 0.95), 10.75, 0.56, "Residual path: Identity or 1x1 Conv + BN", fc="#ffffff", ec="#888888", fs=9)
    arrow(ax, (0.78, 3.1), (0.78, 1.51), color="#777777", dashed=True)
    arrow(ax, (11.5, 1.23), (12.1, 3.25), color="#777777", dashed=True, rad=0.1)

    box(ax, (12.1, 3.1), 0.75, 0.62, "+", fc="#ffffff", ec="#333333", fs=16)
    arrow(ax, (12.85, 3.41), (13.55, 3.41))
    box(ax, (13.55, 3.12), 0.9, 0.58, "GELU", fc="#eadcf3", ec="#8d6aa8", fs=10)
    arrow(ax, (14.45, 3.41), (15.05, 3.41))
    box(ax, (15.05, 3.1), 0.35, 0.62, "Z", fc="#dcecff", ec="#75a3d6", fs=12, style="rect")

    # Legend.
    ax.plot([0.5, 1.05], [0.35, 0.35], color="#111111", lw=1.7)
    arrow(ax, (1.05, 0.35), (1.42, 0.35))
    ax.text(1.55, 0.35, "main path", fontsize=8, va="center")
    ax.plot([2.45, 2.95], [0.35, 0.35], color="#777777", lw=1.5, linestyle=(0, (5, 4)))
    ax.text(3.05, 0.35, "optional or residual path", fontsize=8, va="center")
    ax.text(5.25, 0.35, "stable core keeps A_k, P_k and E_k; R_k and multi-scale TCN are not mandatory", fontsize=8, va="center")

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "fig2_block_atstgcn.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_framework()
    draw_block()
