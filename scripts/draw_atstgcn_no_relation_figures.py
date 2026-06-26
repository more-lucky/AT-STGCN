from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("paper/figures")


def box(ax, xy, w, h, text, fc="#ffffff", ec="#333333", lw=1.3, fs=9, alpha=1.0):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.045",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def tensor(ax, xy, text, fs=12):
    patch = Rectangle(xy, 0.72, 0.48, facecolor="#dcecff", edgecolor="#75a3d6", linewidth=1.2)
    ax.add_patch(patch)
    ax.text(xy[0] + 0.36, xy[1] + 0.24, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, color="#111111", lw=1.5, dashed=False, rad=0.0):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        linestyle=(0, (5, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def feature_group(ax, x, y, title, labels):
    box(ax, (x, y), 1.75, 1.12, "", fc="#e9f6df", ec="#5c9b5c", lw=1.1)
    ax.text(x + 0.88, y + 0.96, title, ha="center", va="center", fontsize=8.2, weight="bold")
    for i, label in enumerate(labels):
        yy = y + 0.68 - 0.28 * i
        box(ax, (x + 0.18, yy), 0.72, 0.18, label, fc="#ffffff", ec="#333333", lw=0.8, fs=6.8)
        for j, c in enumerate(["#7be35f", "#9bf86d", "#ffdf7f", "#ffeab2"]):
            ax.add_patch(Rectangle((x + 1.0 + 0.075 * j, yy + 0.015), 0.04, 0.15, facecolor=c, edgecolor="none"))


def draw_framework():
    fig, ax = plt.subplots(figsize=(14.2, 7.5), dpi=170)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    box(ax, (0.4, 5.0), 1.35, 1.15, "RGB video\nframes", fc="#f2f2f2", fs=9)
    box(ax, (0.4, 2.45), 1.35, 1.15, "68-keypoint\nskeleton\nT=64,V=68", fc="#e8f1ff", ec="#4c78a8", fs=8.8)
    arrow(ax, (1.08, 5.0), (1.08, 3.6))
    ax.text(1.35, 4.25, "MediaPipe Holistic\nrepair + normalize", fontsize=7.8, va="center")

    feature_group(ax, 3.05, 5.25, "Spatial descriptors", ["Joint J", "Bone B", "Position P"])
    feature_group(ax, 3.05, 3.55, "Temporal descriptors", ["Motion dJ", "Bone motion dB", "Accel d2J"])
    feature_group(ax, 3.05, 1.85, "Auxiliary descriptors", ["Root motion", "Time pos", "Validity mask"])
    arrow(ax, (1.75, 3.02), (3.05, 4.12))
    arrow(ax, (4.8, 5.82), (5.45, 5.82))
    arrow(ax, (4.8, 4.12), (5.45, 4.12))
    arrow(ax, (4.8, 2.42), (5.45, 2.42))
    box(ax, (5.45, 1.9), 0.55, 4.45, "Concat", fc="#c9fb91", ec="#7ab65b", fs=9)
    tensor(ax, (6.55, 3.85), "X", fs=12)
    arrow(ax, (6.0, 4.12), (6.55, 4.09))

    box(ax, (7.75, 3.55), 1.55, 1.05, "AT-STGCN\nblocks xL", fc="#ffe6a6", ec="#cc9c36", fs=10)
    arrow(ax, (7.27, 4.09), (7.75, 4.08))
    tensor(ax, (9.95, 3.85), "Z", fs=12)
    arrow(ax, (9.3, 4.08), (9.95, 4.09))

    box(ax, (11.15, 4.45), 1.1, 0.38, "STC attention", fc="#faf5ff", ec="#8d6aa8", fs=7.8)
    box(ax, (10.88, 2.72), 1.6, 1.18, "Global + part-aware\npooling\n\nGlobal | Body/Pose\nFace/Head | L/R hands", fc="#fbf0ff", ec="#8d6aa8", fs=7.4)
    arrow(ax, (10.67, 4.09), (11.15, 4.64))
    arrow(ax, (11.7, 4.45), (11.7, 3.9))
    box(ax, (12.88, 3.05), 0.85, 0.48, "Concat\nLN+Dropout", fc="#ffffff", ec="#8d6aa8", fs=7.5)
    arrow(ax, (12.48, 3.18), (12.88, 3.28))
    box(ax, (13.98, 3.05), 0.5, 0.48, "ArcFace", fc="#ffffff", ec="#8d6aa8", fs=7.5)
    arrow(ax, (13.73, 3.29), (13.98, 3.29))

    box(ax, (6.7, 0.55), 5.25, 1.15, "", fc="#fff9d8", ec="#d2a84b", lw=1.2)
    ax.text(9.32, 1.42, "Adaptive topology inside each block", ha="center", va="center", fontsize=8.6, weight="bold")
    for x, text, fc in [(7.0, "A_k\nfixed\nmulti-hop", "#dcecff"), (8.05, "P_k\nlearnable\nadjacency", "#ffdede"), (9.1, "E_k\nedge\nimportance", "#ffe6bf")]:
        box(ax, (x, 0.76), 0.82, 0.45, text, fc=fc, ec="#666666", fs=7.2)
    box(
        ax,
        (10.28, 0.72),
        1.45,
        0.55,
        r"$\widehat{A}_k=(A_k+\alpha\tanh(P_k))\odot E_k$",
        fc="#ffffff",
        ec="#d2a84b",
        lw=0.9,
        fs=6.8,
    )
    arrow(ax, (9.32, 1.7), (8.52, 3.55), color="#d2a84b", dashed=True)

    ax.plot([0.45, 0.95], [0.52, 0.52], color="#111111", lw=1.6)
    arrow(ax, (0.95, 0.52), (1.28, 0.52))
    ax.text(1.38, 0.52, "main flow", va="center", fontsize=7.5)
    ax.plot([2.2, 2.68], [0.52, 0.52], color="#d2a84b", lw=1.3, linestyle=(0, (5, 4)))
    ax.text(2.8, 0.52, "adaptive topology link", va="center", fontsize=7.5)

    fig.tight_layout(pad=0.45)
    fig.savefig(OUT_DIR / "fig1_framework_atstgcn.png", bbox_inches="tight")
    plt.close(fig)


def draw_block():
    fig, ax = plt.subplots(figsize=(14.2, 4.5), dpi=170)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 4.35)
    ax.axis("off")

    tensor(ax, (0.4, 2.65), "X")
    arrow(ax, (1.12, 2.89), (1.8, 2.89))
    box(ax, (1.8, 1.85), 3.55, 2.1, "", fc="#fff8dd", ec="#d2a84b", lw=1.2)
    ax.text(3.58, 3.72, "Stable adaptive topology support", ha="center", va="center", fontsize=10, weight="bold")
    box(ax, (2.15, 3.08), 0.68, 0.45, "A_k", fc="#dcecff", ec="#4c78a8", fs=10)
    box(ax, (3.02, 3.08), 0.68, 0.45, "P_k", fc="#ffdede", ec="#be5c5c", fs=10)
    box(ax, (3.89, 3.08), 0.68, 0.45, "E_k", fc="#ffe6bf", ec="#c7892f", fs=10)
    ax.text(
        3.58,
        2.48,
        r"$\widehat{A}_k=(A_k+\alpha\tanh(P_k))\odot E_k$",
        ha="center",
        va="center",
        fontsize=8.2,
    )
    box(ax, (3.13, 2.02), 0.9, 0.38, r"$\widehat{A}_k$", fc="#fff0b3", ec="#cfa13c", fs=8.5)

    arrow(ax, (5.35, 2.89), (6.0, 2.89))
    box(ax, (6.0, 2.52), 1.25, 0.74, "K-hop\nGraph Conv", fc="#ffffff", ec="#4c78a8", fs=9.2)
    arrow(ax, (7.25, 2.89), (7.95, 2.89))
    tensor(ax, (7.95, 2.65), "Z_s")
    arrow(ax, (8.67, 2.89), (9.5, 2.89))

    box(ax, (9.5, 2.52), 1.35, 0.74, "Temporal Conv\nk=7, d in D", fc="#e9f2ff", ec="#4c78a8", fs=9.0)
    arrow(ax, (10.85, 2.89), (11.52, 2.89))
    box(ax, (11.52, 2.6), 0.68, 0.58, "+", fc="#ffffff", ec="#333333", fs=15)
    box(ax, (0.85, 0.75), 10.3, 0.5, "Residual path: Identity or 1x1 Conv + BN", fc="#ffffff", ec="#888888", fs=8.5)
    arrow(ax, (0.76, 2.65), (0.76, 1.25), color="#777777", dashed=True)
    arrow(ax, (11.15, 1.0), (11.52, 2.72), color="#777777", dashed=True, rad=0.08)
    arrow(ax, (12.2, 2.89), (12.88, 2.89))
    box(ax, (12.88, 2.62), 0.85, 0.54, "GELU", fc="#eadcf3", ec="#8d6aa8", fs=9.5)
    arrow(ax, (13.73, 2.89), (14.05, 2.89))
    tensor(ax, (14.05, 2.65), "Z", fs=11)

    ax.plot([0.5, 1.0], [0.35, 0.35], color="#111111", lw=1.6)
    arrow(ax, (1.0, 0.35), (1.32, 0.35))
    ax.text(1.42, 0.35, "main path", fontsize=7.5, va="center")
    ax.plot([2.15, 2.65], [0.35, 0.35], color="#777777", lw=1.3, linestyle=(0, (5, 4)))
    ax.text(2.75, 0.35, "residual path", fontsize=7.5, va="center")
    ax.text(4.2, 0.35, "AT-STGCN uses fixed multi-hop supports, learnable adjacency and edge importance", fontsize=7.5, va="center")

    fig.tight_layout(pad=0.45)
    fig.savefig(OUT_DIR / "fig2_block_atstgcn.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_framework()
    draw_block()
