"""Render the carving — cross-sections, montages, and the final worn stone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .world import World


# A palette that reads as wet stone: cold air, warm silt, pale mineral.
STONE_CMAP = LinearSegmentedColormap.from_list(
    "riverstone",
    ["#0b1416", "#1d2b2e", "#3a5552", "#8a9b8e", "#d8cdb4", "#f4ecd2"],
    N=256,
)


def save_topdown(world: World, path: str, step: int | None = None) -> None:
    """Top-down view: max projection of stone along z (depth), overlaid with
    a whisper of the current's speed to show the river still running."""
    stone = world.stone.max(axis=2)
    speed = np.sqrt(world.u ** 2 + world.v ** 2 + world.w ** 2).mean(axis=2)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    ax.imshow(speed.T, origin="upper", cmap="bone", alpha=0.45,
              extent=(0, world.nx, world.ny, 0))
    ax.imshow(stone.T, origin="upper", cmap=STONE_CMAP, alpha=0.95,
              extent=(0, world.nx, world.ny, 0))
    ax.set_xticks([]); ax.set_yticks([])
    title = "WORN · carved by a river"
    if step is not None:
        title += f"   ·   iteration {step:>5d}"
    ax.set_title(title, fontsize=11, color="#2b2b2b", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("#e9e4d8")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_topdown_montage(frames: list[np.ndarray], path: str,
                         steps: list[int]) -> None:
    """Stack saved top-down stone projections into a single contact sheet."""
    n = len(frames)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 1.7),
                             dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for ax, frame, step in zip(axes, frames, steps):
        ax.imshow(frame.T, origin="upper", cmap=STONE_CMAP)
        ax.set_title(f"{step:>5d}", fontsize=8, color="#2b2b2b")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("WORN — ten thousand iterations of a river",
                 fontsize=11, color="#2b2b2b", y=0.995)
    fig.patch.set_facecolor("#e9e4d8")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_cross_section(world: World, path: str, axis: int = 1,
                       idx: int | None = None) -> None:
    """A single slice through the stone — the river's view from the side."""
    if idx is None:
        idx = world.shape[axis] // 2
    if axis == 1:
        slc = world.stone[:, idx, :]
    elif axis == 2:
        slc = world.stone[:, :, idx]
    else:
        slc = world.stone[idx, :, :]
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=130)
    ax.imshow(slc.T, origin="upper", cmap=STONE_CMAP)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"WORN · cross-section (axis {axis}, slice {idx})",
                 fontsize=10, color="#2b2b2b")
    fig.patch.set_facecolor("#e9e4d8")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_mesh_render(mesh, path: str) -> None:
    """A reliable 3D preview of the extracted mesh using matplotlib."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    faces = mesh.faces
    verts = mesh.vertices
    # subsample for performance if the mesh is heavy
    if len(faces) > 40000:
        idx = np.random.default_rng(0).choice(len(faces), 40000, replace=False)
        faces = faces[idx]
    coll = Poly3DCollection(verts[faces], alpha=0.95,
                            facecolor="#c9bea0", edgecolor="#6b6354",
                            linewidth=0.05)
    ax.add_collection3d(coll)
    mins = verts.min(axis=0); maxs = verts.max(axis=0)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((maxs - mins).clip(min=1))
    ax.view_init(elev=22, azim=-60)
    ax.set_axis_off()
    fig.patch.set_facecolor("#10181a")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
