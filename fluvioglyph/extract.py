"""Extract a smooth triangle mesh from the worn voxel stone via marching cubes."""

from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes
import trimesh

from .world import World


def extract_mesh(world: World, level: float | None = None,
                 pad: int = 2) -> trimesh.Trimesh:
    """Run marching cubes on the stone field and return a watertight-ish mesh.

    A light Gaussian smoothing is applied first so the extracted surface is
    organic rather than voxel-stepped — the river's smoothing, made literal.
    If ``level`` is None, an isovalue is chosen adaptively from the surviving
    stone so the worn form is always captured.
    """
    from scipy.ndimage import gaussian_filter

    stone = world.stone.astype(np.float32)
    padded = np.pad(stone, pad, mode="constant", constant_values=0.0)
    smoothed = gaussian_filter(padded, sigma=1.0, mode="constant")

    if level is None:
        lo = float(smoothed.min())
        hi = float(smoothed.max())
        # an isovalue near the lower end keeps the worn, low-density rind;
        # the form is already soft, so we don't want a high cut that erases it.
        level = lo + 0.35 * (hi - lo)
        level = max(level, 1e-3)

    verts, faces, _, _ = marching_cubes(smoothed, level=level)
    # marching_cubes returns faces as (n,3) int with 1-based-ish vertex refs
    # in older skimage; in 0.26 they are 0-based directly. Guard both:
    if faces.size and faces.max() >= len(verts):
        faces = faces - 1

    mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces)
    # tidy up: merge vertices, drop degenerates, fix winding/normals
    mesh.process()
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh
