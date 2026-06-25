"""Fluviglyph — drop a word into a river and let it wear.

    uv run python main.py --word WORN --iterations 10000

Checkpoints are rendered as top-down views; the final worn stone is extracted
into a mesh (.glb / .obj / .stl) and previewed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fluviglyph import (
    World,
    FluidSolver,
    ErosionModel,
    extract_mesh,
    save_topdown,
    save_topdown_montage,
    save_cross_section,
    save_mesh_render,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Carve a word with a simulated river.")
    ap.add_argument("--word", default="WORN")
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nz", type=int, default=48)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints"; ckpt.mkdir(exist_ok=True)

    print(f"fluviglyph · carving '{args.word}' in a river of {args.iterations} iterations")
    world = World.from_word(args.word, nx=args.nx, ny=args.ny, nz=args.nz)
    fluid = FluidSolver(world, viscosity=1e-5, jacobi_proj=8, jacobi_diff=3)
    erosion = ErosionModel(world, dt=args.dt)

    # show the raw blocky stone before the river starts
    save_topdown(world, str(out / "00_before.png"), step=0)

    frames: list[np.ndarray] = []
    steps_logged: list[int] = []
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        fluid.step(args.dt)
        erosion.step()

        # occasional very light smoothing so newly exposed interior isn't
        # voxel-crufty, but gentle enough that the river keeps finding edges.
        if it % 500 == 0:
            from scipy.ndimage import gaussian_filter
            world.stone = gaussian_filter(world.stone, sigma=0.3).astype(np.float32)
            np.clip(world.stone, 0.0, 1.0, out=world.stone)

        if it % args.checkpoint_every == 0 or it == args.iterations:
            elapsed = time.time() - t0
            remaining = elapsed / it * (args.iterations - it)
            stone_mass = world.stone.sum()
            print(f"  iter {it:>5d}/{args.iterations}  "
                  f"stone={stone_mass:8.0f}  "
                  f"elapsed={elapsed:5.1f}s  eta={remaining:5.1f}s")
            save_topdown(world, str(ckpt / f"iter_{it:05d}.png"), step=it)
            frames.append(world.stone.max(axis=2).copy())
            steps_logged.append(it)

    save_topdown_montage(frames, str(out / "01_montage.png"), steps_logged)
    save_topdown(world, str(out / "02_after.png"), step=args.iterations)
    save_cross_section(world, str(out / "03_cross_section.png"), axis=2,
                       idx=world.nz // 2)

    print("extracting worn mesh via marching cubes…")
    mesh = extract_mesh(world, level=None)
    glb = out / "worn.glb"; obj = out / "worn.obj"; stl = out / "worn.stl"
    mesh.export(glb); mesh.export(obj); mesh.export(stl)
    save_mesh_render(mesh, str(out / "04_mesh.png"))
    print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"wrote {glb}, {obj}, {stl}")
    print(f"done in {time.time()-t0:.1f}s — see {out}/")


if __name__ == "__main__":
    main()
