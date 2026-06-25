"""Run the full fluviglyph carving and emit a single self-contained HTML viewer.

    uv run python run_web.py --word WORN --iterations 100

Produces output/fluviglyph.html — open it in any browser.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np

from fluviglyph import World, FluidSolver, ErosionModel, extract_mesh
from fluviglyph.web import build_html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="WORN")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nz", type=int, default=48)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=60,
                    help="number of animation frames to capture")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"fluviglyph · carving '{args.word}' over {args.iterations} iterations")
    world = World.from_word(args.word, nx=args.nx, ny=args.ny, nz=args.nz)
    fluid = FluidSolver(world, viscosity=1e-5, jacobi_proj=8, jacobi_diff=3)
    erosion = ErosionModel(world, dt=args.dt)

    n_frames = args.frames
    capture_every = max(1, args.iterations // n_frames)
    frames: list[np.ndarray] = []
    frame_u: list[np.ndarray] = []
    frame_v: list[np.ndarray] = []
    frame_iters: list[int] = []
    # always capture the pristine state first
    frames.append(world.stone.max(axis=2).copy())
    frame_u.append(world.u.mean(axis=2).copy())
    frame_v.append(world.v.mean(axis=2).copy())
    frame_iters.append(0)

    t0 = time.time()
    for it in range(1, args.iterations + 1):
        fluid.step(args.dt)
        erosion.step()
        if it % 250 == 0:
            from scipy.ndimage import gaussian_filter
            world.stone = gaussian_filter(world.stone, sigma=0.3).astype(np.float32)
            np.clip(world.stone, 0.0, 1.0, out=world.stone)
        if it % capture_every == 0:
            frames.append(world.stone.max(axis=2).copy())
            frame_u.append(world.u.mean(axis=2).copy())
            frame_v.append(world.v.mean(axis=2).copy())
            frame_iters.append(it)

    # always capture the final state
    if frame_iters[-1] != args.iterations:
        frames.append(world.stone.max(axis=2).copy())
        frame_u.append(world.u.mean(axis=2).copy())
        frame_v.append(world.v.mean(axis=2).copy())
        frame_iters.append(args.iterations)

    print(f"  carved in {time.time()-t0:.1f}s · {len(frames)} frames captured")

    print("extracting worn mesh via marching cubes…")
    mesh = extract_mesh(world, level=None)
    glb_buf = io.BytesIO()
    mesh.export(glb_buf, file_type="glb")
    glb_bytes = glb_buf.getvalue()
    print(f"  mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"{len(glb_bytes)/1024:.0f} KB glb")

    html_path = out / "fluviglyph.html"
    build_html(args.word, args.iterations, frames, frame_u, frame_v,
               frame_iters, glb_bytes, str(html_path))
    print(f"wrote {html_path} ({html_path.stat().st_size/1024:.0f} KB) — open it in a browser")


if __name__ == "__main__":
    main()
