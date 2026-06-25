"""Run the carving and emit a single self-contained 3D HTML viewer.

    uv run python run_web3d.py --word TIME --iterations 300 --frames 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes

from fluviglyph import World, FluidSolver, ErosionModel
from fluviglyph.web3d import build_html


MODES = {"anaglyph", "sbs", "flat"}


def _mesh_to_frame(stone: np.ndarray) -> dict:
    """Run marching cubes; return verts (float32, in [0,1]^3) + indices (uint32).

    A FIXED iso level of 0.5 is used (not adaptive) so only real solid stone
    (density >= 0.5) is rendered. As the word erodes below 0.5 it vanishes and
    stays gone — no low-density halos or sediment residue can surface as new
    bodies. A light smoothing (sigma=0.5) keeps the surface organic; it spreads
    a little density into air but well below the 0.5 cut, so it never renders.
    """
    from scipy.ndimage import gaussian_filter
    padded = np.pad(stone, 2, mode="constant", constant_values=0.0)
    smoothed = gaussian_filter(padded, sigma=0.5, mode="constant")
    try:
        verts, faces, _, _ = marching_cubes(smoothed, level=0.5)
    except ValueError:
        return {"verts": np.zeros((0,), np.float32),
                "indices": np.zeros((0,), np.uint32)}
    pad = 2
    nx, ny, nz = stone.shape
    verts = verts.astype(np.float32)
    verts[:, 0] = (verts[:, 0] - pad) / max(1, nx - 1)
    verts[:, 1] = (verts[:, 1] - pad) / max(1, ny - 1)
    verts[:, 2] = (verts[:, 2] - pad) / max(1, nz - 1)
    return {"verts": verts.reshape(-1).astype(np.float32),
            "indices": faces.reshape(-1).astype(np.uint32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="TIME")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--nz", type=int, default=36)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="output")
    ap.add_argument("--mode", default="anaglyph", choices=sorted(MODES))
    ap.add_argument("--sbs-cam-scale", type=float, default=1.0,
                    help="camera pullback for square-eye SBS framing (1.0 = no pullback)")
    ap.add_argument("--eye-sep", type=float, default=0.06,
                    help="stereo eye separation (bigger = bolder depth; 0.06 default)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"fluviglyph · carving '{args.word}' over {args.iterations} iterations")
    world = World.from_word(args.word, nx=args.nx, ny=args.ny, nz=args.nz)
    fluid = FluidSolver(world, viscosity=1e-5, jacobi_proj=8, jacobi_diff=3)
    erosion = ErosionModel(world, dt=args.dt)

    capture_every = max(1, args.iterations // args.frames)
    mesh_frames: list[dict] = []
    frame_iters: list[int] = []

    def capture(it: int) -> None:
        stone = np.clip(world.stone, 0.0, 1.0)
        mesh_frames.append(_mesh_to_frame(stone))
        frame_iters.append(it)

    capture(0)
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        fluid.step(args.dt); erosion.step()
        if it % capture_every == 0:
            capture(it)
    if frame_iters[-1] != args.iterations:
        capture(args.iterations)

    print(f"  carved in {time.time()-t0:.1f}s · {len(mesh_frames)} mesh frames")

    # flat mode renders a single mono view through the original camera (no SBS
    # pullback) — it's the 2D reference a viewer can compare against the stereo
    # outputs. SBS needs the pullback for square-eye framing; flat/anaglyph do not.
    scale = args.sbs_cam_scale if args.mode == "sbs" else 1.0
    name = {"anaglyph": "fluviglyph", "sbs": "fluviglyph_sbs",
            "flat": "fluviglyph_flat"}[args.mode]
    html_path = out / f"{name}.html"
    build_html(args.word, args.iterations, mesh_frames, frame_iters, str(html_path),
               mode=args.mode, sbs_cam_scale=scale, eye_sep=args.eye_sep)
    size_mb = html_path.stat().st_size / (1024 * 1024)
    print(f"wrote {html_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
