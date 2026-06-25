# fluviglyph

Typography carved by rivers. A letterform is dropped into a virtual Navier–Stokes
current; shear stress ablates the stone and sediment is carried off. After many
iterations the blocky glyph becomes something organic — a marching-cubes mesh
that erodes over time.

Three renderings of the same carving (the word `TIME`, 300 iterations of a
stable-fluids current), all frame-locked on the same timeline:

| video | format | view with |
|---|---|---|
| `output/fluviglyph_flat.mp4` | 2D mono reference (1920×1080) | plain |
| `output/anaglyph.mp4` | red/cyan anaglyph (1920×1080) | red/cyan glasses, red over left |
| `output/fluviglyph_sbs.mp4` | side-by-side, two 1280×1280 square eyes | 3D-SBS player / headset, or `mpv --vf=stereo3d=sbsl:arcd` |

## Pipeline

```bash
# carve + emit a self-contained three.js viewer (flat / anaglyph / sbs)
uv run python run_web3d.py --word TIME --iterations 300 --frames 200 --mode anaglyph --eye-sep 0.05
uv run python run_web3d.py --word TIME --iterations 300 --frames 200 --mode sbs --sbs-cam-scale 2.0 --eye-sep 0.22
uv run python run_web3d.py --word TIME --iterations 300 --frames 200 --mode flat

# capture deterministic frames + stitch to mp4 (GPU-accelerated, 6 parallel workers)
uv run python capture_anaglyph.py --mode sbs --frames 301 --width 2560 --height 1280 --workers 6
ffmpeg -framerate 30 -i output/sbs_frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output/fluviglyph_sbs.mp4
```

The 3D viewer (`fluviglyph/web3d.py`) builds a single self-contained HTML with
three.js. Stereo modes use an off-axis asymmetric-frustum `StereoCamera`
(keystone-free) converged on the word:

- **anaglyph** — red/cyan Dubois encoding of left/right eyes into one full frame.
- **sbs** — two square (1:1) eyes side by side in a 2:1 container; the square
  aspect narrows horizontal FOV, so `--sbs-cam-scale` pulls the camera back to
  keep the word clear in both eyes with parallax headroom.
- **flat** — a single mono camera through the original framing, for reference.

`--eye-sep` is the depth knob (bigger = bolder parallax, more eyestrain).

### Why GPU launch flags

Headless Chromium on macOS defaults to SwiftShader (CPU software rendering) for
WebGL — leaving the GPU idle and ~17× slower. `capture_anaglyph.py` forces the
Metal-backed ANGLE backend, then shards frames across parallel browser pages.

## Requirements

Python ≥ 3.13, uv. Simulation deps are in `pyproject.toml`; Playwright is a dev
dependency (`uv run playwright install chromium`).