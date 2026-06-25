"""Deterministically capture the fluviglyph 3D viewer as frames.

    uv run python capture_anaglyph.py --mode anaglyph --sanity
    uv run python capture_anaglyph.py --mode sbs --frames 301 --width 2560 --height 1280
    uv run python capture_anaglyph.py --mode sbs --frames 301 --workers 6

The page's wall-clock playback (DURATION) is bypassed: for each frac in [0,1]
we call window.__fluviglyph.setFrac(f) then renderNow() then screenshot the
<canvas> element directly (clean WebGL pixels, no HTML overlay chrome), so
frame<->content is exact and reproducible. Stitch with ffmpeg afterwards.

  anaglyph: red/cyan full-frame, single view. View with red/cyan glasses
            (red over left eye).
  sbs:      two square (1:1) eyes side-by-side in a 2:1 container, off-axis
            keystone-free StereoCamera. View in a 3D-SBS player / headset, or
            free-view / cross-eye on screen.

Performance: headless Chromium on macOS defaults to SwiftShader (CPU software
rendering) for WebGL, leaving the GPU idle. We force the ANGLE-Metal backend
(see GPU_ARGS) so the Apple Silicon GPU does the work (~17x faster), and shard
frames across --workers parallel browser pages to saturate the system.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).parent

# Force the real Metal-backed ANGLE path instead of SwiftShader (CPU software).
# Verified: unmasked renderer -> "ANGLE Metal Renderer: Apple M4 Pro".
GPU_ARGS = ["--use-gl=angle", "--use-angle=metal", "--enable-gpu",
            "--ignore-gpu-blocklist"]

HTML_BY_MODE = {
    "flat": ROOT / "output" / "fluviglyph_flat.html",
    "anaglyph": ROOT / "output" / "fluviglyph.html",
    "sbs": ROOT / "output" / "fluviglyph_sbs.html",
}
OUT_BY_MODE = {
    "flat": ROOT / "output" / "flat_frames",
    "anaglyph": ROOT / "output" / "anaglyph_frames",
    "sbs": ROOT / "output" / "sbs_frames",
}
SANITY_BY_MODE = {
    "flat": ROOT / "output" / "flat_sanity",
    "anaglyph": ROOT / "output" / "anaglyph_sanity",
    "sbs": ROOT / "output" / "sbs_sanity",
}
# flat & anaglyph are full-frame mono views at 16:9; sbs is two square eyes (2:1).
DEFAULT_SIZE = {
    "flat": (1920, 1080),
    "anaglyph": (1920, 1080),
    "sbs": (2560, 1280),
}


def _sanity_fracs() -> list[float]:
    return [0.0, 0.25, 0.5, 0.9]


async def _worker(browser, mode: str, indices: list[int], fracs: list[float],
                  width: int, height: int, out: Path) -> list[str]:
    """One page rendering+screenshotting a shard of frames in parallel."""
    ctx = await browser.new_context(viewport={"width": width, "height": height})
    page = await ctx.new_page()
    errs: list[str] = []
    page.on("pageerror", lambda e: errs.append(f"[pageerror] {e}"))
    await page.goto(HTML_BY_MODE[mode].as_uri(), wait_until="domcontentloaded",
                    timeout=180_000)
    await page.wait_for_function(
        "window.__fluviglyph && typeof window.__fluviglyph.setFrac==='function'",
        timeout=60_000)
    canvas = page.locator("canvas")
    for idx, f in zip(indices, fracs):
        await page.evaluate(
            "(f) => { window.__fluviglyph.setFrac(f); window.__fluviglyph.renderNow(); }",
            f)
        await page.wait_for_timeout(40)
        await canvas.screenshot(path=str(out / f"frame_{idx:05d}.png"))
    await ctx.close()
    return errs


async def run(mode: str, frames: int, width: int, height: int, sanity: bool,
              workers: int) -> None:
    html = HTML_BY_MODE[mode]
    if not html.exists():
        raise SystemExit(f"missing {html} — run_web3d.py --mode {mode} first")
    out = SANITY_BY_MODE[mode] if sanity else OUT_BY_MODE[mode]
    if out.name.endswith("_frames") and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    fracs = _sanity_fracs() if sanity else [i / max(1, frames - 1) for i in range(frames)]
    if sanity:
        workers = 1
    sharded = list(range(len(fracs)))
    shards_idx = [sharded[i::workers] for i in range(workers)]
    shards_frac = [fracs[i::workers] for i in range(workers)]
    print(f"[{mode}] capturing {len(fracs)} frames at {width}x{height} "
          f"via {workers} worker(s) -> {out}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=GPU_ARGS)
        try:
            t0 = time.time()
            results = await asyncio.gather(*[
                _worker(browser, mode, shards_idx[i], shards_frac[i], width, height, out)
                for i in range(workers)])
            dt = time.time() - t0
        finally:
            await browser.close()
        errs = [e for r in results for e in r]
        print(f"captured {len(fracs)} frames in {dt:.1f}s "
              f"({dt/len(fracs):.3f}s/frame, {workers} workers)")
        print("page errors: " + ("\n" + "\n".join(errs) if errs else "(none)"))
    print(f"done -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sbs", choices=sorted(HTML_BY_MODE))
    ap.add_argument("--frames", type=int, default=301)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--sanity", action="store_true",
                    help="capture just a few frames to eyeball before a full run")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel browser pages rendering frame shards")
    args = ap.parse_args()
    w, h = (args.width, args.height) if args.width and args.height else DEFAULT_SIZE[args.mode]
    asyncio.run(run(args.mode, args.frames, w, h, args.sanity, max(1, args.workers)))


if __name__ == "__main__":
    main()