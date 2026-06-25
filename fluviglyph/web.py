"""Build a single self-contained HTML page that plays the carving as an
animation alongside an interactive 3D view of the final worn mesh."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

STONE_CMAP = LinearSegmentedColormap.from_list(
    "riverstone",
    ["#0b1416", "#1d2b2e", "#3a5552", "#8a9b8e", "#d8cdb4", "#f4ecd2"],
    N=256,
)


def render_frame_png(stone_proj: np.ndarray,
                    u: np.ndarray | None = None,
                    v: np.ndarray | None = None) -> bytes:
    """Render a 2D stone projection (max over z) to PNG bytes.

    If velocity components are given (also max/mean over z, same 2D shape),
    draw the river as streamlines bending around the stone before overlaying
    the worn word on top.
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    ax.set_facecolor("#0b1416")

    if u is not None and v is not None:
        nx, ny = u.shape
        # subtle speed tint beneath the streamlines — the river's body
        speed = np.sqrt(u**2 + v**2)
        ax.imshow(speed.T, origin="upper", cmap="bone", alpha=0.55,
                  extent=(0, nx, ny, 0), aspect="auto", interpolation="bilinear")
        # streamlines: seed across the upstream edge + a grid, let them bend
        # around the stone. A fine grid shows the deflection clearly.
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        seed_x = np.concatenate([np.full(8, 1),
                                 np.linspace(2, nx-2, 14)])
        seed_y = np.concatenate([np.linspace(2, ny-2, 8),
                                 np.full(14, ny//2)])
        seed = np.stack([seed_x, seed_y], axis=1)
        try:
            ax.streamplot(X.T, Y.T, u.T, v.T,
                          start_points=seed, density=1.2,
                          color="#7fb3c4", linewidth=0.7, arrowsize=0.7,
                          integration_direction="forward")
        except Exception:
            pass

    ax.imshow(stone_proj.T, origin="upper", cmap=STONE_CMAP,
              aspect="auto", interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.patch.set_facecolor("#10181a")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{word} · carved by a river</title>
<style>
  :root {{
    --bg: #10181a;
    --panel: #16201f;
    --ink: #e8e2d2;
    --muted: #7d8a85;
    --accent: #c9bea0;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0; min-height: 100vh;
    background: radial-gradient(ellipse at 50% 0%, #1a2422 0%, var(--bg) 70%);
    color: var(--ink);
    font-family: ui-sans-serif, -apple-system, "Helvetica Neue", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{
    max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px;
  }}
  header {{ text-align: center; margin-bottom: 28px; }}
  header .eyebrow {{
    text-transform: uppercase; letter-spacing: .28em; font-size: 11px;
    color: var(--muted); margin-bottom: 10px;
  }}
  header h1 {{
    font-family: "Georgia", "Times New Roman", serif;
    font-weight: 400; font-size: clamp(28px, 5vw, 44px); margin: 0;
    color: var(--ink); letter-spacing: .04em;
  }}
  header .sub {{
    margin-top: 10px; color: var(--muted); font-size: 14px; font-style: italic;
  }}
  .panels {{
    display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px;
  }}
  @media (max-width: 820px) {{ .panels {{ grid-template-columns: 1fr; }} }}
  .panel {{
    background: var(--panel); border: 1px solid #23302e; border-radius: 14px;
    overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,.35);
  }}
  .panel .label {{
    padding: 10px 16px; font-size: 11px; text-transform: uppercase;
    letter-spacing: .22em; color: var(--muted); border-bottom: 1px solid #23302e;
  }}
  .stage {{
    position: relative; width: 100%; aspect-ratio: 2 / 1;
    background: #0b1416; display: flex; align-items: center; justify-content: center;
  }}
  .stage canvas, .stage img {{
    width: 100%; height: 100%; object-fit: contain; display: block;
  }}
  #three {{
    position: absolute; inset: 0;
  }}
  .controls {{
    padding: 14px 16px; display: flex; align-items: center; gap: 14px;
    border-top: 1px solid #23302e;
  }}
  button {{
    appearance: none; background: #1d2b2e; color: var(--ink);
    border: 1px solid #2e3d3b; border-radius: 8px; padding: 8px 14px;
    font-size: 13px; cursor: pointer; transition: background .15s;
  }}
  button:hover {{ background: #28393a; }}
  .timeline {{
    flex: 1; position: relative; height: 28px; display: flex; align-items: center;
  }}
  .timeline input[type=range] {{
    width: 100%; -webkit-appearance: none; appearance: none;
    background: transparent; cursor: pointer;
  }}
  .timeline input[type=range]::-webkit-slider-runnable-track {{
    height: 4px; background: #2e3d3b; border-radius: 4px;
  }}
  .timeline input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; border-radius: 50%; background: var(--accent);
    margin-top: -5px; box-shadow: 0 0 0 3px rgba(201,190,160,.18);
  }}
  .timeline input[type=range]::-moz-range-track {{
    height: 4px; background: #2e3d3b; border-radius: 4px;
  }}
  .timeline input[type=range]::-moz-range-thumb {{
    width: 14px; height: 14px; border: none; border-radius: 50%;
    background: var(--accent);
  }}
  .iter-label {{
    font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted);
    min-width: 92px; text-align: right;
  }}
  .hint {{
    padding: 10px 16px; font-size: 11px; color: var(--muted);
    border-top: 1px solid #23302e;
  }}
  footer {{
    text-align: center; color: var(--muted); font-size: 12px;
    margin-top: 36px; font-style: italic;
  }}
  footer span {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Fluviglyph</div>
    <h1>{word}</h1>
    <div class="sub">carved by a river — {iterations} iterations of a Navier–Stokes current</div>
  </header>

  <div class="panels">
    <div class="panel">
      <div class="label">The carving · top-down</div>
      <div class="stage">
        <canvas id="anim"></canvas>
      </div>
      <div class="controls">
        <button id="play">▶ play</button>
        <div class="timeline">
          <input id="scrub" type="range" min="0" max="{lastFrame}" value="0" step="1">
        </div>
        <div class="iter-label" id="iterlabel">iter 0 / {iterations}</div>
      </div>
      <div class="hint">drag the timeline to scrub the river's work</div>
    </div>

    <div class="panel">
      <div class="label">The worn stone · 3D</div>
      <div class="stage" id="threestage">
        <canvas id="three"></canvas>
      </div>
      <div class="hint">drag to orbit · scroll to zoom — the river's final artifact</div>
    </div>
  </div>

  <footer>
    the river is the sculptor; the code merely watches · <span>fluviglyph</span>
  </footer>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/"
  }}
}}
</script>

<script>
const FRAMES = [ {frame_b64_list} ];
const ITERATIONS = {iterations};
const FRAME_ITERS = [ {frame_iter_list} ];

// ---- 2D carving animation ----------------------------------------------
const canvas = document.getElementById('anim');
const ctx = canvas.getContext('2d');
const frameImages = [];
let loaded = 0;
FRAMES.forEach((src, i) => {{
  const img = new Image();
  img.onload = () => {{ loaded++; if (loaded === FRAMES.length) drawFrame(currentFrame); }};
  img.src = "data:image/png;base64," + src;
  frameImages[i] = img;
}});

function resizeCanvas() {{
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  drawFrame(currentFrame);
}}
function drawFrame(i) {{
  const img = frameImages[i];
  if (!img || !img.complete || img.naturalWidth === 0) return;
  const cw = canvas.width, ch = canvas.height;
  const ir = img.naturalWidth / img.naturalHeight;
  const cr = cw / ch;
  let w, h, x, y;
  if (ir > cr) {{ w = cw; h = cw / ir; x = 0; y = (ch - h)/2; }}
  else {{ h = ch; w = ch * ir; y = 0; x = (cw - w)/2; }}
  ctx.clearRect(0,0,cw,ch);
  ctx.drawImage(img, x, y, w, h);
}}

let currentFrame = 0;
const scrub = document.getElementById('scrub');
const iterlabel = document.getElementById('iterlabel');
const playBtn = document.getElementById('play');
let playing = false;
let rafId = null;
let lastStep = 0;
const STEP_MS = 90;

function setFrame(i) {{
  currentFrame = Math.max(0, Math.min(FRAMES.length-1, i));
  scrub.value = currentFrame;
  drawFrame(currentFrame);
  iterlabel.textContent = "iter " + FRAME_ITERS[currentFrame] + " / " + ITERATIONS;
}}
function tick(t) {{
  if (!playing) return;
  if (t - lastStep > STEP_MS) {{
    lastStep = t;
    if (currentFrame >= FRAMES.length - 1) {{ setFrame(0); }}
    else setFrame(currentFrame + 1);
  }}
  rafId = requestAnimationFrame(tick);
}}
playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? "❚❚ pause" : "▶ play";
  if (playing) {{ lastStep = performance.now(); rafId = requestAnimationFrame(tick); }}
  else if (rafId) cancelAnimationFrame(rafId);
}});
scrub.addEventListener('input', () => setFrame(parseInt(scrub.value)));
window.addEventListener('resize', resizeCanvas);
setTimeout(resizeCanvas, 50);
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const stage = document.getElementById('threestage');
const canvas3 = document.getElementById('three');
const renderer = new THREE.WebGLRenderer({{ canvas: canvas3, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setClearColor(0x0b1416, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1416);
scene.add(new THREE.AmbientLight(0x394a47, 0.55));
// Raking key light from upper-right — casts shadows into the river's gouges
const key = new THREE.DirectionalLight(0xfff4dc, 1.8);
key.position.set(1.5, 0.4, 1.2);
scene.add(key);
// Cool rim/fill from the opposite side to define the silhouette
const rim = new THREE.DirectionalLight(0x8fb3c4, 0.7);
rim.position.set(-1.5, -0.6, 0.8);
scene.add(rim);
// Soft bounce from below so the undersides aren't pure black
const bounce = new THREE.HemisphereLight(0x8fb3c0, 0x20240a, 0.35);
scene.add(bounce);

const camera = new THREE.PerspectiveCamera(45, 2, 0.01, 100);
camera.position.set(0, 0, 130);

const controls = new OrbitControls(camera, canvas3);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.8;

const GLB_B64 = "{glb_b64}";
const glbBytes = Uint8Array.from(atob(GLB_B64), c => c.charCodeAt(0));
new GLTFLoader().parse(glbBytes.buffer, '', (gltf) => {{
  const m = gltf.scene;
  // center & scale to fit
  const box = new THREE.Box3().setFromObject(m);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const s = 60 / Math.max(size.x, size.y, size.z, 1e-6);
  m.scale.setScalar(s);
  m.position.sub(center.multiplyScalar(s));
  // The glyph lies in x-y, extruded thin in z. Face it toward the camera so
  // the worn word reads; a slight tilt reveals thickness as depth.
  m.rotation.x = -0.32;
  m.rotation.y = 0.0;
  scene.add(m);

  const mat = new THREE.MeshStandardMaterial({{
    color: 0xb8ad8e, roughness: 0.95, metalness: 0.0, flatShading: true,
  }});
  m.traverse(o => {{ if (o.isMesh) o.material = mat; }});
}}, (err) => console.error('glb parse error', err));

function resize3() {{
  const r = stage.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resize3);
resize3();

function animate() {{
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}}
animate();
</script>
</body>
</html>
"""


def build_html(
    word: str,
    iterations: int,
    frame_projections: list[np.ndarray],
    frame_u: list[np.ndarray],
    frame_v: list[np.ndarray],
    frame_iters: list[int],
    glb_bytes: bytes,
    out_path: str,
) -> None:
    """Assemble the self-contained viewer HTML.

    frame_projections: list of 2D arrays (max projection of stone over z).
    frame_u / frame_v:  2D velocity components (mean over z) per frame, for the
                        river streamlines.
    frame_iters:        the iteration index each frame corresponds to.
    glb_bytes:          the exported final mesh as GLB.
    """
    frame_b64 = [
        b64(render_frame_png(p, u, v))
        for p, u, v in zip(frame_projections, frame_u, frame_v)
    ]
    frame_iters_str = ", ".join(str(i) for i in frame_iters)
    html = HTML_TEMPLATE.format(
        word=word,
        iterations=iterations,
        lastFrame=len(frame_b64) - 1,
        frame_b64_list=", ".join('"' + s + '"' for s in frame_b64),
        frame_iter_list=frame_iters_str,
        glb_b64=b64(glb_bytes),
    )
    Path(out_path).write_text(html, encoding="utf-8")
