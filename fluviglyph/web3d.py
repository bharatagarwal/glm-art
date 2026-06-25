"""Self-contained 3D viewer: the word eroding over time.

Plain three.js (WebGL1-safe): the stone is a sequence of marching-cubes
meshes (one per checkpoint), hard-swapped per frame (no crossfade => no
z-fighting / flashing). No GLB artifact, no end transition — the word erodes
to its final state and stops.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


_ANAGLYPH_CLASS = r"""
// AnaglyphEffect (three.js r161, trimmed) with tunable eye separation.
class AnaglyphEffect{
  constructor(renderer,width=512,height=512){
    const _camera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
    const _scene=new THREE.Scene();
    const _stereo=new THREE.StereoCamera();
    this.eyeSep=0.06;
    const _params={minFilter:THREE.LinearFilter,magFilter:THREE.NearestFilter,format:THREE.RGBAFormat};
    const _rtL=new THREE.WebGLRenderTarget(width,height,_params);
    const _rtR=new THREE.WebGLRenderTarget(width,height,_params);
    this.colorMatrixLeft=new THREE.Matrix3().fromArray([
      0.456100,-0.0400822,-0.0152161,
      0.500484,-0.0378246,-0.0205971,
      0.176381,-0.0157589,-0.00546856]);
    this.colorMatrixRight=new THREE.Matrix3().fromArray([
      -0.0434706,0.378476,-0.0721527,
      -0.0879388,0.73364,-0.112961,
      -0.00155529,-0.0184503,1.2264]);
    const _material=new THREE.ShaderMaterial({
      uniforms:{mapLeft:{value:_rtL.texture},mapRight:{value:_rtR.texture},
        colorMatrixLeft:{value:this.colorMatrixLeft},colorMatrixRight:{value:this.colorMatrixRight}},
      vertexShader:'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
      fragmentShader:[
        'uniform sampler2D mapLeft;uniform sampler2D mapRight;',
        'uniform mat3 colorMatrixLeft;uniform mat3 colorMatrixRight;',
        'varying vec2 vUv;',
        'void main(){',
        'vec4 cL=texture2D(mapLeft,vUv);vec4 cR=texture2D(mapRight,vUv);',
        'vec3 c=clamp(colorMatrixLeft*cL.rgb+colorMatrixRight*cR.rgb,0.,1.);',
        'gl_FragColor=vec4(c.r,c.g,c.b,max(cL.a,cR.a));',
        '#include <tonemapping_fragment>','#include <colorspace_fragment>','}'].join('\n')
    });
    const _mesh=new THREE.Mesh(new THREE.PlaneGeometry(2,2),_material);_scene.add(_mesh);
    this.setSize=function(w,h){renderer.setSize(w,h);const pr=renderer.getPixelRatio();_rtL.setSize(w*pr,h*pr);_rtR.setSize(w*pr,h*pr);};
    this.render=function(scene,camera){
      const cur=renderer.getRenderTarget();
      if(scene.matrixWorldAutoUpdate)scene.updateMatrixWorld();
      if(camera.parent===null&&camera.matrixWorldAutoUpdate)camera.updateMatrixWorld();
      _stereo.eyeSep=this.eyeSep;_stereo.update(camera);
      renderer.setRenderTarget(_rtL);renderer.clear();renderer.render(scene,_stereo.cameraL);
      renderer.setRenderTarget(_rtR);renderer.clear();renderer.render(scene,_stereo.cameraR);
      renderer.setRenderTarget(null);renderer.render(_scene,_camera);
      renderer.setRenderTarget(cur);
    };
    this.dispose=function(){_rtL.dispose();_rtR.dispose();_mesh.geometry.dispose();_mesh.material.dispose();};
  }
}
"""

_SBS_CLASS = r"""
// SbsEffect: off-axis StereoCamera rendered into two side-by-side viewports.
// stereo.aspect=0.5 => per-eye aspect = windowAspect*0.5, which is 1.0 (square)
// when the window is 2:1. Keystone-free asymmetric frustums, converged at camera.focus.
class SbsEffect{
  constructor(renderer){ this._stereo=new THREE.StereoCamera(); this.eyeSep=0.06; this.aspect=0.5; this._r=renderer; }
  setSize(w,h){ this._r.setSize(w,h); }
  render(scene,camera){
    const r=this._r;
    if(scene.matrixWorldAutoUpdate)scene.updateMatrixWorld();
    if(camera.parent===null&&camera.matrixWorldAutoUpdate)camera.updateMatrixWorld();
    this._stereo.aspect=this.aspect; this._stereo.eyeSep=this.eyeSep; this._stereo.update(camera);
    const pr=r.getPixelRatio(), W=r.domElement.width, H=r.domElement.height, hw=W/2;
    r.setScissorTest(true);
    r.setScissor(0,0,hw,H); r.setViewport(0,0,hw,H); r.setRenderTarget(null); r.clear();
    r.render(scene,this._stereo.cameraL);
    r.setScissor(hw,0,hw,H); r.setViewport(hw,0,hw,H);
    r.render(scene,this._stereo.cameraR);
    r.setScissorTest(false); r.setViewport(0,0,W,H);
  }
}
"""


def _b64(arr) -> str:
    data = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
    return base64.b64encode(data).decode("ascii")


def build_html(
    word: str,
    iterations: int,
    mesh_frames: list[dict],      # each: {verts: float32, indices: uint32}
    frame_iters: list[int],
    out_path: str,
    mode: str = "anaglyph",      # "anaglyph" | "sbs" | "flat"
    sbs_cam_scale: float = 1.0,    # pullback factor for square-eye framing in SBS
    eye_sep: float = 0.06,         # stereo eye separation (bigger = bolder depth)
) -> None:
    payload = {
        "word": word,
        "iterations": iterations,
        "frameIters": [int(i) for i in frame_iters],
        "meshes": [{"v": _b64(m["verts"]), "i": _b64(m["indices"])} for m in mesh_frames],
    }
    html = (HTML_TEMPLATE
            .replace("__PAYLOAD__", json.dumps(payload))
            .replace("__WORD__", word)
            .replace("__ITERS__", str(iterations)))
    html = html.replace("__ANAGLYPH_CLASS__", _ANAGLYPH_CLASS)
    html = html.replace("__SBS_CLASS__", _SBS_CLASS if mode == "sbs" else "")
    html = html.replace("__MODE__", mode)
    html = html.replace("__SBS_CAM_SCALE__", str(sbs_cam_scale))
    html = html.replace("__EYE_SEP__", str(eye_sep))
    Path(out_path).write_text(html, encoding="utf-8")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__WORD__ · carved by a river</title>
<style>
  :root{ --bg:#101418; --ink:#e8e2d2; --muted:#7d8a85; --accent:#c9bea0; }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--ink);
    font-family:ui-sans-serif,-apple-system,"Helvetica Neue",system-ui,sans-serif;
    -webkit-font-smoothing:antialiased;overflow:hidden;}
  #app{position:fixed;inset:0;}
  canvas{display:block;}
  .overlay{position:absolute;inset:0;pointer-events:none;}
  header{position:absolute;top:0;left:0;right:0;text-align:center;padding:26px 16px 0;}
  header .eyebrow{text-transform:uppercase;letter-spacing:.32em;font-size:10px;color:var(--muted);margin-bottom:8px;}
  header h1{font-family:Georgia,"Times New Roman",serif;font-weight:400;margin:0;
    font-size:clamp(24px,4.4vw,40px);letter-spacing:.05em;color:var(--ink);}
  header .sub{margin-top:8px;color:var(--muted);font-size:13px;font-style:italic;}
  .controls{position:absolute;bottom:0;left:0;right:0;padding:18px 22px 22px;
    background:linear-gradient(to top,rgba(16,20,24,.85),transparent);pointer-events:none;}
  .controls .row{max-width:760px;margin:0 auto;display:flex;align-items:center;gap:16px;pointer-events:auto;}
  button{appearance:none;background:#1d2b2e;color:var(--ink);border:1px solid #2e3d3b;
    border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer;transition:background .15s;}
  button:hover{background:#28393a;}
  .timeline{flex:1;position:relative;}
  .timeline input[type=range]{width:100%;-webkit-appearance:none;appearance:none;background:transparent;cursor:pointer;height:20px;}
  .timeline input[type=range]::-webkit-slider-runnable-track{height:3px;background:#2e3d3b;border-radius:3px;}
  .timeline input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;
    background:var(--accent);margin-top:-6px;box-shadow:0 0 0 3px rgba(201,190,160,.18);}
  .timeline input[type=range]::-moz-range-track{height:3px;background:#2e3d3b;border-radius:3px;}
  .timeline input[type=range]::-moz-range-thumb{width:15px;height:15px;border:none;border-radius:50%;background:var(--accent);}
  .iterlabel{font-variant-numeric:tabular-nums;font-size:12px;color:var(--muted);min-width:110px;text-align:right;}
  footer{position:absolute;bottom:64px;left:0;right:0;text-align:center;color:var(--muted);font-size:11px;font-style:italic;opacity:.7;}
  #loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:13px;letter-spacing:.2em;text-transform:uppercase;}
</style>
</head>
<body>
<div id="app"></div>
<div class="overlay">
  <header>
    <div class="eyebrow">Fluviglyph</div>
    <h1>__WORD__</h1>
    <div class="sub">carved by a river — <span id="iterText">0</span> / __ITERS__ iterations of a Navier–Stokes current</div>
  </header>
  <div class="controls">
    <div class="row">
      <button id="play">▶ play</button>
      <div class="timeline"><input id="scrub" type="range" min="0" max="1000" value="0" step="1"></div>
      <div class="iterlabel" id="iterlabel">iter 0 / __ITERS__</div>
    </div>
  </div>
  <footer>the river is the sculptor; the code merely watches · <span style="color:var(--accent)">fluviglyph</span></footer>
</div>
<div id="loading">gathering the river…</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = __PAYLOAD__;
const WORD = DATA.word, ITERS = DATA.iterations, NFRAMES = DATA.meshes.length;
const MODE = '__MODE__';
document.querySelector('h1').textContent = WORD;
__ANAGLYPH_CLASS__
__SBS_CLASS__

// ---------- scene ----------
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias:true, preserveDrawingBuffer:true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101418, 1);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(-0.95*__SBS_CAM_SCALE__, -0.35*__SBS_CAM_SCALE__, 1.55*__SBS_CAM_SCALE__);
camera.lookAt(0,0,0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.target.set(0,0,0); controls.minDistance = 1.5; controls.maxDistance = 10;

// Stereo effect chosen at build time. anaglyph => red/cyan full-frame; sbs => two square
// eyes side-by-side (off-axis, keystone-free). Both reuse StereoCamera geometry.
let effect = null, renderFn = (s,c)=>renderer.render(s,c);
if(MODE==='anaglyph'){
  effect = new AnaglyphEffect(renderer, window.innerWidth, window.innerHeight);
  effect.eyeSep = __EYE_SEP__;
  renderFn = (s,c)=>effect.render(s,c);
} else if(MODE==='sbs'){
  effect = new SbsEffect(renderer);
  effect.eyeSep = __EYE_SEP__;
  camera.focus = camera.position.length();   // converge exactly on the word (target=origin)
  camera.updateProjectionMatrix();
  renderFn = (s,c)=>effect.render(s,c);
}

scene.add(new THREE.AmbientLight(0x2a3633, 0.5));
const key = new THREE.DirectionalLight(0xfff4dc, 1.5); key.position.set(1.6,0.6,1.3); scene.add(key);
const rim = new THREE.DirectionalLight(0x8fb3c4, 0.6); rim.position.set(-1.5,-0.5,0.9); scene.add(rim);
const hemi = new THREE.HemisphereLight(0x8fb3c0, 0x141405, 0.3); scene.add(hemi);

const SIZE = new THREE.Vector3(2.0, 1.0, 0.75);
function b64ToTyped(b, Ctor){ return new Ctor(Uint8Array.from(atob(b), c=>c.charCodeAt(0)).buffer); }

// ---------- stone meshes (one per checkpoint) ----------
const stoneMeshes = [];
const stoneMat = new THREE.MeshStandardMaterial({
  color:0xb8ad8e, roughness:0.92, metalness:0.0, flatShading:true,
  transparent:true, opacity:0.0, side: THREE.DoubleSide,
});
for(let i=0;i<NFRAMES;i++){
  const v = b64ToTyped(DATA.meshes[i].v, Float32Array);
  const idx = b64ToTyped(DATA.meshes[i].i, Uint32Array);
  const g = new THREE.BufferGeometry();
  const verts = new Float32Array(v.length);
  for(let j=0;j<v.length;j+=3){
    verts[j+0] = (v[j+0] - 0.5) * SIZE.x;
    verts[j+1] = (0.5 - v[j+1]) * SIZE.y;   // flip y: voxel top -> world up
    verts[j+2] = (v[j+2] - 0.5) * SIZE.z;
  }
  g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, stoneMat.clone());
  m.material.transparent = true;
  m.material.opacity = 0.0;
  m.visible = false;
  m.frustumCulled = false;
  scene.add(m);
  stoneMeshes.push(m);
}

// ---------- timeline / playback ----------
const scrub=document.getElementById('scrub'), playBtn=document.getElementById('play');
const iterlabel=document.getElementById('iterlabel'), iterText=document.getElementById('iterText');
let playing=false, last=0, frac=0; const DURATION=16.0; let finished=false;

function setFrac(f){
  frac=Math.max(0,Math.min(1,f));
  const fi = frac*(NFRAMES-1);
  const a = Math.max(0, Math.min(NFRAMES-1, Math.round(fi)));
  // hard-swap to the nearest frame — no crossfade (avoids z-fighting/flashing).
  for(let i=0;i<NFRAMES;i++){ stoneMeshes[i].visible=false; stoneMeshes[i].material.opacity=0; }
  stoneMeshes[a].visible=true; stoneMeshes[a].material.opacity = 1.0;
  // no artifact crossfade — the last frame IS the final worn form.
  const itDisp = Math.round(frac*ITERS);
  iterlabel.textContent = 'iter '+itDisp+' / '+ITERS;
  if(iterText) iterText.textContent = itDisp;
  scrub.value = Math.round(frac*1000);
  finished = frac >= 0.999;
}
setFrac(0);

playBtn.addEventListener('click', ()=>{
  if(finished){ setFrac(0); finished=false; }
  playing=!playing; playBtn.textContent = playing?'❚❚ pause':'▶ play';
  if(playing) last=performance.now();
});
scrub.addEventListener('input', ()=>{ playing=false; playBtn.textContent='▶ play'; setFrac(parseInt(scrub.value)/1000); });

// ---------- render loop ----------
const clock = new THREE.Clock();
function animate(){
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if(playing){
    frac += dt / DURATION;
    if(frac >= 1){ frac=1; playing=false; playBtn.textContent='▶ replay'; }
    setFrac(frac);
  }
  controls.update();
  renderFn(scene, camera);
}

document.getElementById('loading').style.display='none';
function renderNow(){ controls.update(); renderFn(scene, camera); }
window.__fluviglyph = { scene, camera, stoneMeshes, renderer, effect, renderFn, renderNow, setFrac, get frac(){return frac;} };
animate();

window.addEventListener('resize', ()=>{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if(typeof effect!=='undefined') effect.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""
