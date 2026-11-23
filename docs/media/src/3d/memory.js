// Scene 4 (memory): the 32 GiB budget as a wall of 512 tiles, one column per GiB, 64 MiB per tile.
//
// Totals are measured (docs/benchmarks.md): default server (4 slots x 32k, DSpark, vision) 26.4 GiB after start,
// 28.8 GiB peak; long context (--kv-fp8 --max-ctx 262144 --slots 1 --no-vision) 25.3 GiB after start, 28.6 peak.
// Segments inside those totals are computed:
//   weights      = checkpoint files on the Hub: model-00001-of-00001.safetensors 19,526,019,168 B = 18.185 GiB,
//                  DSpark drafter mtp.safetensors 635,801,416 B = 0.592 GiB -> 18.78 GiB (1 GiB = 2^30 B);
//                  --no-vision drops the 0.6 GB vision tower (docs/serving.md; 0.6e9 B = 0.56 GiB) -> 18.22 GiB
//   KV per slot  = CacheSpec.bytes (inkling_engine/cache.py): full-attention KV 2 * 1024 * 2 B * 8 visits * 32768
//                  = 1,073,741,824 B = 1.000 GiB, plus sliding rings 4 * 1024 * 40 visits * 1088 = 178,257,920 B
//                  = 0.166 GiB -> 1.166 GiB per slot, x 4 slots;
//                  256k FP8, one slot: 2 * 1024 * 1 B * 8 * 262144 = 4,294,967,296 B = 4.000 GiB + 0.166 GiB ring = 4.166 GiB
//   graphs + workspace = after-start total minus the above; peak = the measured peak.
import * as THREE from 'three';
import { setup, rng, fitCamera } from './common3d.js';

const q = new URLSearchParams(location.search);
const GIF = q.get('gif') === '1';
const DURATION = GIF ? 8.0 : 3.0;
const { W, H, ramp } = window.S;
const wide = W / H > 1.3;

const COLS = 32, ROWS = 16, TPG = 16;  // tiles per GiB
const A = { weights: 18.185 + 0.592, kv: 4 * (1.0 + 0.166), after: 26.4, peak: 28.8, slots: 4 };
const B = { weights: 18.185 + 0.592 - 0.56, kv: 4.0 + 0.166, after: 25.3, peak: 28.6, slots: 1 };
// kinds: 0 weights, 1 kv, 2 graphs/workspace, 3 peak headroom, 4 free
function layout(P) {
  const cut = g => Math.round(g * TPG);
  const w = cut(P.weights), k = cut(P.weights + P.kv), a = cut(P.after), p = cut(P.peak);
  const kinds = [], slot = [];
  for (let i = 0; i < COLS * ROWS; i++) {
    kinds.push(i < w ? 0 : i < k ? 1 : i < a ? 2 : i < p ? 3 : 4);
    slot.push(i >= w && i < k ? Math.min(P.slots - 1, Math.floor((i - w) / ((k - w) / P.slots))) : -1);
  }
  return { kinds, slot, b: [0, w, k, a, p, COLS * ROWS] };
}
const LA = layout(A), LB = layout(B);

// ---- timeline ---------------------------------------------------------------------------------------------
// fill(t): how far the A layout has filled in (0..1, tiles beyond the weights); merge(t): A -> B (0..1).
const TL = GIF
  ? { fill0: 0.2, fill1: 1.5, merge0: 3.6, merge1: 4.6, back0: 6.8, back1: 7.3, unfill0: 7.25, unfill1: 7.9, h1: [0.0, 3.3], h1again: 7.45, h2: [3.75, 6.75] }
  : { fill0: 0.35, fill1: 1.15, merge0: 1.45, merge1: 2.2, back0: 99, back1: 99, h1: [0.4, 1.4], h2: [1.6, null] };
// GIF loop: un-merge, then empty the KV again, so the last frame matches frame 0
const fillAt = t => GIF && t >= TL.unfill0 ? 1 - ramp(t, TL.unfill0, TL.unfill1 - TL.unfill0) : ramp(t, TL.fill0, TL.fill1 - TL.fill0);
const mergeAt = t => ramp(t, TL.merge0, TL.merge1 - TL.merge0) * (1 - ramp(t, TL.back0, TL.back1 - TL.back0));

// ---- scene ------------------------------------------------------------------------------------------------
const { scene, camera, render } = setup({ fov: wide ? 26 : 30, bloom: { strength: 0.7, radius: 0.45, threshold: 0.6 } });
scene.fog = null;
const N = COLS * ROWS, S = 0.5, GAP = 0.06;
const tileGeo = new THREE.BoxGeometry(S - GAP, S - GAP, 0.1);
const aCol = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
const aHollow = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
const aSeed = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
const aTex = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
{ const R = rng(32); for (let i = 0; i < N; i++) aSeed.array[i] = R() * 100 }
tileGeo.setAttribute('aCol', aCol); tileGeo.setAttribute('aHollow', aHollow); tileGeo.setAttribute('aSeed', aSeed); tileGeo.setAttribute('aTex', aTex);
const mat = new THREE.ShaderMaterial({
  vertexShader: `
    attribute vec3 aCol; attribute float aHollow, aSeed, aTex;
    varying vec3 vCol; varying float vHollow, vSeed, vTex, vFront; varying vec2 vUv;
    void main(){
      vCol = aCol; vHollow = aHollow; vSeed = aSeed; vTex = aTex; vUv = uv; vFront = step(0.5, normal.z);
      gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: `
    varying vec3 vCol; varying float vHollow, vSeed, vTex, vFront; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    void main(){
      float rowN = hash(vec2(floor(vUv.y * 9.0), vSeed));
      float run = hash(vec2(floor(vUv.x * (4.0 + 9.0 * rowN)), floor(vUv.y * 9.0) + vSeed * 1.7));
      float tex = mix(1.0, 0.45 + 0.55 * run * (0.6 + 0.4 * rowN), vTex);
      vec3 c = vCol * tex;
      float e = min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y));
      float rim = 1.0 - smoothstep(0.035, 0.075, e);
      c = mix(c, vCol * 1.25, rim * 0.35);
      c = mix(c, vCol * rim * 1.1, vHollow);
      c *= mix(0.55, 1.0, vFront);
      gl_FragColor = vec4(pow(max(c, 0.0), vec3(2.2)), 1.0);
    }`,
});
const tiles = new THREE.InstancedMesh(tileGeo, mat, N);
scene.add(tiles);
const tilePos = i => new THREE.Vector3((Math.floor(i / ROWS) - (COLS - 1) / 2) * S, (i % ROWS - (ROWS - 1) / 2) * S, 0);

// mirrored floor
const FLOOR_Y = -ROWS * S / 2 - 0.35;
const mir = new THREE.InstancedMesh(tileGeo, mat, N);
mir.scale.y = -1; mir.position.y = 2 * FLOOR_Y; scene.add(mir);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshBasicMaterial({ color: 0x05080f, transparent: true, opacity: 0.72 }));
floor.rotation.x = -Math.PI / 2; floor.position.y = FLOOR_Y; scene.add(floor);

// colours (sRGB, the shader converts): weights = idle-expert grey, KV = blue, graphs = slate, peak = hollow, free = socket
const C = {
  weights: [0.30, 0.33, 0.38], kv: [[0.25, 0.53, 0.98], [0.13, 0.33, 0.74], [0.25, 0.53, 0.98], [0.13, 0.33, 0.74]],
  kvB: [0.26, 0.55, 1.0], graphs: [0.15, 0.19, 0.26], peak: [0.42, 0.48, 0.58], free: [0.085, 0.10, 0.135],
};
const mix3 = (a, b, x) => a.map((v, i) => v + (b[i] - v) * x);

// ---- camera -----------------------------------------------------------------------------------------------
const corners = [[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([u, v]) => new THREE.Vector3(u * COLS * S / 2, v * ROWS * S / 2 + (v > 0 ? 1.0 : 0), 0));
const BOX = GIF ? { x0: 0.0, x1: 0.92, y0: -0.40, y1: 0.70 } : wide ? { x0: -0.12, x1: 0.90, y0: -0.60, y1: 0.70 } : { x0: -0.88, x1: 0.88, y0: -0.34, y1: 0.70 };
const K0 = fitCamera(camera, new THREE.Vector3(-0.62, 0.22, 0.75), corners, BOX);
const K1 = fitCamera(camera, new THREE.Vector3(-0.50, 0.17, 0.85), corners, { x0: BOX.x0 - 0.03, x1: BOX.x1 + 0.03, y0: BOX.y0 - 0.02, y1: BOX.y1 + 0.02 });

// ---- HTML tags over segments --------------------------------------------------------------------------------
const tagsEl = document.getElementById('tags');
function tag(cls, label, value) {
  const d = document.createElement('div');
  d.className = 'tag ' + cls;
  d.innerHTML = `<div class="label">${label}</div>${value ? `<div class="v">${value}</div>` : ''}<div class="stem"></div>`;
  tagsEl.appendChild(d); return d;
}
const fmt = g => g.toFixed(1) + ' GiB';
const TAGS = [
  { el: tag('', 'Weights + drafter', fmt(A.weights)), seg: 0, lift: 0, show: t => 1 },
  { el: tag('kv', '4 × 32k KV', '4 × ' + (1.166).toFixed(2) + ' GiB'), seg: 1, lift: 2, show: t => 1 - ramp(t, TL.merge0 - 0.05, 0.3) + (GIF ? ramp(t, TL.back0 + 0.3, 0.4) : 0) },
  { el: tag('kv', '256k FP8 KV', fmt(B.kv)), seg: 1, lift: 2, layout: 'B', show: t => ramp(t, TL.merge1 - 0.25, 0.4) * (1 - (GIF ? ramp(t, TL.back0, 0.3) : 0)) },
  { el: tag('', 'Graphs', ''), seg: 2, lift: 0, show: t => 1 },
  { el: tag('', 'Peak', ''), seg: 3, lift: 1, show: t => 1 },
  { el: tag('', 'Free', ''), seg: 4, lift: 0, show: t => 1 },
];
const AXIS = [0, 8, 16, 24, 32].map(g => {
  const d = document.createElement('div');
  d.className = 'axis label'; d.textContent = g === 32 ? '32 GiB' : String(g);
  tagsEl.appendChild(d); return { g, el: d };
});

// ---- per-frame ----------------------------------------------------------------------------------------------
const m4 = new THREE.Matrix4(), v3 = new THREE.Vector3();
const ids = ['h1', 'h2', 's1', 's2'].map(id => document.getElementById(id));

function renderFrame(t) {
  // the GIF holds the camera still: a moving camera changes every pixel and defeats GIF frame diffing
  const u = GIF ? 0.6 : 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, t / DURATION));
  camera.position.copy(K0.pos).lerp(K1.pos, u);
  camera.lookAt(K0.target.clone().lerp(K1.target, u));
  camera.updateMatrixWorld();

  const f = fillAt(t), mg = mergeAt(t);
  const fillEnd = LA.b[1] + f * (N - LA.b[1]);              // tiles beyond the weights appear in order
  for (let i = 0; i < N; i++) {
    // merge sweeps left to right over the tiles that change
    const local = Math.min(1, Math.max(0, mg * 1.6 - 0.6 * ((i - LB.b[1]) / (LA.b[3] - LB.b[1] + 1))));
    const ka = LA.kinds[i], kb = LB.kinds[i];
    const colOf = (k, s, isB) => k === 0 ? C.weights : k === 1 ? (isB ? C.kvB : C.kv[s]) : k === 2 ? C.graphs : k === 3 ? C.peak : C.free;
    let col = mix3(colOf(ka, LA.slot[i], false), colOf(kb, LB.slot[i], true), local);
    let hollow = (ka === 3 ? 1 : 0) * (1 - local) + (kb === 3 ? 1 : 0) * local;
    let tex = (ka <= 1 ? 1 : 0) * (1 - local) + (kb <= 1 ? 1 : 0) * local;
    // appearance in fill order: unfilled tiles look free
    const appear = i < LA.b[1] ? 1 : Math.min(1, Math.max(0, (fillEnd - i) / 10));
    if (appear < 1) { col = mix3(C.free, col, appear); hollow *= appear; tex *= appear }
    // KV glows when it lands and while it merges
    const isKV = (ka === 1 ? 1 - local : 0) + (kb === 1 ? local : 0);
    const flash = isKV * (Math.max(0, 1 - Math.abs(fillEnd - i) / 14) * 0.9 + Math.sin(Math.PI * local) * 0.9);
    col = col.map(c => c * (1 + flash));
    const z = isKV * (0.06 + 0.12 * Math.sin(Math.PI * local)) + (appear < 1 ? (1 - appear) * -0.1 : 0);
    v3.copy(tilePos(i)); v3.z += z;
    m4.makeTranslation(v3.x, v3.y, v3.z);
    tiles.setMatrixAt(i, m4); mir.setMatrixAt(i, m4);
    aCol.setXYZ(i, col[0], col[1], col[2]); aHollow.setX(i, hollow); aTex.setX(i, tex);
  }
  tiles.instanceMatrix.needsUpdate = mir.instanceMatrix.needsUpdate = true;
  aCol.needsUpdate = aHollow.needsUpdate = aTex.needsUpdate = true;

  render();

  // tags: centred over their segment's columns, above the wall
  for (const tg of TAGS) {
    const L = tg.layout === 'B' || mg > 0.5 ? LB : LA;
    const c0 = L.b[tg.seg] / TPG, c1 = L.b[tg.seg + 1] / TPG;
    const p = new THREE.Vector3(((c0 + c1) / 2 - COLS / 2) * S, ROWS * S / 2 + 0.12, 0).project(camera);
    const x = (p.x + 1) / 2 * W, y = (1 - p.y) / 2 * H;
    const fillShown = tg.seg === 0 ? 1 : ramp(fillEnd, L.b[tg.seg] + 4, 10);
    tg.el.style.left = x.toFixed(1) + 'px';
    tg.el.style.top = (y - 4).toFixed(1) + 'px';
    tg.el.querySelector('.stem').style.height = [18, 46, 84][tg.lift] + 'px';   // three heights so neighbouring tags never touch
    tg.el.style.opacity = Math.max(0, Math.min(1, tg.show(t) * fillShown)).toFixed(3);
  }
  TAGS[0].el.querySelector('.v').textContent = fmt(mg > 0.5 ? B.weights : A.weights);
  for (const ax of AXIS) {
    const p = new THREE.Vector3((ax.g - COLS / 2) * S, -ROWS * S / 2 - 0.06, 0.2).project(camera);
    ax.el.style.left = ((p.x + 1) / 2 * W).toFixed(1) + 'px';
    ax.el.style.top = ((1 - p.y) / 2 * H).toFixed(1) + 'px';
  }

  // text
  const [h1, h2, s1, s2] = ids;
  const inOut = (el, a, b) => {
    const o = (a <= 0 ? 1 : ramp(t, a, 0.4)) * (b == null ? 1 : 1 - ramp(t, b, 0.25));
    el.style.opacity = o.toFixed(4);
    el.style.transform = `translateY(${a <= 0 ? 0 : ((1 - ramp(t, a, 0.4)) * 12).toFixed(2)}px)`;
  };
  if (GIF && t >= TL.h1again) { inOut(h1, TL.h1again, null); inOut(s1, TL.h1again, null) }
  else { inOut(h1, TL.h1[0], TL.h1[1]); inOut(s1, TL.h1[0] + (GIF ? 0 : 0.1), TL.h1[1]) }
  inOut(h2, TL.h2[0], TL.h2[1]); inOut(s2, TL.h2[0] + 0.1, TL.h2[1]);
}

window.renderFrame = renderFrame;
window.SCENE = { duration: DURATION };
