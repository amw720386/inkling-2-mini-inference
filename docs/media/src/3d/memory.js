// Scene 4 (memory): the 32 GiB budget as one slab of blocks, width proportional to GiB (1 grid column = 1 GiB).
// Beat 1: the default server profile, four 32k KV slots as four separate blocks. Beat 2: the slots slide together
// and merge into the single 256k FP8 slot of the long-context profile.
//
// Totals are measured (docs/benchmarks.md, tests/gpu/memcheck.py, server defaults: FP8 dense weights, FP4 prefill,
// DSpark, vision): 4 slots x 32k: 25.9 GiB after start, 27.4 GiB peak; --kv-fp8 --max-ctx 262144 --slots 1:
// 24.9 GiB after start, 27.1 GiB peak (with a 30k prefill).
// Segments inside those totals are computed (1 GiB = 2^30 B):
//   weights      = checkpoint files on the Hub: model-00001-of-00001.safetensors 19,526,019,168 B = 18.185 GiB +
//                  DSpark drafter mtp.safetensors 635,801,416 B = 0.592 GiB = 18.777 GiB as bf16, then the FP8
//                  default (inkling_engine/fused_model.py, compact()): the padded attention projections
//                  (20 x 8640 + 4 x 8768 = 207,872 rows x 2304) and wo (24 x 2304 x 3072) become row-scaled E4M3
//                  (1 B per weight + a 4 B scale per row) and the bf16 copies are dropped: -0.445 and -0.158 GiB;
//                  the LM head keeps its bf16 copy and adds an FP8 one (201,024 x 2304 + scales): +0.432 GiB
//                  -> 18.61 GiB in both profiles (both keep vision)
//   KV per slot  = CacheSpec.bytes (inkling_engine/cache.py): full-attention KV 2 * 1024 * 2 B * 8 visits * 32768
//                  = 1,073,741,824 B = 1.000 GiB, plus sliding rings 4 * 1024 * 40 visits * 1088 = 178,257,920 B
//                  = 0.166 GiB -> 1.166 GiB per slot, x 4 slots;
//                  256k FP8, one slot: 2 * 1024 * 1 B * 8 * 262144 = 4,294,967,296 B = 4.000 GiB + 0.166 GiB ring = 4.166 GiB
//   graphs + workspace = after-start total minus the above; peak = measured peak minus after-start; free = 32 - peak.
import * as THREE from 'three';
import { setup, fitCamera } from './common3d.js';

const q = new URLSearchParams(location.search);
const GIF = q.get('gif') === '1';
const DURATION = GIF ? 8.0 : 3.0;
const { W, H, ramp } = window.S;
const wide = W / H > 1.3;

// ---- the two layouts (GiB) ----------------------------------------------------------------------------------
const SLOT = 1.0 + 0.166, FP8SLOT = 4.0 + 0.166;
const WEIGHTS = 18.185 + 0.592 - 0.445 - 0.158 + 0.432;   // FP8 default, see the header
const A = { weights: WEIGHTS, slots: [SLOT, SLOT, SLOT, SLOT], after: 25.9, peak: 27.4 };
const B = { weights: WEIGHTS, slots: [FP8SLOT / 4, FP8SLOT / 4, FP8SLOT / 4, FP8SLOT / 4], after: 24.9, peak: 27.1 };
for (const P of [A, B]) {
  P.kv = P.slots.reduce((s, x) => s + x, 0);
  P.graphs = P.after - P.weights - P.kv;
  P.peakH = P.peak - P.after;
  P.free = 32 - P.peak;
}
// block order along the slab: weights, slot 1-4, graphs, peak, free
const KIND = ['weights', 'kv', 'kv', 'kv', 'kv', 'graphs', 'peak', 'free'];
const sizes = P => [P.weights, ...P.slots, P.graphs, P.peakH, P.free];
const GAP = 0.45, SLOT_GAP = 0.28;      // between segments; between KV slots in the 4-slot layout

// ---- timeline -----------------------------------------------------------------------------------------------
const TL = GIF
  ? { merge0: 3.6, merge1: 4.7, back0: 6.8, back1: 7.3, unfill0: 7.25, h1: [0.0, 3.3], h1again: 7.45, h2: [3.75, 6.75] }
  : { merge0: 1.45, merge1: 2.25, back0: 99, back1: 99, unfill0: 99, h1: [0.4, 1.4], h2: [1.6, null] };
// appearance of block i (0 = empty socket, 1 = filled): slots drop in one by one, then graphs and peak
const APPEAR_AT = GIF ? [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.45, 0] : [0, 0.35, 0.5, 0.65, 0.8, 0.95, 1.05, 0];
function appearAt(i, t) {
  if (i === 0 || i === 7) return 1;
  const a = ramp(t, APPEAR_AT[i], GIF ? 0.35 : 0.25);
  return GIF && t >= TL.unfill0 ? a * (1 - ramp(t, TL.unfill0 + (6 - i) * 0.04, 0.4)) : a;
}
const mergeAt = t => ramp(t, TL.merge0, TL.merge1 - TL.merge0) * (1 - ramp(t, TL.back0, TL.back1 - TL.back0));

// ---- scene --------------------------------------------------------------------------------------------------
const { scene, camera, render } = setup({ fov: wide ? 24 : 28, bloom: { strength: 0.55, radius: 0.4, threshold: 0.62 } });
scene.fog = null;
const ROWS = 8, CELL_H = wide ? 1.15 : 2.6, SLAB_H = ROWS * CELL_H, DEPTH = 0.9;

const blockVS = `
  varying vec2 vUv; varying float vFront;
  void main(){ vUv = uv; vFront = step(0.5, normal.z); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const blockFS = `
  uniform vec3 uCol; uniform float uOff, uW, uTex, uHollow, uAppear, uRim, uGlow, uSeed;
  varying vec2 vUv; varying float vFront;
  float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  void main(){
    float gx = uOff + vUv.x * uW;                     // position in GiB along the slab
    vec2 c = vec2(gx, vUv.y * ${ROWS}.0);
    vec2 id = floor(c + 1e-4), f = fract(c);
    float g = 0.06;
    float inCell = step(g, f.x) * step(f.x, 1.0 - g) * step(0.04, f.y) * step(f.y, 0.96);
    float rowN = hash(vec2(floor(f.y * 10.0), id.x * 13.0 + id.y * 7.0 + uSeed));
    float run = hash(vec2(floor(f.x * (5.0 + 12.0 * rowN)), floor(f.y * 10.0) + id.y * 3.0 + id.x + uSeed));
    float tex = mix(1.0, 0.45 + 0.55 * run * (0.6 + 0.4 * rowN), uTex);
    vec3 empty = vec3(0.085, 0.10, 0.135);
    vec3 full = uCol * tex;
    // hollow cells (prefill peak headroom): only the cell outline
    float cellEdge = 1.0 - step(0.12, min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y) * 1.6));
    full = mix(full, uCol * cellEdge, uHollow);
    vec3 col = mix(empty, full, uAppear) * inCell;
    // block outline (separates the KV slots)
    float ex = min(vUv.x, 1.0 - vUv.x) * uW, ey = min(vUv.y, 1.0 - vUv.y) * ${SLAB_H.toFixed(3)};
    float rim = 1.0 - smoothstep(0.03, 0.07, min(ex, ey));
    col = mix(col, vec3(0.55, 0.75, 1.0), rim * uRim);
    col *= (1.0 + uGlow) * mix(0.5, 1.0, vFront);
    gl_FragColor = vec4(pow(max(col, 0.0), vec3(2.2)), 1.0);
  }`;
const C = { weights: [0.30, 0.33, 0.38], kv: [0.25, 0.53, 0.98], graphs: [0.17, 0.21, 0.28], peak: [0.45, 0.52, 0.62], free: [0.085, 0.10, 0.135] };
const boxGeo = new THREE.BoxGeometry(1, SLAB_H, DEPTH);
const blocks = KIND.map((k, i) => {
  const u = {
    uCol: { value: new THREE.Vector3(...C[k]) }, uOff: { value: 0 }, uW: { value: 1 }, uTex: { value: k === 'weights' || k === 'kv' ? 1 : 0 },
    uHollow: { value: k === 'peak' ? 1 : 0 }, uAppear: { value: 1 }, uRim: { value: 0 }, uGlow: { value: 0 }, uSeed: { value: i * 17 },
  };
  const m = new THREE.Mesh(boxGeo, new THREE.ShaderMaterial({ uniforms: u, vertexShader: blockVS, fragmentShader: blockFS }));
  scene.add(m);
  return { m, u, kind: k };
});

// geometry of a layout blend: x centre and width of every block, plus GiB offsets for the grid
function placement(mg) {
  const sa = sizes(A), sb = sizes(B), out = [];
  let x = 0, off = 0;
  for (let i = 0; i < 8; i++) {
    const w = sa[i] + (sb[i] - sa[i]) * mg;
    out.push({ x: x + w / 2, w, off });
    x += w; off += w;
    if (i < 7) x += (i >= 1 && i <= 3) ? SLOT_GAP * (1 - mg) : GAP;
  }
  const total = x;
  out.forEach(p => { p.x -= total / 2 });
  return { blocks: out, total };
}

// ---- camera: slab spans a fixed share of the width, its bottom edge sits on a fixed line ----------------------
const slabFrac = wide ? 0.70 : 0.86;
const textTopPx = H - 64 - 14 - 30 - 53 - 20 - 18;            // top of the caption block (label + headline + sub)
const labelsPx = 108;                                          // leader lines + two label lines
const bottomNdc = 1 - 2 * (textTopPx - labelsPx) / H;
const P0 = placement(0);
const half = P0.total / 2 + 0.2;
const corners = [];
for (const x of [-half, half]) for (const y of [-SLAB_H / 2, SLAB_H / 2]) for (const z of [-DEPTH / 2, DEPTH / 2]) corners.push(new THREE.Vector3(x, y, z));
function frame(dir) {
  const k = fitCamera(camera, dir, corners, { x0: -slabFrac, x1: slabFrac, y0: -0.95, y1: 0.95 });
  // slide the view so the slab's lowest projected point lands on bottomNdc
  camera.position.copy(k.pos); camera.lookAt(k.target); camera.updateMatrixWorld();
  const yMin = Math.min(...corners.map(p => p.clone().project(camera).y));
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const hh = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * k.pos.distanceTo(k.target);
  const d = up.multiplyScalar((yMin - bottomNdc) * hh);
  return { pos: k.pos.add(d), target: k.target.add(d) };
}
const K0 = frame(new THREE.Vector3(-0.16, 0.26, 0.95));
const K1 = frame(new THREE.Vector3(-0.10, 0.22, 0.97));

// ---- HTML labels ----------------------------------------------------------------------------------------------
const tagsEl = document.getElementById('tags'), svg = document.getElementById('leaders');
svg.setAttribute('width', W); svg.setAttribute('height', H);
const fmt = g => g.toFixed(1) + ' GiB';
function seg(cls, label, value) {
  const d = document.createElement('div');
  d.className = 'seg ' + cls;
  d.innerHTML = `<div class="label">${label}</div>${value ? `<div class="v">${value}</div>` : ''}`;
  tagsEl.appendChild(d);
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  path.setAttribute('fill', 'none'); path.setAttribute('stroke-width', '1');
  path.setAttribute('stroke', cls === 'kv' ? '#60a5fa' : '#7d8899');
  svg.appendChild(path);
  return { el: d, path };
}
const LABELS = [
  { ...seg('', 'Weights + drafter', fmt(A.weights)), of: [0] },
  { ...seg('kv', '4 × 32k KV', '4 × ' + SLOT.toFixed(2) + ' GiB'), of: [1, 2, 3, 4], phase: 'A' },
  { ...seg('kv', '256k FP8 KV', fmt(FP8SLOT)), of: [1, 2, 3, 4], phase: 'B' },
  { ...seg('', 'Graphs', ''), of: [5] },
  { ...seg('', 'Peak', ''), of: [6] },
  { ...seg('', 'Free', ''), of: [7] },
];
const slotTags = [1, 2, 3, 4].map(n => { const d = document.createElement('div'); d.className = 'slot'; d.textContent = String(n); tagsEl.appendChild(d); return d });
const mergedTag = (() => { const d = document.createElement('div'); d.className = 'slot'; d.textContent = '256K'; tagsEl.appendChild(d); return d })();
const EDGE = 24;                                               // nothing closer than this to the frame edge
const toPx = v => { const p = v.clone().project(camera); return { x: (p.x + 1) / 2 * W, y: (1 - p.y) / 2 * H } };

// ---- per-frame ----------------------------------------------------------------------------------------------
const ids = ['h1', 'h2', 's1', 's2'].map(id => document.getElementById(id));

function renderFrame(t) {
  const u = GIF ? 0.5 : 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, t / DURATION));   // the GIF holds the camera
  camera.position.copy(K0.pos).lerp(K1.pos, u);
  camera.lookAt(K0.target.clone().lerp(K1.target, u));
  camera.updateMatrixWorld();

  const mg = mergeAt(t), sm = mg * mg * (3 - 2 * mg);
  const P = placement(sm);
  blocks.forEach((b, i) => {
    const p = P.blocks[i], a = appearAt(i, t);
    b.m.position.set(p.x, 0, b.kind === 'kv' ? 0.12 * a + 0.25 * Math.sin(Math.PI * sm) : 0);
    b.m.scale.set(Math.max(1e-3, p.w), 1, 1);
    b.u.uOff.value = p.off; b.u.uW.value = p.w; b.u.uAppear.value = a;
    // slot outlines fade as the slots merge; KV glows as it lands and while it merges
    b.u.uRim.value = b.kind === 'kv' ? 0.75 * a * (1 - sm) : 0;
    b.u.uGlow.value = b.kind === 'kv' ? 0.9 * Math.max(0, 1 - Math.abs(t - APPEAR_AT[i] - 0.15) / 0.3) * a + 0.7 * Math.sin(Math.PI * sm) : 0;
  });
  render();

  // slot numbers over each 32k slot; one "256K" over the merged block
  const top = SLAB_H / 2 + 0.25;
  slotTags.forEach((d, k) => {
    const b = blocks[k + 1], p = toPx(new THREE.Vector3(b.m.position.x, top, b.m.position.z + DEPTH / 2));
    d.style.left = p.x.toFixed(1) + 'px'; d.style.top = p.y.toFixed(1) + 'px';
    d.style.opacity = (appearAt(k + 1, t) * (1 - ramp(sm, 0.05, 0.35))).toFixed(3);
  });
  {
    const x = (blocks[1].m.position.x - blocks[1].m.scale.x / 2 + blocks[4].m.position.x + blocks[4].m.scale.x / 2) / 2;
    const p = toPx(new THREE.Vector3(x, top, blocks[1].m.position.z + DEPTH / 2));
    mergedTag.style.left = p.x.toFixed(1) + 'px'; mergedTag.style.top = p.y.toFixed(1) + 'px';
    mergedTag.style.opacity = ramp(sm, 0.7, 0.3).toFixed(3);
  }

  // labels: anchors at the bottom centre of their blocks; one baseline; spread so they never touch
  const live = LABELS.filter(L => !L.phase || (L.phase === 'A' ? sm < 0.5 : sm >= 0.5));
  LABELS.forEach(L => { if (!live.includes(L)) { L.el.style.opacity = '0'; L.path.setAttribute('points', '') } });
  const base = Math.max(...blocks.map(b => toPx(new THREE.Vector3(b.m.position.x, -SLAB_H / 2, DEPTH / 2)).y)) + 34;
  const items = live.map(L => {
    const bs = L.of.map(i => blocks[i]);
    const x0 = bs[0].m.position.x - bs[0].m.scale.x / 2, x1 = bs.at(-1).m.position.x + bs.at(-1).m.scale.x / 2;
    const anchor = toPx(new THREE.Vector3((x0 + x1) / 2, -SLAB_H / 2, DEPTH / 2));
    return { L, anchor, w: L.el.offsetWidth, x: anchor.x };
  }).sort((p, q) => p.anchor.x - q.anchor.x);
  const MIN_GAP = 22;
  for (let k = 1; k < items.length; k++) {
    const need = items[k - 1].x + items[k - 1].w / 2 + MIN_GAP + items[k].w / 2;
    if (items[k].x < need) items[k].x = need;
  }
  const over = items.at(-1).x + items.at(-1).w / 2 - (W - EDGE);
  if (over > 0) for (let k = items.length - 1; k >= 0; k--) {
    items[k].x -= over;
    if (k > 0) { const lim = items[k].x - items[k].w / 2 - MIN_GAP - items[k - 1].w / 2; if (items[k - 1].x > lim) items[k - 1].x = lim; else break }
  }
  const under = EDGE - (items[0].x - items[0].w / 2);
  if (under > 0) items.forEach(it => { it.x += under });
  for (const it of items) {
    const { L, anchor } = it;
    const shown = Math.min(...L.of.map(i => appearAt(i, t)));
    const o = L.phase === 'A' ? 1 - ramp(sm, 0.3, 0.2) : L.phase === 'B' ? ramp(sm, 0.5, 0.25) : 1;
    L.el.style.left = it.x.toFixed(1) + 'px';
    L.el.style.top = (base + 10).toFixed(1) + 'px';
    L.el.style.opacity = (shown * o).toFixed(3);
    const midY = anchor.y + (base - anchor.y) * 0.55;
    L.path.setAttribute('points', `${anchor.x.toFixed(1)},${(anchor.y + 6).toFixed(1)} ${anchor.x.toFixed(1)},${midY.toFixed(1)} ${it.x.toFixed(1)},${(base + 2).toFixed(1)}`);
    L.path.setAttribute('opacity', (0.6 * shown * o).toFixed(3));
  }
  LABELS[0].el.querySelector('.v').textContent = fmt(sm > 0.5 ? B.weights : A.weights);

  // captions
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
