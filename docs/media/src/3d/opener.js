// Scene 1 (opener): one token through Inkling-2-Mini's looped MoE.
//
// Facts drawn (docs/architecture.md, inkling_engine/config.py, tests/test_config.py):
//   24 stored layers = prelude 0-5, loop core 6-17 run 3 passes, coda 18-23 -> 48 layer visits per token.
//   128 routed experts per layer, top-8 per visit: each layer is a wall of 8 x 16 = 128 tiles, 8 light up per visit.
// Which 8 experts light up is illustrative (a seeded draw), not a recorded routing trace.
import * as THREE from 'three';
import { setup, rng, textLabel, glowTexture, fitCamera, lerp, COLORS } from './common3d.js';

await document.fonts.load('400 64px JBMono');
// ?gif=1: the README loop (fixed wide camera, one caption, everything fades back to idle so the loop is seamless)
const GIF = new URLSearchParams(location.search).get('gif') === '1';
// Tile texture: by default real checkpoint data (expert-atlas.png, built by build_expert_atlas.py: per expert and layer,
// brightness = w13 amax, grain = w13 block scales). ?tex=procedural (or a missing atlas) uses a procedural texture
// instead. Nothing else changes between the two.
const DATA_TEX = new URLSearchParams(location.search).get('tex') !== 'procedural';
const atlas = DATA_TEX ? await new THREE.TextureLoader().loadAsync('expert-atlas.png').catch(() => null) : null;
if (atlas) { atlas.magFilter = THREE.NearestFilter; atlas.minFilter = THREE.LinearMipmapLinearFilter; atlas.colorSpace = THREE.NoColorSpace }
const DURATION = GIF ? 6.6 : 6.0;
const { W, H, ramp } = window.S;
const wide = W / H > 1.3;

// ---- layout -------------------------------------------------------------------------------------------------
const COLS = 8, ROWS = 16, CELL = 0.2;            // 128 experts per layer
const PW = COLS * CELL, PH = ROWS * CELL;         // plane size in world units
const STEP = 0.92, GROUP_GAP = 1.9;               // spacing between layers, between prelude/core/coda
const xs = [];
for (let l = 0; l < 24; l++) xs.push(l * STEP + (l >= 6 ? GROUP_GAP - STEP : 0) + (l >= 18 ? GROUP_GAP - STEP : 0));
const PLANE_YAW = -Math.PI / 2 + 0.62;            // planes face back along the beam, turned toward the camera
const FLOOR_Y = -PH / 2 - 0.42;

// ---- the token's path and the 48 visits ---------------------------------------------------------------------
const T0 = 0.45, T1 = 5.35, TA = 0.36;            // enter, leave, time on each return arc
const xIn = xs[0] - 3.0, xOut = xs[23] + 3.0;
const xLoopEnd = xs[17] + 0.55, xLoopStart = xs[6] - 0.7;
const straights = [[xIn, xLoopEnd, 0, 17], [xLoopStart, xLoopEnd, 6, 17], [xLoopStart, xOut, 6, 23]];
const LS = straights.reduce((s, [a, b]) => s + (b - a), 0);
const V = LS / (T1 - T0 - 2 * TA);
const arc = new THREE.CubicBezierCurve3(
  // steep legs and a high apex keep the arc clear of the LOOP and CODA labels
  new THREE.Vector3(xLoopEnd, 0, 0), new THREE.Vector3(xLoopEnd + 0.6, 4.6, 2.0),
  new THREE.Vector3(xLoopStart - 0.6, 4.6, 2.0), new THREE.Vector3(xLoopStart, 0, 0));
const segs = [];                                   // {t0, t1, kind, ...}
const visits = [];                                 // {layer, t, pass, experts}
{
  const R = rng(20261123);
  let t = T0;
  straights.forEach(([a, b, l0, l1], i) => {
    const d = (b - a) / V;
    segs.push({ t0: t, t1: t + d, kind: 'line', a, b, pass: i + 1 });
    for (let l = l0; l <= l1; l++) {
      const pick = new Set();
      while (pick.size < 8) pick.add(Math.floor(R() * 128));
      visits.push({ layer: l, t: t + (xs[l] - a) / V, pass: l >= 6 && l <= 17 ? i + 1 : 0, experts: [...pick] });
    }
    t += d;
    if (i < 2) { segs.push({ t0: t, t1: t + TA, kind: 'arc', pass: i + 1, n: i }); t += TA }
  });
  visits.sort((p, q) => p.t - q.t);
  if (visits.length !== 48) throw new Error('expected 48 visits, got ' + visits.length);
}
function headAt(t) {
  if (t <= T0) return { p: new THREE.Vector3(xIn, 0, 0), seg: segs[0], u: 0 };
  for (const s of segs) {
    if (t <= s.t1) {
      const u = (t - s.t0) / (s.t1 - s.t0);
      const p = s.kind === 'line' ? new THREE.Vector3(lerp(s.a, s.b, u), 0, 0) : arc.getPointAt(u);
      return { p, seg: s, u };
    }
  }
  const s = segs.at(-1);
  return { p: new THREE.Vector3(s.b + (t - s.t1) * V, 0, 0), seg: s, u: 1 };
}

// ---- scene ------------------------------------------------------------------------------------------------
const { scene, camera, render } = setup({ fov: wide ? 27 : 36, bloom: { strength: 1.0, radius: 0.5, threshold: 0.55 } });

// Expert intensities: one texel per (expert, layer).
const intData = new Uint8Array(128 * 24 * 4);
const intTex = new THREE.DataTexture(intData, 128, 24, THREE.RGBAFormat);
intTex.magFilter = intTex.minFilter = THREE.NearestFilter;

const planeVS = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`;
const planeFS = `
  uniform sampler2D uInt, uAtlas; uniform float uLayer, uPulse, uDim, uLoop, uData;
  varying vec2 vUv;
  float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  void main(){
    vec2 c = vUv * vec2(${COLS}.0, ${ROWS}.0);
    vec2 id = floor(c), f = fract(c);
    float e = (${ROWS}.0 - 1.0 - id.y) * ${COLS}.0 + id.x;
    float it = texture2D(uInt, vec2((e + 0.5) / 128.0, (uLayer + 0.5) / 24.0)).r;
    float g = 0.075;
    float inCell = step(g, f.x) * step(f.x, 1.0 - g) * step(g, f.y) * step(f.y, 1.0 - g);
    // weight-like texture: short horizontal runs of varying brightness
    float rowN = hash(vec2(floor(f.y * 22.0), id.x * 13.0 + id.y * 7.0 + uLayer * 31.0));
    float run = hash(vec2(floor(f.x * (6.0 + 18.0 * rowN)), floor(f.y * 22.0) + id.y * 3.0 + uLayer));
    float tex = 0.35 + 0.65 * run * (0.6 + 0.4 * rowN);
    // real data: atlas row block = layer, column block = expert, top row first
    float d = texture2D(uAtlas, vec2((e + f.x) / 128.0, 1.0 - (uLayer + 1.0 - f.y) / 24.0)).r;
    tex = mix(tex, 0.30 + 0.70 * d, uData);
    vec3 idle = mix(vec3(0.07, 0.08, 0.10), vec3(0.21, 0.23, 0.27), tex);
    idle = mix(idle, idle * vec3(1.05, 0.98, 0.9), uLoop * 0.35);
    vec3 hot = mix(vec3(0.14, 0.38, 0.92), vec3(0.30, 0.56, 1.0), tex);
    float k = clamp(it * 1.15, 0.0, 1.0);
    vec3 col = mix(idle, hot * (1.0 + 0.9 * it * it), k);
    col *= inCell;
    col += vec3(0.25, 0.5, 1.0) * uPulse * 0.35 * (1.0 - inCell);
    gl_FragColor = vec4(pow(col * uDim, vec3(2.2)), 1.0);  // tuned in sRGB, output linear
  }`;

const planeGeo = new THREE.PlaneGeometry(PW, PH);
const cardGeo = new THREE.PlaneGeometry(PW + 0.16, PH + 0.16);
const planes = [];
function makeLayers(dim, mirror) {
  const grp = new THREE.Group();
  for (let l = 0; l < 24; l++) {
    const u = { uInt: { value: intTex }, uAtlas: { value: atlas || intTex }, uData: { value: atlas ? 1 : 0 }, uLayer: { value: l }, uPulse: { value: 0 }, uDim: { value: dim }, uLoop: { value: l >= 6 && l <= 17 ? 1 : 0 } };
    const m = new THREE.Mesh(planeGeo, new THREE.ShaderMaterial({ uniforms: u, vertexShader: planeVS, fragmentShader: planeFS, side: THREE.DoubleSide }));
    const card = new THREE.Mesh(cardGeo, new THREE.MeshBasicMaterial({ color: 0x0a111c, side: THREE.DoubleSide }));
    card.position.set(0, 0, -0.004);
    const holder = new THREE.Group();
    holder.add(card, m);
    holder.position.set(xs[l], 0, 0);
    holder.rotation.y = PLANE_YAW;
    grp.add(holder);
    if (!mirror) planes.push(u);
    else (planes.mirror ||= []).push(u);
  }
  return grp;
}
scene.add(makeLayers(1, false));
const mirror = makeLayers(0.45, true);
mirror.scale.y = -1; mirror.position.y = 2 * FLOOR_Y;
scene.add(mirror);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), new THREE.MeshBasicMaterial({ color: COLORS.bg, transparent: true, opacity: 0.55 }));
floor.rotation.x = -Math.PI / 2; floor.position.y = FLOOR_Y;
scene.add(floor);

// Faint guide along the straight path.
{
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(xIn, 0, 0), new THREE.Vector3(xOut, 0, 0)]);
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x1f2a3c, transparent: true, opacity: 0.8 })));
}

// The return arc (orange): drawn as the token first travels it, then stays.
const ARC_SEG = 200, ARC_RAD = 8;
const arcGeo = new THREE.TubeGeometry(arc, ARC_SEG, 0.028, ARC_RAD, false);
const arcMat = new THREE.MeshBasicMaterial({ color: COLORS.orange, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false });
const arcMesh = new THREE.Mesh(arcGeo, arcMat);
scene.add(arcMesh);
const arcHalo = new THREE.Mesh(new THREE.TubeGeometry(arc, ARC_SEG, 0.09, ARC_RAD, false),
  new THREE.MeshBasicMaterial({ color: COLORS.orange, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false }));
scene.add(arcHalo);
const arcLabel = textLabel('× 3 passes', { height: 0.3, color: '#f5a524' });
arcLabel.position.copy(arc.getPointAt(0.5)).add(new THREE.Vector3(0, 0.38, 0));
scene.add(arcLabel);

// Group labels above the walls.
const groups = [['Prelude · 0-5', 0, 5], ['Loop · 6-17', 6, 17], ['Coda · 18-23', 18, 23]];
const groupLabels = groups.map(([txt, a, b]) => {
  const m = textLabel(txt, { height: 0.42, color: '#b4c0d2', chip: true });
  m.geometry.translate(m.geometry.parameters.width / 2, 0, 0);   // anchored at its left end, over the group's first layer
  m.position.set(xs[a] - 0.35, PH / 2 + 0.45, 0.2);
  if (a === 6) m.position.x = xs[a] + 0.8;   // the loop label sits inside the arc, clear of both legs
  scene.add(m); return m;
});

// Token head and its trail.
const glow = glowTexture();
const head = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color: 0xdbe8ff, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
head.scale.set(0.9, 0.9, 1);
scene.add(head);
const TRAIL = 90;
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
trailGeo.setAttribute('size', new THREE.BufferAttribute(new Float32Array(TRAIL), 1));
const trailMat = new THREE.ShaderMaterial({
  uniforms: { uMap: { value: glow }, uScale: { value: H / 2 } },
  vertexShader: `attribute float size; attribute vec3 color; varying vec3 vC;
    void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position,1.0); gl_PointSize = size * ${(H / 1080).toFixed(4)} * 300.0 / -mv.z; gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform sampler2D uMap; varying vec3 vC; void main(){ gl_FragColor = vec4(vC, 1.0) * texture2D(uMap, gl_PointCoord).a; }`,
  blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
});
scene.add(new THREE.Points(trailGeo, trailMat));

// Routing rays: from the token to each of the 8 chosen tiles, flashing at the visit.
const RAYS = 48 * 8;
const rayGeo = new THREE.BufferGeometry();
const rayPos = new Float32Array(RAYS * 6), rayCol = new Float32Array(RAYS * 6);
const tilePos = (l, e) => {
  const col = e % COLS, row = Math.floor(e / COLS);
  const local = new THREE.Vector3(-PW / 2 + (col + 0.5) * CELL, PH / 2 - (row + 0.5) * CELL, 0);
  return local.applyAxisAngle(new THREE.Vector3(0, 1, 0), PLANE_YAW).add(new THREE.Vector3(xs[l], 0, 0));
};
visits.forEach((v, i) => v.experts.forEach((e, j) => {
  const k = (i * 8 + j) * 6, from = new THREE.Vector3(xs[v.layer] - 0.5, 0, 0), to = tilePos(v.layer, e);
  rayPos.set([from.x, from.y, from.z, to.x, to.y, to.z], k);
}));
rayGeo.setAttribute('position', new THREE.BufferAttribute(rayPos, 3));
rayGeo.setAttribute('color', new THREE.BufferAttribute(rayCol, 3));
scene.add(new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })));

// ---- camera -----------------------------------------------------------------------------------------------
// Keyframes, each fitted to a set of cards: close on the prelude, then the loop core with its arc, then everything.
const cardPts = (l0, l1) => {
  const out = [];
  for (let l = l0; l <= l1; l++) for (const [u, v] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    out.push(new THREE.Vector3(u * (PW / 2 + 0.08), v * (PH / 2 + 0.08), 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), PLANE_YAW).add(new THREE.Vector3(xs[l], 0, 0)));
  }
  return out;
};
const labelPts = groupLabels.flatMap(m => { const w = m.geometry.parameters.width; return [m.position.clone().add(new THREE.Vector3(0, 0.2, 0)), m.position.clone().add(new THREE.Vector3(w, 0.2, 0))] });
const arcPts = Array.from({ length: 21 }, (_, i) => arc.getPointAt(i / 20)).concat([arcLabel.position.clone().add(new THREE.Vector3(0, 0.2, 0))]);
const BOX = wide ? { x0: -0.86, x1: 0.86, y0: -0.36, y1: 0.76 } : { x0: -0.86, x1: 0.86, y0: -0.40, y1: 0.76 };
const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
const KEYS = [
  fitCamera(camera, V3(-0.62, 0.30, 0.72), cardPts(0, 8), { ...BOX, x1: BOX.x1 + 0.6 }),
  fitCamera(camera, V3(-0.58, 0.30, 0.76), [...cardPts(4, 18), ...arcPts], BOX),
  fitCamera(camera, wide ? V3(-0.70, 0.26, 0.66) : V3(-0.84, 0.30, 0.45), [...cardPts(0, 23), ...labelPts, ...arcPts], BOX),
];
KEYS.push({ pos: KEYS[2].pos.clone().lerp(KEYS[2].target, 0.06), target: KEYS[2].target.clone() });
const camPath = new THREE.CatmullRomCurve3(KEYS.map(k => k.pos), false, 'centripetal');
const tgtPath = new THREE.CatmullRomCurve3(KEYS.map(k => k.target), false, 'centripetal');

// ---- per-frame ----------------------------------------------------------------------------------------------
const flash = d => d < 0 ? 0 : Math.min(1, d / 0.05) * (0.55 + 0.45 * Math.exp(-d / 0.32));
const hud = { el: document.getElementById('hud'), visit: document.getElementById('visit'), pass: document.getElementById('pass') };
const beats = [['b1', 0.4, 1.85], ['b2', 2.2, 3.75], ['b3', 4.1, null]].map(([id, a, b]) => [document.getElementById(id), a, b]);
const tmp = new THREE.Color();

function renderFrame(t) {
  // camera dolly
  const u = GIF ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, t / DURATION));
  camera.position.copy(camPath.getPoint(u));
  camera.lookAt(tgtPath.getPoint(u));
  const fadeOut = GIF ? 1 - ramp(t, 5.6, 0.8) : 1;
  [...groupLabels, arcLabel].forEach(m => m.quaternion.copy(camera.quaternion));
  camera.updateMatrixWorld();
  // a label near the frame edge fades out instead of being clipped
  groupLabels.forEach(m => {
    const w = m.geometry.parameters.width;
    const r = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    const ends = [m.position.clone(), m.position.clone().addScaledVector(r, w)].map(p => p.project(camera));
    const out = Math.max(...ends.map(q => Math.max(Math.abs(q.x), q.y)));
    m.material.opacity = Math.min(1, Math.max(0, (0.97 - out) / 0.07));
  });

  // expert tiles
  const val = new Float32Array(128 * 24);
  const pulse = new Float32Array(24);
  for (const v of visits) {
    const d = t - v.t;
    if (d < 0) continue;
    const f = flash(d);
    for (const e of v.experts) val[v.layer * 128 + e] = Math.max(val[v.layer * 128 + e], f);
    pulse[v.layer] = Math.max(pulse[v.layer], Math.exp(-Math.abs(d) / 0.12));
  }
  for (let i = 0; i < val.length; i++) intData[i * 4] = Math.round(Math.min(1, val[i]) * fadeOut * 255);
  intTex.needsUpdate = true;
  planes.forEach((p, l) => { p.uPulse.value = pulse[l] });
  planes.mirror.forEach((p, l) => { p.uPulse.value = pulse[l] });

  // rays
  visits.forEach((v, i) => {
    const d = t - v.t, a = d < 0 || d > 0.6 ? 0 : Math.min(1, d / 0.03) * Math.exp(-d / 0.11) * 0.9;
    for (let j = 0; j < 8; j++) {
      const k = (i * 8 + j) * 6;
      rayCol.set([0.55 * a, 0.75 * a, 1.0 * a, 0.25 * a, 0.5 * a, 1.0 * a], k);
    }
  });
  rayGeo.attributes.color.needsUpdate = true;

  // head + trail
  const h = headAt(t);
  const onArc = h.seg.kind === 'arc' && t > h.seg.t0 && t < h.seg.t1;
  const vis = ramp(t, 0.4, 0.25) * (1 - ramp(t, T1 + 0.1, 0.45));
  head.position.copy(h.p);
  head.material.opacity = vis;
  head.material.color.set(onArc ? 0xffc46b : 0xdbe8ff);
  const tp = trailGeo.attributes.position.array, tc = trailGeo.attributes.color.array, ts = trailGeo.attributes.size.array;
  for (let i = 0; i < TRAIL; i++) {
    const tt = t - i * 0.0042, q = headAt(Math.max(0, tt)), fall = Math.pow(1 - i / TRAIL, 1.6) * vis * (tt < T0 - 0.3 ? 0 : 1);
    const arcPt = q.seg.kind === 'arc' && tt > q.seg.t0 && tt < q.seg.t1;
    tmp.set(arcPt ? COLORS.orange : 0x8fb8ff);
    tp.set([q.p.x, q.p.y, q.p.z], i * 3);
    tc.set([tmp.r * fall * 0.9, tmp.g * fall * 0.9, tmp.b * fall * 0.9], i * 3);
    ts[i] = 0.55 * (1 - i / TRAIL) + 0.12;
  }
  ['position', 'color', 'size'].forEach(k => { trailGeo.attributes[k].needsUpdate = true });

  // arc: revealed on the first return, re-lit on the second
  const a1 = segs.find(s => s.kind === 'arc' && s.n === 0), a2 = segs.find(s => s.kind === 'arc' && s.n === 1);
  const reveal = Math.min(1, Math.max(0, (t - a1.t0) / (a1.t1 - a1.t0)));
  const per = ARC_RAD * 6;
  arcGeo.setDrawRange(0, Math.floor(reveal * ARC_SEG) * per);
  arcHalo.geometry.setDrawRange(0, Math.floor(reveal * ARC_SEG) * per);
  const hot = Math.max(...[a1, a2].map(s => t < s.t0 ? 0 : t <= s.t1 ? 1 : Math.exp(-(t - s.t1) / 0.4)));
  const k = 0.55 + 1.6 * hot;
  arcMat.color.setRGB(0.91 * k, 0.34 * k, 0.004 * k);  // #f59e0b in linear
  arcLabel.material.opacity = ramp(t, a1.t1 - 0.1, 0.4) * fadeOut;
  arcMat.opacity = fadeOut; arcHalo.material.opacity = 0.12 * fadeOut;

  render();

  // overlay text
  const n = visits.filter(v => v.t <= t).length;
  hud.visit.textContent = String(n);
  hud.pass.textContent = String(t < a1.t1 ? 1 : t < a2.t1 ? 2 : 3);
  hud.el.style.opacity = (GIF ? Math.min(ramp(t, 0.15, 0.3), fadeOut) : ramp(t, 0.4, 0.4)).toFixed(3);
  if (GIF) {
    beats.forEach(([el], i) => { el.style.opacity = i === 1 ? '1' : '0' });
    return;
  }
  for (const [el, a, b] of beats) {
    const o = ramp(t, a, 0.4) * (b == null ? 1 : 1 - ramp(t, b, 0.3));
    el.style.opacity = o.toFixed(4);
    el.style.transform = `translateY(${((1 - ramp(t, a, 0.4)) * 12).toFixed(2)}px)`;
  }
}

window.renderFrame = renderFrame;
window.SCENE = { duration: DURATION };
