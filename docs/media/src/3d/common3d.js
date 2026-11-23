// Shared three.js setup for the 3D scenes: renderer + bloom, text labels, glow sprites, a seeded RNG.
// Every scene is a pure function of time t, so frames can be rendered in any order.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export const COLORS = {
  bg: 0x05080f, blue: 0x3b82f6, blueGlow: 0x60a5fa, grey: 0x6b7280, orange: 0xf59e0b, text: 0xe8edf4, muted: 0x7d8899,
};

export function setup({ fov = 30, bloom = { strength: 0.9, radius: 0.55, threshold: 0.62 } } = {}) {
  const { W, H } = window.S;
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(1);
  renderer.setSize(W, H);
  renderer.setClearColor(COLORS.bg, 1);
  document.body.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.bg);  // color-managed (setClearColor alone is not)
  scene.fog = new THREE.FogExp2(COLORS.bg, 0.028);
  const camera = new THREE.PerspectiveCamera(fov, W / H, 0.1, 200);
  const rt = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(W, H), bloom.strength, bloom.radius, bloom.threshold);
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());
  return { renderer, scene, camera, composer, bloomPass, render: () => composer.render() };
}

// Deterministic PRNG (mulberry32).
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 };
}

// A flat text label in the 3D scene (monospace caps, letter-spaced), `height` world units tall.
export function textLabel(text, { height = 0.22, color = '#9aa6b8', weight = 400, spacing = 0.2, font = 'JBMono', chip = false } = {}) {
  const px = 64, c = document.createElement('canvas'), g = c.getContext('2d');
  const f = `${weight} ${px}px ${font}`;
  g.font = f;
  const chars = [...text.toUpperCase()];
  const sp = spacing * px;
  const w = Math.ceil(chars.reduce((s, ch) => s + g.measureText(ch).width + sp, 0) - sp) + 8;
  const pad = chip ? Math.round(px * 0.45) : 0;
  c.width = w + 2 * pad; c.height = Math.ceil(px * 1.3) + (chip ? Math.round(px * 0.3) : 0);
  if (chip) {   // dark rounded backing so the label reads over tiles
    g.fillStyle = 'rgba(5,8,15,0.82)'; g.strokeStyle = 'rgba(60,74,96,0.9)'; g.lineWidth = 3;
    g.beginPath(); g.roundRect(1.5, 1.5, c.width - 3, c.height - 3, c.height * 0.22); g.fill(); g.stroke();
  }
  g.font = f; g.fillStyle = color; g.textBaseline = 'middle';
  let x = 4 + pad;
  for (const ch of chars) { g.fillText(ch, x, c.height / 2); x += g.measureText(ch).width + sp }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false, fog: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(height * c.width / c.height, height), mat);
  mesh.renderOrder = 10;   // labels draw over everything
  return mesh;
}

// Radial glow texture for sprites and points.
export function glowTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d'), r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.15, 'rgba(255,255,255,0.85)');
  r.addColorStop(0.4, 'rgba(255,255,255,0.25)'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); return t;
}

export const lerp = (a, b, x) => a + (b - a) * x;
export const smooth = x => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x) };

// Camera framing: place a camera looking along -dir so that every point in `pts` projects inside the NDC box
// {x0, x1, y0, y1}. Returns {pos, target}. Deterministic (bisection + recentering), so framing survives edits.
export function fitCamera(camera, dir, pts, box) {
  const d = dir.clone().normalize();
  const c = new THREE.Vector3();
  pts.forEach(p => c.add(p)); c.multiplyScalar(1 / pts.length);
  const target = c.clone();
  const bbox = dist => {
    camera.position.copy(target).addScaledVector(d, dist);
    camera.lookAt(target); camera.updateMatrixWorld(); camera.updateProjectionMatrix();
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const p of pts) { const q = p.clone().project(camera); x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y) }
    return { x0, x1, y0, y1 };
  };
  let dist = 20;
  for (let it = 0; it < 6; it++) {
    let lo = 0.5, hi = 400;
    for (let k = 0; k < 50; k++) {
      const m = (lo + hi) / 2, b = bbox(m);
      if (b.x1 - b.x0 <= box.x1 - box.x0 && b.y1 - b.y0 <= box.y1 - box.y0) hi = m; else lo = m;
    }
    dist = hi;
    const b = bbox(dist);
    // shift the target so the bbox centre moves to the box centre (camera right/up axes)
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
    const hh = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * dist, hw = hh * camera.aspect;
    target.addScaledVector(right, ((b.x0 + b.x1) / 2 - (box.x0 + box.x1) / 2) * hw);
    target.addScaledVector(up, ((b.y0 + b.y1) / 2 - (box.y0 + box.y1) / 2) * hh);
  }
  bbox(dist);
  return { pos: camera.position.clone(), target: target.clone() };
}
