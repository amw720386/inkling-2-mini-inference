#!/usr/bin/env node
// Frame-by-frame scene renderer: headless Chrome draws each frame at a fixed time t, ffmpeg encodes the frames.
// Nothing is screen-captured in real time, so every render of a scene is identical.
//
//   node render.js <scene.html> <out.mp4> [--w 1080] [--h 1080] [--fps 30] [--crf 12] [--stills 0.5,3,5.9]
//
// A scene page defines:
//   window.SCENE = { duration: <seconds>, ready: <optional Promise> }
//   window.renderFrame(t)   // draw the frame at time t (seconds); may return a Promise
// It reads its frame size from the ?w=&h= query string.
// Chrome: $CHROME, else the newest chrome-headless-shell under ~/.cache (npx @puppeteer/browsers install chrome-headless-shell@stable).
const fs = require('fs'), path = require('path'), os = require('os'), { spawn } = require('child_process');
const puppeteer = require('puppeteer-core');

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i < 0 ? d : argv[i + 1] };
const [sceneArg, outArg] = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--')));
if (!sceneArg || !outArg) { console.error('usage: render.js <scene.html> <out.mp4> [--w W --h H --fps F --crf C --stills t1,t2]'); process.exit(2) }
const W = +opt('w', 1080), H = +opt('h', 1080), FPS = +opt('fps', 30), CRF = String(opt('crf', 12));

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const roots = [path.join(os.homedir(), '.cache')];
  const found = [];
  const walk = (d, depth) => {
    if (depth > 6) return;
    let ents = []; try { ents = fs.readdirSync(d, { withFileTypes: true }) } catch { return }
    for (const e of ents) {
      const p = path.join(d, e.name);
      if (e.isFile() && e.name === 'chrome-headless-shell') found.push(p);
      else if (e.isDirectory() && (depth < 1 ? /chrome|chs|puppeteer/.test(e.name) : true)) walk(p, depth + 1);
    }
  };
  roots.forEach(r => walk(r, 0));
  if (!found.length) throw new Error('no chrome-headless-shell found; set CHROME=/path/to/chrome');
  return found.sort().at(-1);
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: findChrome(), headless: 'shell',
    args: ['--hide-scrollbars', '--force-device-scale-factor=1', '--font-render-hinting=none', '--allow-file-access-from-files',
      '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--no-sandbox', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warn') console.error('[page]', m.text()) });
  page.on('pageerror', e => { console.error('[page error]', e.message); process.exitCode = 1 });
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  await page.goto('file://' + path.resolve(sceneArg) + `?w=${W}&h=${H}`, { waitUntil: 'load' });
  await page.waitForFunction('window.SCENE && window.renderFrame');
  await page.evaluate(async () => { await document.fonts.ready; if (window.SCENE.ready) await window.SCENE.ready });
  const dur = await page.evaluate(() => window.SCENE.duration);
  const out = path.resolve(outArg);

  if (opt('stills')) {
    fs.mkdirSync(out, { recursive: true });
    for (const t of String(opt('stills')).split(',').map(Number)) {
      await page.evaluate(t => window.renderFrame(t), t);
      await page.screenshot({ path: path.join(out, `t${t.toFixed(2)}.png`) });
    }
    await browser.close(); return;
  }

  const n = Math.round(dur * FPS);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', CRF, '-pix_fmt', 'yuv420p', '-r', String(FPS), '-movflags', '+faststart', out],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', c => c === 0 ? res() : rej(new Error('ffmpeg exit ' + c))));
  for (let i = 0; i < n; i++) {
    await page.evaluate(t => window.renderFrame(t), i / FPS);
    const buf = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  }
  ff.stdin.end(); await done; await browser.close();
  console.log(`${out}: ${n} frames, ${W}x${H} @ ${FPS} fps`);
})().catch(e => { console.error(e); process.exit(1) });
