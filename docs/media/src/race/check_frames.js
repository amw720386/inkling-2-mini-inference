#!/usr/bin/env node
// Steps every frame of a race page and reports visible-text defects: raw markdown/LaTeX characters in the streamed
// text, elements spilling out of their card, overlapping cards/punchline, and text clipped at the frame edge.
//   node race/check_frames.js scenes/scene2-race.html 1080 1080 [fps]
const path = require('path'), os = require('os'), fs = require('fs');
const puppeteer = require('puppeteer-core');
const [page_, W, H, FPS = 30] = process.argv.slice(2);
function chrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const base = path.join(os.homedir(), '.cache');
  const out = [];
  const walk = (d, n) => { if (n > 6) return; let es = []; try { es = fs.readdirSync(d, { withFileTypes: true }) } catch { return }
    for (const e of es) { const p = path.join(d, e.name); if (e.isFile() && e.name === 'chrome-headless-shell') out.push(p); else if (e.isDirectory()) walk(p, n + 1) } };
  walk(base, 0); return out.sort().at(-1);
}
(async () => {
  const b = await puppeteer.launch({ executablePath: chrome(), headless: 'shell', args: ['--no-sandbox', '--allow-file-access-from-files'] });
  const p = await b.newPage();
  await p.setViewport({ width: +W, height: +H });
  await p.goto('file://' + path.resolve(page_) + `?w=${W}&h=${H}`, { waitUntil: 'load' });
  await p.evaluate(async () => { await document.fonts.ready });
  const dur = await p.evaluate(() => window.SCENE.duration);
  let bad = 0;
  for (let i = 0; i < Math.round(dur * FPS); i++) {
    const t = i / FPS;
    const issues = await p.evaluate(t => {
      window.renderFrame(t);
      const out = [];
      document.querySelectorAll('.card').forEach((c, k) => {
        const txt = c.querySelector('.stream').innerText;
        if (/[\\{}]|\*\*/.test(txt)) out.push(`card${k} raw chars: ${JSON.stringify(txt.slice(-80))}`);
        const cr = c.getBoundingClientRect();
        c.querySelectorAll('.pill, .bar, .meta, .timer').forEach(e => {
          const r = e.getBoundingClientRect();
          if (getComputedStyle(e).opacity !== '0' && (r.bottom > cr.bottom + 0.5 || r.right > cr.right + 0.5)) out.push(`card${k} ${e.className} spills`);
        });
      });
      const last = document.querySelectorAll('.card')[document.querySelectorAll('.card').length - 1].getBoundingClientRect();
      const pl = document.querySelector('#punchline').getBoundingClientRect();
      if (pl.top < last.bottom) out.push('punchline overlaps cards');
      document.querySelectorAll('#root *').forEach(e => {
        const r = e.getBoundingClientRect();
        if (r.width && (r.right > innerWidth || r.bottom > innerHeight) && getComputedStyle(e).opacity !== '0') out.push(`${e.id || e.className} past frame edge`);
      });
      return out;
    }, t);
    if (issues.length) { bad++; if (bad <= 20) console.log(t.toFixed(3), issues.join(' | ')) }
  }
  console.log(`${page_} ${W}x${H}: ${bad} frames with issues`);
  await b.close();
  process.exit(bad ? 1 : 0);
})();
