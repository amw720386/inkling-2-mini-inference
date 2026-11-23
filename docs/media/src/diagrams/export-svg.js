#!/usr/bin/env node
// Renders each architecture diagram in headless Chrome (real font metrics for wrapping), checks its layout, and saves
// it as a standalone SVG with the fonts embedded, so GitHub shows it exactly as rendered.
//
//   node export-svg.js <out_dir> [--png <png_dir>] [name ...]
//
// Writes <out_dir>/<name>-dark.svg and <name>-light.svg. Fails if a label leaves the canvas or two labels overlap.
const fs = require('fs'), path = require('path');
const puppeteer = require('puppeteer-core');
const { findChrome, LAUNCH_ARGS } = require('./chrome.js');

const argv = process.argv.slice(2);
const pngI = argv.indexOf('--png');
const pngDir = pngI >= 0 ? argv.splice(pngI, 2)[1] : null;
const [outDir, ...only] = argv;
if (!outDir) { console.error('usage: export-svg.js <out_dir> [--png dir] [name ...]'); process.exit(2) }
const NAMES = only.length ? only : ['request-path', 'decode-step', 'memory-layout', 'kernel-fusion'];
const FONTS = path.join(__dirname, '..', 'fonts');
const face = (fam, w, file) => `@font-face{font-family:${fam};font-weight:${w};src:url(data:font/woff2;base64,${
  fs.readFileSync(path.join(FONTS, file)).toString('base64')}) format("woff2")}`;
const STYLE = [face('Inter', 300, 'inter-latin-300-normal.woff2'), face('Inter', 400, 'inter-latin-400-normal.woff2'),
  face('JBMono', 400, 'jetbrains-mono-latin-400-normal.woff2')].join('');

(async () => {
  const browser = await puppeteer.launch({ executablePath: findChrome(), headless: 'shell', args: LAUNCH_ARGS });
  const page = await browser.newPage();
  page.on('pageerror', e => { console.error('[page error]', e.message); process.exitCode = 1 });
  page.on('console', m => { if (m.type() === 'error') console.error('[page]', m.text()) });
  await page.setViewport({ width: 1600, height: 1200, deviceScaleFactor: 1 });
  fs.mkdirSync(outDir, { recursive: true });
  if (pngDir) fs.mkdirSync(pngDir, { recursive: true });
  let bad = 0;
  for (const name of NAMES) for (const theme of ['dark', 'light']) {
    await page.goto('file://' + path.join(__dirname, 'diagram.html') + `?name=${name}&theme=${theme}`, { waitUntil: 'load' });
    await page.waitForFunction('window.DIAGRAM && window.AUDIT');
    const { svg, audit, weights } = await page.evaluate(() => {
      const ws = new Set([...window.DIAGRAM.querySelectorAll('text')].map(t => t.getAttribute('font-family').split(',')[0] + ' ' + t.getAttribute('font-weight')));
      return { svg: new XMLSerializer().serializeToString(window.DIAGRAM), audit: window.AUDIT, weights: [...ws] };
    });
    const extra = weights.filter(w => !['Inter 300', 'Inter 400', 'JBMono 400'].includes(w));
    if (extra.length) audit.push('font weight not embedded: ' + extra.join(', '));
    const out = svg.replace(/^<svg([^>]*)>/, (m, a) => `<svg${a}><style>${STYLE}</style>`);
    const file = path.join(outDir, `${name}-${theme}.svg`);
    fs.writeFileSync(file, out);
    if (pngDir) await (await page.$('svg')).screenshot({ path: path.join(pngDir, `${name}-${theme}.png`) });
    console.log(`${file}: ${(out.length / 1024).toFixed(0)} KiB${audit.length ? '  PROBLEMS:\n  ' + audit.join('\n  ') : ''}`);
    if (audit.length) bad++;
  }
  await browser.close();
  if (bad) { console.error(`${bad} diagram(s) failed the layout check`); process.exit(1) }
})().catch(e => { console.error(e); process.exit(1) });
