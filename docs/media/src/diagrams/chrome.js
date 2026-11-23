// Finds headless Chrome: $CHROME, else the newest chrome-headless-shell under ~/.cache
// (install: npx @puppeteer/browsers install chrome-headless-shell@stable).
const fs = require('fs'), path = require('path'), os = require('os');

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
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
  walk(path.join(os.homedir(), '.cache'), 0);
  if (!found.length) throw new Error('no chrome-headless-shell found; set CHROME=/path/to/chrome');
  return found.sort().at(-1);
}

const LAUNCH_ARGS = ['--hide-scrollbars', '--force-device-scale-factor=1', '--font-render-hinting=none',
  '--allow-file-access-from-files', '--no-sandbox', '--disable-dev-shm-usage'];

module.exports = { findChrome, LAUNCH_ARGS };
