// SVG building blocks for the architecture diagrams, the explanatory GIFs and the how-it-works scene.
// Runs in the browser after theme.css: colors come from theme.css's custom properties, and text is measured with the
// real fonts so wrapping and layout checks match what is drawn.
(function () {
  const NS = 'http://www.w3.org/2000/svg';

  // ---- colors: read from theme.css (the one palette for every scene, GIF and diagram)
  const VARS = ['bg', 'bg-card', 'border', 'text', 'body', 'muted', 'dim', 'blue', 'blue-glow', 'grey', 'orange', 'green', 'red'];
  function palette(light) {
    const probe = document.createElement('div');
    if (light) probe.className = 'light';
    document.body.appendChild(probe);
    const cs = getComputedStyle(probe), P = {};
    for (const v of VARS) P[v.replace('-', '_')] = cs.getPropertyValue('--' + v).trim();
    probe.remove();
    P.light = !!light;
    return P;
  }

  // ---- fonts (sizes in SVG user units)
  const FAM = { sans: 'Inter, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
                mono: 'JBMono, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' };
  const font = (fam, weight, size, ls = 0, caps = false) => ({ fam, weight, size, ls, caps });
  const F = {
    label: (s = 12) => font('mono', 400, s, 0.2, true),
    code: (s = 13) => font('mono', 400, s),
    head: (s = 30) => font('sans', 300, s, -0.01),
    title: (s = 17) => font('sans', 400, s),
    body: (s = 14) => font('sans', 400, s),
    light: (s = 14) => font('sans', 300, s),
  };

  const ctx = document.createElement('canvas').getContext('2d');
  function measure(str, f) {
    ctx.font = `${f.weight} ${f.size}px ${f.fam === 'mono' ? 'JBMono' : 'Inter'}`;
    ctx.letterSpacing = (f.ls * f.size) + 'px';
    return ctx.measureText(f.caps ? str.toUpperCase() : str).width;
  }
  function wrap(str, width, f) {
    const out = [];
    for (const para of String(str).split('\n')) {
      let line = '';
      for (const w of para.split(' ')) {
        const tryL = line ? line + ' ' + w : w;
        if (line && measure(tryL, f) > width) { out.push(line); line = w } else line = tryL;
      }
      out.push(line);
    }
    return out;
  }

  // ---- elements
  function el(tag, attrs = {}, kids = []) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    for (const k of [].concat(kids)) if (k != null) e.appendChild(typeof k === 'string' ? document.createTextNode(k) : k);
    return e;
  }
  function fontAttrs(f) {
    return { 'font-family': FAM[f.fam], 'font-weight': f.weight, 'font-size': f.size,
             'letter-spacing': f.ls ? +(f.ls * f.size).toFixed(2) : null };
  }
  // One line of text. y is the baseline.
  function text(x, y, str, f, fill, o = {}) {
    const halo = o.halo ? { stroke: o.halo, 'stroke-width': 6, 'paint-order': 'stroke', 'stroke-linejoin': 'round' } : {};
    return el('text', { x, y, fill, 'text-anchor': o.anchor, opacity: o.opacity, ...fontAttrs(f), ...halo, ...(o.attrs || {}) },
      [f.caps ? String(str).toUpperCase() : String(str)]);
  }
  // Wrapped paragraph; y is the top of the first line. Returns {g, h, lines}.
  function para(x, y, str, width, f, fill, o = {}) {
    const lh = o.lh || 1.42, lines = wrap(str, width, f);
    const g = el('g');
    lines.forEach((l, i) => g.appendChild(text(x, y + f.size * 0.93 + i * f.size * lh, l, f, fill, o)));
    return { g, h: lines.length * f.size * lh, lines };
  }
  // Mixed runs on one line: [[str, font, fill], ...]
  function runs(x, y, parts, o = {}) {
    const g = el('g'); let cx = x;
    const total = parts.reduce((s, [str, f]) => s + measure(str, f), 0);
    if (o.anchor === 'middle') cx -= total / 2;
    if (o.anchor === 'end') cx -= total;
    for (const [str, f, fill] of parts) {
      const lead = str.length - str.trimStart().length;       // SVG collapses leading spaces: advance instead
      cx += measure(str.slice(0, lead), f);
      g.appendChild(text(cx, y, str.trimStart(), f, fill)); cx += measure(str.trimStart(), f);
    }
    return g;
  }
  function rect(x, y, w, h, o = {}) {
    return el('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), rx: o.rx, fill: o.fill || 'none',
      stroke: o.stroke, 'stroke-width': o.sw, 'stroke-dasharray': o.dash, opacity: o.opacity, 'fill-opacity': o.fo });
  }
  function card(P, x, y, w, h, o = {}) {
    return rect(x + 0.5, y + 0.5, w - 1, h - 1, { rx: o.rx ?? 10, fill: o.fill ?? P.bg_card, stroke: o.stroke ?? P.border, sw: o.sw ?? 1, opacity: o.opacity });
  }
  // Polyline arrow through points [[x,y],...], arrowhead at the end.
  function arrow(pts, color, o = {}) {
    const g = el('g', { opacity: o.opacity });
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
    const [a, b] = [pts[pts.length - 2], pts[pts.length - 1]];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), hl = o.head ?? 7, hw = hl * 0.62;
    // stop the line at the arrowhead base so the round cap does not poke through the tip
    const end = [b[0] - Math.cos(ang) * hl * 0.8, b[1] - Math.sin(ang) * hl * 0.8];
    const d2 = pts.slice(0, -1).concat([end]).map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
    g.appendChild(el('path', { d: d2, fill: 'none', stroke: color, 'stroke-width': o.sw || 1.4, 'stroke-dasharray': o.dash,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    if (o.head !== 0) {
      const p1 = [b[0] - Math.cos(ang) * hl + Math.sin(ang) * hw, b[1] - Math.sin(ang) * hl - Math.cos(ang) * hw];
      const p2 = [b[0] - Math.cos(ang) * hl - Math.sin(ang) * hw, b[1] - Math.sin(ang) * hl + Math.cos(ang) * hw];
      g.appendChild(el('path', { d: `M${b[0]} ${b[1]} L${p1[0]} ${p1[1]} L${p2[0]} ${p2[1]} Z`, fill: color }));
    }
    void d;
    return g;
  }
  // Grid of square tiles; color(i, col, row) -> fill (or [fill, opacity]).
  function tiles(x, y, cols, rows, size, gap, color) {
    const g = el('g');
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      let f = color(r * cols + c, c, r); if (f == null) continue;
      const [fill, op] = Array.isArray(f) ? f : [f, 1];
      g.appendChild(rect(x + c * (size + gap), y + r * (size + gap), size, size, { rx: Math.min(2, size / 4), fill, opacity: op }));
    }
    return g;
  }
  // ✓ and → are drawn, not typed (they are not in the font subset).
  function check(x, y, s, color, sw = 1.8) {
    return el('path', { d: `M${x} ${y + s * 0.55} L${x + s * 0.38} ${y + s * 0.9} L${x + s} ${y + s * 0.12}`, fill: 'none',
      stroke: color, 'stroke-width': sw, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
  }
  function cross(x, y, s, color, sw = 1.8) {
    return el('path', { d: `M${x} ${y} L${x + s} ${y + s} M${x + s} ${y} L${x} ${y + s}`, fill: 'none', stroke: color,
      'stroke-width': sw, 'stroke-linecap': 'round' });
  }

  // A titled card: caps label, optional title, bullet/paragraph body. Returns {g, h}.
  // If h is not given the card grows to fit.
  function box(P, x, y, w, o) {
    const pad = o.pad ?? 16, g = el('g');
    const inner = el('g'); let cy = y + pad;
    if (o.label) { inner.appendChild(text(x + pad, cy + 10, o.label, F.label(o.labelSize || 11), o.labelColor || P.muted)); cy += 22 }
    if (o.title) {
      const p = para(x + pad, cy, o.title, w - 2 * pad, F.title(o.titleSize || 17), P.text, { lh: 1.25 });
      inner.appendChild(p.g); cy += p.h + 6;
    }
    for (const b of o.bullets || []) {
      const bf = F.body(o.bodySize || 13.5);
      const indent = o.plain ? 0 : 12;
      const p = para(x + pad + indent, cy, b, w - 2 * pad - indent, bf, P.body, { lh: 1.4 });
      if (!o.plain) inner.appendChild(rect(x + pad + 1, cy + bf.size * 0.5, 4, 4, { rx: 1, fill: o.dot || P.dim }));
      inner.appendChild(p.g); cy += p.h + 4;
    }
    if (o.extra) { const e = o.extra(x + pad, cy, w - 2 * pad); inner.appendChild(e.g); cy += e.h }
    const h = Math.max(o.h || 0, cy - y + pad - 4);
    inner.setAttribute('data-box', [x + 2, y + 2, w - 4, h - 4].join(','));   // audit: text must stay inside the card
    g.appendChild(card(P, x, y, w, h, { stroke: o.stroke, fill: o.fill, sw: o.sw }));
    if (o.accent) g.appendChild(rect(x + 0.5, y + 10, 2, h - 20, { rx: 1, fill: o.accent }));
    g.appendChild(inner);
    return { g, h };
  }

  // Root svg with background.
  function svg(P, W, H, o = {}) {
    const s = el('svg', { xmlns: NS, width: o.pxW || W, height: o.pxH || H, viewBox: `0 0 ${W} ${H}`, role: 'img',
                          'aria-label': o.title || null });
    if (o.title) s.appendChild(el('title', {}, [o.title]));
    s.appendChild(rect(0, 0, W, H, { fill: P.bg, rx: o.rx ?? 0 }));
    return s;
  }

  // Layout check: text that leaves the canvas, or two text lines whose boxes overlap. Returns a list of problems.
  function audit(root) {
    const vb = root.viewBox.baseVal, out = [];
    const ts = [...root.querySelectorAll('text')].filter(t => +(t.getAttribute('opacity') ?? 1) > 0.05 && t.textContent.trim());
    // screen boxes (transforms applied), mapped back to viewBox units
    const R = root.getBoundingClientRect(), k = vb.width / R.width;
    const box = e => { const r = e.getBoundingClientRect(); return { x: (r.x - R.x) * k + vb.x, y: (r.y - R.y) * k + vb.y, width: r.width * k, height: r.height * k } };
    const bb = ts.map(t => ({ t, b: box(t) }));
    for (const { t, b } of bb) {
      if (b.x < vb.x - 0.5 || b.y < vb.y - 0.5 || b.x + b.width > vb.x + vb.width + 0.5 || b.y + b.height > vb.y + vb.height + 0.5)
        out.push('outside canvas: ' + t.textContent);
    }
    for (let i = 0; i < bb.length; i++) for (let j = i + 1; j < bb.length; j++) {
      const a = bb[i].b, c = bb[j].b;
      const ox = Math.min(a.x + a.width, c.x + c.width) - Math.max(a.x, c.x);
      const oy = Math.min(a.y + a.height, c.y + c.height) - Math.max(a.y, c.y);
      // text boxes include ascender/descender space; ignore slivers
      if (ox > 1 && oy > Math.min(a.height, c.height) * 0.35) out.push(`overlap: "${bb[i].t.textContent}" / "${bb[j].t.textContent}"`);
    }
    // text inside a [data-box] group must stay in its box
    for (const g of root.querySelectorAll('[data-box]')) {
      const [x, y, w, h] = g.getAttribute('data-box').split(',').map(Number);
      for (const t of g.querySelectorAll('text')) {
        const b = box(t);
        if (b.x < x - 0.5 || b.x + b.width > x + w + 0.5 || b.y < y - 0.5 || b.y + b.height > y + h + 0.5)
          out.push('outside its box: ' + t.textContent);
      }
    }
    return out;
  }

  window.D = { NS, palette, F, FAM, measure, wrap, el, text, para, runs, rect, card, arrow, tiles, check, cross, box, svg, audit };
})();
