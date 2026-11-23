// The four architecture diagrams. Each takes a palette (dark or light, from theme.css) and returns an <svg>.
// Facts: docs/architecture.md, docs/serving.md, docs/benchmarks.md; numbers with their sources inline.
(function () {
  const { el, text, para, runs, rect, card, arrow, tiles, box, svg, F, measure } = D;
  const at = (g, x, y) => { g.setAttribute('transform', `translate(${x},${y})`); return g };

  function header(s, P, x, label, headline, sub) {
    s.appendChild(text(x, 52, label, F.label(12), P.muted));
    s.appendChild(text(x, 92, headline, F.head(30), P.text));
    if (sub) s.appendChild(text(x, 122, sub, F.light(15), P.body));
  }
  function footer(s, P, x, y, str, w) {
    const p = para(x, y, str, w, F.body(12.5), P.muted, { lh: 1.45 });
    s.appendChild(p.g); return p.h;
  }
  // Short label on an arrow.
  function tag(s, P, x, y, str, anchor = 'middle', color) {
    s.appendChild(text(x, y, str, F.label(10), color || P.muted, { anchor }));
  }

  // ---------------------------------------------------------------- request path
  function requestPath(P) {
    const W = 1320, H = 640, s = svg(P, W, H, { title: 'Request path of the inkling-2-mini inference engine' });
    header(s, P, 40, 'REQUEST PATH', 'From an HTTP request to one CUDA graph replay per step.');
    const top = 160;
    const col = { client: [40, 150], server: [226, 232], sched: [494, 272], eng: [802, 236], ker: [1074, 206] };
    const B = (k, o, y = top) => box(P, col[k][0], y, col[k][1], o);

    const client = B('client', { label: 'CLIENT', title: 'OpenAI API', bullets: ['chat and completions', 'SSE streaming', 'tools, images, JSON schema'] });
    const server = B('server', { label: 'SERVER', title: 'server.py', bullets: [
      'chat template; reasoning split into reasoning_content',
      'tool calls; response_format through xgrammar',
      'images through the vision tower'] });
    const sched = B('sched', { label: 'SCHEDULER', title: 'scheduler.py', accent: P.blue, bullets: [
      '4 slots, continuous batching',
      'prompts prefill in 2048-token chunks between the other slots’ decode steps',
      'each request goes to the slot with the longest reusable prefix',
      'speculates when calibrated costs say it beats a plain step'],
      extra: (ex, ey) => {      // 4 slots, each a row of context tiles (filled = tokens in that slot's cache)
        const fill = [11, 5, 14, 8];
        const g = tiles(ex, ey + 8, 20, 4, 8, 3, (i, c, r) => c < fill[r] ? [P.blue, 0.85] : [P.grey, 0.18]);
        return { g, h: 4 * 11 + 8 };
      } });
    const pre = B('eng', { label: 'PREFILL', title: 'engine.py', bullets: [
      '2048-token chunks; cuBLAS projections, grouped expert GEMMs',
      'counting-sort expert tiles, no host sync',
      'chunks of up to 32 tokens replay a graph'] });
    const decY = top + pre.h + 22;
    const dec = B('eng', { label: 'DECODE', title: 'one CUDA graph replay per step', accent: P.blue, bullets: [
      'graphs for 1 to 4 requests and for verify widths',
      'DSpark drafts 7 tokens, the target verifies'] }, decY);
    const ker = B('ker', { label: 'KERNELS · ops/', title: 'Triton, no custom\u00a0C++', bullets: [
      'attention: split-K, sinks, gate',
      'MoE: FP4 GEMV, grouped tiles, FP4 tensor cores',
      'GEMV bf16 / FP8, short convs',
      'fused per-token glue, PDL'] });
    const cacheY = top + ker.h + 22;
    const cache = B('ker', { label: 'CACHES · cache.py', title: 'static, per slot', bullets: [
      'full attention: contiguous KV',
      'sliding window: ring of 1088',
      'conv states, prefix snapshots'] }, cacheY);
    for (const b of [client, server, sched, pre, dec, ker, cache]) s.appendChild(b.g);

    const mid = (k) => [col[k][0], col[k][0] + col[k][1]];
    const A = (pts, c = P.muted, o) => s.appendChild(arrow(pts, c, o));
    const yRow = top + 48;
    A([[mid('client')[1] + 4, yRow], [mid('server')[0] - 4, yRow]]);
    A([[mid('server')[1] + 4, yRow], [mid('sched')[0] - 4, yRow]]);
    // scheduler fans out to prefill and decode
    const fx = mid('sched')[1] + 16;
    A([[mid('sched')[1] + 4, yRow], [mid('eng')[0] - 4, yRow]]);
    A([[fx, yRow], [fx, decY + 48], [mid('eng')[0] - 4, decY + 48]], P.blue);
    s.appendChild(el('circle', { cx: fx, cy: yRow, r: 2.5, fill: P.muted }));
    // engine to kernels
    A([[mid('eng')[1] + 4, yRow], [mid('ker')[0] - 4, yRow]]);
    A([[mid('eng')[1] + 4, decY + 48], [mid('eng')[1] + 18, decY + 48], [mid('eng')[1] + 18, yRow + 30], [mid('ker')[0] - 4, yRow + 30]], P.blue);
    // kernels read and write the caches
    const cx = col.ker[0] + col.ker[1] / 2;
    A([[cx - 10, top + ker.h + 3], [cx - 10, cacheY - 3]], P.muted, { head: 6 });
    A([[cx + 10, cacheY - 3], [cx + 10, top + ker.h + 3]], P.muted, { head: 6 });
    tag(s, P, cx - 20, top + ker.h + 15, 'read / write', 'end');

    // tokens stream back
    const by = Math.max(decY + dec.h, cacheY + cache.h, top + sched.h) + 46;
    const dx = col.eng[0] + 40;
    A([[dx, decY + dec.h + 3], [dx, by], [col.server[0] + 40, by], [col.server[0] + 40, top + server.h + 4]], P.blue, { dash: '4 4' });
    tag(s, P, (dx + col.server[0] + 40) / 2, by - 9, 'sampled tokens stream back over SSE');
    footer(s, P, 40, by + 26, 'Python and Triton on PyTorch. Defaults: 4 slots × 32k context, DSpark and image input on. ' +
      'Flags and API: docs/serving.md.', W - 80);
    s.setAttribute('viewBox', `0 0 ${W} ${by + 64}`); s.setAttribute('height', by + 64);
    s.querySelector('rect').setAttribute('height', by + 64);
    return s;
  }

  // ---------------------------------------------------------------- decode step
  // Visit plan (config.json loop_config): prelude layers 0-5, core 6-17 three times, coda 18-23 = 48 visits.
  // Full-attention layers are 5, 11, 17, 23 (local_layer_ids lists the other 20). DSpark taps visits 5, 23, 41, 48.
  const VISITS = [];
  for (let l = 0; l < 6; l++) VISITS.push({ layer: l, grp: 0 });
  for (let p = 0; p < 3; p++) for (let l = 6; l < 18; l++) VISITS.push({ layer: l, grp: 1 + p });
  for (let l = 18; l < 24; l++) VISITS.push({ layer: l, grp: 4 });
  const FULL = new Set([5, 11, 17, 23]), TAPS = [5, 23, 41, 48];

  function decodeStep(P) {
    const W = 1320, s = svg(P, W, 900, { title: 'One decode step of Inkling-2-Mini in the engine' });
    header(s, P, 40, 'ONE DECODE STEP', '48 layer visits, ~12 fused kernels each, one CUDA graph replay.');
    const A = (pts, c = P.muted, o) => s.appendChild(arrow(pts, c, o));

    // --- the visit strip
    const sy = 420, ch = 40, x0 = 182, x1 = 1112, gg = 16;
    const cw = (x1 - x0 - 4 * gg) / 48;
    const cx = (i) => x0 + i * cw + VISITS[i].grp * gg;          // left edge of visit i (0-based)
    const grpX = (g) => [cx(VISITS.findIndex(v => v.grp === g)), cx(VISITS.length - 1 - [...VISITS].reverse().findIndex(v => v.grp === g)) + cw];
    // graph bracket
    s.appendChild(rect(26, sy - 64, W - 52, ch + 120, { rx: 12, stroke: P.border, sw: 1, dash: '3 5' }));
    s.appendChild(text(40, sy - 42, 'ONE CUDA GRAPH REPLAY', F.label(10.5), P.blue, { halo: P.bg }));
    VISITS.forEach((v, i) => {
      const loop = v.grp >= 1 && v.grp <= 3, full = FULL.has(v.layer);
      s.appendChild(rect(cx(i) + 1, sy, cw - 3, ch, { rx: 2.5, fill: loop ? P.orange : P.blue, fo: full ? 0.9 : 0.28,
        stroke: loop ? P.orange : P.blue, sw: 1, opacity: 1 }));
    });
    // group labels
    const glab = ['PRELUDE', 'LOOP PASS 1', 'LOOP PASS 2', 'LOOP PASS 3', 'CODA'];
    const gsub = ['layers 0-5', 'layers 6-17', 'layers 6-17', 'layers 6-17', 'layers 18-23'];
    for (let g = 0; g < 5; g++) {
      const [a, b] = grpX(g), m = (a + b) / 2;
      s.appendChild(text(m, sy + ch + 20, glab[g], F.label(10.5), g >= 1 && g <= 3 ? P.orange : P.blue, { anchor: 'middle', halo: P.bg }));
      s.appendChild(text(m, sy + ch + 38, gsub[g], F.body(12.5), P.muted, { anchor: 'middle', halo: P.bg }));
    }
    // loop-back arcs with gated messages
    for (let g = 1; g <= 2; g++) {
      const a = grpX(g)[1] - cw / 2, b = grpX(g + 1)[0] + cw / 2;
      const start = grpX(g)[0] + cw / 2;
      void start;
      s.appendChild(el('path', { d: `M${a} ${sy - 3} C ${a} ${sy - 30}, ${b} ${sy - 30}, ${b} ${sy - 6}`, fill: 'none', stroke: P.orange, 'stroke-width': 1.4 }));
      s.appendChild(el('path', { d: `M${b - 4} ${sy - 12} L${b} ${sy - 4} L${b + 4} ${sy - 12} Z`, fill: P.orange }));
    }
    s.appendChild(text((grpX(2)[0] + grpX(3)[1]) / 2, sy - 36, 'GATED RANK-64 MESSAGES BETWEEN PASSES', F.label(10), P.orange, { anchor: 'middle', halo: P.bg }));
    // visit numbers
    for (const n of [1, 6, 18, 30, 42, 48]) s.appendChild(text(cx(n - 1) + cw / 2 - 1, sy + ch / 2 + 4, String(n), F.code(10), P.light ? '#ffffff' : P.text, { anchor: 'middle', opacity: 0 }));
    // embedding and LM head
    const emb = box(P, 40, sy - 12, 112, { label: 'TOKEN', title: 'embed', titleSize: 15, h: 64, pad: 12 });
    s.appendChild(emb.g);
    A([[154, sy + ch / 2], [x0 - 4, sy + ch / 2]]);
    const lm = box(P, 1142, sy - 12, 138, { label: 'LM HEAD', title: 'softcap 30', titleSize: 15, h: 64, pad: 12 });
    s.appendChild(lm.g);
    A([[x1 + 4 * 0 + 4, sy + ch / 2], [1138, sy + ch / 2]]);

    // legend for the strip
    const ly = sy + ch + 70;
    s.appendChild(rect(182, ly - 10, 12, 12, { rx: 2, fill: P.blue, fo: 0.28, stroke: P.blue }));
    s.appendChild(text(200, ly, 'sliding-window attention (1024)', F.body(12.5), P.body, { halo: P.bg }));
    s.appendChild(rect(410, ly - 10, 12, 12, { rx: 2, fill: P.blue, fo: 0.9, stroke: P.blue }));
    s.appendChild(text(428, ly, 'full attention', F.body(12.5), P.body, { halo: P.bg }));
    s.appendChild(rect(540, ly - 10, 12, 12, { rx: 2, fill: P.orange, fo: 0.28, stroke: P.orange }));
    s.appendChild(text(558, ly, 'loop core visit', F.body(12.5), P.body, { halo: P.bg }));
    s.appendChild(el('circle', { cx: 680, cy: ly - 4, r: 4, fill: P.orange }));
    s.appendChild(text(690, ly, 'hidden state read by the drafter', F.body(12.5), P.body, { halo: P.bg }));

    // --- one visit, zoomed (above the strip)
    const zi = 20, zy = 150, zh = 180;               // zoom from visit 21 (loop pass 2, layer 8)
    s.appendChild(el('path', { d: `M${cx(zi) + 1} ${sy - 2} L40 ${zy + zh + 2} M${cx(zi) + cw - 2} ${sy - 2} L${W - 40} ${zy + zh + 2}`,
      stroke: P.dim, 'stroke-width': 1, fill: 'none', 'stroke-dasharray': '2 4', opacity: 0.8 }));
    s.appendChild(rect(cx(zi) - 1, sy - 2, cw + 1, ch + 4, { rx: 3, stroke: P.text, sw: 1.2 }));
    const steps = [
      ['DEPTH ATTENTION', 'input from the block residuals, plus the gated messages on passes 2 and 3'],
      ['PROJECTIONS', 'q, gate, bias, sinks, k, v and conv taps in one GEMV'],
      ['ATTENTION', 'short convs on K, V; split-K over the cache; sink and sigmoid gate'],
      ['ROUTER', 'router GEMV and top-8 selection in one launch'],
      ['EXPERTS', null],
      ['POST-MLP', 'sum of the 8 expert rows, MLP conv, residual'],
    ];
    const n = steps.length, gap = 26, bw = (W - 80 - (n - 1) * gap) / n;
    s.appendChild(rect(40, zy - 6, W - 80, zh + 8, { rx: 12, fill: P.bg_card, stroke: P.border, fo: 0.6 }));
    s.appendChild(text(56, zy + 16, 'ONE LAYER VISIT · ~12 FUSED LAUNCHES', F.label(10.5), P.muted));
    steps.forEach(([lab, body], i) => {
      const x = 56 + i * (bw + gap) - i * (32 / n), w = bw - 32 / n;
      const isEx = lab === 'EXPERTS';
      const b = box(P, x, zy + 30, w, { label: lab, labelSize: 10.5, plain: true, bodySize: 12.5, pad: 12, h: zh - 44,
        stroke: isEx ? P.blue : undefined,
        bullets: body ? [body] : ['8 of 128, NVFP4 or MXFP8, SwiGLU fused'],
        extra: isEx ? (ex, ey, ew) => {
          const g = tiles(ex, ey + 6, 16, 8, 6.5, 2.2, (k) => [P.blue, [3, 17, 38, 52, 77, 90, 101, 124].includes(k) ? 1 : 0.13].map((v, j) => j ? v : (v)) );
          return { g, h: 76 };
        } : null });
      // grey the idle experts
      s.appendChild(b.g);
      if (i < n - 1) A([[x + w + 3, zy + 30 + 30], [x + w + gap - 32 / n - 3, zy + 30 + 30]], P.muted, { head: 6 });
    });

    // --- bottom row: drafter, verify, sampler
    const by = 600;
    const busY = 560;
    const tapX = TAPS.map(t => cx(t - 1) + cw / 2);
    for (const x of tapX) {
      s.appendChild(el('circle', { cx: x, cy: sy + ch + 4 + 46 - 46, r: 0 }));
    }
    // taps: dots under the visits, lines down to a bus
    tapX.forEach((x, k) => {
      s.appendChild(el('path', { d: `M${x} ${sy + ch + 1} L${x} ${busY}`, stroke: P.orange, 'stroke-width': 1.2, fill: 'none', 'stroke-dasharray': '2 3' }));
      s.appendChild(el('circle', { cx: x, cy: sy + ch + 1, r: 3.6, fill: P.orange }));
    });
    const dr = { x: 182, w: 330 }, vf = { x: 572, w: 330 }, sp = { x: 962, w: 318 };
    s.appendChild(el('path', { d: `M${dr.x + dr.w / 2} ${busY} L${tapX[3]} ${busY}`, stroke: P.orange, 'stroke-width': 1.2, fill: 'none', 'stroke-dasharray': '2 3' }));
    A([[dr.x + dr.w / 2, busY], [dr.x + dr.w / 2, by - 4]], P.orange);
    s.appendChild(text(dr.x + dr.w / 2 + 10, busY - 8, 'VISITS 5, 23, 41, 48', F.label(10), P.orange));
    const drafter = box(P, dr.x, by, dr.w, { label: 'DSPARK DRAFTER', title: 'drafts the next 7 tokens in one pass', titleSize: 15.5,
      accent: P.orange, bullets: ['3 bidirectional block layers and a Markov RNN head', 'candidates from a 32k-token shortlist'],
      extra: (ex, ey) => ({ g: tiles(ex, ey + 4, 7, 1, 16, 5, () => P.orange), h: 26 }) });
    const verify = box(P, vf.x, by, vf.w, { label: 'VERIFY (NEXT STEP)', title: 'the target checks anchor + drafts', titleSize: 15.5,
      accent: P.green, bullets: ['width 2, 4 or 8, chosen from measured graph costs', 'greedy: longest matching prefix; sampled: rejection sampling, exact'],
      extra: (ex, ey) => ({ g: tiles(ex, ey + 4, 7, 1, 16, 5, (k) => k < 4 ? P.green : k === 4 ? P.red : [P.grey, 0.35]), h: 26 }) });
    const sampler = box(P, sp.x, by, sp.w, { label: 'SAMPLER', title: 'temperature, top-k, top-p, penalties', titleSize: 15.5,
      bullets: ['greedy or sampled token per row', 'tokens stream to the client'] });
    for (const b of [drafter, verify, sampler]) s.appendChild(b.g);
    const rowMid = by + 40;
    A([[dr.x + dr.w + 4, rowMid], [vf.x - 4, rowMid]], P.orange);
    A([[1211, sy + 52 + 4], [1211, by - 4]]);
    A([[sp.x - 4, rowMid + 30], [vf.x + vf.w + 4, rowMid + 30]], P.muted);
    tag(s, P, (vf.x + vf.w + sp.x) / 2, rowMid + 22, 'logits');

    const fy = by + Math.max(drafter.h, verify.h, sampler.h) + 32;
    footer(s, P, 40, fy, 'A plain decode step is one CUDA graph replay (graphs for 1 to 4 requests); a verify step replays a graph for ' +
      '(sequences, width). Source: docs/architecture.md. Visit plan and attention layers from the checkpoint’s config.json.', W - 80);
    const H = fy + 50;
    for (const t of [...s.querySelectorAll('text[paint-order]')]) s.appendChild(t);   // haloed labels sit above the lines
    s.setAttribute('viewBox', `0 0 ${W} ${H}`); s.setAttribute('height', H); s.querySelector('rect').setAttribute('height', H);
    return s;
  }

  // ---------------------------------------------------------------- memory layout
  // Totals are measured at the server defaults (FP8 dense weights, FP4 prefill, vision): tests/gpu/memcheck.py runs both
  // profiles (default: 25.9 GiB after start, 27.4 peak; --kv-fp8 --max-ctx 262144 --slots 1: 24.9 / 27.1),
  // docs/benchmarks.md. Segments are computed:
  //   checkpoint tensors (model-00001-of-00001.safetensors header): experts 14.59 GiB, embedding + LM head 1.73,
  //   other dense weights 1.27, vision tower 0.59; drafter mtp.safetensors 0.59.
  //   FP8 dense (fused_model.py, compact()): the padded attention projections (207,872 rows x 2304) and wo
  //   (24 x 2304 x 3072) become E4M3 with a 4 B scale per row and the bf16 copies are dropped (-0.445, -0.158 GiB:
  //   other dense 1.27 -> 0.67); the LM head keeps bf16 and adds an FP8 copy (201,024 x 2304 + scales, +0.432 GiB:
  //   embedding + LM head 1.73 -> 2.16).
  //   KV (cache.py CacheSpec.bytes): 8 full-attention visits x 8 KV heads x 128 x (K, V) = 16 Ki values per token:
  //   bf16 32 KiB/token x 32k x 4 slots = 4.0 GiB; FP8 16 KiB/token x 256k x 1 slot = 4.0 GiB.
  //   Sliding rings: 40 visits x 1088 positions x 1024 x (K, V) x bf16 = 0.166 GiB per slot.
  //   "graphs, workspaces, snapshots" = measured after-start total minus the computed segments.
  const MEM = {
    def: { title: 'DEFAULT SERVER', sub: '4 slots × 32k context, bf16 KV, DSpark, vision', after: 25.9, peak: 27.4,
      segs: [['expert weights', 14.59, 'w', ['expert weights', '14.6 GiB']], ['embedding + LM head', 2.16, 'w2'], ['other dense', 0.67, 'w3'], ['vision tower', 0.59, 'w3'],
             ['drafter', 0.59, 'd'], ['KV', 1.0, 'kv', ['32k', '1.0']], ['KV', 1.0, 'kv', ['32k', '1.0']], ['KV', 1.0, 'kv', ['32k', '1.0']], ['KV', 1.0, 'kv', ['32k', '1.0']],
             ['sliding rings', 0.664, 'ring']] },
    long: { title: 'LONG CONTEXT', sub: '--kv-fp8 --max-ctx 262144 --slots 1, vision', after: 24.9, peak: 27.1,
      segs: [['expert weights', 14.59, 'w', ['expert weights', '14.6 GiB']], ['embedding + LM head', 2.16, 'w2'], ['other dense', 0.67, 'w3'], ['vision tower', 0.59, 'w3'],
             ['drafter', 0.59, 'd'], ['KV', 4.0, 'kv', ['one 256k slot, FP8', '4.0 GiB']], ['sliding ring', 0.166, 'ring']] },
  };
  function memoryLayout(P) {
    const W = 1320, s = svg(P, W, 800, { title: 'GPU memory layout of the engine in a 32 GiB budget' });
    header(s, P, 40, 'MEMORY · 32 GiB BUDGET', 'Four 32k contexts, or one 256k context, in 32 GiB.');
    const X0 = 40, X1 = W - 40, k = (X1 - X0) / 32, gx = (g) => X0 + g * k;
    const fills = { w: [P.grey, 0.55], w2: [P.grey, 0.4], w3: [P.grey, 0.28], d: [P.orange, 0.75], kv: [P.blue, 0.85], ring: [P.blue, 0.45] };
    const hatchId = 'hatch-' + (P.light ? 'l' : 'd');
    const defs = el('defs', {}, [el('pattern', { id: hatchId, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' },
      [el('rect', { width: 6, height: 6, fill: P.bg_card }), el('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: P.dim, 'stroke-width': 2 })])]);
    s.appendChild(defs);

    function bar(m, y) {
      s.appendChild(text(X0, y, m.title, F.label(12), P.text));
      s.appendChild(text(X0 + D.measure(m.title, F.label(12)) + 14, y, m.sub, F.code(12.5), P.muted));
      const by = y + 16, bh = 56;
      s.appendChild(rect(X0, by, X1 - X0, bh, { rx: 6, fill: P.bg_card, stroke: P.border }));
      let g = 0; const below = [];
      for (const [name, gib, kind, inner] of m.segs) {
        const [f, op] = fills[kind];
        const x = gx(g) + 1, w = gib * k - 2;
        s.appendChild(rect(x, by + 1, w, bh - 2, { rx: 3, fill: f, fo: op }));
        const num = gib.toFixed(gib < 0.95 ? 2 : 1);
        const white = kind === 'kv' || kind === 'd' || !P.light;
        if (inner) {        // [line 1, line 2] drawn inside the segment
          const mid = w < 60;
          s.appendChild(text(mid ? x + w / 2 : x + 9, by + 24, inner[0], F.body(13), white ? '#ffffff' : P.text, { anchor: mid ? 'middle' : null }));
          s.appendChild(text(mid ? x + w / 2 : x + 9, by + 42, inner[1], F.code(11.5), white ? '#ffffffcc' : P.body, { anchor: mid ? 'middle' : null }));
        } else below.push([x + w / 2, name, num]);
        g += gib;
      }
      // the rest of the measured after-start total
      const other = m.after - g;
      s.appendChild(rect(gx(g) + 1, by + 1, other * k - 2, bh - 2, { rx: 3, fill: P.grey, fo: 0.18, stroke: P.grey, sw: 1, dash: '3 3' }));
      below.push([gx(g) + other * k / 2, 'graphs, workspaces, snapshots', other.toFixed(1)]);
      s.appendChild(rect(gx(m.after) + 1, by + 1, (m.peak - m.after) * k - 2, bh - 2, { rx: 3, fill: `url(#${hatchId})` }));
      for (const [v, c] of [[m.after, P.text], [m.peak, P.muted]])
        s.appendChild(el('path', { d: `M${gx(v)} ${by - 6} L${gx(v)} ${by + bh + 6}`, stroke: c, 'stroke-width': 1.4 }));
      s.appendChild(text(gx(m.after) - 6, by - 10, 'after start ' + m.after.toFixed(1) + ' GiB', F.code(11.5), P.text, { anchor: 'end' }));
      s.appendChild(text(gx(m.peak) + 6, by - 10, 'peak ' + m.peak.toFixed(1), F.code(11.5), P.muted));
      s.appendChild(text(gx(32) - 8, by + 34, 'free ' + (32 - m.peak).toFixed(1), F.code(11.5), P.muted, { anchor: 'end' }));
      // Callouts: one row of labels in segment order, with elbow leaders that never cross. A leader whose label
      // sits left of its segment turns lower the farther right it starts; one whose label sits right turns lower
      // the farther left it starts.
      const lf = F.body(12.5), nf = F.code(11.5), gapL = 26;
      const widths = below.map(([, n, v]) => D.measure(n, lf) + D.measure(' ' + v + ' GiB', nf));
      const total = widths.reduce((a, b) => a + b, 0) + gapL * (below.length - 1);
      const centre = (below[0][0] + below[below.length - 1][0]) / 2;
      let lx = Math.min(Math.max(centre - total / 2, X0), X1 - total);
      const lxs = widths.map(w => { const x = lx; lx += w + gapL; return x });
      const step = 7, base = by + bh + 10;
      const left = below.map(([cx], i) => lxs[i] + 10 < cx);
      const li = below.map((_, i) => i).filter(i => left[i]), ri = below.map((_, i) => i).filter(i => !left[i]);
      const lvl = {};
      li.forEach((i, r) => { lvl[i] = r });                       // rightmost left-goer lowest
      ri.slice().reverse().forEach((i, r) => { lvl[i] = r });     // leftmost right-goer lowest
      const levels = Math.max(li.length, ri.length);
      const labY = base + step * levels + 18;
      below.forEach(([cx, n, v], i) => {
        const tx = lxs[i] + 10, yt = base + step * lvl[i];
        s.appendChild(el('path', { d: `M${cx} ${by + bh + 2} L${cx} ${yt} L${tx} ${yt} L${tx} ${labY - 13}`, fill: 'none', stroke: P.dim, 'stroke-width': 1 }));
        s.appendChild(el('circle', { cx, cy: by + bh + 2, r: 2, fill: P.dim }));
        s.appendChild(runs(lxs[i], labY, [[n, lf, P.body], [' ' + v + ' GiB', nf, P.muted]]));
      });
      return labY + 12;
    }
    // axis
    const ay = 190;
    for (let v = 0; v <= 32; v += 4) {
      s.appendChild(el('path', { d: `M${gx(v)} ${ay} L${gx(v)} ${ay + 6}`, stroke: P.dim, 'stroke-width': 1 }));
      s.appendChild(text(gx(v), ay - 6, v + (v === 32 ? ' GiB' : ''), F.code(11), P.muted, { anchor: v === 0 ? 'start' : v === 32 ? 'end' : 'middle' }));
    }
    let y = bar(MEM.def, 236);
    y = bar(MEM.long, y + 44);
    // the equation
    const ey = y + 30;
    const eq = box(P, X0, ey, X1 - X0, { label: 'WHY BOTH FIT', plain: true, title: '4 × 32k tokens × 32 KiB (bf16)  =  1 × 256k tokens × 16 KiB (FP8)  =  4.0 GiB of KV',
      titleSize: 17, bodySize: 13, accent: P.blue, bullets: [
      'Only the 8 full-attention visits keep KV for the whole context (layer 5, layers 11 and 17 in each of the 3 loop passes, layer 23): 8 visits × 8 KV heads × 128 × (K, V) per token. ' +
      'The 40 sliding-window visits keep a ring of 1088 positions per slot, whatever the context length.'] });
    s.appendChild(eq.g);
    const fy = ey + eq.h + 24;
    footer(s, P, 40, fy, 'Totals measured on an RTX PRO 6000 with a 32 GiB cap at the server defaults (tests/gpu/memcheck.py; ' +
      'docs/benchmarks.md). Segments computed from the checkpoint’s tensor sizes and cache.py; the dashed segment is the measured total minus them.', W - 80);
    const H = fy + 50;
    for (const t of [...s.querySelectorAll('text[paint-order]')]) s.appendChild(t);
    s.setAttribute('viewBox', `0 0 ${W} ${H}`); s.setAttribute('height', H); s.querySelector('rect').setAttribute('height', H);
    return s;
  }

  // ---------------------------------------------------------------- kernel fusion
  // Launch order of one decode visit: fused_model.py forward(); what each kernel does: docs/architecture.md "Decode
  // step"; times: docs/experiments.md (depth attention 3.7 us, attention 9.7 us, split combine 1.4 us, decode MoE 28.7 us).
  const LAUNCHES = [
    ['depth_attn', 'attention input', ['depth attention over the block residuals', 'RMS norm'], '3.7 µs'],
    ['seg_gemv', 'projections', ['q', 'gate', 'relative bias', 'sinks', 'k', 'v', 'conv taps', 'up to 3 inputs, 1 launch'], ''],
    ['qkv_post', 'attention prep', ['q/k clip', 'log-scaling tau', 'short conv K', 'short conv V', 'V norm', 'cache write'], ''],
    ['attention', 'split-K attention', ['cache + new tokens', 'relative bias', 'sink', 'sigmoid gate'], '9.7 µs'],
    ['combine', 'split merge', ['merge the split-K partials'], '1.4 µs'],
    ['o_proj', 'output GEMV', ['bf16, or FP8 with --dense-fp8'], ''],
    ['post_attn', 'residual', ['output conv', 'residual', 'MLP input'], ''],
    ['route', 'router', ['router GEMV', 'top-8 of 128', 'routing weights'], ''],
    ['moe w13', 'experts, up', ['8 expert GEMVs', 'FP4 hardware decode', 'gate/up interleaved', 'SwiGLU'], '28.7 µs'],
    ['moe w2', 'experts, down', ['8 expert GEMVs'], ''],
    ['post_mlp', 'residual', ['sum of 8 expert rows', 'MLP conv', 'residual'], ''],
  ];
  function kernelFusion(P) {
    const W = 1320, s = svg(P, W, 900, { title: 'The fused kernels of one decode layer visit' });
    header(s, P, 40, 'KERNEL FUSION · ONE LAYER VISIT', 'About 12 launches per visit, 48 visits, one graph replay.');
    const x0 = 40, top = 160, rh = 44, nameW = 130, roleW = 140, tx = W - 330;
    s.appendChild(text(x0 + 36, top - 14, 'KERNEL', F.label(10.5), P.muted));
    s.appendChild(text(x0 + 36 + nameW, top - 14, 'STEP', F.label(10.5), P.muted));
    s.appendChild(text(x0 + 36 + nameW + roleW, top - 14, 'FUSED INTO THE ONE LAUNCH', F.label(10.5), P.muted));
    s.appendChild(text(tx - 24, top - 14, 'TIME', F.label(10.5), P.muted, { anchor: 'end' }));
    LAUNCHES.forEach(([name, role, parts, t], i) => {
      const y = top + i * rh;
      const moe = name.startsWith('moe');
      s.appendChild(card(P, x0, y, tx - 10 - x0, rh - 8, { rx: 8, stroke: moe ? P.blue : P.border }));
      s.appendChild(text(x0 + 14, y + 23, String(i + 1).padStart(2, '0'), F.code(11.5), P.dim));
      s.appendChild(text(x0 + 36, y + 23, name, F.code(13), moe ? (P.light ? P.blue : P.blue_glow) : P.text));
      s.appendChild(text(x0 + 36 + nameW, y + 23, role, F.body(13), P.muted));
      let cx = x0 + 36 + nameW + roleW;
      for (const part of parts) {
        const f = F.body(12.5), w = D.measure(part, f) + 18;
        s.appendChild(rect(cx, y + 7, w, rh - 22, { rx: (rh - 22) / 2, fill: moe ? P.blue : P.grey, fo: moe ? 0.18 : 0.14 }));
        s.appendChild(text(cx + 9, y + 23, part, f, P.body));
        cx += w + 6;
      }
      if (t && name !== 'moe w13') s.appendChild(text(tx - 24, y + 23, t, F.code(12.5), P.text, { anchor: 'end' }));
    });
    // MoE time spans both expert launches: bracket in the gap right of the cards
    const my = top + 8 * rh, bx = tx - 4;
    s.appendChild(el('path', { d: `M${bx} ${my + 6} L${bx + 5} ${my + 6} L${bx + 5} ${my + rh * 2 - 14} L${bx} ${my + rh * 2 - 14}`, fill: 'none', stroke: P.blue, 'stroke-width': 1.2 }));
    s.appendChild(text(tx - 24, my + 23, '28.7 \u00b5s', F.code(12.5), P.light ? P.blue : P.blue_glow, { anchor: 'end' }));
    s.appendChild(text(tx - 24, my + rh + 23, 'w13 + w2', F.code(11), P.muted, { anchor: 'end' }));
    const endY = top + LAUNCHES.length * rh;
    const lp = para(x0, endY + 8, 'Loop passes 2 and 3 add the gated rank-64 messages: a two-input depth attention and two message-gate launches ' +
      'in front of the projections and of the MLP. With fewer than 4 tokens the experts run as GEMVs; verify steps of 4 to 32 tokens ' +
      'use grouped expert tiles, so each expert’s weights are read once for all tokens.', tx - 10 - x0, F.body(13), P.body, { lh: 1.45 });
    s.appendChild(lp.g);

    // PDL panel
    const px = tx + 10, pw = W - 40 - px;
    const pdl = box(P, px, top - 30, pw, { label: 'PDL \u00b7 PROGRAMMATIC LAUNCH', title: 'Each kernel starts before the last one ends',
      titleSize: 16, accent: P.blue, plain: true, bodySize: 13, bullets: [
        'It prefetches its weights or cached K/V into L2, then waits on its predecessor. None of that load depends on the previous kernel.'],
      extra: (ex, ey, ew) => {
        const g = el('g'); const bh = 18, gap = 9;
        const ks = [['seg_gemv', 0, 0.3], ['qkv_post', 0.25, 0.42], ['attention', 0.37, 0.66], ['combine', 0.61, 0.72]];
        ks.forEach(([n, a, b], i) => {
          const y = ey + 14 + i * (bh + gap), x = ex + a * ew, w = (b - a) * ew;
          const pre = Math.min(0.07 * ew, w * 0.35);
          if (i) g.appendChild(rect(x, y, pre, bh, { rx: 3, fill: P.blue, fo: 0.28 }));
          g.appendChild(rect(x + (i ? pre : 0), y, w - (i ? pre : 0), bh, { rx: 3, fill: P.blue, fo: 0.85 }));
          g.appendChild(text(x + w + 6, y + 13, n, F.code(11), P.muted));
        });
        const ly = ey + 14 + ks.length * (bh + gap) + 8;
        g.appendChild(rect(ex, ly, 12, 12, { rx: 2, fill: P.blue, fo: 0.28 }));
        g.appendChild(text(ex + 18, ly + 10, 'prefetch into L2', F.body(12), P.body));
        g.appendChild(rect(ex + 140, ly, 12, 12, { rx: 2, fill: P.blue, fo: 0.85 }));
        g.appendChild(text(ex + 158, ly + 10, 'after the wait', F.body(12), P.body));
        return { g, h: ly + 20 - ey };
      } });
    s.appendChild(pdl.g);
    const res = box(P, px, top - 30 + pdl.h + 16, pw, { label: 'MEASURED \u00b7 bench_pdl.py', plain: true, bodySize: 13,
      title: 'decode step: 6.08 to 5.66 ms', titleSize: 16, bullets: ['verify step, width 8: 8.68 to 8.27 ms'] });
    s.appendChild(res.g);

    const fy = Math.max(endY + 8 + lp.h, top - 30 + pdl.h + 16 + res.h) + 28;
    footer(s, P, 40, fy, 'Launch order from fused_model.py; times on the RTX PRO 6000 proxy from docs/experiments.md (one launch, decode). ' +
      'The whole decode step, 48 visits and the LM head, is captured once as a CUDA graph and replayed.', W - 80);
    const H = fy + 50;
    s.setAttribute('viewBox', `0 0 ${W} ${H}`); s.setAttribute('height', H); s.querySelector('rect').setAttribute('height', H);
    return s;
  }

  window.DIAGRAMS = { 'request-path': requestPath, 'decode-step': decodeStep, 'memory-layout': memoryLayout, 'kernel-fusion': kernelFusion };
})();
