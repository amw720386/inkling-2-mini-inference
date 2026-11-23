// Race scene: draws the frame at time t from window.RACE (race-data.js). Shared by scenes/scene2-race.html and
// scenes/gif-race.html; window.RACE_OPTS = {scale, gif} may override the type scale and the layout.
(function () {
  const R = window.RACE, { $, ramp, fadeUp, wide } = S;
  const O = window.RACE_OPTS || {};
  if (O.scale) document.documentElement.style.setProperty('--s', String(O.scale));
  if (O.gif) document.body.classList.add('gif');
  if (wide) document.body.classList.add('wide');
  const T0 = R.layout.t0;                        // scene time at which every request is sent
  const tDone = Math.max(...R.engines.map(e => e.t_end));
  const tPunch = T0 + tDone + R.layout.punch_delay;
  window.SCENE = { duration: O.gif ? R.layout.gif_duration : R.layout.duration };

  $('#headline').textContent = R.headline;
  $('#question').textContent = R.question;
  $('#punchline').textContent = R.punchline;
  $('#punchsub').textContent = R.punchsub || '';
  $('#foot').innerHTML = R.footer.map(l => `<div class="foot">${l}</div>`).join('');
  const check = '<svg viewBox="0 0 16 16"><path d="M2.5 8.5l3.6 3.6 7.4-8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  const cards = R.engines.map(e => {
    const el = document.createElement('div');
    el.className = 'card' + (e.ours ? ' ours' : '');
    el.innerHTML = `<div class="top"><div><div class="lab">${e.label}</div><div class="name">${e.name}</div></div>
      <div class="right"><div class="status"></div><div class="timer"></div></div></div>
      <div class="bar"><div class="fill"></div></div><div class="meta"></div>
      <div class="textrow"><div class="stream"><span class="th"></span><span class="ans"></span></div>
      <div class="pill">${check}<span>correct</span></div></div>`;
    $('#cards').appendChild(el);
    const q = s => el.querySelector(s);
    return { e, status: q('.status'), timer: q('.timer'), fill: q('.fill'), meta: q('.meta'), stream: q('.stream'),
             th: q('.th'), ans: q('.ans'), pill: q('.pill') };
  });

  // Last event at or before request time u: [t, tokens, chars].
  function at(ev, u) {
    let lo = 0, hi = ev.length - 1, k = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (ev[m][0] <= u) { k = m; lo = m + 1 } else hi = m - 1 }
    return k < 0 ? [0, 0, 0] : ev[k];
  }

  // The answer is markdown: show **bold** as bold, and hide a lone ** while its closing pair is still streaming.
  const esc = x => x.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // Inline LaTeX is shown the way a chat UI renders it: \( \) and \[ \] dropped, \times -> ×, \frac{a}{b} -> a/b.
  const tex = x => {
    x = x.replace(/\\[()\[\]]/g, '').replace(/\\times/g, '×').replace(/\\cdot/g, '·').replace(/\\div/g, '÷')
      .replace(/\{,\}/g, ',').replace(/\\([$%])/g, '$1').replace(/\\[,;!]/g, ' ');
    for (let k = 0; k < 4; k++)  // innermost first: \boxed{4 \text{ minutes}}
      x = x.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '$1/$2').replace(/\\(?:text|mathrm|boxed)\{([^{}]*)\}/g, '$1');
    // a group or command still streaming in: drop its opener / the partial command at the end
    return x.replace(/\\(?:text|mathrm|boxed)\{/g, '').replace(/\\[a-zA-Z]*$/, '').replace(/\{,?$/, '');
  };
  const md = x => esc(tex(x)).replace(/\*\*(.+?)\*\*/gs, '<strong>$1</strong>').replace(/\*\*/g, '');

  // Show the tail of the streamed text: one line in 1:1 (drop leading characters until it fits), the box in 16:9.
  function setText(c, chars) {
    const e = c.e, cut = Math.min(chars, e.text.length);
    const thEnd = Math.min(cut, e.answer_start);
    let from = 0;
    const put = f => {
      c.th.textContent = (f > 0 ? '…' : '') + tex(e.text.slice(f, thEnd));
      c.ans.innerHTML = cut > e.answer_start ? md(e.text.slice(Math.max(e.answer_start, f), cut)) : '';
    };
    if (!wide) {
      // binary search the smallest start that fits on one line
      let lo = 0, hi = cut;
      put(0);
      if (c.stream.scrollWidth > c.stream.clientWidth) {
        while (lo < hi) { const m = (lo + hi) >> 1; put(m); if (c.stream.scrollWidth > c.stream.clientWidth) lo = m + 1; else hi = m }
        from = lo;
        // start at a word boundary
        const sp = e.text.indexOf(' ', from);
        if (sp >= 0 && sp < cut && sp - from < 12) from = sp + 1;
      }
      put(from);
    } else {
      // keep ~the last 520 characters; the box clips the top under a fade
      from = Math.max(0, cut - 520);
      const sp = e.text.indexOf(' ', from);
      if (from > 0 && sp >= 0 && sp < cut) from = sp + 1;
      put(from);
    }
  }

  const fmt = x => x.toFixed(2);
  window.renderFrame = function (t) {
    const u = t - T0;                             // request time
    for (const c of cards) {
      const e = c.e;
      const [, tok, chars] = at(e.events, u);
      const done = u >= e.t_end;
      const shown = Math.max(0, Math.min(u, e.t_end));
      c.fill.style.width = (100 * tok / e.tokens).toFixed(3) + '%';
      c.status.textContent = u < 0 ? 'ready' : done ? 'done' : tok === 0 ? 'prefill' : 'decoding';
      c.status.style.color = done && e.ours ? 'var(--blue-glow)' : '';
      c.timer.innerHTML = `${fmt(shown)}<small>s</small>`;
      c.timer.classList.toggle('done', done);
      const rate = done ? e.tok_s : null;
      c.meta.textContent = `${tok} / ${e.tokens} tokens` + (rate ? ` · ${Math.round(rate)} tok/s` : '');
      setText(c, chars);
      const a = e.correct ? ramp(u, e.t_end + 0.05, 0.4) : 0;
      c.pill.style.opacity = a.toFixed(4);
    }
    fadeUp($('#punchline'), t, tPunch);
    fadeUp($('#punchsub'), t, tPunch + 0.15);
    fadeUp($('#foot'), t, tPunch + 0.25);
  };
})();
