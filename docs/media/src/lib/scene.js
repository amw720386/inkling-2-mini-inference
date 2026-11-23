// Helpers shared by the HTML scenes. Load after theme.css; then define window.SCENE and window.renderFrame.
(function () {
  const q = new URLSearchParams(location.search);
  const W = +(q.get('w') || 1080), H = +(q.get('h') || 1080);
  document.documentElement.style.setProperty('--W', W + 'px');
  document.documentElement.style.setProperty('--H', H + 'px');
  // Type is specified for a 1080 px tall frame; --s scales it (1 for both 1080x1080 and 1920x1080).
  document.documentElement.style.setProperty('--s', String(H / 1080));
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const ease = x => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2 };
  // 0 before t0, 1 after t0 + d, eased in between.
  const ramp = (t, t0, d = 0.4) => ease((t - t0) / d);
  // Fade + 12 px rise in at t0, optional fade out at t1.
  function fadeUp(el, t, t0, t1, d = 0.4) {
    const a = ramp(t, t0, d) * (t1 == null ? 1 : 1 - ramp(t, t1, d));
    el.style.opacity = a.toFixed(4);
    el.style.transform = `translateY(${((1 - ramp(t, t0, d)) * 12 * H / 1080).toFixed(2)}px)`;
  }
  window.S = { W, H, wide: W / H > 1.3, clamp, ease, ramp, fadeUp, $: s => document.querySelector(s) };
})();
