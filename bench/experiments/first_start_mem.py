"""Where device memory goes on a first start: per kernel load and per autotuned config, the device memory used outside
PyTorch, the config's spill (`.local` bytes per thread) and its time; then the configs each kernel ends up with.
Modal: parity --args "TRITON_CACHE_DIR=/tmp/tc first_start_mem.py [default|bf16|long] [sweep]"
"sweep" also runs one eager chunk for every attention shape (tokens per tile, key splits) a prompt can reach.
"""

import collections
import json
import random
import sys
import time

import torch

sys.path.insert(0, "/root")
import triton  # noqa: E402
from triton import knobs  # noqa: E402
from triton.runtime.autotuner import Autotuner  # noqa: E402

profile = sys.argv[2] if len(sys.argv) > 2 else "default"
sweep = "sweep" in sys.argv[3:]
if profile == "all":  # one process per profile
    import subprocess
    sys.exit(max(subprocess.call([sys.executable, __file__, sys.argv[1], p, *sys.argv[3:]]) for p in ("default", "bf16", "long")))  # noqa: E501
G, M = 2**30, 2**20
torch.cuda.init()


def outside():  # device memory in use by this process and outside PyTorch's allocator (CUDA, modules, local memory)
    free, total = torch.cuda.mem_get_info()
    return total - free - torch.cuda.memory_reserved()


loads = collections.defaultdict(lambda: [0, 0, 0])  # kernel -> [modules, bytes of binary, outside growth at load]
rows, held, spills = [], {}, {}
t_start = time.time()


def on_load(module, function, name, group, h):
    held[h] = outside()


def on_loaded(module, function, name, group, h):
    r = loads[name]
    r[0] += 1
    r[2] += outside() - held.pop(h)


knobs.runtime.kernel_load_start_hook = on_load
knobs.runtime.kernel_load_end_hook = on_loaded

from inkling_engine.ops.tuned import Tuner  # noqa: E402

bench = Tuner._bench
tuned = []


def _bench(self, *args, config, **meta):
    fn = self.fn
    while not isinstance(fn, triton.runtime.jit.JITFunction):
        fn = fn.fn
    cache = fn.device_caches[torch.cuda.current_device()][0]
    before_keys = set(cache)
    torch.cuda.synchronize()
    o0 = outside()
    t0 = time.time()
    out = bench(self, *args, config=config, **meta)
    torch.cuda.synchronize()
    o1 = outside()
    new = [cache[k] for k in cache if k not in before_keys]
    spill = max([getattr(k, "n_spills", -1) or 0 for k in new], default=-1)
    size = sum(len(k.kernel) for k in new if hasattr(k, "kernel"))
    rows.append((o1 - o0, fn.__name__, str(config), spill, size, out[0] if isinstance(out, (list, tuple)) else out,
                 time.time() - t0))
    return out


Tuner._bench = _bench
check = Tuner.check_disk_cache


def check_disk_cache(self, key, configs, bench_fn):
    n0, o0, t0 = len(rows), outside(), time.time()
    hit = check(self, key, configs, bench_fn)
    tuned.append((self.base_fn.__name__, key, "pin" if self.pinned(key) else "disk" if hit else "tuned", len(rows) - n0, outside() - o0, time.time() - t0,
                  str(self.cache[key]), time.time() - t_start))
    return hit


Tuner.check_disk_cache = check_disk_cache

from inkling_engine.engine import Engine  # noqa: E402
from inkling_engine.sampling import Sampling  # noqa: E402
from inkling_engine.scheduler import Scheduler  # noqa: E402


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"[{time.time() - t_start:6.1f} s] {tag}: used {(total - free) / G:.2f}, PyTorch allocated "
          f"{torch.cuda.memory_allocated() / G:.2f} reserved {torch.cuda.memory_reserved() / G:.2f}, outside PyTorch "
          f"{outside() / G:.2f} GiB, {sum(r[0] for r in loads.values())} modules, {len(tuned)} tuned keys", flush=True)


mem("cuda init")
fp = profile != "bf16"
if profile == "long":
    eng = Engine(sys.argv[1], max_ctx=262144, slots=1, kv_fp8=True, spec=True, dense_fp8=True, fp4_prefill=True)
else:
    eng = Engine(sys.argv[1], max_ctx=32768, slots=4, spec=True, vision=True, dense_fp8=fp, fp4_prefill=fp)
sched = Scheduler(eng)
mem("after start")
rng = random.Random(0)
for n in (8, 100, 300, 1500, 5000, 31000):
    sched.generate([rng.randrange(1000, 60000) for _ in range(n)], 64, (), Sampling())
    mem(f"after a {n}-token prompt")
if sweep:
    from inkling_engine.batch import Batch, SeqChunk
    from inkling_engine.fast import pick_splits

    done = set()
    for n in range(1, 2049):
        tok = 4 if n <= 4 else 8 if n <= 8 else 16 if n <= 256 else 32
        for start in [*range(0, 2200, 16), 4000, 8000]:
            nl, ng = pick_splits(start + n, (n + tok - 1) // tok, eng.cfg.num_kv_heads, eng.model.sms, eng.cfg.sliding_window)
            if not {(tok, nl, "local"), (tok, ng, "global")} <= done:
                done |= {(tok, nl, "local"), (tok, ng, "global")}
                part = [rng.randrange(1000, 60000) for _ in range(n)]
                eng.model.forward(Batch.build([SeqChunk(0, start, n, "last")], [part]), eng.cache, eng.pws)
    mem(f"after the sweep ({len(done)} shapes)")
print("\n== autotuned keys, in order: kernel, key, source, configs tried, growth outside PyTorch (MiB), seconds, pick, at")
for name, key, src, n, d, s, cfg, at in tuned:
    print(f"{name} {key} {src} {n} {d / M:+.0f} {s:.1f}s -> {cfg}  @{at:.0f}s")
print("\n== configs that grew memory outside PyTorch by >= 8 MiB: MiB, kernel, config, spill bytes/thread, binary, ms")
for d, name, cfg, spill, size, ms, s in sorted(rows, reverse=True):
    if abs(d) >= 8 * M:
        print(f"{d / M:+.0f} {name} {cfg} spill {spill} binary {size / 1024:.0f} KiB {ms}")
print("\n== spilling configs:", sum(1 for r in rows if r[3] > 0), "of", len(rows))
print("largest spills:", sorted(((r[3], r[1], r[2]) for r in rows), reverse=True)[:12])
print("binary MiB over all tried configs", sum(r[4] for r in rows) / M)
print("\n== modules: kernel, count, growth at load (MiB)")
for name, (n, _, g) in sorted(loads.items(), key=lambda x: -x[1][2]):
    print(f"{name} {n} {g / M:+.1f}")
for name, mod in list(sys.modules.items()):
    if name.startswith("inkling_engine.ops"):
        for attr, k in vars(mod).items():
            if isinstance(k, triton.runtime.jit.JITFunction):
                for x in k.device_caches[0][0].values():
                    if (getattr(x, "n_spills", 0) or 0) > 0:
                        print(f"loaded and spilling: {attr} {x.n_spills} bytes/thread, {x.metadata.num_warps} warps")
            elif isinstance(k, Autotuner):
                f = k.fn
                while not isinstance(f, triton.runtime.jit.JITFunction):
                    f = f.fn
                for x in f.device_caches[0][0].values():
                    if (getattr(x, "n_spills", 0) or 0) > 0:
                        print(f"loaded and spilling: {attr} {x.n_spills} bytes/thread, {x.metadata.num_warps} warps")
print("PICKS", profile, json.dumps([(name, list(key), cfg) for name, key, *_, cfg, _ in tuned]))
