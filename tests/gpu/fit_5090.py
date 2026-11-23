"""Does the engine fit an RTX 5090's memory? (Modal: parity --args "fit_5090.py")

A second process fills the GPU until only BUDGET_GIB is free, so the engine runs out of device memory at a real
limit: CUDA context, CUDA graphs and allocator together. A 5090 has 32607 MiB (31.84 GiB) in total; the default
budget of 31.0 GiB leaves about 0.8 GiB for the driver and a display. Under that limit it runs
  default   4 slots x 32k (FP8 dense, FP4 prefill, vision, DSpark): 4 requests of 31k tokens at once, twice
  bf16      the same with --no-dense-fp8 --no-fp4-prefill
  long      256k context (FP8 KV, 1 slot): a 250k-token prompt
and then the server end-to-end test and the soak test. Prints the highest device memory used by the engine.
COLD=1 gives every job its own empty Triton cache (a first start). INKLING_AUTOTUNE=1 also ignores the pinned configs.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time

BUDGET = float(os.environ.get("BUDGET_GIB", "31.0"))
G = 2**30
BALLAST = """
import sys, time, torch
budget = float(sys.argv[1]) * 2**30
free, total = torch.cuda.mem_get_info()
keep = [torch.empty(max(0, int(free - budget)), dtype=torch.uint8, device="cuda")]
free, _ = torch.cuda.mem_get_info()
while free > budget + 2**20:
    keep.append(torch.empty(int(min(free - budget, 2**30)), dtype=torch.uint8, device="cuda"))
    free, _ = torch.cuda.mem_get_info()
print("ballast ready", free / 2**30, flush=True)
time.sleep(10**6)
"""


class Peak:
    """Highest device memory used (MiB), minus the ballast's usage."""

    def __init__(self, base):
        self.base, self.peak, self.stop = base, 0, False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            self.peak = max(self.peak, int(out.split()[0]) - self.base)
            time.sleep(0.2)


def used():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])


WORK = """
import random, sys, threading, torch
sys.path.insert(0, "/root")
from inkling_engine.engine import Engine
from inkling_engine.sampling import Sampling
from inkling_engine.scheduler import Scheduler
path, profile = sys.argv[1], sys.argv[2]
fp8 = profile != "bf16"
if profile == "long":
    eng = Engine(path, max_ctx=262144, slots=1, kv_fp8=True, spec=True, dense_fp8=True, fp4_prefill=True)
else:
    eng = Engine(path, max_ctx=32768, slots=4, spec=True, vision=True, dense_fp8=fp8, fp4_prefill=fp8)
sched = Scheduler(eng)
def mem(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"{profile} {tag}: pytorch allocated {torch.cuda.memory_allocated() / 2**30:.2f} (max {torch.cuda.max_memory_allocated() / 2**30:.2f}) reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB, free {free / 2**30:.2f}", flush=True)
mem("after start")
rng = random.Random(0)
n, k, out = (250000, 1, 256) if profile == "long" else (31000, 4, 512)
for rep in range(1 if profile == "long" else 2):
    res, err = {}, []
    def go(i):
        try:
            res[i] = len(sched.generate([rng.randrange(1000, 60000) for _ in range(n)], out, (), Sampling())[0])
        except Exception as e:  # noqa: BLE001
            err.append(repr(e))
    th = [threading.Thread(target=go, args=(i,)) for i in range(k)]
    [t.start() for t in th]
    [t.join() for t in th]
    mem(f"rep {rep}")
    print(profile, "rep", rep, "tokens", sorted(res.values()), "errors", err, flush=True)
    if err or len(res) != k:
        raise SystemExit(1)
"""

if __name__ == "__main__":
    path = sys.argv[1]
    only = sys.argv[2:] or ["default", "bf16", "long", "server_e2e", "soak"]
    open("/tmp/ballast.py", "w").write(BALLAST)
    open("/tmp/work.py", "w").write(WORK)
    ok = True
    for job in only:
        b = subprocess.Popen([sys.executable, "/tmp/ballast.py", str(BUDGET)], stdout=subprocess.PIPE, text=True)
        line = b.stdout.readline()
        base = used()
        pk = Peak(base)
        cmd = ([sys.executable, f"/root/tests/gpu/{job}.py", path] if job in ("server_e2e", "soak")
               else [sys.executable, "/tmp/work.py", path, job])
        env = dict(os.environ)
        if os.environ.get("COLD") == "1":
            env["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
        t0 = time.time()
        cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
        pk.stop = True
        tail = (cp.stdout + cp.stderr).strip().splitlines()[-9:]
        good = cp.returncode == 0
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {job:10s} device peak {pk.peak / 1024:.2f} GiB of {BUDGET} budget "
              f"({time.time() - t0:.0f} s; {line.strip()}) | {tail}", flush=True)
        b.terminate()
        b.wait()
    print("FIT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
