"""Kernel configs: pinned for sm_120, autotuned elsewhere.

Autotuning on the first start took minutes, its picks varied between runs (19 of 48 keys differed between two
runs on the same GPU model; configs change the order of fp32 sums, so logits too), and it cost device memory for good: the
driver sizes its local-memory pool for the largest register spill it has launched and never shrinks it, and a losing
attention config spills 2.7 KB per thread (2.3 GiB on an RTX PRO 6000; bench/experiments/first_start_mem.py).

PINS holds the config of every shape the server's profiles reach, recorded from the RTX PRO 6000 the numbers in the
docs were measured on. Shapes outside the table are autotuned as before, except that configs spilling more than
MAX_SPILL bytes per thread are not launched. INKLING_AUTOTUNE=1 ignores the table.
"""

import os

import torch
import triton
from triton.runtime.autotuner import Autotuner

USE_PINS = os.environ.get("INKLING_AUTOTUNE", "0") != "1"
MAX_SPILL = 64  # bytes of register spill per thread a config may have to be launched by the autotuner


class Tuner(Autotuner):
    """Autotuner that takes a pinned config for a known shape; `pin_args`: arguments whose dtype is part of the pin."""

    pin_args = ()

    def pinned(self, key):
        if not USE_PINS or torch.cuda.get_device_capability() != (12, 0):
            return None
        table = PINS.get(self.base_fn.__name__, {})
        pin = table.get(tuple(key[:len(self.keys)]) + tuple(str(self.nargs[a].dtype)[6:] for a in self.pin_args))
        if pin is None:
            return None
        names = list(self.configs[0].kwargs)
        return triton.Config(dict(zip(names, pin)), num_warps=pin[-2], num_stages=pin[-1])

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        pin = self.pinned(tuning_key)
        if pin is None:
            return super().check_disk_cache(tuning_key, configs, bench_fn)
        self.cache[tuning_key] = pin
        return True

    def _bench(self, *args, config, **meta):
        try:
            kernel = self.fn.run(*args, **{**meta, **config.all_kwargs(), "warmup": True})  # compile, do not launch
            kernel._init_handles()
            if kernel.n_spills > MAX_SPILL:
                return [float("inf")] * 3
        except Exception:  # noqa: BLE001 - the launch below reports what the autotuner expects
            pass
        return super()._bench(*args, config=config, **meta)


def autotune(configs, key, pin_args=(), **kw):
    """triton.autotune with pins (always caching results on disk)."""

    def decorator(fn):
        t = Tuner(fn, fn.arg_names, configs, key, kw.pop("reset_to_zero", None), kw.pop("restore_value", None),
                  cache_results=True, **kw)
        t.pin_args = pin_args
        return t

    return decorator


# kernel -> {key values (+ dtypes of pin_args): (config kwargs in order, num_warps, num_stages)}
PINS = {
    "_seg_gemv_kernel": {
        (8640, 2304, 16, False): (32, 256, 4, 4), (8640, 2304, 16, True): (32, 256, 4, 4),
        (8640, 2304, 32, False): (64, 256, 8, 3), (8640, 2304, 32, True): (64, 256, 8, 4),
        (8768, 2304, 16, False): (32, 256, 4, 4), (8768, 2304, 16, True): (32, 256, 4, 4),
        (8768, 2304, 32, False): (64, 256, 4, 3), (8768, 2304, 32, True): (64, 256, 4, 4),
    },
    "_attn_kernel": {
        (16, 1, 0, 0, 128, 24, "bfloat16"): (128, 8, 1), (16, 1, 0, 0, 128, 24, "float8_e4m3fn"): (32, 8, 1),
        (16, 1, 1024, 1088, 128, 24, "bfloat16"): (128, 8, 1), (16, 16, 0, 0, 128, 24, "bfloat16"): (32, 4, 2),
        (16, 16, 0, 0, 128, 24, "float8_e4m3fn"): (32, 8, 2), (16, 16, 1024, 1088, 128, 24, "bfloat16"): (64, 4, 2),
        (16, 2, 0, 0, 128, 24, "bfloat16"): (64, 8, 1), (16, 2, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 1),
        (16, 2, 1024, 1088, 128, 24, "bfloat16"): (64, 8, 1), (16, 32, 0, 0, 128, 24, "bfloat16"): (64, 8, 2),
        (16, 32, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 2), (16, 4, 0, 0, 128, 24, "bfloat16"): (64, 4, 1),
        (16, 4, 0, 0, 128, 24, "float8_e4m3fn"): (128, 8, 1), (16, 4, 1024, 1088, 128, 24, "bfloat16"): (32, 8, 3),
        (16, 8, 0, 0, 128, 24, "bfloat16"): (64, 4, 1), (16, 8, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 1),
        (16, 8, 1024, 1088, 128, 24, "bfloat16"): (64, 8, 1), (32, 1, 0, 0, 128, 24, "bfloat16"): (32, 8, 2),
        (32, 1, 0, 0, 128, 24, "float8_e4m3fn"): (32, 8, 3), (32, 1, 1024, 1088, 128, 24, "bfloat16"): (32, 8, 2),
        (32, 2, 0, 0, 128, 24, "bfloat16"): (32, 8, 2), (32, 2, 0, 0, 128, 24, "float8_e4m3fn"): (32, 8, 3),
        (32, 2, 1024, 1088, 128, 24, "bfloat16"): (32, 8, 2), (32, 4, 0, 0, 128, 24, "bfloat16"): (32, 4, 3),
        (32, 4, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 2), (32, 4, 1024, 1088, 128, 24, "bfloat16"): (32, 8, 2),
        (4, 1, 0, 0, 128, 24, "bfloat16"): (128, 8, 1), (4, 1, 0, 0, 128, 24, "float8_e4m3fn"): (32, 4, 1),
        (4, 1, 1024, 1088, 128, 24, "bfloat16"): (128, 8, 1), (4, 16, 0, 0, 128, 24, "bfloat16"): (32, 4, 2),
        (4, 16, 0, 0, 128, 24, "float8_e4m3fn"): (32, 8, 2), (4, 16, 1024, 1088, 128, 24, "bfloat16"): (64, 4, 2),
        (4, 2, 0, 0, 128, 24, "bfloat16"): (64, 8, 1), (4, 2, 0, 0, 128, 24, "float8_e4m3fn"): (64, 4, 1),
        (4, 2, 1024, 1088, 128, 24, "bfloat16"): (64, 4, 2), (4, 32, 0, 0, 128, 24, "bfloat16"): (64, 4, 2),
        (4, 32, 0, 0, 128, 24, "float8_e4m3fn"): (64, 4, 1), (4, 4, 0, 0, 128, 24, "bfloat16"): (64, 4, 2),
        (4, 4, 0, 0, 128, 24, "float8_e4m3fn"): (64, 4, 1), (4, 4, 1024, 1088, 128, 24, "bfloat16"): (32, 4, 3),
        (4, 8, 0, 0, 128, 24, "bfloat16"): (64, 4, 1), (4, 8, 0, 0, 128, 24, "float8_e4m3fn"): (64, 4, 2),
        (4, 8, 1024, 1088, 128, 24, "bfloat16"): (64, 4, 1), (8, 1, 0, 0, 128, 24, "bfloat16"): (32, 8, 2),
        (8, 1, 0, 0, 128, 24, "float8_e4m3fn"): (32, 8, 2), (8, 1, 1024, 1088, 128, 24, "bfloat16"): (32, 8, 2),
        (8, 16, 0, 0, 128, 24, "bfloat16"): (32, 8, 2), (8, 16, 0, 0, 128, 24, "float8_e4m3fn"): (32, 8, 3),
        (8, 16, 1024, 1088, 128, 24, "bfloat16"): (64, 4, 1), (8, 2, 0, 0, 128, 24, "bfloat16"): (64, 4, 2),
        (8, 2, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 1), (8, 2, 1024, 1088, 128, 24, "bfloat16"): (64, 8, 1),
        (8, 32, 0, 0, 128, 24, "bfloat16"): (64, 8, 2), (8, 32, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 1),
        (8, 4, 0, 0, 128, 24, "bfloat16"): (64, 4, 1), (8, 4, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 1),
        (8, 4, 1024, 1088, 128, 24, "bfloat16"): (32, 8, 3), (8, 8, 0, 0, 128, 24, "bfloat16"): (64, 4, 2),
        (8, 8, 0, 0, 128, 24, "float8_e4m3fn"): (64, 8, 1), (8, 8, 1024, 1088, 128, 24, "bfloat16"): (64, 4, 1),
    },
    "_fp8_gemv_kernel": {
        (18432, 2304, 16): (64, 256, 8, 3), (18432, 2304, 32): (64, 128, 4, 4), (200058, 2304, 16): (32, 256, 4, 4),
        (2304, 3072, 16): (32, 512, 4, 4), (2304, 3072, 32): (16, 512, 4, 3), (2304, 9216, 16): (32, 512, 4, 4),
        (2304, 9216, 32): (16, 512, 4, 3), (32827, 2304, 16): (32, 128, 4, 4), (32827, 2304, 32): (64, 128, 4, 4),
        (5120, 2304, 16): (16, 512, 4, 3), (5120, 2304, 32): (32, 512, 8, 3),
    },
    "_moe_gemv_kernel": {
        (2304, 1152, 1, 0, True): (32, 128, 8, 2), (2304, 2304, 1, 1, True): (32, 128, 8, 2),
    },
    "_grouped_kernel": {
        (2304, 1152, 0, 0, 16): (64, 128, 4, 3), (2304, 1152, 0, 0, 32): (128, 128, 4, 2),
        (2304, 1152, 0, 0, 64): (128, 128, 4, 2), (2304, 1152, 1, 0, 16): (64, 128, 4, 3),
        (2304, 1152, 1, 0, 32): (32, 128, 8, 3), (2304, 1152, 1, 0, 64): (128, 128, 4, 2),
        (2304, 2304, 0, 1, 16): (64, 256, 8, 3), (2304, 2304, 0, 1, 32): (128, 256, 8, 2),
        (2304, 2304, 0, 1, 64): (128, 128, 4, 2), (2304, 2304, 1, 1, 16): (64, 256, 4, 3),
        (2304, 2304, 1, 1, 32): (64, 256, 4, 2), (2304, 2304, 1, 1, 64): (128, 128, 4, 2),
    },
    "_dense_gemv_kernel": {
        (200058, 2304, 16): (64, 256, 4, 3), (200058, 2304, 32): (64, 128, 8, 3), (2304, 3072, 16): (16, 512, 4, 3),
        (2304, 3072, 32): (16, 512, 8, 3),
    },
    "_fp8_mm_kernel": {
        (1024, 2304): (128, 128, 128, 8, 4), (2304, 3072): (128, 128, 64, 4, 3), (64, 2304): (64, 64, 128, 8, 4),
        (6592, 2304): (128, 64, 128, 4, 3), (8640, 2304): (128, 128, 64, 8, 4),
    },
    "_grouped_fp4_kernel": {
        (2304, 1152, 0, 128): (64, 128, 4, 3), (2304, 2304, 1, 128): (128, 256, 4, 2),
    },
}
