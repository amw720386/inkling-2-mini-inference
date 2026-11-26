// Shared device helpers of the single-stream decode kernels (sm_120a).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace inkling {

// ------------------------------------------------------------------------------------- PDL
// griddepcontrol.wait: wait for the preceding kernel's completion and memory flush (no-op without the launch
// attribute). launch_dependents: let the next kernel be scheduled.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// ------------------------------------------------------------------------------------- loads
// read-only weight streams: no L1 allocation
__device__ __forceinline__ uint4 ld_stream16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
// data written by the previous kernel: coherent, not hoistable above the PDL wait
__device__ __forceinline__ uint4 ld_dep16(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float ld_dep(const float* p) {
  float v;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_dep(const int* p) {
  int v;
  asm volatile("ld.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint4 lds16(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint16_t lds2(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ uint8_t lds1(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=h"(v) : "r"(a));
  return uint8_t(v);
}

// ------------------------------------------------------------------------- TMA bulk + mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\nLAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT_%=;\n}" ::"r"(smem_u32(bar)),
      "r"(phase)
      : "memory");
}
// evict-first L2 policy for weights read once per step
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// global -> shared bulk copy (bytes % 16 == 0, both addresses 16-byte aligned), completes on `bar`
__device__ __forceinline__ void tma_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}

// ------------------------------------------------------------------------------------ conversions
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint16_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
  return r;
}
// 8 E2M1 codes (byte b: elements 2b in the low nibble, 2b + 1) -> 4 f16x2
__device__ __forceinline__ void cvt_e2m1x8_f16x2(uint32_t w, uint32_t (&h)[4]) {
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(w));
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// bf16x2 -> f16x2
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(__uint_as_float(v & 0xffff0000u)), "f"(__uint_as_float(v << 16)));
  return r;
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// ------------------------------------------------------------------------------------------ mma
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float tanh_f(float x) { return 2.f / (1.f + __expf(-2.f * x)) - 1.f; }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

}  // namespace inkling
