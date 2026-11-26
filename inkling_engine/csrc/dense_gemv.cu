// Row-scaled FP8 (E4M3) GEMV for decode-sized steps (M = 1..8 tokens): out[m, r] = bf16(scale[r] * W[r] . x_i[m])
// where the input i of row r comes from a segment table (the attention projections read up to three differently
// normalised inputs; segments are 16-row aligned).
//
// Same machinery as the expert kernels (moe_decode.cu): persistent grid, 16-row tiles streamed into a shared-memory
// ring by TMA bulk copies (a ring entry is a tile's 16 rows over K / KSPLIT columns), f16 mma (E4M3 -> f16 is exact;
// activations converted to f16). The weights do not depend on the preceding kernel, so a block fills its ring before
// griddepcontrol.wait: with programmatic dependent launch a large part of the weights is in flight (or already in
// shared memory) while the predecessor runs.

#include "common.cuh"

namespace inkling {

template <int K, int KSPLIT>
struct DGeo {
  static constexpr int KS = K / KSPLIT;  // columns per ring entry
  static constexpr int RB = KS;          // bytes per row per entry
  static constexpr int RBP = KS + 16;    // padded shared-memory row (quarter-warp quads hit distinct banks)
  static constexpr int STAGE = 16 * RBP;
  static constexpr int GROUPS = KS / 128;
};

struct Segs {
  int b[4];  // input index of row r: r < b0: 0, < b1: 1, < b2: 2, < b3: 1, else 2
};

__device__ __forceinline__ int seg_input(const Segs& s, int r) {
  return r < s.b[0] ? 0 : r < s.b[1] ? 1 : r < s.b[2] ? 2 : r < s.b[3] ? 1 : 2;
}

template <int K, int KSPLIT, int WARPS, int STAGES>
__global__ void __launch_bounds__(WARPS * 32, 1)
    fp8_gemv_kernel(const __nv_bfloat16* __restrict__ x, long x_stride_i, int x_stride_m, Segs segs,
                    const uint8_t* __restrict__ w, const float* __restrict__ wscale, int R, int M,
                    __nv_bfloat16* __restrict__ out, int out_stride) {
  using G = DGeo<K, KSPLIT>;
  static_assert(G::GROUPS % WARPS == 0, "");
  constexpr int GPW = G::GROUPS / WARPS;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bars[STAGES];
  __shared__ float red[WARPS][32][4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int tiles = R / 16, grid = gridDim.x;
  const int mine = tiles > int(blockIdx.x) ? (tiles - 1 - int(blockIdx.x)) / grid + 1 : 0;
  const int chunks = mine * KSPLIT;
  const uint64_t pol = policy_evict_first();
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(&bars[s], 1);
    mbar_fence_init();
  }
  __syncthreads();
  auto issue = [&](int q, int stage) {
    const int u = blockIdx.x + (q / KSPLIT) * grid, h = q % KSPLIT;
    uint8_t* dst = smem + stage * G::STAGE;
    if (lane == 0) mbar_expect_tx(&bars[stage], 16 * G::RB);
    __syncwarp();
    if (lane < 16)
      tma_load(dst + lane * G::RBP, w + (size_t(u) * 16 + lane) * K + h * G::KS, G::RB, &bars[stage], pol);
  };
  if (warp == 0)
    for (int s = 0; s < STAGES && s < chunks; ++s) issue(s, s);
  pdl_wait();  // x comes from the predecessor
  pdl_trigger();

  float acc[GPW][2][4];
  for (int q = 0; q < chunks; ++q) {
    const int u = blockIdx.x + (q / KSPLIT) * grid, h = q % KSPLIT;
    const int stage = q % STAGES;
    if (h == 0) {
#pragma unroll
      for (int r = 0; r < GPW; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[r][0][i] = acc[r][1][i] = 0.f;
    }
    uint32_t xb[GPW][16];
    {
      const __nv_bfloat16* xrow =
          g < M ? x + size_t(seg_input(segs, u * 16)) * x_stride_i + size_t(g) * x_stride_m : nullptr;
#pragma unroll
      for (int r = 0; r < GPW; ++r) {
        const int c = h * (G::KS / 32) + 4 * (warp + r * WARPS) + t;
        if (xrow) {
          const uint8_t* p = reinterpret_cast<const uint8_t*>(xrow + 32 * c);
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const uint4 l = ld_dep16(p + 16 * v);
            xb[r][4 * v] = l.x, xb[r][4 * v + 1] = l.y, xb[r][4 * v + 2] = l.z, xb[r][4 * v + 3] = l.w;
          }
#pragma unroll
          for (int v = 0; v < 16; ++v) xb[r][v] = bf2_to_h2(xb[r][v]);
        } else {
#pragma unroll
          for (int v = 0; v < 16; ++v) xb[r][v] = 0u;
        }
      }
    }
    mbar_wait(&bars[stage], (q / STAGES) & 1);
    const uint32_t a0 = smem_u32(smem + stage * G::STAGE) + g * G::RBP;
#pragma unroll
    for (int r = 0; r < GPW; ++r) {
      const int c = 4 * (warp + r * WARPS) + t;  // 32-column chunk within the entry
      const uint4 q00 = lds16(a0 + 32 * c), q01 = lds16(a0 + 32 * c + 16);
      const uint4 q10 = lds16(a0 + 8 * G::RBP + 32 * c), q11 = lds16(a0 + 8 * G::RBP + 32 * c + 16);
      const uint32_t wa[8] = {q00.x, q00.y, q00.z, q00.w, q01.x, q01.y, q01.z, q01.w};
      const uint32_t wb[8] = {q10.x, q10.y, q10.z, q10.w, q11.x, q11.y, q11.z, q11.w};
      uint32_t a[16], b[16];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        a[2 * j] = cvt_e4m3x2_f16x2(uint16_t(wa[j])), a[2 * j + 1] = cvt_e4m3x2_f16x2(uint16_t(wa[j] >> 16));
        b[2 * j] = cvt_e4m3x2_f16x2(uint16_t(wb[j])), b[2 * j + 1] = cvt_e4m3x2_f16x2(uint16_t(wb[j] >> 16));
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        mma_f16(acc[r][i & 1], a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1], xb[r][2 * i], xb[r][2 * i + 1]);
    }
    const bool last = h == KSPLIT - 1;
    if (last) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float v = 0.f;
#pragma unroll
        for (int r = 0; r < GPW; ++r) v += acc[r][0][i] + acc[r][1][i];
        red[warp][lane][i] = v;
      }
    }
    __syncthreads();  // the entry is consumed by every warp (and red is written)
    if (warp == 0 && q + STAGES < chunks) issue(q + STAGES, stage);
    if (last) {
      for (int idx = tid; idx < 128; idx += WARPS * 32) {
        const int row = idx >> 3, n = idx & 7;
        if (n >= M) continue;
        const int ln = (row & 7) * 4 + (n >> 1), el = (row >> 3) * 2 + (n & 1);
        float sum = 0.f;
#pragma unroll
        for (int ww = 0; ww < WARPS; ++ww) sum += red[ww][ln][el];
        const int rr = u * 16 + row;
        out[size_t(n) * out_stride + rr] = __float2bfloat16_rn(sum * __ldg(wscale + rr));
      }
      __syncthreads();  // red is rewritten by the next tile
    }
  }
}

}  // namespace inkling

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

int sm_count() {
  static int n = 0;
  if (!n) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

template <int K, int KSPLIT, int WARPS, int STAGES>
void run_fp8(const __nv_bfloat16* x, long si, int sm, inkling::Segs segs, const uint8_t* w, const float* sc, int R,
             int M, __nv_bfloat16* out, int os, bool pdl, int grid) {
  auto k = inkling::fp8_gemv_kernel<K, KSPLIT, WARPS, STAGES>;
  constexpr int SMEM = STAGES * inkling::DGeo<K, KSPLIT>::STAGE;
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    init = true;
  }
  const int tiles = R / 16;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(std::min(tiles, grid > 0 ? grid : sm_count()));
  cfg.blockDim = dim3(WARPS * 32);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, x, si, sm, segs, w, sc, R, M, out, os));
}

}  // namespace

// x bf16 [I, M, K] (stride_i may be 0: one shared input), w fp8 [R, K] (R % 16 == 0), scale fp32 [R],
// out bf16 [M, R] (row stride out.stride(0)); bounds: 4 row boundaries (see seg_input), all >= R for one input.
void fp8_gemv(torch::Tensor x, int64_t stride_i, int64_t stride_m, torch::Tensor w, torch::Tensor scale,
              torch::Tensor out, int64_t M, std::vector<int64_t> bounds, bool pdl, int64_t variant, int64_t grid) {
  const int K = w.size(1), R = w.size(0);
  TORCH_CHECK(R % 16 == 0 && M >= 1 && M <= 8 && bounds.size() == 4);
  inkling::Segs s;
  for (int i = 0; i < 4; ++i) s.b[i] = int(bounds[i]);
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto wp = reinterpret_cast<const uint8_t*>(w.data_ptr());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const float* sp = scale.data_ptr<float>();
  const int os = int(out.stride(0)), sm = int(stride_m), m = int(M), gr = int(grid);
  switch (K * 16 + int(variant)) {
    case 2304 * 16 + 0: run_fp8<2304, 1, 6, 2>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    case 2304 * 16 + 1: run_fp8<2304, 2, 9, 4>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    case 2304 * 16 + 2: run_fp8<2304, 2, 3, 4>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    case 2304 * 16 + 3: run_fp8<2304, 3, 6, 6>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    case 3072 * 16 + 0: run_fp8<3072, 2, 6, 3>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    case 3072 * 16 + 1: run_fp8<3072, 4, 6, 6>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    case 3072 * 16 + 2: run_fp8<3072, 2, 3, 3>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    case 3072 * 16 + 3: run_fp8<3072, 3, 4, 4>(xp, stride_i, sm, s, wp, sp, R, m, op, os, pdl, gr); break;
    default: TORCH_CHECK(false, "fp8_gemv: unsupported K / variant ", K, " ", variant);
  }
}
