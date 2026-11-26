// Loop message gates for decode-sized steps (T <= 8 tokens), same math as ops/fused_triton.msg_gates:
//   g[t, j]   = bf16(Wd_j . bf16([rms(h_j[t]), rms(m_j[t])]))                       (rank 64)
//   o[t, j]   = (1 - tanh(Wf_j . g)) * h_j[t] + tanh(Wg_j . g) * m_j[t]
//   out[j, t] = bf16(rms(o) * norm_w[j])   (or o in fp32 without a norm weight)
// Two kernels. Each block copies its slice of the (static) gate weights into shared memory with TMA before
// griddepcontrol.wait, so the weight traffic overlaps the predecessor; after the wait only small reductions remain.
// msg_up: every block also writes its channel block's sums of squares; the last block of a gate (atomic counter,
// reset by that block) applies the RMS norm to the gate's rows.

#include "common.cuh"

namespace inkling {

constexpr int MH = 2304, RANK = 64, MAXT = 8;
constexpr int DOWN_ROWS = 2;   // rows of Wd per block
constexpr int UP_CH = 64;      // channels per block
constexpr int UP_BLOCKS = MH / UP_CH;

__device__ __forceinline__ float block_sum(float v, float* sh) {
  // blockDim.x == 256
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += sh[i];
  return s;
}

// grid (64 / DOWN_ROWS, n_gate), 256 threads
__global__ void __launch_bounds__(256) msg_down_kernel(const float* __restrict__ hs, long hs_g, int hs_t,
                                                       const float* __restrict__ ms, long ms_g, int ms_t,
                                                       const __nv_bfloat16* __restrict__ wd, int gate0, int T,
                                                       float eps, float* __restrict__ gout) {
  __shared__ __align__(128) __nv_bfloat16 w[DOWN_ROWS][2 * MH];
  __shared__ __align__(8) uint64_t bar;
  __shared__ float sh[8];
  const int j = blockIdx.y, r0 = blockIdx.x * DOWN_ROWS, tid = threadIdx.x;
  const int t_lo = blockIdx.z, t_hi = t_lo + 1;  // one token per grid slice
  if (tid == 0) {
    mbar_init(&bar, 1);
    mbar_fence_init();
    mbar_expect_tx(&bar, sizeof(w));
    tma_load(&w[0][0], wd + (size_t(gate0 + j) * RANK + r0) * 2 * MH, sizeof(w), &bar, policy_evict_first());
  }
  pdl_wait();
  pdl_trigger();
  __syncthreads();
  const int ng = gridDim.y;
  for (int t = t_lo; t < t_hi; ++t) {
    const float* h = hs + j * hs_g + size_t(t) * hs_t;
    const float* m = ms + j * ms_g + size_t(t) * ms_t;
    float hv[9], mv[9], sh2 = 0.f, sm2 = 0.f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      hv[i] = ld_dep(h + tid + 256 * i), mv[i] = ld_dep(m + tid + 256 * i);
      sh2 += hv[i] * hv[i], sm2 += mv[i] * mv[i];
    }
    const float rx = rsqrtf(block_sum(sh2, sh) / MH + eps);
    const float rm = rsqrtf(block_sum(sm2, sh) / MH + eps);
    if (t == t_lo) mbar_wait(&bar, 0);
    float acc[DOWN_ROWS] = {};
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const int k = tid + 256 * i;
      const float xn = bf16r(hv[i] * rx), mn = bf16r(mv[i] * rm);
#pragma unroll
      for (int r = 0; r < DOWN_ROWS; ++r)
        acc[r] += __bfloat162float(w[r][k]) * xn + __bfloat162float(w[r][MH + k]) * mn;
    }
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r) {
      const float s = block_sum(acc[r], sh);
      if (tid == 0) gout[(size_t(t) * ng + j) * RANK + r0 + r] = bf16r(s);
    }
  }
}

// grid (UP_BLOCKS, n_gate), 256 threads: 64 channels x 4 quarters of the rank (reduced by shuffles).
// Writes o (fp32) and, when a norm follows (msg_norm_kernel), the block's per-token sums of squares.
__global__ void __launch_bounds__(256) msg_up_kernel(const float* __restrict__ hs, long hs_g, int hs_t,
                                                     const float* __restrict__ ms, long ms_g, int ms_t,
                                                     const float* __restrict__ gin, const __nv_bfloat16* __restrict__ wg,
                                                     const __nv_bfloat16* __restrict__ wf, int gate0, int T,
                                                     float* __restrict__ o_out, long o_g, int o_t,
                                                     float* __restrict__ ss) {
  __shared__ __align__(128) __nv_bfloat16 sf[UP_CH][RANK];
  __shared__ __align__(128) __nv_bfloat16 sg[UP_CH][RANK];
  __shared__ float gsh[MAXT][RANK];
  __shared__ float part[MAXT][8];
  __shared__ __align__(8) uint64_t bar;
  const int j = blockIdx.y, blk = blockIdx.x, tid = threadIdx.x, ng = gridDim.y;
  const int t_lo = blockIdx.z, t_hi = t_lo + 1;  // one token per grid slice
  const int c0 = blk * UP_CH;
  if (tid == 0) {
    mbar_init(&bar, 1);
    mbar_fence_init();
    mbar_expect_tx(&bar, sizeof(sf) + sizeof(sg));
    const uint64_t pol = policy_evict_first();
    tma_load(&sf[0][0], wf + (size_t(gate0 + j) * MH + c0) * RANK, sizeof(sf), &bar, pol);
    tma_load(&sg[0][0], wg + (size_t(gate0 + j) * MH + c0) * RANK, sizeof(sg), &bar, pol);
  }
  pdl_wait();
  pdl_trigger();
  for (int i = tid; i < T * RANK; i += 256)
    gsh[i / RANK][i % RANK] = ld_dep(gin + (size_t(i / RANK) * ng + j) * RANK + i % RANK);
  const int c = tid >> 2, q = tid & 3, ch = c0 + c;
  float hv[MAXT], mv[MAXT];
  for (int t = t_lo; t < t_hi; ++t)
    if (q == 0) hv[t] = ld_dep(hs + j * hs_g + size_t(t) * hs_t + ch), mv[t] = ld_dep(ms + j * ms_g + size_t(t) * ms_t + ch);
  __syncthreads();
  mbar_wait(&bar, 0);
  // the lane's 16 rank entries of both weight rows, as 2 x 8 bf16x2 words
  const uint4* pf = reinterpret_cast<const uint4*>(&sf[c][16 * q]);
  const uint4* pg = reinterpret_cast<const uint4*>(&sg[c][16 * q]);
  const uint4 f0 = pf[0], f1 = pf[1], g0 = pg[0], g1 = pg[1];
  const uint32_t fw[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
  const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
  for (int t = t_lo; t < t_hi; ++t) {
    float f = 0.f, a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float x0 = gsh[t][16 * q + 2 * i], x1 = gsh[t][16 * q + 2 * i + 1];
      f += __uint_as_float(fw[i] << 16) * x0 + __uint_as_float(fw[i] & 0xffff0000u) * x1;
      a += __uint_as_float(gw[i] << 16) * x0 + __uint_as_float(gw[i] & 0xffff0000u) * x1;
    }
    f += __shfl_xor_sync(0xffffffffu, f, 1);
    a += __shfl_xor_sync(0xffffffffu, a, 1);
    f += __shfl_xor_sync(0xffffffffu, f, 2);
    a += __shfl_xor_sync(0xffffffffu, a, 2);
    float s = 0.f;
    if (q == 0) {
      const float o = (1.f - tanh_f(f)) * hv[t] + tanh_f(a) * mv[t];
      o_out[j * o_g + size_t(t) * o_t + ch] = o;
      s = o * o;
    }
    if (ss) {
#pragma unroll
      for (int x = 16; x; x >>= 1) s += __shfl_xor_sync(0xffffffffu, s, x);
      if ((tid & 31) == 0) part[t][tid >> 5] = s;
    }
  }
  if (!ss) return;
  __syncthreads();
  if (tid >= t_lo && tid < t_hi) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += part[tid][w];
    ss[(size_t(j) * MAXT + tid) * UP_BLOCKS + blk] = s;
  }
}

// grid (H / 256, n_gate), 256 threads: out[j, t, c] = bf16(o * rms_inv(o[j, t]) * norm_w[j, c])
__global__ void __launch_bounds__(256) msg_norm_kernel(const float* __restrict__ o, long o_g, int o_t,
                                                       const float* __restrict__ ss, int T,
                                                       const __nv_bfloat16* __restrict__ norm_w, long nw_g,
                                                       __nv_bfloat16* __restrict__ out, long out_g, int out_t,
                                                       float eps) {
  __shared__ float rinv[MAXT];
  pdl_wait();
  pdl_trigger();
  const int j = blockIdx.y, ch = blockIdx.x * 256 + threadIdx.x, tid = threadIdx.x;
  const int t_lo = blockIdx.z, t_hi = t_lo + 1;  // one token per grid slice
  if (tid < 32) {
    for (int t = t_lo; t < t_hi; ++t) {
      float s = 0.f;
      for (int b = tid; b < UP_BLOCKS; b += 32) s += ld_dep(ss + (size_t(j) * MAXT + t) * UP_BLOCKS + b);
#pragma unroll
      for (int x = 16; x; x >>= 1) s += __shfl_xor_sync(0xffffffffu, s, x);
      if (tid == 0) rinv[t] = rsqrtf(s / MH + eps);
    }
  }
  const float w = __bfloat162float(norm_w[j * nw_g + ch]);
  float ov[MAXT];
  for (int t = t_lo; t < t_hi; ++t) ov[t] = ld_dep(o + j * o_g + size_t(t) * o_t + ch);
  __syncthreads();
  for (int t = t_lo; t < t_hi; ++t) out[j * out_g + size_t(t) * out_t + ch] = __float2bfloat16_rn(ov[t] * rinv[t] * w);
}

}  // namespace inkling

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {
template <typename K, typename... A>
void launch_pdl(K k, dim3 grid, dim3 block, bool pdl, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
}
}  // namespace

// hs, ms: fp32 [G, T, H] views (gate stride may be 0); down bf16 [4, 64, 2H]; g fp32 [T, G, 64]
void msg_down(torch::Tensor hs, torch::Tensor ms, torch::Tensor down, int64_t gate0, torch::Tensor g, double eps,
              bool pdl) {
  const int G = hs.size(0), T = hs.size(1);
  TORCH_CHECK(T <= inkling::MAXT && hs.size(2) == inkling::MH);
  launch_pdl(inkling::msg_down_kernel, dim3(inkling::RANK / inkling::DOWN_ROWS, G, T), dim3(256), pdl,
             (const float*)hs.data_ptr(), (long)hs.stride(0), (int)hs.stride(1), (const float*)ms.data_ptr(),
             (long)ms.stride(0), (int)ms.stride(1), (const __nv_bfloat16*)down.data_ptr(), (int)gate0, T, (float)eps,
             g.data_ptr<float>());
}

// gain/forget bf16 [4, H, 64]; norm_w bf16 [G, H] or empty (then out is fp32 [G, T, H] and gets o directly);
// tmp fp32 [G, T, H] (o before the norm); ss fp32 [G, 8, H/64]
void msg_up(torch::Tensor hs, torch::Tensor ms, torch::Tensor g, torch::Tensor gain, torch::Tensor forget, int64_t gate0,
            torch::Tensor norm_w, torch::Tensor tmp, torch::Tensor ss, torch::Tensor counter, torch::Tensor out,
            double eps, bool pdl) {
  const int G = hs.size(0), T = hs.size(1);
  TORCH_CHECK(T <= inkling::MAXT);
  const bool norm = norm_w.numel() > 0;
  torch::Tensor o = norm ? tmp : out;
  launch_pdl(inkling::msg_up_kernel, dim3(inkling::UP_BLOCKS, G, T), dim3(256), pdl, (const float*)hs.data_ptr(),
             (long)hs.stride(0), (int)hs.stride(1), (const float*)ms.data_ptr(), (long)ms.stride(0), (int)ms.stride(1),
             (const float*)g.data_ptr(), (const __nv_bfloat16*)gain.data_ptr(), (const __nv_bfloat16*)forget.data_ptr(),
             (int)gate0, T, o.data_ptr<float>(), (long)o.stride(0), (int)o.stride(1),
             norm ? ss.data_ptr<float>() : nullptr);
  if (!norm) return;
  launch_pdl(inkling::msg_norm_kernel, dim3(inkling::MH / 256, G, T), dim3(256), pdl, (const float*)tmp.data_ptr(),
             (long)tmp.stride(0), (int)tmp.stride(1), (const float*)ss.data_ptr(), T,
             (const __nv_bfloat16*)norm_w.data_ptr(), (long)norm_w.stride(0), (__nv_bfloat16*)out.data_ptr(),
             (long)out.stride(0), (int)out.stride(1), (float)eps);
}
