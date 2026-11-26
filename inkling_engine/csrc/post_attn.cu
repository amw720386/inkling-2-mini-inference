// Post-attention step and router of a decode-sized call (one sequence, T <= 8 consecutive tokens).
// Same math as ops/fused_triton.post_attn and route:
//   a      = bf16((conv(o) + o) * conv_scale)                (short conv over the attention output, state per visit)
//   resid  = a (block boundary) or partial + a                 -> resid fp32
//   mi     = softmax-weighted mix of the normalised sources [blocks..., resid] under the MLP query
//   x      = bf16(rms(mi) * norm_w)          (or mi in fp32 when a message gate follows)
//   router: logits = W_r . x, top-8 of sigmoid(logits) + bias, weights = scale * s / sum(s)
//
// post_attn_kernel: one 8-CTA thread-block cluster; CTA r owns channels [288 r, 288 r + 288), one per thread. The
// reductions over H (source norms and query dots, the output norm) are block-reduced, then a few threads of every CTA
// sum the 8 CTAs' partials through distributed shared memory in rank order, so every CTA gets identical totals.
// router_kernel: 32 blocks x 4 experts; each block copies its 4 weight rows into shared memory with TMA before
// griddepcontrol.wait; the last block to finish (atomic counter, reset by it) selects the top 8 of every token.

#include <cooperative_groups.h>

#include "common.cuh"

namespace cg = cooperative_groups;

namespace inkling {

constexpr int PH = 2304, PCL = 8, PCH = PH / PCL, PE = 128, PT = 8, PMAXS = 7, PTOPK = 8;
constexpr int RB_EXPERTS = 4, RB_BLOCKS = PE / RB_EXPERTS;

struct PostAttnArgs {
  const __nv_bfloat16* o;  // [T, H] attention output projection
  int o_stride;
  const float* cw;         // conv weight fp32 [H, 4] (tap 3 = current token)
  __nv_bfloat16* state;    // conv state [3, H] (oldest first)
  __nv_bfloat16* cand;     // verify: candidate states, + t * cand_t + r * H (null: decode, shift the state)
  long cand_t;
  const __nv_bfloat16* blk;  // sources [S, T', H] bf16
  long blk_s;
  int blk_t;
  int ns;                  // sources before resid
  int part;                // slot of the partial in blk (unused at a boundary)
  int boundary;
  float* resid;            // [T, H] fp32
  const float* q;          // MLP query fp32 [H] (pre-scaled)
  const float* norm_w;     // fp32 [H] or null (then out_f32)
  __nv_bfloat16* out;      // x bf16 [T, H]
  int out_stride;
  float* out_f32;          // mi fp32 [T, H] (message visits)
  float conv_scale, eps;
  int T;
  const int* slot;  // device slot index of the sequence (state += slot * state_slot); null: slot 0
  long state_slot;
  int tok;  // per-token launch: this cluster's token (pointers above already point at its row)
};

template <int T>
__device__ __forceinline__ void post_attn_body(const PostAttnArgs& p, cg::cluster_group& cluster) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = int(cluster.block_rank()) * PCH + tid;
  constexpr int NW = PCH / 32;
  __shared__ float red[NW][T][2 * PMAXS];
  __shared__ float part[T][2 * PMAXS];  // this CTA's partial sums (read by every CTA of the cluster)
  __shared__ float tot[T][2 * PMAXS];   // cluster totals
  const int S = p.ns + 1;
  float xsrc[T][PMAXS];
  const float w0 = p.cw[4 * c], w1 = p.cw[4 * c + 1], w2 = p.cw[4 * c + 2], w3 = p.cw[4 * c + 3];
  float st[3], ov[T];
#pragma unroll
  for (int r = 0; r < 3; ++r) st[r] = __bfloat162float(p.state[r * PH + c]);
#pragma unroll
  for (int t = 0; t < T; ++t) ov[t] = __bfloat162float(p.o[size_t(t) * p.o_stride + c]);
  const float qc = p.q[c];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float x = ov[t];
    float x1, x2, x3;
    if (T == 1) {  // token p.tok of the call: earlier tokens are rows above it
      const int k = p.tok;
      auto row = [&](int back) { return __bfloat162float(p.o[-long(back) * p.o_stride + c]); };
      x1 = k >= 1 ? row(1) : st[2 + k];
      x2 = k >= 2 ? row(2) : st[1 + k];
      x3 = k >= 3 ? row(3) : st[k];
    } else {
      x1 = t >= 1 ? ov[t - 1 < 0 ? 0 : t - 1] : st[2 + t];
      x2 = t >= 2 ? ov[t - 2 < 0 ? 0 : t - 2] : st[1 + t];
      x3 = t >= 3 ? ov[t - 3 < 0 ? 0 : t - 3] : st[t < 3 ? t : 2];
    }
    float a = x3 * w0 + x2 * w1 + x1 * w2 + x * w3;
    a = bf16r((a + x) * p.conv_scale);
    const float rs = p.boundary ? a : __bfloat162float(p.blk[p.part * p.blk_s + size_t(t) * p.blk_t + c]) + a;
    p.resid[size_t(t) * PH + c] = rs;
#pragma unroll
    for (int s = 0; s < PMAXS; ++s) {
      const float v = s < p.ns ? __bfloat162float(p.blk[s * p.blk_s + size_t(t) * p.blk_t + c]) : s == p.ns ? rs : 0.f;
      xsrc[t][s] = v;
      float sq = v * v, dq = v * qc;
#pragma unroll
      for (int o = 16; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o), dq += __shfl_xor_sync(0xffffffffu, dq, o);
      if (lane == 0) red[warp][t][2 * s] = sq, red[warp][t][2 * s + 1] = dq;
    }
    if (p.cand) {
#pragma unroll
      for (int r = 0; r < 3; ++r) {
        if (T == 1) {
          const int s = p.tok - 2 + r;  // token index of the candidate row
          p.cand[r * PH + c] = s >= 0 ? p.o[-long(p.tok - s) * p.o_stride + c] : __float2bfloat16_rn(st[3 + s]);
        } else {
          const int s = t - 2 + r;
          p.cand[t * p.cand_t + r * PH + c] = __float2bfloat16_rn(s >= 0 ? ov[s < 0 ? 0 : s] : st[3 + (s < -3 ? -3 : s)]);
        }
      }
    }
  }
  if (!p.cand) {  // decode: state <- [row1, row2, o]
    p.state[c] = __float2bfloat16_rn(st[1]);
    p.state[PH + c] = __float2bfloat16_rn(st[2]);
    p.state[2 * PH + c] = p.o[c];
  }
  __syncthreads();
  if (tid < T * 2 * PMAXS) {
    const int t = tid / (2 * PMAXS), k = tid % (2 * PMAXS);
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) v += red[w][t][k];
    part[t][k] = v;
  }
  cluster.sync();
  if (tid < T * 2 * PMAXS) {
    const int t = tid / (2 * PMAXS), k = tid % (2 * PMAXS);
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < PCL; ++r) v += cluster.map_shared_rank(&part[0][0], r)[t * 2 * PMAXS + k];
    tot[t][k] = v;
  }
  __syncthreads();
  float mix[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    float lg[PMAXS], rinv[PMAXS], m = -INFINITY;
#pragma unroll
    for (int s = 0; s < PMAXS; ++s) {
      rinv[s] = rsqrtf(tot[t][2 * s] / PH + p.eps);
      lg[s] = tot[t][2 * s + 1] * rinv[s];
      if (s < S) m = fmaxf(m, lg[s]);
    }
    float l = 0.f, o = 0.f;
#pragma unroll
    for (int s = 0; s < PMAXS; ++s)
      if (s < S) {
        const float e = __expf(lg[s] - m);
        l += e;
        o += e * rinv[s] * xsrc[t][s];
      }
    mix[t] = o / l;
  }
  if (!p.norm_w) {
#pragma unroll
    for (int t = 0; t < T; ++t) p.out_f32[size_t(t) * PH + c] = mix[t];
    cluster.sync();  // no CTA exits while another still reads its partials
    return;
  }
#pragma unroll
  for (int t = 0; t < T; ++t) {
    float v = mix[t] * mix[t];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[warp][t][0] = v;
  }
  cluster.sync();  // every CTA has read `part`; red is complete
  if (tid < T) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) v += red[w][tid][0];
    part[tid][0] = v;
  }
  cluster.sync();
  if (tid < T) {
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < PCL; ++r) v += cluster.map_shared_rank(&part[0][0], r)[tid * 2 * PMAXS];
    tot[tid][0] = v;
  }
  __syncthreads();
  const float nw = p.norm_w[c];
#pragma unroll
  for (int t = 0; t < T; ++t)
    p.out[size_t(t) * p.out_stride + c] = __float2bfloat16_rn(mix[t] * rsqrtf(tot[t][0] / PH + p.eps) * nw);
  cluster.sync();
}

// One cluster per token (T clusters): every token's post-attention runs in parallel at one-token latency.
__global__ void __cluster_dims__(PCL, 1, 1) __launch_bounds__(PCH) post_attn_tok_kernel(PostAttnArgs p) {
  cg::cluster_group cluster = cg::this_cluster();
  pdl_wait();
  pdl_trigger();
  if (p.slot) p.state += long(*p.slot) * p.state_slot;
  const int k = blockIdx.x / PCL;
  p.tok = k;
  p.o += size_t(k) * p.o_stride;
  p.blk += size_t(k) * p.blk_t;
  p.resid += size_t(k) * PH;
  if (p.out) p.out += size_t(k) * p.out_stride;
  if (p.out_f32) p.out_f32 += size_t(k) * PH;
  if (p.cand) p.cand += k * p.cand_t;
  post_attn_body<1>(p, cluster);
}

__global__ void __cluster_dims__(PCL, 1, 1) __launch_bounds__(PCH) post_attn_kernel(PostAttnArgs p) {
  cg::cluster_group cluster = cg::this_cluster();
  pdl_wait();
  pdl_trigger();
  if (p.slot) p.state += long(*p.slot) * p.state_slot;
  switch (p.T) {
    case 1: post_attn_body<1>(p, cluster); break;
    case 2: post_attn_body<2>(p, cluster); break;
    case 3: post_attn_body<3>(p, cluster); break;
    case 4: post_attn_body<4>(p, cluster); break;
    default: post_attn_body<8>(p, cluster); break;
  }
}

// grid RB_BLOCKS, 256 threads. x bf16 [T, H] (row stride), W bf16 [E, H], logits fp32 [T, E] workspace.
__global__ void __launch_bounds__(256) router_kernel(const __nv_bfloat16* __restrict__ x, int x_stride, int T,
                                                     const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
                                                     float* __restrict__ logits, int* __restrict__ counter,
                                                     int* __restrict__ ids, float* __restrict__ wts, float scale) {
  __shared__ __align__(128) __nv_bfloat16 sw[RB_EXPERTS][PH];
  __shared__ __align__(8) uint64_t bar;
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, e0 = blockIdx.x * RB_EXPERTS;
  if (tid == 0) {
    mbar_init(&bar, 1);
    mbar_fence_init();
    mbar_expect_tx(&bar, sizeof(sw));
    tma_load(&sw[0][0], w + size_t(e0) * PH, sizeof(sw), &bar, policy_evict_first());
  }
  pdl_wait();
  pdl_trigger();
  // warp w: token w (T <= 8), all 4 experts; lane covers 72 channels as 9 chunks of 8
  float acc[RB_EXPERTS] = {};
  uint4 xv[9];
  if (warp < T) {
#pragma unroll
    for (int i = 0; i < 9; ++i) xv[i] = ld_dep16(x + size_t(warp) * x_stride + 8 * (lane + 32 * i));
  }
  __syncthreads();
  mbar_wait(&bar, 0);
  if (warp < T) {
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const uint32_t xw[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
#pragma unroll
      for (int e = 0; e < RB_EXPERTS; ++e) {
        const uint4 wq = *reinterpret_cast<const uint4*>(&sw[e][8 * (lane + 32 * i)]);
        const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          acc[e] += __uint_as_float(ww[k] << 16) * __uint_as_float(xw[k] << 16) +
                    __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xw[k] & 0xffff0000u);
      }
    }
#pragma unroll
    for (int e = 0; e < RB_EXPERTS; ++e) {
#pragma unroll
      for (int o = 16; o; o >>= 1) acc[e] += __shfl_xor_sync(0xffffffffu, acc[e], o);
      if (lane == 0) logits[warp * PE + e0 + e] = acc[e];
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(counter, 1) == RB_BLOCKS - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (warp < T) {
    const int t = warp;
    float lgv[4], sel[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int e = lane + 32 * j;
      lgv[j] = __ldcg(logits + t * PE + e);
      sel[j] = 1.f / (1.f + __expf(-lgv[j])) + bias[e];
    }
    int pick = 0;
    float swt = 0.f, tot = 0.f;
#pragma unroll
    for (int k = 0; k < PTOPK; ++k) {
      float best = -INFINITY;
      int be = PE;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (sel[j] > best || (sel[j] == best && lane + 32 * j < be)) best = sel[j], be = lane + 32 * j;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oe = __shfl_xor_sync(0xffffffffu, be, o);
        if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
      }
      const int j = be >> 5;
      const float mine = j == 0 ? lgv[0] : j == 1 ? lgv[1] : j == 2 ? lgv[2] : lgv[3];
      const float lgb = __shfl_sync(0xffffffffu, mine, be & 31);
      const float s = 1.f / (1.f + __expf(-lgb));
      tot += s;
      if (lane == k) pick = be, swt = s;
      if ((be & 31) == lane) {
        if (j == 0) sel[0] = -INFINITY;
        else if (j == 1) sel[1] = -INFINITY;
        else if (j == 2) sel[2] = -INFINITY;
        else sel[3] = -INFINITY;
      }
    }
    if (lane < PTOPK) {
      ids[t * PTOPK + lane] = pick;
      wts[t * PTOPK + lane] = swt / (tot + 1e-20f) * scale;
    }
  }
  if (tid == 0) *counter = 0;
}

}  // namespace inkling

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {
template <typename K, typename... A>
void launch_k(K k, dim3 grid, dim3 block, bool pdl, A... args) {
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
template <typename T>
T* ptr_or_null(const torch::Tensor& t) {
  return t.numel() ? reinterpret_cast<T*>(t.data_ptr()) : nullptr;
}
}  // namespace

// See PostAttnArgs. Empty tensors stand for null pointers.
void post_attn(torch::Tensor o, torch::Tensor cw, torch::Tensor state, torch::Tensor cand, torch::Tensor blk, int64_t ns,
               int64_t part, bool boundary, torch::Tensor resid, torch::Tensor q, torch::Tensor norm_w,
               torch::Tensor out, torch::Tensor out_f32, double conv_scale, double eps, int64_t T, bool pdl,
               torch::Tensor slot, int64_t state_slot) {
  TORCH_CHECK(T >= 1 && T <= inkling::PT, "post_attn: T must be 1-8");
  TORCH_CHECK(ns + 1 <= inkling::PMAXS);
  inkling::PostAttnArgs a{};
  a.o = ptr_or_null<const __nv_bfloat16>(o);
  a.o_stride = int(o.stride(0));
  a.cw = ptr_or_null<const float>(cw);
  a.state = ptr_or_null<__nv_bfloat16>(state);
  a.cand = ptr_or_null<__nv_bfloat16>(cand);
  a.cand_t = cand.numel() ? cand.stride(0) : 0;
  a.blk = ptr_or_null<const __nv_bfloat16>(blk);
  a.blk_s = blk.stride(0);
  a.blk_t = int(blk.stride(1));
  a.ns = int(ns);
  a.part = int(part);
  a.boundary = boundary;
  a.resid = ptr_or_null<float>(resid);
  a.q = ptr_or_null<const float>(q);
  a.norm_w = ptr_or_null<const float>(norm_w);
  a.out = ptr_or_null<__nv_bfloat16>(out);
  a.out_stride = out.numel() ? int(out.stride(0)) : 0;
  a.out_f32 = ptr_or_null<float>(out_f32);
  a.conv_scale = float(conv_scale);
  a.eps = float(eps);
  a.T = int(T);
  a.slot = ptr_or_null<const int>(slot);
  a.state_slot = state_slot;
  if (T == 1) launch_k(inkling::post_attn_kernel, dim3(inkling::PCL), dim3(inkling::PCH), pdl, a);
  else launch_k(inkling::post_attn_tok_kernel, dim3(inkling::PCL * T), dim3(inkling::PCH), pdl, a);
}

// x bf16 [T, H]; w bf16 [128, H]; bias fp32 [128]; logits fp32 [>= T, 128]; counter int32 [1] (zero, left zero)
void router(torch::Tensor x, int64_t T, torch::Tensor w, torch::Tensor bias, torch::Tensor logits, torch::Tensor counter,
            torch::Tensor ids, torch::Tensor wts, double scale, bool pdl) {
  TORCH_CHECK(T >= 1 && T <= inkling::PT);
  launch_k(inkling::router_kernel, dim3(inkling::RB_BLOCKS), dim3(256), pdl,
           (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (int)T, (const __nv_bfloat16*)w.data_ptr(),
           bias.data_ptr<float>(), logits.data_ptr<float>(), counter.data_ptr<int>(), ids.data_ptr<int>(),
           wts.data_ptr<float>(), (float)scale);
}
