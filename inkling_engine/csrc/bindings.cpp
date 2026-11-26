// Python bindings of the one-request CUDA decode kernels (built at first use by ops/cuda_ext.py).
#include <torch/extension.h>

void fp8_gemv(torch::Tensor x, int64_t stride_i, int64_t stride_m, torch::Tensor w, torch::Tensor scale,
              torch::Tensor out, int64_t M, std::vector<int64_t> bounds, bool pdl, int64_t variant, int64_t grid);
void msg_down(torch::Tensor hs, torch::Tensor ms, torch::Tensor down, int64_t gate0, torch::Tensor g, double eps,
              bool pdl);
void msg_up(torch::Tensor hs, torch::Tensor ms, torch::Tensor g, torch::Tensor gain, torch::Tensor forget, int64_t gate0,
            torch::Tensor norm_w, torch::Tensor tmp, torch::Tensor ss, torch::Tensor counter, torch::Tensor out,
            double eps, bool pdl);
void post_attn(torch::Tensor o, torch::Tensor cw, torch::Tensor state, torch::Tensor cand, torch::Tensor blk, int64_t ns,
               int64_t part, bool boundary, torch::Tensor resid, torch::Tensor q, torch::Tensor norm_w,
               torch::Tensor out, torch::Tensor out_f32, double conv_scale, double eps, int64_t T, bool pdl,
               torch::Tensor slot, int64_t state_slot);
void router(torch::Tensor x, int64_t T, torch::Tensor w, torch::Tensor bias, torch::Tensor logits, torch::Tensor counter,
            torch::Tensor ids, torch::Tensor wts, double scale, bool pdl);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemv", &fp8_gemv, "row-scaled FP8 GEMV over segmented inputs (M <= 8)");
  m.def("msg_down", &msg_down, "loop message gate down projection (T <= 8)");
  m.def("msg_up", &msg_up, "loop message gate up projection + combine + norm (T <= 8)");
  m.def("post_attn", &post_attn, "post-attention conv/residual/depth mix/norm (8-CTA cluster, T <= 8)");
  m.def("router", &router, "router logits + top-8 (T <= 8)");
}
