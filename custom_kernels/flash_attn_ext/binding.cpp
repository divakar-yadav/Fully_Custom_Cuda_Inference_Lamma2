#include <torch/extension.h>

torch::Tensor sdp_single_q(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdp_single_q", &sdp_single_q, "Single-query Flash-Attn (naive)");
}


