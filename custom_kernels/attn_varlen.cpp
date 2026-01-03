#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
// Kernel entry (defined in .cu)
extern "C" void attn_varlen_host(
    const __half* q,          // [H, D]
    const __half* k,          // [H, T, D]
    const __half* v,          // [H, T, D]
    int32_t H, int32_t T, int32_t D,
    float scale,
    __half* ctx_out,          // [H, D]
    cudaStream_t stream);
 
static at::Tensor attn_varlen_forward(
    at::Tensor q,              // [H, D] fp16, cuda
    at::Tensor k,              // [H, T, D] fp16, cuda
    at::Tensor v,              // [H, T, D] fp16, cuda
    int64_t seq_len,           // T (<= k.size(1))
    double scale               // 1/sqrt(D)
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA");
    TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf && v.dtype() == at::kHalf, "q,k,v must be fp16");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 3 && v.dim() == 3, "shapes: q[H,D], k[H,T,D], v[H,T,D]");
    int64_t H = q.size(0);
    int64_t D = q.size(1);
    TORCH_CHECK(k.size(0) == H && v.size(0) == H, "head mismatch");
    TORCH_CHECK(k.size(2) == D && v.size(2) == D, "head_dim mismatch");
    TORCH_CHECK(seq_len >= 1 && seq_len <= k.size(1), "invalid seq_len");
    auto opts = q.options();
    at::Tensor ctx = at::empty({H, D}, opts);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    attn_varlen_host(
        reinterpret_cast<const __half*>(q.data_ptr()),
        reinterpret_cast<const __half*>(k.data_ptr()),
        reinterpret_cast<const __half*>(v.data_ptr()),
        static_cast<int32_t>(H),
        static_cast<int32_t>(seq_len),
        static_cast<int32_t>(D),
        static_cast<float>(scale),
        reinterpret_cast<__half*>(ctx.data_ptr()),
        stream);
    return ctx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attn_varlen_forward, "Varlen attention (Q·K^T->softmax->P·V) for one token");
}

