#include <torch/extension.h>
#include <vector>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void qkv_gemm_host(
    const __half* input_h,
    const __half* qkv_weight_h,
    int32_t hidden,
    __half* q_out,
    __half* k_out,
    __half* v_out);
extern "C" void qkv_gemm_host_stream(
    const __half* input_h,
    const __half* qkv_weight_h,
    int32_t hidden,
    __half* q_out,
    __half* k_out,
    __half* v_out,
    cudaStream_t stream);
extern "C" void qkv_set_stream(cudaStream_t stream);

static void qkv_project(
    at::Tensor input_h,       // [H], cuda, float16
    at::Tensor qkv_weight_h,  // [3H, H], cuda, float16
    at::Tensor q_out,         // [H], cuda, float16
    at::Tensor k_out,         // [H], cuda, float16
    at::Tensor v_out) {       // [H], cuda, float16
    TORCH_CHECK(input_h.is_cuda() && qkv_weight_h.is_cuda() && q_out.is_cuda() && k_out.is_cuda() && v_out.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(input_h.scalar_type() == at::kHalf && qkv_weight_h.scalar_type() == at::kHalf,
                "input and weight must be fp16");
    int32_t H = input_h.size(0);
    TORCH_CHECK(qkv_weight_h.size(0) == 3 * H && qkv_weight_h.size(1) == H, "qkv weight shape mismatch");
    TORCH_CHECK(q_out.size(0) == H && k_out.size(0) == H && v_out.size(0) == H, "qkv out shape mismatch");

    // Bind to current CUDA stream for stream-local execution
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // Initialize/rebind cached handle for this stream
    qkv_set_stream(stream);
    qkv_gemm_host_stream(
        reinterpret_cast<const __half*>(input_h.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(qkv_weight_h.data_ptr<at::Half>()),
        H,
        reinterpret_cast<__half*>(q_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(k_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_project", &qkv_project, "Q/K/V projections (cuBLASEx)");
}


