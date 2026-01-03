#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>

extern "C" void capture_decode_host(
    const int64_t* ctrl_input,
    void* arena_k,
    void* arena_v,
    int32_t* seq_len_ptr,
    const int32_t* position_ptr,
    float* logits_out,
    int32_t L, int32_t H, int32_t max_len, int32_t D, int32_t vocab,
    cudaStream_t stream);

// Extended host launcher accepting scratch buffers and packed weight pointers
extern "C" void capture_decode_host_ext(
    const int64_t* ctrl_input,
    void* arena_k,
    void* arena_v,
    int32_t* seq_len_ptr,
    const int32_t* position_ptr,
    float* logits_out,
    // scratch buffers (fp16 except logits_out)
    void* x_norm_scratch,
    void* gate_out_scratch,
    void* up_out_scratch,
    void* act_scratch,
    void* mlp_out_scratch,
    // packed weight pointers (fp16)
    const void* rms_in_w,
    const void* rms_post_w,
    const void* rms_final_w,
    const void* mlp_gate_w,
    const void* mlp_up_w,
    const void* mlp_down_w,
    const void* lm_head_w,
    // dims and params
    int32_t L, int32_t H, int32_t max_len, int32_t D, int32_t vocab,
    int32_t I,
    float rms_eps,
    cudaStream_t stream);

extern "C" void rmsnorm_host(
    const void* x,
    const void* w,
    void* y,
    int32_t H,
    float eps,
    cudaStream_t stream);

extern "C" void prepare_cublas_handle(cudaStream_t stream);

extern "C" void swiglu_host(
    const void* gate,
    const void* up,
    void* act,
    int32_t I,
    cudaStream_t stream);

static void launch_capture_decode(
    at::Tensor ctrl_input, at::Tensor arena_k, at::Tensor arena_v,
    at::Tensor seq_len_dev, at::Tensor pos_dev, at::Tensor logits_out)
{
    TORCH_CHECK(ctrl_input.is_cuda() && arena_k.is_cuda() && arena_v.is_cuda() && seq_len_dev.is_cuda() && pos_dev.is_cuda() && logits_out.is_cuda(),
                "all tensors must be CUDA");
    auto L = arena_k.size(0);
    auto H = arena_k.size(2);
    auto max_len = arena_k.size(3);
    auto D = arena_k.size(4);
    auto vocab = logits_out.size(1);
    // Use PyTorch's current CUDA stream to make this capture-friendly
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    capture_decode_host(
        ctrl_input.data_ptr<int64_t>(),
        arena_k.data_ptr(),
        arena_v.data_ptr(),
        seq_len_dev.data_ptr<int32_t>(),
        pos_dev.data_ptr<int32_t>(),
        logits_out.data_ptr<float>(),
        (int32_t)L, (int32_t)H, (int32_t)max_len, (int32_t)D, (int32_t)vocab,
        stream);
}

static void launch_capture_decode_ext(
    at::Tensor ctrl_input, at::Tensor arena_k, at::Tensor arena_v,
    at::Tensor seq_len_dev, at::Tensor pos_dev, at::Tensor logits_out,
    at::Tensor x_norm_s, at::Tensor gate_out_s, at::Tensor up_out_s,
    at::Tensor act_s, at::Tensor mlp_out_s,
    uint64_t rms_in_w, uint64_t rms_post_w, uint64_t rms_final_w,
    uint64_t w_gate, uint64_t w_up, uint64_t w_down, uint64_t w_vocab,
    int64_t I, double rms_eps)
{
    TORCH_CHECK(ctrl_input.is_cuda() && arena_k.is_cuda() && arena_v.is_cuda() && seq_len_dev.is_cuda() && pos_dev.is_cuda() && logits_out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(x_norm_s.is_cuda() && gate_out_s.is_cuda() && up_out_s.is_cuda() && act_s.is_cuda() && mlp_out_s.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(x_norm_s.scalar_type() == at::kHalf &&
                gate_out_s.scalar_type() == at::kHalf &&
                up_out_s.scalar_type() == at::kHalf &&
                act_s.scalar_type() == at::kHalf &&
                mlp_out_s.scalar_type() == at::kHalf,
                "scratch must be fp16");
    auto L = arena_k.size(0);
    auto H = arena_k.size(2);
    auto max_len = arena_k.size(3);
    auto D = arena_k.size(4);
    auto vocab = logits_out.size(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    capture_decode_host_ext(
        ctrl_input.data_ptr<int64_t>(),
        arena_k.data_ptr(),
        arena_v.data_ptr(),
        seq_len_dev.data_ptr<int32_t>(),
        pos_dev.data_ptr<int32_t>(),
        logits_out.data_ptr<float>(),
        x_norm_s.data_ptr(),
        gate_out_s.data_ptr(),
        up_out_s.data_ptr(),
        act_s.data_ptr(),
        mlp_out_s.data_ptr(),
        reinterpret_cast<const void*>(rms_in_w),
        reinterpret_cast<const void*>(rms_post_w),
        reinterpret_cast<const void*>(rms_final_w),
        reinterpret_cast<const void*>(w_gate),
        reinterpret_cast<const void*>(w_up),
        reinterpret_cast<const void*>(w_down),
        reinterpret_cast<const void*>(w_vocab),
        (int32_t)L, (int32_t)H, (int32_t)max_len, (int32_t)D, (int32_t)vocab,
        (int32_t)I,
        static_cast<float>(rms_eps),
        stream);
}

static void launch_rmsnorm(at::Tensor x, at::Tensor w, at::Tensor y, double eps)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf && y.scalar_type() == at::kHalf, "x,w,y must be fp16");
    int32_t H = static_cast<int32_t>(x.size(-1));
    TORCH_CHECK(w.numel() == H && y.numel() == H, "shape mismatch");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rmsnorm_host(
        x.data_ptr(), w.data_ptr(), y.data_ptr(),
        H, static_cast<float>(eps), stream);
}

// Variant that accepts weight as raw device pointer (uint64)
static void launch_rmsnorm_ptr(at::Tensor x, uint64_t w_ptr, at::Tensor y, double eps)
{
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf && y.scalar_type() == at::kHalf, "x,y must be fp16");
    int32_t H = static_cast<int32_t>(x.size(-1));
    TORCH_CHECK(y.numel() == H, "shape mismatch");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rmsnorm_host(
        x.data_ptr(), reinterpret_cast<const void*>(w_ptr), y.data_ptr(),
        H, static_cast<float>(eps), stream);
}

static void launch_swiglu(at::Tensor gate, at::Tensor up, at::Tensor act)
{
    TORCH_CHECK(gate.is_cuda() && up.is_cuda() && act.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(gate.scalar_type() == at::kHalf && up.scalar_type() == at::kHalf && act.scalar_type() == at::kHalf, "gate/up/act must be fp16");
    int32_t I = static_cast<int32_t>(gate.numel());
    TORCH_CHECK(up.numel() == I && act.numel() == I, "shape mismatch");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    swiglu_host(gate.data_ptr(), up.data_ptr(), act.data_ptr(), I, stream);
}

static void cublas_check(cublasStatus_t st, const char* where) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error at ") + where);
    }
}

// GEMM: [1,H] x [H,I] -> [1,I], fp16 inputs/weights, fp32 accum
static void gemm_1xH_HxI_fp16_fp32(
    const at::Tensor& x_norm,       // [H] fp16
    uintptr_t w_ptr,                // device ptr to W [I,H] fp16 (row-major)
    at::Tensor& out,                // [I] fp16
    int H, int I)
{
    TORCH_CHECK(x_norm.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x_norm.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf, "fp16 required");
    cublasHandle_t handle = nullptr;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cublas_check(cublasSetStream(handle, stream), "cublasSetStream");
    const __half* x = reinterpret_cast<const __half*>(x_norm.data_ptr()); // [H] as (H x 1)
    const __half* W = reinterpret_cast<const __half*>(w_ptr);             // [I,H] row-major in memory
    __half* y = reinterpret_cast<__half*>(out.data_ptr());                // [I]
    // Compute y(I,1) = W(I,H) * x(H,1)
    // Use col-major cublas: op(W)=T with provided dims (H x I) and lda=H to interpret row-major correctly
    int m = I, n = 1, k = H;
    int lda = H; // rows of op(A)==I, original A is (H x I)
    int ldb = H; // rows of B (x) as (H x 1)
    int ldc = I; // rows of C (y) as (I x 1)
    float alpha = 1.0f, beta = 0.0f;
    cublas_check(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            W, CUDA_R_16F, lda,
            x, CUDA_R_16F, ldb,
            &beta,
            y, CUDA_R_16F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx gate/up"
    );
    cublas_check(cublasDestroy(handle), "cublasDestroy");
}

static void launch_gate_up_gemm(at::Tensor x_norm, at::Tensor gate_out, at::Tensor up_out,
                                uint64_t w_gate_ptr, uint64_t w_up_ptr, int64_t H, int64_t I)
{
    TORCH_CHECK(x_norm.dim() == 1 && x_norm.size(0) == H, "x_norm shape");
    TORCH_CHECK(gate_out.dim() == 1 && gate_out.size(0) == I, "gate_out shape");
    TORCH_CHECK(up_out.dim() == 1 && up_out.size(0) == I, "up_out shape");
    gemm_1xH_HxI_fp16_fp32(x_norm, (uintptr_t)w_gate_ptr, gate_out, (int)H, (int)I);
    gemm_1xH_HxI_fp16_fp32(x_norm, (uintptr_t)w_up_ptr, up_out, (int)H, (int)I);
}

// GEMM: [1,I] x [I,H] -> [1,H], fp16 inputs/weights, fp32 accum
static void gemm_1xI_IxH_fp16_fp32(
    const at::Tensor& act,      // [I] fp16
    uintptr_t w_ptr,            // device ptr to W_down [H,I] fp16 (row-major)
    at::Tensor& out,            // [H] fp16
    int I, int H)
{
    TORCH_CHECK(act.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(act.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf, "fp16 required");
    cublasHandle_t handle = nullptr;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cublas_check(cublasSetStream(handle, stream), "cublasSetStream");
    const __half* x = reinterpret_cast<const __half*>(act.data_ptr());  // [I] as (I x 1)
    const __half* W = reinterpret_cast<const __half*>(w_ptr);           // [H,I] row-major
    __half* y = reinterpret_cast<__half*>(out.data_ptr());              // [H]
    // y(H,1) = W(H,I) * x(I,1); use op(W)=T with (I x H), lda=I
    int m = H, n = 1, k = I;
    int lda = I; // rows of op(A)==H, original A is (I x H)
    int ldb = I; // rows of x (I x 1)
    int ldc = H; // rows of y (H x 1)
    float alpha = 1.0f, beta = 0.0f;
    cublas_check(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            W, CUDA_R_16F, lda,
            x, CUDA_R_16F, ldb,
            &beta,
            y, CUDA_R_16F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx down"
    );
    cublas_check(cublasDestroy(handle), "cublasDestroy");
}

static void launch_down_gemm(at::Tensor act, at::Tensor mlp_out,
                             uint64_t w_down_ptr, int64_t I, int64_t H)
{
    TORCH_CHECK(act.dim() == 1 && act.size(0) == I, "act shape");
    TORCH_CHECK(mlp_out.dim() == 1 && mlp_out.size(0) == H, "mlp_out shape");
    gemm_1xI_IxH_fp16_fp32(act, (uintptr_t)w_down_ptr, mlp_out, (int)I, (int)H);
}

// GEMM logits: [1,H] x W^T where W is [V,H] row-major -> [1,V]
static void logits_gemm_vocab(at::Tensor y, at::Tensor logits_out, uint64_t w_vocab_ptr, int64_t H, int64_t V)
{
    TORCH_CHECK(y.dim() == 1 && y.size(0) == H, "y shape");
    TORCH_CHECK(logits_out.dim() == 1 && logits_out.size(0) == V, "logits shape");
    TORCH_CHECK(y.is_cuda() && logits_out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(y.scalar_type() == at::kHalf && logits_out.scalar_type() == at::kFloat, "y fp16, logits fp32");
    cublasHandle_t handle = nullptr;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cublas_check(cublasSetStream(handle, stream), "cublasSetStream");
    const __half* x = reinterpret_cast<const __half*>(y.data_ptr());  // [H] as (H x 1)
    const __half* W = reinterpret_cast<const __half*>(w_vocab_ptr);   // [V,H] row-major
    float* out = reinterpret_cast<float*>(logits_out.data_ptr());     // [V] fp32
    // out(V,1) = W(V,H) * x(H,1); use op(W)=T with dims (H x V), lda=H
    int m = V, n = 1, k = H;
    int lda = H; // rows of op(A)==V, original A is (H x V)
    int ldb = H; // rows of x (H x 1)
    int ldc = V; // rows of out (V x 1)
    float alpha = 1.0f, beta = 0.0f;
    cublas_check(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)m, (int)n, (int)k,
            &alpha,
            W, CUDA_R_16F, lda,
            x, CUDA_R_16F, ldb,
            &beta,
            out, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx logits"
    );
    cublas_check(cublasDestroy(handle), "cublasDestroy");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("capture_decode", &launch_capture_decode, "Case4 custom capture/replay decode");
    m.def("capture_decode_ext", &launch_capture_decode_ext, "Case4 custom capture/replay decode (extended args)");
    m.def("prepare_cublas", [](){
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        prepare_cublas_handle(stream);
    }, "Prepare global cuBLAS handle on current stream (call before capture)");
    m.def("rmsnorm", &launch_rmsnorm, "Case4 fused RMSNorm (fp16)");
    m.def("rmsnorm_ptr", &launch_rmsnorm_ptr, "Case4 fused RMSNorm with weight pointer (fp16)");
    m.def("swiglu", &launch_swiglu, "Case4 SwiGLU (fp16)");
    m.def("gate_up_gemm", &launch_gate_up_gemm, "Case4 MLP gate/up GEMMs (fp16, fp32 accum)");
    m.def("down_gemm", &launch_down_gemm, "Case4 MLP down GEMM (fp16, fp32 accum)");
    m.def("logits_gemm_vocab", &logits_gemm_vocab, "Case4 logits GEMM to vocab (fp16->fp32)");
}
