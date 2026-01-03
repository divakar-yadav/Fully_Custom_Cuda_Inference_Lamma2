#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
// Optional global handle bound to a stream (lightweight cache)
static cublasHandle_t g_handle = nullptr;
static cudaStream_t g_handle_stream = nullptr;

extern "C" void qkv_set_stream(cudaStream_t stream) {
    if (g_handle == nullptr) {
        if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
            g_handle = nullptr;
            return;
        }
    }
    g_handle_stream = stream;
    cublasSetStream(g_handle, stream);
}


static void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error in ") + msg);
    }
}

extern "C" void qkv_gemm_host(
    const __half* input_h,           // [H]
    const __half* qkv_weight_h,      // [3H, H] (row-major: out x in)
    int32_t hidden,
    __half* q_out,                   // [H]
    __half* k_out,                   // [H]
    __half* v_out                    // [H]
) {
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    // Default stream; no explicit binding

    const int H = hidden;
    const __half* Wq = qkv_weight_h + 0 * H * H;
    const __half* Wk = qkv_weight_h + 1 * H * H;
    const __half* Wv = qkv_weight_h + 2 * H * H;

    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    // y = W * x with row-major W: compute GEMM with transposed W
    cublas_check(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H, 1, H,
        &alpha,
        Wq, CUDA_R_16F, H,
        input_h, CUDA_R_16F, H,
        &beta,
        q_out, CUDA_R_16F, H,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx Wq*x");

    cublas_check(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H, 1, H,
        &alpha,
        Wk, CUDA_R_16F, H,
        input_h, CUDA_R_16F, H,
        &beta,
        k_out, CUDA_R_16F, H,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx Wk*x");

    cublas_check(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H, 1, H,
        &alpha,
        Wv, CUDA_R_16F, H,
        input_h, CUDA_R_16F, H,
        &beta,
        v_out, CUDA_R_16F, H,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx Wv*x");

    cublasDestroy(handle);
}

extern "C" void qkv_gemm_host_stream(
    const __half* input_h,           // [H]
    const __half* qkv_weight_h,      // [3H, H]
    int32_t hidden,
    __half* q_out,                   // [H]
    __half* k_out,                   // [H]
    __half* v_out,                   // [H]
    cudaStream_t stream
) {
    cublasHandle_t handle = g_handle;
    if (handle == nullptr || g_handle_stream != stream) {
        // Create or rebind handle for this stream
        if (handle == nullptr) {
            cublas_check(cublasCreate(&handle), "cublasCreate(stream)");
        }
        cublas_check(cublasSetStream(handle, stream), "cublasSetStream");
        g_handle = handle;
        g_handle_stream = stream;
    }

    const int H = hidden;
    const __half* Wq = qkv_weight_h + 0 * H * H;
    const __half* Wk = qkv_weight_h + 1 * H * H;
    const __half* Wv = qkv_weight_h + 2 * H * H;

    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    cublas_check(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        H, 1, H,
        &alpha, Wq, CUDA_R_16F, H,
        input_h, CUDA_R_16F, H,
        &beta, q_out, CUDA_R_16F, H,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx Wq*x (stream)");
    cublas_check(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        H, 1, H,
        &alpha, Wk, CUDA_R_16F, H,
        input_h, CUDA_R_16F, H,
        &beta, k_out, CUDA_R_16F, H,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx Wk*x (stream)");
    cublas_check(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        H, 1, H,
        &alpha, Wv, CUDA_R_16F, H,
        input_h, CUDA_R_16F, H,
        &beta, v_out, CUDA_R_16F, H,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx Wv*x (stream)");
    // Do not destroy cached handle
}


