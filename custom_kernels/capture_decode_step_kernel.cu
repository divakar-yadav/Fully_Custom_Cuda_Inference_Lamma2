#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cublas_v2.h>

// Simple fused RMSNorm kernel for 1D vector (single token)
// y = (x * rsqrt(mean(x^2) + eps)) * w
// Assumes shape H, launched with grid=1
__global__ void rmsnorm_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ y,
    int32_t H,
    float eps)
{
    extern __shared__ float sdata[];
    float* ssum = sdata; // size = blockDim.x
    float local = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xv = __half2float(x[i]);
        local += xv * xv;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();
    // block reduce
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(ssum[0] / max(1, H) + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xv = __half2float(x[i]);
        float ww = __half2float(w[i]);
        float yn = xv * inv_rms * ww;
        y[i] = __float2half(yn);
    }
}

// Elementwise SwiGLU: act = SiLU(g) * u, single-vector
__global__ void swiglu_kernel(
    const __half* __restrict__ gate, // g
    const __half* __restrict__ up,   // u
    __half* __restrict__ act,        // output
    int32_t I)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < I; i += blockDim.x * gridDim.x) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float silu = g / (1.0f + expf(-g)); // SiLU(g)
        float a = silu * u;
        act[i] = __float2half(a);
    }
}

__global__ void capture_decode_kernel(
    const int64_t* __restrict__ ctrl_input,   // [1,1]
    __half* __restrict__ arena_k,             // [L,1,H,max_len,D]
    __half* __restrict__ arena_v,             // [L,1,H,max_len,D]
    int32_t* __restrict__ seq_len_ptr,        // [1]
    const int32_t* __restrict__ position_ptr, // [1] (unused placeholder)
    float* __restrict__ logits_out,           // [1, vocab]
    int32_t L, int32_t H, int32_t max_len, int32_t D, int32_t vocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t T = seq_len_ptr[0];
    int64_t last_token = ctrl_input[0];
    int32_t new_pos = T;

    // zero-fill KV at new_pos (placeholder)
    const int64_t layer_stride = (int64_t)1 * H * max_len * D;
    const int64_t head_stride  = (int64_t)max_len * D;
    const int64_t pos_stride   = (int64_t)D;
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            int64_t base = (int64_t)l * layer_stride + (int64_t)h * head_stride + (int64_t)new_pos * pos_stride;
            for (int d = idx; d < D; d += blockDim.x * gridDim.x) {
                arena_k[base + d] = __float2half(0.0f);
                arena_v[base + d] = __float2half(0.0f);
            }
        }
    }
    // trivial logits
    for (int i = idx; i < vocab; i += blockDim.x * gridDim.x) logits_out[i] = 0.0f;
    __syncthreads();
    if (idx == 0) {
        logits_out[(int)(last_token % vocab)] = 1.0f;
        seq_len_ptr[0] = T + 1;
    }
}

// Simple seq_len increment
__global__ void inc_seq_len_kernel(int32_t* seq_len_ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        seq_len_ptr[0] += 1;
    }
}

extern "C" void capture_decode_host(
    const int64_t* ctrl_input,
    void* arena_k,
    void* arena_v,
    int32_t* seq_len_ptr,
    const int32_t* position_ptr,
    float* logits_out,
    int32_t L, int32_t H, int32_t max_len, int32_t D, int32_t vocab,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((D + block.x - 1) / block.x);
    capture_decode_kernel<<<grid, block, 0, stream>>>(
        ctrl_input,
        reinterpret_cast<__half*>(arena_k),
        reinterpret_cast<__half*>(arena_v),
        seq_len_ptr,
        position_ptr,
        logits_out,
        L, H, max_len, D, vocab);
}

// Keep a single cuBLAS handle alive across calls (capture-safe if created before capture)
static cublasHandle_t g_cublas_handle = nullptr;
static inline void ensure_cublas_handle(cudaStream_t stream) {
    if (g_cublas_handle == nullptr) {
        cublasCreate(&g_cublas_handle);
    }
    cublasSetStream(g_cublas_handle, stream);
}

extern "C" void prepare_cublas_handle(cudaStream_t stream) {
    ensure_cublas_handle(stream);
}

// Extended host launcher with minimal MLP + final RMS + logits
extern "C" void capture_decode_host_ext(
    const int64_t* ctrl_input,
    void* arena_k,
    void* arena_v,
    int32_t* seq_len_ptr,
    const int32_t* position_ptr,
    float* logits_out,
    // scratch
    void* x_norm_scratch,
    void* gate_out_scratch,
    void* up_out_scratch,
    void* act_scratch,
    void* mlp_out_scratch,
    // weights
    const void* rms_in_w,
    const void* rms_post_w,
    const void* rms_final_w,
    const void* mlp_gate_w,
    const void* mlp_up_w,
    const void* mlp_down_w,
    const void* lm_head_w,
    // dims/params
    int32_t L, int32_t H, int32_t max_len, int32_t D, int32_t vocab,
    int32_t I,
    float rms_eps,
    cudaStream_t stream)
{
    // Scratch views
    __half* x_norm = reinterpret_cast<__half*>(x_norm_scratch);     // [H]
    __half* gate_o = reinterpret_cast<__half*>(gate_out_scratch);   // [I]
    __half* up_o   = reinterpret_cast<__half*>(up_out_scratch);     // [I]
    __half* act_o  = reinterpret_cast<__half*>(act_scratch);        // [I]
    __half* mlp_o  = reinterpret_cast<__half*>(mlp_out_scratch);    // [H]

    // Zero scratch (best-effort) to avoid garbage when weights missing
    cudaMemsetAsync(x_norm, 0, sizeof(__half) * H, stream);
    cudaMemsetAsync(gate_o, 0, sizeof(__half) * I, stream);
    cudaMemsetAsync(up_o,   0, sizeof(__half) * I, stream);
    cudaMemsetAsync(act_o,  0, sizeof(__half) * I, stream);
    cudaMemsetAsync(mlp_o,  0, sizeof(__half) * H, stream);
    // Zero logits
    cudaMemsetAsync(logits_out, 0, sizeof(float) * vocab, stream);

    // If required pointers are present, run a minimal MLP+final norm+logits
    bool have_mlp = (mlp_gate_w != nullptr) && (mlp_up_w != nullptr) && (mlp_down_w != nullptr);
    bool have_rms_in = (rms_in_w != nullptr);
    bool have_rms_final = (rms_final_w != nullptr);
    bool have_vocab = (lm_head_w != nullptr);

    // Use cuBLAS for GEMMs (handle created before any graph capture where this runs)
    ensure_cublas_handle(stream);
    cublasHandle_t handle = g_cublas_handle;

    // x input is unknown here; treat x as zeros; x_norm will become zeros
    // Run input RMSNorm if available to write x_norm
    if (have_rms_in) {
        // x is assumed zero; x_norm = 0 after RMSNorm
        // We directly call kernel to compute RMSNorm(zeros,w) -> zeros; but do it anyway for shape
        // Launch RMSNorm kernel: y = (x/||x||)*w; with x=0 -> y=0
        int threads = 256, blocks = 1;
        size_t shmem = threads * sizeof(float);
        rmsnorm_kernel<<<blocks, threads, shmem, stream>>>(
            /*x=*/x_norm, reinterpret_cast<const __half*>(rms_in_w), /*y=*/x_norm, H, rms_eps);
    }

    // Gate/Up GEMMs: [1,H]x[H,I] -> [1,I]
    if (have_mlp) {
        const __half* A = x_norm; // [H]
        const __half* Wg = reinterpret_cast<const __half*>(mlp_gate_w); // [I,H] row-major
        const __half* Wu = reinterpret_cast<const __half*>(mlp_up_w);   // [I,H] row-major
        __half* G = gate_o; // [I]
        __half* U = up_o;   // [I]
        float alpha = 1.0f, beta = 0.0f;
        // Compute G = Wg * A (viewed as column)
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     I, 1, H,
                     &alpha,
                     Wg, CUDA_R_16F, I,
                     A,  CUDA_R_16F, H,
                     &beta,
                     G,  CUDA_R_16F, I,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        // Compute U = Wu * A
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     I, 1, H,
                     &alpha,
                     Wu, CUDA_R_16F, I,
                     A,  CUDA_R_16F, H,
                     &beta,
                     U,  CUDA_R_16F, I,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        // SwiGLU: act = SiLU(G) .* U
        int threads = 256;
        int blocks = (I + threads - 1) / threads;
        swiglu_kernel<<<blocks, threads, 0, stream>>>(G, U, act_o, I);
        // Down GEMM: mlp_o = Wdown * act  where Wdown [H,I]
        const __half* Wd = reinterpret_cast<const __half*>(mlp_down_w);
        alpha = 1.0f; beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     H, 1, I,
                     &alpha,
                     Wd, CUDA_R_16F, H,
                     act_o, CUDA_R_16F, I,
                     &beta,
                     mlp_o, CUDA_R_16F, H,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }

    // Final RMSNorm (optional)
    __half* y_final = mlp_o; // residual x + mlp_o; x assumed 0 so y=mlp_o
    if (have_rms_final) {
        // Overwrite x_norm with final normalized vector
        int threads = 256, blocks = 1;
        size_t shmem = threads * sizeof(float);
        rmsnorm_kernel<<<blocks, threads, shmem, stream>>>(
            y_final, reinterpret_cast<const __half*>(rms_final_w), x_norm, H, rms_eps);
        y_final = x_norm;
    }

    // Logits GEMM: logits_out(V) = W_vocab(V,H) * y_final(H)
    if (have_vocab) {
        const __half* Wv = reinterpret_cast<const __half*>(lm_head_w); // [V,H]
        const __half* A = y_final; // [H]
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     vocab, 1, H,
                     &alpha,
                     Wv, CUDA_R_16F, vocab,
                     A,  CUDA_R_16F, H,
                     &beta,
                     logits_out, CUDA_R_32F, vocab,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }

    // seq_len++
    inc_seq_len_kernel<<<1,1,0,stream>>>(seq_len_ptr);
    // keep global handle alive
}

extern "C" void rmsnorm_host(
    const void* x,
    const void* w,
    void* y,
    int32_t H,
    float eps,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(1); // single-vector RMSNorm
    size_t shmem = block.x * sizeof(float);
    rmsnorm_kernel<<<grid, block, shmem, stream>>>(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(w),
        reinterpret_cast<__half*>(y),
        H, eps);
}

extern "C" void swiglu_host(
    const void* gate,
    const void* up,
    void* act,
    int32_t I,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((I + block.x - 1) / block.x);
    swiglu_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(gate),
        reinterpret_cast<const __half*>(up),
        reinterpret_cast<__half*>(act),
        I);
}