#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#
static __device__ inline float h2f(__half x) { return __half2float(x); }
static __device__ inline __half f2h(float x) { return __float2half(x); }
#
__global__ void attn_varlen_kernel_impl(
    const __half* __restrict__ q,  // [H,D]
    const __half* __restrict__ k,  // [H,T,D]
    const __half* __restrict__ v,  // [H,T,D]
    int32_t H, int32_t T, int32_t D,
    float scale,
    __half* __restrict__ ctx_out   // [H,D]
) {
    int h = blockIdx.x; // one block per head
    if (h >= H) return;
    // Rotary base (LLaMA-style)
    // Step 1: max score
    __shared__ float max_score_sh;
    if (threadIdx.x == 0) max_score_sh = -INFINITY;
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const __half* qh = q + h * D;
        const __half* kth = k + ( (h * T + t) * D );
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            dot += h2f(qh[d]) * h2f(kth[d]);
        }
        __shared__ float red_buf[256];
        int tid = threadIdx.x;
        red_buf[tid] = dot;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) red_buf[tid] += red_buf[tid + s];
            __syncthreads();
        }
        float score = red_buf[0] * scale;
        if (tid == 0 && score > max_score_sh) max_score_sh = score;
        __syncthreads();
    }
    __syncthreads();
    // Step 2: sum_exp
    float sum_exp = 0.0f;
    for (int t = 0; t < T; ++t) {
        const __half* qh = q + h * D;
        const __half* kth = k + ( (h * T + t) * D );
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            dot += h2f(qh[d]) * h2f(kth[d]);
        }
        __shared__ float red_buf2[256];
        int tid = threadIdx.x;
        red_buf2[tid] = dot;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) red_buf2[tid] += red_buf2[tid + s];
            __syncthreads();
        }
        float score = red_buf2[0] * scale;
        float e = expf(score - max_score_sh);
        if (tid == 0) sum_exp += e;
        __syncthreads();
    }
    __shared__ float sum_exp_sh;
    if (threadIdx.x == 0) sum_exp_sh = sum_exp;
    __syncthreads();
    // Step 3: context
    for (int d0 = threadIdx.x; d0 < D; d0 += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T; ++t) {
            const __half* qh = q + h * D;
            const __half* kth = k + ( (h * T + t) * D );
            const __half* vht = v + ( (h * T + t) * D );
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += h2f(qh[d]) * h2f(kth[d]);
            }
            float score = dot * scale;
            float p = expf(score - max_score_sh) / sum_exp_sh;
            acc += p * h2f(vht[d0]);
        }
        ctx_out[h * D + d0] = f2h(acc);
    }
}
#
extern "C" void attn_varlen_host(
    const __half* q,
    const __half* k,
    const __half* v,
    int32_t H, int32_t T, int32_t D,
    float scale,
    __half* ctx_out,
    cudaStream_t stream) {
    dim3 grid(H);
    dim3 block(min(256, ((D + 31) / 32) * 32));
    attn_varlen_kernel_impl<<<grid, block, 0, stream>>>(q, k, v, H, T, D, scale, ctx_out);
}

