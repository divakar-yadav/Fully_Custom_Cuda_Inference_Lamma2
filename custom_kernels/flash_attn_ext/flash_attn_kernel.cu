#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void sdp_single_q_kernel(
    const scalar_t* __restrict__ q,   // [B,H,1,D]
    const scalar_t* __restrict__ k,   // [B,H,T,D]
    const scalar_t* __restrict__ v,   // [B,H,T,D]
    scalar_t* __restrict__ out,       // [B,H,1,D]
    int B, int H, int T, int D, float scale)
{
    int b = blockIdx.x;
    int h = blockIdx.y;

    const scalar_t* q_ptr = q + (b*H + h)*D; // q len=1
    const scalar_t* k_ptr = k + (b*H)*T*D + h*T*D;
    const scalar_t* v_ptr = v + (b*H)*T*D + h*T*D;
    extern __shared__ float shared[];
    float* scores = shared; // size T

    // compute scores[t] = dot(q, k[t]) * scale
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float acc = 0.0f;
        const scalar_t* kt = k_ptr + t*D;
        for (int d = 0; d < D; ++d) {
            acc += static_cast<float>(q_ptr[d]) * static_cast<float>(kt[d]);
        }
        scores[t] = acc * scale;
    }
    __syncthreads();

    // softmax over scores
    // find max
    float maxv = -1e30f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        maxv = fmaxf(maxv, scores[t]);
    }
    __shared__ float smax;
    if (threadIdx.x == 0) smax = -1e30f;
    __syncthreads();
    atomicMax((int*)&smax, __float_as_int(maxv));
    __syncthreads();
    float sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float e = expf(scores[t] - smax);
        scores[t] = e;
        sum += e;
    }
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.0f;
    __syncthreads();
    atomicAdd(&ssum, sum);
    __syncthreads();

    // context = sum_t (scores[t]/ssum) * v[t]
    for (int d0 = threadIdx.x; d0 < D; d0 += blockDim.x) {
        float c = 0.0f;
        for (int t = 0; t < T; ++t) {
            const scalar_t* vt = v_ptr + t*D;
            c += (scores[t] / ssum) * static_cast<float>(vt[d0]);
        }
        out[(b*H + h)*D + d0] = static_cast<scalar_t>(c);
    }
}

torch::Tensor sdp_single_q(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4, "q,k,v must be [B,H,L,D]");
    int64_t B = q.size(0);
    int64_t H = q.size(1);
    int64_t Lq = q.size(2);
    int64_t D = q.size(3);
    int64_t T = k.size(2);
    TORCH_CHECK(Lq==1, "Only single-query supported");
    auto out = torch::empty({B,H,1,D}, q.options());

    dim3 blocks(B, H);
    int threads = 128;
    size_t shmem = sizeof(float) * T;
    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "sdp_single_q", ([&] {
        sdp_single_q_kernel<scalar_t><<<blocks, threads, shmem>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            (int)B,(int)H,(int)T,(int)D,(float)scale);
    }));
    return out;
}


