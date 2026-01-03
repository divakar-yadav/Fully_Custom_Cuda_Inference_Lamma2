// Minimal D2D row copy for embedding: copies one row (token_id) of size hidden
// from a packed embedding base pointer (device) into a preallocated CUDA tensor.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#include <cuda.h>
#endif

static void cuda_check(cudaError_t st, const char* msg) {
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + msg + ": " + cudaGetErrorString(st));
    }
}

// emb_ptr: device address to base of embedding [rows, hidden] fp16
// rows, hidden: shape dims
// token_id: which row to copy
// out: preallocated CUDA tensor [hidden], fp16, contiguous
static void copy_emb_row(
    uint64_t emb_ptr,
    int64_t rows,
    int64_t hidden,
    int64_t token_id,
    at::Tensor out
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA tensor");
    TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be fp16 (Half)");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out.numel() == hidden, "out.numel() must equal hidden");
    TORCH_CHECK(token_id >= 0 && token_id < rows, "token_id out of range");

    const size_t elem_size = 2; // fp16
    const size_t row_bytes = static_cast<size_t>(hidden) * elem_size;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(emb_ptr);
    const uint8_t* src = base + static_cast<size_t>(token_id) * row_bytes;
    void* dst = out.data_ptr();

    auto stream = at::cuda::getCurrentCUDAStream();
    cuda_check(cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyDeviceToDevice, stream.stream()),
               "cudaMemcpyAsync(copy_emb_row)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_emb_row", &copy_emb_row, "Device-to-device copy of one embedding row");
}


