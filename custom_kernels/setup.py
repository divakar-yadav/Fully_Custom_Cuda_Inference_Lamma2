from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extra_cflags = []
extra_nvcc = ['-O3', '--use_fast_math']

# Match venv ABI if set
if os.environ.get('PYTORCH_CXX11_ABI', '0') == '0':
    extra_cflags.append('-D_GLIBCXX_USE_CXX11_ABI=0')
else:
    extra_cflags.append('-D_GLIBCXX_USE_CXX11_ABI=1')

setup(
    name='case4_custom_kernels',
    ext_modules=[
        CUDAExtension(
            name='capture_decode_step',
            sources=['capture_decode_step.cpp', 'capture_decode_step_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'] + extra_cflags, 'nvcc': extra_nvcc}
        ),
        CUDAExtension(
            name='qkv_gemm',
            sources=['qkv_gemm.cpp', 'qkv_gemm_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'] + extra_cflags, 'nvcc': extra_nvcc}
        ),
        CUDAExtension(
            name='d2d_row_copy',
            sources=['d2d_row_copy.cpp'],
            extra_compile_args={'cxx': ['-O3'] + extra_cflags, 'nvcc': extra_nvcc}
        ),
        CUDAExtension(
            name='attn_varlen',
            sources=['attn_varlen.cpp', 'attn_varlen_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'] + extra_cflags, 'nvcc': extra_nvcc}
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)


