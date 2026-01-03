from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attn_ext',
    ext_modules=[
        CUDAExtension(
            name='flash_attn_ext',
            sources=['binding.cpp', 'flash_attn_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)


