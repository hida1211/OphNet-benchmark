import torch
from pathlib import Path

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent


setup(
    name='nms_1d_cpu',
    ext_modules=[
        CppExtension(
            name='nms_1d_cpu',
            sources=[str(ROOT / 'csrc' / 'nms_cpu.cpp')],
            extra_compile_args=['-fopenmp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
