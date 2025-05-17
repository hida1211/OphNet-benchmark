from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="ophnet",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="nms_1d_cpu",
            sources=[
                str(ROOT / "baselines/task2/talnets/TriDet/libs/utils/csrc/nms_cpu.cpp")
            ],
            extra_compile_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
