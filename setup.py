import subprocess
import os
import torch
import platform
from packaging.version import parse, Version
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# package name managed by pip, which can be remove by `pip uninstall tiny_pkg`
PACKAGE_NAME = "q8_kernels"
system_name = platform.system()

ext_modules = []
generator_flag = []
cc_flag = []
cc_flag.append("--gpu-architecture=native")

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

# Original imports at the top of setup.py:
# import subprocess
# import os
# import torch
# import platform
# from packaging.version import parse, Version
# from pathlib import Path
# from setuptools import setup, find_packages
# from torch.utils.cpp_extension import (
#     BuildExtension,
#     CUDAExtension,
#     CUDA_HOME,
# )
# from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# ... (other functions like get_cuda_bare_metal_version remain the same) ...

def get_device_arch():
    forced_arch_cc = os.environ.get('FORCE_CUDA_ARCH_CC')
    major = -1
    minor = -1

    if forced_arch_cc:
        print(f"Attempting to use forced CUDA compute capability from FORCE_CUDA_ARCH_CC: '{forced_arch_cc}'")
        try:
            major_str, minor_str = forced_arch_cc.strip().split('.')
            major, minor = int(major_str), int(minor_str)
            print(f"Successfully parsed forced CUDA CC: {major}.{minor}")
        except ValueError:
            print(f"Error: Invalid FORCE_CUDA_ARCH_CC format. Expected 'MAJ.MIN', got '{forced_arch_cc}'.")
            raise ValueError(f"Invalid FORCE_CUDA_ARCH_CC format: '{forced_arch_cc}'. Please set it like '7.5', '8.6', etc.")
    else:
        print("FORCE_CUDA_ARCH_CC not set. Attempting to detect GPU architecture via torch.cuda...")
        try:
            if not torch.cuda.is_available():
                print("Warning: torch.cuda.is_available() returned False.")
                print("Cannot detect GPU architecture. Please set FORCE_CUDA_ARCH_CC (e.g., '7.5').")
                raise RuntimeError("CUDA not available, and FORCE_CUDA_ARCH_CC not set. Cannot determine target architecture.")

            major, minor = torch.cuda.get_device_capability(0) # type: ignore
            print(f"Detected GPU architecture via torch.cuda: {major}.{minor}")
        except Exception as e:
            print(f"Error: Could not detect GPU architecture via torch.cuda: {e}")
            print("Please ensure an NVIDIA GPU is available and drivers are installed, or set FORCE_CUDA_ARCH_CC (e.g., '7.5').")
            raise # Re-raise the original exception

    # Now, convert major, minor to the architecture name string expected by the script
    if major == 8 and (minor >= 0 and minor < 9): # Covers 8.0, 8.6, 8.7 etc.
        return "ampere"
    if major == 8 and minor == 9:
        return "ada"
    if major == 9 and minor == 0:
        return "hopper"
    # The original script listed 12 for Blackwell, let's assume Blackwell is CC 10.0 or similar for this example,
    # as CC 12.x is very future. If Blackwell is indeed major 12, the original logic holds.
    # For now, let's stick to what's common or adjust if specific Blackwell CC is known.
    # The original script doesn't have a direct mapping for CC 12.x to "blackwell" string by checking minor.
    # Let's assume a future CC for Blackwell for placeholder.
    # Given the original script's explicit `if major == 12: return "blackwell"`, we'll honor that.
    if major == 12: # Based on original script logic
        return "blackwell"
    
    # For older architectures like Turing (7.5), Pascal (6.x), etc.,
    # they are not "ampere", "ada", "hopper", or "blackwell".
    # The script will correctly not enable `should_compile_fp8_fast_acc` for these.
    # We can return a generic name or the CC string itself if no specific name matches.
    # Or, if no specific name is needed for older archs, we can just let it fall through
    # and the `should_compile_fp8_fast_acc` logic will handle it.
    # Let's return a generic "older_arch" or the CC for clarity if none of the special names match.
    print(f"Architecture CC {major}.{minor} does not map to a specific named architecture (ampere, ada, hopper, blackwell). Proceeding with general compilation.")
    return f"cc_{major}_{minor}" # This ensures `device_arch` is a string, and won't match special ones.

# ... (the rest of the setup.py, like `this_dir = Path(__file__).parent` etc. remains the same)
    
this_dir = Path(__file__).parent
device_arch = get_device_arch()
should_compile_fp8_fast_acc = device_arch in ["ada", "blackwell"]
if should_compile_fp8_fast_acc:
    subprocess.run(["git", "submodule", "update", "--init", "third_party/cutlass"], check=True)

ext_modules.append(
    CUDAExtension(
        # package name for import
        name="q8_kernels_cuda.ops._C",
        sources=[
            "csrc/fast_hadamard/fast_hadamard_transform.cpp",
            "csrc/ops/ops_api.cpp",
            "csrc/fast_hadamard/fast_hadamard_transform_cuda.cu",
            "csrc/fast_hadamard/fused_hadamard_transform_cuda.cu",
            "csrc/fast_hadamard/rms_norm_rope_cuda.cu",
            "csrc/fast_hadamard/dequant_fast_hadamard_transform_cuda.cu"
        ],
          extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-lineinfo",
                    "--ptxas-options=-v",
                    "--ptxas-options=-O2",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",

                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            this_dir / "csrc" / "fast_hadamard",
        ],
    )
)

if should_compile_fp8_fast_acc:
    ext_modules.append(
        CUDAExtension(
            name="q8_kernels_cuda.gemm._C",
            sources=[
                "csrc/gemm/fp8_gemm.cpp",
                "csrc/gemm/fp8_gemm_cuda.cu",
                "csrc/gemm/fp8_gemm_bias.cu",
                
            ],
            extra_compile_args={
                # add c compile flags
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-lineinfo",
                        "--ptxas-options=-v",
                        "--ptxas-options=-O2",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    ]
                    + generator_flag
                    + cc_flag,
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "gemm",
                Path(this_dir) / "third_party/cutlass/include",
                Path(this_dir) / "third_party/cutlass/tools/utils/include" ,
                Path(this_dir) / "third_party/cutlass/examples/common" ,
            ],
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="q8_kernels_cuda.flash_attention._C",
            sources=[
                "csrc/flash_attention/flash_attention.cpp",
                "csrc/flash_attention/flash_attention_cuda.cu",
                
            ],
            extra_compile_args={
                # add c compile flags
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-lineinfo",
                        "--ptxas-options=-v",
                        "--ptxas-options=-O2",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    ]
                    + generator_flag
                    + cc_flag,
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attention",
                Path(this_dir) / "third_party/cutlass/include",
                Path(this_dir) / "third_party/cutlass/tools/utils/include" ,
                Path(this_dir) / "third_party/cutlass/examples/common" ,
            ],
        )
    )

setup(
    name=PACKAGE_NAME,
    version="0.0.5",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="8bit kernels",
    ext_modules=ext_modules,
    cmdclass={ "build_ext": BuildExtension, "bdist_wheel": _bdist_wheel},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "wheel",
        "packaging",
        "ninja",
        "setuptools",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    author="KONAKONA666/Aibek Bekbayev",
)