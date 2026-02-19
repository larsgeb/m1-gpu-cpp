# Metal GPU Computing for Scientific C++ on Apple Silicon

GPU-accelerated scientific computing on Apple Silicon using Metal Shading Language (MSL) and C++ ([metal-cpp](https://github.com/bkaradzic/metal-cpp)), with Python bindings via [pybind11](https://pybind11.readthedocs.io/).

This repository accompanies _"Seamless GPU acceleration for C++ based physics using the M1's unified processing units"_ ([arXiv:2206.01791](https://arxiv.org/abs/2206.01791)).

Blog posts:
- [Getting started](https://larsgeb.github.io/2022/04/20/m1-gpu.html)
- [SAXPY and FD](https://larsgeb.github.io/2022/04/22/m1-gpu.html)

## Examples

| Directory | Description | Metal Kernels |
|---|---|---|
| `01-MetalAdder` | Basic GPU array addition | `add_arrays` |
| `02-GeneralArrayOperations` | 1D ops: add, multiply, SAXPY, central difference | `add_arrays`, `multiply_arrays`, `saxpy`, `central_diff` |
| `03-2DKernels` | 2D spatial kernels: quadratic, Laplacian (5pt & 9pt) | `quadratic2d`, `laplacian2d`, `laplacian2d9p` |
| `04-Compute` | Compute-heavy: Mandelbrot fractal, N-body gravity | `mandelbrot`, `nbody_forces`, `nbody_integrate` |
| `05-WavePropagation` | 2D elastic wave propagation (Virieux staggered grid) | `stress_update`, `velocity_update`, `apply_damping` |

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Xcode and Command Line Tools (`xcode-select --install`)
- CMake 3.21+ (`brew install cmake`)
- For CPU baselines: `brew install llvm libomp`
- For Python bindings: `python3`, `numpy`, `pybind11`

## Building

```bash
git clone --recurse-submodules https://github.com/larsgeb/m1-gpu-cpp.git
cd m1-gpu-cpp

cmake -B build
cmake --build build
```

Run an example:
```bash
cd build/02-GeneralArrayOperations && ./example_02
```

## Tests

```bash
cd build && ctest --output-on-failure
```

Tests verify GPU results against CPU reference implementations.

## Python bindings

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pybind11 matplotlib jupytext

cmake -B build -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build
```

Quick test:
```python
import sys; sys.path.insert(0, "build")
import numpy as np
import m1_gpu_ops as metal

ctx = metal.MetalContext()
ctx.load_library("build/02-GeneralArrayOperations/ops.metallib")

x = np.random.rand(100_000).astype(np.float32)
y = np.random.rand(100_000).astype(np.float32)
result = metal.add_arrays(ctx, x, y)
```

### Available functions

| Function | Library | Description |
|---|---|---|
| `add_arrays(ctx, x, y)` | `02-…/ops.metallib` | Element-wise addition |
| `multiply_arrays(ctx, x, y)` | `02-…/ops.metallib` | Element-wise multiplication |
| `saxpy(ctx, a, x, y)` | `02-…/ops.metallib` | Scalar a*x + y |
| `laplacian2d(ctx, u)` | `03-…/ops.metallib` | 5-point Laplacian stencil |
| `laplacian2d9p(ctx, u)` | `03-…/ops.metallib` | 9-point Laplacian stencil |
| `mandelbrot(ctx, …)` | `04-…/ops.metallib` | Mandelbrot set rendering |
| `nbody_step(ctx, …)` | `04-…/ops.metallib` | Single N-body gravity step |
| `nbody_simulate(ctx, …)` | `04-…/ops.metallib` | Multi-step N-body (data on GPU) |
| `diffuse_steps(ctx, …)` | `03-…/ops.metallib` | Multi-step heat diffusion (data on GPU) |
| `elastic_wave_propagate(ctx, …)` | `05-…/ops.metallib` | 2D elastic wave propagation |

## Notebooks

The demo notebooks are stored as [jupytext](https://jupytext.readthedocs.io/) light-format `.py` files for clean version control. To open them:

```bash
# Convert to .ipynb and open in Jupyter
jupytext --to notebook demo.py
jupyter notebook demo.ipynb
```

| Notebook | Description |
|---|---|
| `demo.py` | 1D/2D operations, Mandelbrot, N-body, diffusion — with GPU vs NumPy benchmarks |
| `wave_demo.py` | 2D elastic wave propagation: homogeneous/layered media, seismograms, GPU benchmarks |

## Project structure

```
01-MetalAdder/              # Example 1: Basic GPU array addition
02-GeneralArrayOperations/  # Example 2: 1D array operations
03-2DKernels/               # Example 3: 2D spatial kernels
04-Compute/                 # Example 4: Mandelbrot & N-body
05-WavePropagation/         # Example 5: 2D elastic wave propagation
metal-cpp/                  # metal-cpp headers (git submodule)
python/                     # pybind11 bindings & Python package
tests/                      # CTest correctness tests
paper/                      # Benchmark scripts & figures for arXiv paper
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).
