"""
m1_gpu_ops: Metal GPU operations for scientific computing on Apple Silicon.

Usage:
    import m1_gpu_ops

    ctx = m1_gpu_ops.MetalContext()
    ctx.load_library("path/to/ops.metallib")

    result = m1_gpu_ops.add_arrays(ctx, x, y)
"""

from ._m1_gpu_ops import (
    GpuArray,
    MetalContext,
    add_arrays,
    diffuse_steps,
    elastic_wave_propagate,
    laplacian2d,
    laplacian2d9p,
    mandelbrot,
    multiply_arrays,
    nbody_simulate,
    nbody_step,
    saxpy,
)

__all__ = [
    "GpuArray",
    "MetalContext",
    "add_arrays",
    "diffuse_steps",
    "elastic_wave_propagate",
    "laplacian2d",
    "laplacian2d9p",
    "mandelbrot",
    "multiply_arrays",
    "nbody_simulate",
    "nbody_step",
    "saxpy",
]
