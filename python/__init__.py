"""
m1_gpu_ops: Metal GPU operations for scientific computing on Apple Silicon.

Usage:
    import m1_gpu_ops

    ctx = m1_gpu_ops.MetalContext()
    ctx.load_library("path/to/ops.metallib")

    result = m1_gpu_ops.add_arrays(ctx, x, y)
"""

from ._m1_gpu_ops import (
    MetalContext,
    GpuArray,
    add_arrays,
    multiply_arrays,
    saxpy,
    laplacian2d,
    laplacian2d9p,
    mandelbrot,
    nbody_step,
    nbody_simulate,
    diffuse_steps,
    elastic_wave_propagate,
)

__all__ = [
    "MetalContext",
    "GpuArray",
    "add_arrays",
    "multiply_arrays",
    "saxpy",
    "laplacian2d",
    "laplacian2d9p",
    "mandelbrot",
    "nbody_step",
    "nbody_simulate",
    "diffuse_steps",
    "elastic_wave_propagate",
]
