"""
Type stubs for the _m1_gpu_ops pybind11 C extension.

These stubs describe the Python-visible interface so that type checkers (Ty, mypy,
Pyright) can analyse code that imports from m1_gpu_ops without needing the compiled
.so/.dylib to be present.
"""

from collections.abc import Sequence

from numpy import ndarray

# ---------------------------------------------------------------------------
# MetalContext — lightweight GPU context
# ---------------------------------------------------------------------------

class MetalContext:
    def __init__(self) -> None: ...
    def load_library(self, path: str) -> None: ...
    @property
    def device_name(self) -> str: ...

# ---------------------------------------------------------------------------
# GpuArray — zero-copy wrapper around a shared-memory Metal buffer
# ---------------------------------------------------------------------------

class GpuArray:
    shape: list[int]
    size: int

    def __init__(self, ctx: MetalContext, shape: Sequence[int]) -> None: ...
    @staticmethod
    def zeros(ctx: MetalContext, shape: Sequence[int]) -> GpuArray: ...
    @staticmethod
    def from_numpy(ctx: MetalContext, arr: ndarray) -> GpuArray: ...
    def numpy(self) -> ndarray: ...
    def to_numpy(self) -> ndarray: ...

# ---------------------------------------------------------------------------
# 1D operations (02-GeneralArrayOperations)
# ---------------------------------------------------------------------------

def add_arrays(ctx: MetalContext, x: ndarray, y: ndarray) -> ndarray: ...
def multiply_arrays(ctx: MetalContext, x: ndarray, y: ndarray) -> ndarray: ...
def saxpy(ctx: MetalContext, alpha: float, x: ndarray, y: ndarray) -> ndarray: ...

# ---------------------------------------------------------------------------
# 2D operations (03-2DKernels)
# ---------------------------------------------------------------------------

def laplacian2d(ctx: MetalContext, X: ndarray) -> ndarray: ...
def laplacian2d9p(ctx: MetalContext, X: ndarray) -> ndarray: ...
def diffuse_steps(
    ctx: MetalContext,
    field: ndarray,
    dt: float = ...,
    n_steps: int = ...,
) -> ndarray: ...

# ---------------------------------------------------------------------------
# Compute-heavy operations (04-Compute)
# ---------------------------------------------------------------------------

def mandelbrot(
    ctx: MetalContext,
    width: int,
    height: int,
    x_min: float = ...,
    x_max: float = ...,
    y_min: float = ...,
    y_max: float = ...,
    max_iter: int = ...,
) -> ndarray: ...
def nbody_step(
    ctx: MetalContext,
    pos_mass: ndarray,
    velocities: ndarray,
    dt: float = ...,
    softening: float = ...,
) -> tuple[ndarray, ndarray]: ...
def nbody_simulate(
    ctx: MetalContext,
    pos_mass: ndarray,
    velocities: ndarray,
    dt: float = ...,
    softening: float = ...,
    n_steps: int = ...,
) -> tuple[ndarray, ndarray]: ...

# ---------------------------------------------------------------------------
# Elastic wave propagation (05-WavePropagation)
# ---------------------------------------------------------------------------

def elastic_wave_propagate(
    ctx: MetalContext,
    vp: ndarray,
    vs: ndarray,
    rho: ndarray,
    src_x: int,
    src_z: int,
    wavelet: ndarray,
    recv_x: ndarray,
    recv_z: ndarray,
    dx: float,
    dz: float,
    dt: float,
    snapshot_interval: int = ...,
    n_boundary: int = ...,
) -> tuple[ndarray, ndarray, ndarray, ndarray]: ...

__all__ = [
    "MetalContext",
    "GpuArray",
    "add_arrays",
    "multiply_arrays",
    "saxpy",
    "laplacian2d",
    "laplacian2d9p",
    "diffuse_steps",
    "mandelbrot",
    "nbody_step",
    "nbody_simulate",
    "elastic_wave_propagate",
]
