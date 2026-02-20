# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Metal GPU Computing on Apple Silicon — Python Demo
#
# This notebook demonstrates GPU-accelerated scientific computing operations on Apple Silicon using Metal Shading Language (MSL) via C++ (metal-cpp) and Python (pybind11).
#
# **Operations available:**
# - **1D:** `add_arrays`, `multiply_arrays`, `saxpy` (scalar-alpha-x-plus-y)
# - **2D:** `laplacian2d` (5-point stencil), `laplacian2d9p` (9-point stencil)
# - **Compute-heavy:** `mandelbrot` (fractal rendering), `nbody_step` / `nbody_simulate` (gravitational N-body)
# - **Multi-step:** `diffuse_steps` (heat diffusion with data kept on GPU)
#
# All GPU kernels run on the Apple GPU via Metal compute shaders and are compared against NumPy CPU implementations.

# +
import sys, os

sys.path.insert(0, os.path.join(os.getcwd(), "build"))

import numpy as np
import matplotlib.pyplot as plt
import time

import _m1_gpu_ops as metal

# %matplotlib inline
plt.rcParams["figure.dpi"] = 120
# -

# ## 1. Initialize Metal Context
#
# We create a `MetalContext` which connects to the default Metal GPU device, then load the pre-compiled `.metallib` shader libraries.

# +
# Context for 1D operations (add, multiply, saxpy, central_difference)
ctx_1d = metal.MetalContext()
ctx_1d.load_library("build/02-GeneralArrayOperations/ops.metallib")

# Context for 2D operations (laplacian2d, laplacian2d9p, quadratic2d)
ctx_2d = metal.MetalContext()
ctx_2d.load_library("build/03-2DKernels/ops.metallib")

print(f"GPU device: {ctx_1d.device_name}")
# -

# ## 2. 1D Array Operations — Correctness
#
# Let's verify that the GPU produces the same results as NumPy for all 1D operations.

# +
N = 100_000
x = np.random.rand(N).astype(np.float32)
y = np.random.rand(N).astype(np.float32)
alpha = 2.5

# Addition: x + y
gpu_add = metal.add_arrays(ctx_1d, x, y)
cpu_add = x + y
print(
    f"add_arrays  — max error: {np.max(np.abs(gpu_add - cpu_add)):.2e}  {'PASS' if np.allclose(gpu_add, cpu_add) else 'FAIL'}"
)

# Multiplication: x * y
gpu_mul = metal.multiply_arrays(ctx_1d, x, y)
cpu_mul = x * y
print(
    f"multiply    — max error: {np.max(np.abs(gpu_mul - cpu_mul)):.2e}  {'PASS' if np.allclose(gpu_mul, cpu_mul) else 'FAIL'}"
)

# SAXPY: alpha * x + y
gpu_saxpy = metal.saxpy(ctx_1d, alpha, x, y)
cpu_saxpy = alpha * x + y
print(
    f"saxpy       — max error: {np.max(np.abs(gpu_saxpy - cpu_saxpy)):.2e}  {'PASS' if np.allclose(gpu_saxpy, cpu_saxpy) else 'FAIL'}"
)


# -

# ## 3. GPU vs NumPy Benchmarks — 1D Operations
#
# We benchmark Metal GPU against NumPy (which uses Accelerate/BLAS on macOS) across varying array sizes. The GPU has overhead for small arrays but should win on large ones.


# +
def benchmark(func, *args, repeats=100, warmup=15):
    """Time a function call, returning median time in microseconds."""
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    return np.median(times), np.std(times)


sizes = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
]
ops = {
    "add": {
        "gpu": lambda x, y: metal.add_arrays(ctx_1d, x, y),
        "numpy": lambda x, y: x + y,
    },
    "multiply": {
        "gpu": lambda x, y: metal.multiply_arrays(ctx_1d, x, y),
        "numpy": lambda x, y: x * y,
    },
    "saxpy": {
        "gpu": lambda x, y: metal.saxpy(ctx_1d, 2.5, x, y),
        "numpy": lambda x, y: 2.5 * x + y,
    },
}

results = {op: {"gpu": [], "numpy": [], "gpu_std": [], "numpy_std": []} for op in ops}

for n in sizes:
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    for op_name, funcs in ops.items():
        for backend in ["gpu", "numpy"]:
            med, std = benchmark(funcs[backend], x, y, repeats=30)
            results[op_name][backend].append(med)
            results[op_name][f"{backend}_std"].append(std)
    print(f"  N = {n:>12,d}  done")

print("Benchmarking complete.")

# +
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

for ax, (op_name, data) in zip(axes, results.items()):
    ax.loglog(sizes, data["gpu"], "o-", label="Metal GPU", color="#FF6B35", linewidth=2)
    ax.loglog(sizes, data["numpy"], "s--", label="NumPy (CPU)", color="#004E89", linewidth=2)
    ax.set_xlabel("Array length")
    ax.set_title(op_name.upper(), fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

axes[0].set_ylabel("Time (microseconds)")
fig.suptitle(f"Metal GPU vs NumPy — {ctx_1d.device_name}", fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()
# -

# ## 4. 2D Laplacian Stencils
#
# The Laplacian operator is fundamental to PDE solvers (heat equation, wave equation, etc.). We implement two variants on the GPU:
#
# - **5-point stencil** (`laplacian2d`): Uses 4 direct neighbors (N, S, E, W)
# - **9-point stencil** (`laplacian2d9p`): Also uses 4 diagonal neighbors, more accurate
#
# $$\nabla^2 f \approx f_{i-1,j} + f_{i+1,j} + f_{i,j-1} + f_{i,j+1} - 4f_{i,j}$$

# +
# Create a smooth 2D test function: a Gaussian bump
rows, cols = 256, 256
yy, xx = np.meshgrid(np.linspace(-2, 2, cols), np.linspace(-2, 2, rows))
sigma = 0.5
field = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)

# Compute Laplacians on GPU
lap_5pt = metal.laplacian2d(ctx_2d, field)
lap_9pt = metal.laplacian2d9p(ctx_2d, field)

# Analytical Laplacian of a Gaussian: nabla^2 exp(-r^2/2s^2) = (r^2 - 2s^2) / s^4 * exp(-r^2/2s^2)
r2 = xx**2 + yy**2
lap_analytical = ((r2 - 2 * sigma**2) / sigma**4) * field

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

im0 = axes[0, 0].imshow(field, cmap="RdBu_r", extent=[-2, 2, -2, 2])
axes[0, 0].set_title("Input: Gaussian bump")
plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

vmax = np.max(np.abs(lap_5pt)) * 0.8
im1 = axes[0, 1].imshow(lap_5pt, cmap="RdBu_r", extent=[-2, 2, -2, 2], vmin=-vmax, vmax=vmax)
axes[0, 1].set_title("GPU: 5-point Laplacian")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

im2 = axes[1, 0].imshow(lap_9pt, cmap="RdBu_r", extent=[-2, 2, -2, 2], vmin=-vmax, vmax=vmax)
axes[1, 0].set_title("GPU: 9-point Laplacian")
plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

im3 = axes[1, 1].imshow(lap_analytical, cmap="RdBu_r", extent=[-2, 2, -2, 2], vmin=-vmax, vmax=vmax)
axes[1, 1].set_title("Analytical Laplacian")
plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)

for ax in axes.flat:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle("Laplacian of a Gaussian — GPU Metal compute shaders", fontweight="bold")
fig.tight_layout()
plt.show()


# -

# ## 5. 2D Laplacian Benchmark — GPU vs NumPy
#
# For 2D stencil operations, the GPU dispatch uses a 2D thread grid that maps directly to the array dimensions. Let's benchmark against a NumPy implementation of the same 5-point stencil.


# +
def numpy_laplacian5(X):
    """NumPy 5-point Laplacian stencil."""
    result = np.zeros_like(X)
    result[1:-1, 1:-1] = X[:-2, 1:-1] + X[2:, 1:-1] + X[1:-1, :-2] + X[1:-1, 2:] - 4 * X[1:-1, 1:-1]
    return result


grid_sizes = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4000, 4000),
]

gpu_times = []
numpy_times = []
labels = []

for r, c in grid_sizes:
    X = np.random.rand(r, c).astype(np.float32)

    t_gpu, _ = benchmark(metal.laplacian2d, ctx_2d, X, repeats=30)
    t_np, _ = benchmark(numpy_laplacian5, X, repeats=30)

    gpu_times.append(t_gpu)
    numpy_times.append(t_np)
    labels.append(f"{r}x{c}")
    total_elements = r * c
    print(
        f"  {r:>5d} x {c:<5d} ({total_elements:>12,d} elements)  GPU: {t_gpu:>10.0f} us   NumPy: {t_np:>10.0f} us   speedup: {t_np / t_gpu:.1f}x"
    )

# +
total_elements = [r * c for r, c in grid_sizes]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.loglog(total_elements, gpu_times, "o-", label="Metal GPU", color="#FF6B35", linewidth=2)
ax1.loglog(total_elements, numpy_times, "s--", label="NumPy (CPU)", color="#004E89", linewidth=2)
ax1.set_xlabel("Grid elements (rows x cols)")
ax1.set_ylabel("Time (microseconds)")
ax1.set_title("2D Laplacian (5-point stencil)", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, which="both")

speedup = [n / g for n, g in zip(numpy_times, gpu_times)]
ax2.semilogx(total_elements, speedup, "D-", color="#2D936C", linewidth=2, markersize=8)
ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Grid elements (rows x cols)")
ax2.set_ylabel("Speedup (NumPy time / GPU time)")
ax2.set_title("GPU Speedup Factor", fontweight="bold")
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle(f"2D Laplacian Benchmark — {ctx_2d.device_name}", fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()
# -

# ## 6. Iterative Diffusion — GPU in a Loop
#
# A common use case: repeatedly applying the Laplacian to simulate diffusion. Each iteration is a single GPU dispatch. We evolve the heat equation $\frac{\partial u}{\partial t} = \alpha \nabla^2 u$ with a simple forward Euler step.

# +
# Initial condition: hot spot in the center of a 512x512 grid
N = 512
u = np.zeros((N, N), dtype=np.float32)
u[N // 2 - 20 : N // 2 + 20, N // 2 - 20 : N // 2 + 20] = 1.0  # hot square

dt = 0.1  # time step
n_steps = 500  # number of iterations

# Store snapshots for visualization
snapshots = [u.copy()]
snapshot_steps = [0, 50, 150, 500]

t0 = time.perf_counter()
for step in range(1, n_steps + 1):
    lap = metal.laplacian2d(ctx_2d, u)
    # SAXPY: u_new = dt * lap + u  (forward Euler step)
    u = metal.saxpy(ctx_1d, dt, lap.ravel(), u.ravel()).reshape(N, N)
    if step in snapshot_steps:
        snapshots.append(u.copy())
elapsed = time.perf_counter() - t0
print(
    f"Ran {n_steps} diffusion steps on {N}x{N} grid in {elapsed:.2f}s ({elapsed / n_steps * 1e3:.1f} ms/step)"
)

fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.5))
for ax, snap, step in zip(axes, snapshots, [0] + snapshot_steps):
    im = ax.imshow(snap, cmap="inferno", vmin=0, vmax=0.5, extent=[0, N, 0, N])
    ax.set_title(f"Step {step}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.colorbar(im, ax=axes, shrink=0.8, label="Temperature")
fig.suptitle("GPU-accelerated 2D Heat Diffusion (Metal compute shaders)", fontweight="bold")
fig.tight_layout()
plt.show()
# -

# ## 7. 5-point vs 9-point Stencil Comparison
#
# The 9-point stencil includes diagonal neighbors, producing a more isotropic (rotationally symmetric) approximation of the Laplacian. Let's compare both on a circular test function where the analytical Laplacian is known.

# +
# Compare error of 5-point vs 9-point against analytical Laplacian
rows, cols = 512, 512
yy, xx = np.meshgrid(np.linspace(-3, 3, cols), np.linspace(-3, 3, rows))
sigma = 1.0
field = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
r2 = xx**2 + yy**2
lap_exact = ((r2 - 2 * sigma**2) / sigma**4 * np.exp(-r2 / (2 * sigma**2))).astype(np.float32)

lap_5 = metal.laplacian2d(ctx_2d, field)
lap_9 = metal.laplacian2d9p(ctx_2d, field)

# Only compare interior (stencils leave boundaries at 0)
interior = slice(2, -2), slice(2, -2)
err_5 = np.abs(lap_5[interior] - lap_exact[interior])
err_9 = np.abs(lap_9[interior] - lap_exact[interior])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(err_5, cmap="magma", extent=[-3, 3, -3, 3])
axes[0].set_title(f"5-point error\n(mean: {err_5.mean():.4f})")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(err_9, cmap="magma", extent=[-3, 3, -3, 3])
axes[1].set_title(f"9-point error\n(mean: {err_9.mean():.4f})")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# Cross-section through center
mid = rows // 2
axes[2].plot(np.linspace(-3, 3, cols)[2:-2], err_5[mid - 2, :], label="5-point", alpha=0.8)
axes[2].plot(np.linspace(-3, 3, cols)[2:-2], err_9[mid - 2, :], label="9-point", alpha=0.8)
axes[2].set_xlabel("x")
axes[2].set_ylabel("|error|")
axes[2].set_title("Error cross-section (y=0)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.suptitle("Stencil Accuracy: 5-point vs 9-point Laplacian", fontweight="bold")
fig.tight_layout()
plt.show()


# -

# ## 8. Mandelbrot Set — Compute-Heavy GPU Win
#
# The Mandelbrot set is the ideal GPU workload: each pixel iterates `z = z² + c` independently for up to `max_iter` steps. That's 100-1000+ floating-point operations per output element — far beyond the memory-bandwidth ceiling that limits simple element-wise ops.
#
# NumPy must either use a slow Python loop or an awkward mask-based vectorization. The GPU runs all pixels in parallel.


# +
def logn1p(x, n=2):
    y = np.asarray(x, dtype=np.float64)
    for _ in range(n):
        y = np.log1p(np.maximum(y, 0.0))  # defined at 0, ignores negatives
    return y


def logstar(x, base=np.e, threshold=1.0):
    x = np.asarray(x, dtype=np.float64)
    if base == np.e:
        logf = np.log
    else:
        logf = lambda z: np.log(z) / np.log(base)

    k = np.zeros_like(x, dtype=np.int64)
    y = x.copy()
    m = y > threshold
    while np.any(m):
        y[m] = logf(y[m])
        k[m] += 1
        m = y > threshold
    return k


# Load compute-heavy kernels
ctx_compute = metal.MetalContext()
ctx_compute.load_library("build/04-Compute/ops.metallib")
print(f"Compute library loaded on: {ctx_compute.device_name}")

# GPU Mandelbrot — single dispatch
width, height = 2048, 2048
t0 = time.perf_counter()
mandel_gpu = metal.mandelbrot(
    ctx_compute, width, height, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, max_iter=1000000
)
t_gpu = time.perf_counter() - t0

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(logn1p(mandel_gpu, n=50), cmap="inferno", extent=[-2, 1, -1.5, 1.5], origin="lower")
ax.set_xlabel("Re(c)")
ax.set_ylabel("Im(c)")
ax.set_title(f"Mandelbrot Set — {width}x{height}, max_iter=1000\nGPU time: {t_gpu * 1e3:.1f} ms")
plt.tight_layout()
plt.show()


# +
def mandelbrot_numpy(width, height, x_min, x_max, y_min, y_max, max_iter):
    """Vectorized NumPy Mandelbrot — iterates all pixels with mask."""
    cx = np.linspace(x_min, x_max, width, dtype=np.float32)
    cy = np.linspace(y_min, y_max, height, dtype=np.float32)
    cx, cy = np.meshgrid(cx, cy)
    c = cx + 1j * cy

    z = np.zeros_like(c)
    result = np.zeros(c.shape, dtype=np.float32)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c[mask]
        escaped = mask & (np.abs(z) > 2.0)
        result[escaped] = i + 1
        mask &= ~escaped

    # Points still in set get max_iter
    result[mask] = max_iter
    return result


# Benchmark across resolutions and max_iter values
configs = [
    (512, 512, 200),
    (1024, 1024, 500),
    # (2048, 2048, 1000),
    # (4096, 4096, 1000),
]

gpu_times_m = []
numpy_times_m = []
labels_m = []

for w, h, mi in configs:
    # GPU
    t_g, _ = benchmark(
        metal.mandelbrot, ctx_compute, w, h, -2.0, 1.0, -1.5, 1.5, mi, repeats=5, warmup=2
    )
    # NumPy
    t_n, _ = benchmark(mandelbrot_numpy, w, h, -2.0, 1.0, -1.5, 1.5, mi, repeats=3, warmup=1)
    gpu_times_m.append(t_g)
    numpy_times_m.append(t_n)
    labels_m.append(f"{w}x{h}\niter={mi}")
    print(
        f"  {w}x{h} max_iter={mi:>4d}  GPU: {t_g / 1e3:>8.1f} ms   NumPy: {t_n / 1e3:>8.1f} ms   speedup: {t_n / t_g:.0f}x"
    )

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

x_pos = np.arange(len(labels_m))
bar_w = 0.35
ax1.bar(
    x_pos - bar_w / 2, [t / 1e3 for t in gpu_times_m], bar_w, label="Metal GPU", color="#FF6B35"
)
ax1.bar(
    x_pos + bar_w / 2, [t / 1e3 for t in numpy_times_m], bar_w, label="NumPy (CPU)", color="#004E89"
)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels_m, fontsize=9)
ax1.set_ylabel("Time (ms)")
ax1.set_title("Mandelbrot: Absolute Times", fontweight="bold")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

speedups_m = [n / g for n, g in zip(numpy_times_m, gpu_times_m)]
bars = ax2.bar(x_pos, speedups_m, color="#2D936C", width=0.5)
ax2.bar_label(bars, fmt="%.0fx", fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels_m, fontsize=9)
ax2.set_ylabel("Speedup (NumPy / GPU)")
ax2.set_title("Mandelbrot: GPU Speedup", fontweight="bold")
ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Mandelbrot Benchmark — {ctx_compute.device_name}", fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()


# -

# ## 9. N-body Gravitational Simulation — O(N²) GPU Powerhouse
#
# The N-body problem computes gravitational accelerations between all particle pairs. Each thread processes one particle's interaction with all N others — O(N) work per thread, O(N²) total.
#
# For N=4096 that's ~16.8 million pairwise interactions. For N=16384, ~268 million. The GPU parallelizes across particles while NumPy must either loop (very slow) or create massive O(N²) temporary arrays.
#
# We provide two functions:
# - `nbody_step(ctx, pos_mass, velocities, dt, softening)` — single step, returns new arrays
# - `nbody_simulate(ctx, pos_mass, velocities, dt, softening, n_steps)` — multi-step, data stays on GPU


# +
# Set up initial conditions: particles in a disk with slight rotation
def make_galaxy(N, seed=42):
    """Create N particles in a disk with Keplerian-ish velocities."""
    rng = np.random.default_rng(seed)
    r = rng.exponential(1.0, N).astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
    z_pos = rng.normal(0, 0.1, N).astype(np.float32)

    pos_mass = np.zeros((N, 4), dtype=np.float32)
    pos_mass[:, 0] = r * np.cos(theta)
    pos_mass[:, 1] = r * np.sin(theta)
    pos_mass[:, 2] = z_pos
    pos_mass[:, 3] = 1.0 / N  # equal mass

    # Circular-ish velocities: v ~ sqrt(M_enclosed / r)
    v_circ = np.sqrt(np.clip(r, 0.1, None))
    velocities = np.zeros((N, 4), dtype=np.float32)
    velocities[:, 0] = -v_circ * np.sin(theta) * 0.5
    velocities[:, 1] = v_circ * np.cos(theta) * 0.5
    return pos_mass, velocities


# Visualize a short simulation
N_vis = 2048
pos, vel = make_galaxy(N_vis)

n_steps = 200
dt = 0.005

# Run simulation — data stays on GPU for all steps
t0 = time.perf_counter()
pos_final, vel_final = metal.nbody_simulate(
    ctx_compute, pos, vel, dt=dt, softening=0.05, n_steps=n_steps
)
t_sim = time.perf_counter() - t0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(pos[:, 0], pos[:, 1], s=0.5, alpha=0.5, c="steelblue")
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect("equal")
ax1.set_title("Initial positions")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.scatter(pos_final[:, 0], pos_final[:, 1], s=0.5, alpha=0.5, c="coral")
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_aspect("equal")
ax2.set_title(f"After {n_steps} steps")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

fig.suptitle(
    f"N-body: {N_vis} particles, {n_steps} steps in {t_sim:.2f}s "
    f"({t_sim / n_steps * 1e3:.1f} ms/step)",
    fontweight="bold",
)
fig.tight_layout()
plt.show()


# +
def nbody_step_numpy(pos_mass, velocities, dt, softening):
    """Vectorized NumPy N-body: O(N²) pairwise forces + leapfrog integration."""
    N = pos_mass.shape[0]
    pos = pos_mass[:, :3]  # (N, 3)
    mass = pos_mass[:, 3]  # (N,)

    # All-pairs displacement: diff[i,j] = pos[j] - pos[i]
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, 3)
    dist_sq = np.sum(diff**2, axis=2) + softening**2  # (N, N)
    inv_dist = 1.0 / np.sqrt(dist_sq)
    inv_dist3 = inv_dist**3

    # Acceleration: a_i = sum_j m_j * (r_j - r_i) / |r_ij|^3
    acc = np.sum(diff * (mass[np.newaxis, :, np.newaxis] * inv_dist3[:, :, np.newaxis]), axis=1)

    vel_new = velocities.copy()
    pos_new = pos_mass.copy()
    vel_new[:, :3] += acc * dt
    pos_new[:, :3] += vel_new[:, :3] * dt
    return pos_new, vel_new


# Benchmark: GPU single-step vs NumPy single-step at various N
particle_counts = [256, 512, 1024, 2048, 4096, 8192]
gpu_times_nb = []
numpy_times_nb = []

for N in particle_counts:
    pos, vel = make_galaxy(N)

    # GPU
    t_g, _ = benchmark(metal.nbody_step, ctx_compute, pos, vel, 0.001, 0.05, repeats=5, warmup=2)
    gpu_times_nb.append(t_g)

    # NumPy — fewer repeats because it's expensive
    reps = max(1, min(5, int(2e7 / (N * N))))
    t_n, _ = benchmark(nbody_step_numpy, pos, vel, 0.001, 0.05, repeats=reps, warmup=1)
    numpy_times_nb.append(t_n)

    print(
        f"  N = {N:>5d}  GPU: {t_g / 1e3:>8.1f} ms   NumPy: {t_n / 1e3:>8.1f} ms   speedup: {t_n / t_g:.0f}x"
    )

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.loglog(
    particle_counts,
    [t / 1e3 for t in gpu_times_nb],
    "o-",
    label="Metal GPU",
    color="#FF6B35",
    linewidth=2,
)
ax1.loglog(
    particle_counts,
    [t / 1e3 for t in numpy_times_nb],
    "s--",
    label="NumPy (CPU)",
    color="#004E89",
    linewidth=2,
)
ax1.set_xlabel("Number of particles (N)")
ax1.set_ylabel("Time per step (ms)")
ax1.set_title("N-body Forces: Absolute Times", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, which="both")

speedups_nb = [n / g for n, g in zip(numpy_times_nb, gpu_times_nb)]
ax2.semilogx(particle_counts, speedups_nb, "D-", color="#2D936C", linewidth=2, markersize=8)
for x, s in zip(particle_counts, speedups_nb):
    ax2.annotate(
        f"{s:.0f}x",
        (x, s),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
    )
ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Number of particles (N)")
ax2.set_ylabel("Speedup (NumPy / GPU)")
ax2.set_title("N-body: GPU Speedup", fontweight="bold")
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle(f"N-body Benchmark — {ctx_compute.device_name}", fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()
# -

# ## 10. Multi-step Diffusion — Eliminating Per-Call Overhead
#
# In section 6, we ran diffusion by calling `laplacian2d` and `saxpy` from Python in a loop — each call copies data in and out. The new `diffuse_steps` function keeps all buffers on GPU and dispatches N steps in C++, only copying once at the start and end.
#
# This eliminates the per-step overhead of:
# - NumPy → Metal buffer copy (memcpy in)
# - Metal → NumPy buffer copy (memcpy out)
# - Python function call overhead
# - Buffer allocation/deallocation

# +
# Need both 1D (saxpy) and 2D (laplacian2d) kernels for diffuse_steps
ctx_diffuse = metal.MetalContext()
ctx_diffuse.load_library("build/02-GeneralArrayOperations/ops.metallib")
ctx_diffuse.load_library("build/03-2DKernels/ops.metallib")


def diffuse_numpy(field, dt, n_steps):
    """Pure NumPy diffusion: forward Euler with 5-point Laplacian."""
    u = field.copy()
    for _ in range(n_steps):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]
        )
        u += dt * lap
    return u


def diffuse_python_loop(ctx, field, dt, n_steps):
    """Per-call GPU diffusion (from section 6): Python loop calling GPU each step."""
    u = field.copy()
    for _ in range(n_steps):
        lap = metal.laplacian2d(ctx, u)
        u = metal.saxpy(ctx, dt, lap.ravel(), u.ravel()).reshape(u.shape)
    return u


# Benchmark: diffuse_steps (C++ loop) vs Python-loop GPU vs NumPy
grid_size = 512
n_steps_list = [10, 50, 100, 500]
dt = 0.1

gpu_fused = []
gpu_pyloop = []
numpy_diff = []

for ns in n_steps_list:
    field = np.zeros((grid_size, grid_size), dtype=np.float32)
    field[grid_size // 2 - 20 : grid_size // 2 + 20, grid_size // 2 - 20 : grid_size // 2 + 20] = (
        1.0
    )

    t_fused, _ = benchmark(metal.diffuse_steps, ctx_diffuse, field, dt, ns, repeats=3, warmup=1)
    gpu_fused.append(t_fused)

    t_pyloop, _ = benchmark(diffuse_python_loop, ctx_diffuse, field, dt, ns, repeats=3, warmup=1)
    gpu_pyloop.append(t_pyloop)

    t_np, _ = benchmark(diffuse_numpy, field, dt, ns, repeats=3, warmup=1)
    numpy_diff.append(t_np)

    print(
        f"  {ns:>4d} steps: C++ loop {t_fused / 1e3:>8.1f} ms | "
        f"Py loop {t_pyloop / 1e3:>8.1f} ms ({t_pyloop / t_fused:.1f}x slower) | "
        f"NumPy {t_np / 1e3:>8.1f} ms (GPU fused {t_np / t_fused:.1f}x faster)"
    )

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.loglog(
    n_steps_list,
    [t / 1e3 for t in gpu_fused],
    "o-",
    label="GPU C++ loop (diffuse_steps)",
    color="#FF6B35",
    linewidth=2,
)
ax1.loglog(
    n_steps_list,
    [t / 1e3 for t in gpu_pyloop],
    "^--",
    label="GPU Python loop",
    color="#9B59B6",
    linewidth=2,
)
ax1.loglog(
    n_steps_list,
    [t / 1e3 for t in numpy_diff],
    "s--",
    label="NumPy (CPU)",
    color="#004E89",
    linewidth=2,
)
ax1.set_xlabel("Number of diffusion steps")
ax1.set_ylabel("Total time (ms)")
ax1.set_title(f"Diffusion on {grid_size}x{grid_size} grid", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, which="both")

sp_vs_numpy = [n / g for n, g in zip(numpy_diff, gpu_fused)]
sp_vs_pyloop = [p / g for p, g in zip(gpu_pyloop, gpu_fused)]
ax2.plot(
    n_steps_list, sp_vs_numpy, "D-", label="vs NumPy", color="#2D936C", linewidth=2, markersize=8
)
ax2.plot(
    n_steps_list,
    sp_vs_pyloop,
    "o--",
    label="vs GPU Python loop",
    color="#9B59B6",
    linewidth=2,
    markersize=8,
)
ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Number of diffusion steps")
ax2.set_ylabel("Speedup over diffuse_steps")
ax2.set_title("Benefit of Keeping Data on GPU", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle(f"Multi-step Diffusion — {ctx_diffuse.device_name}", fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()
# -

# ## Summary
#
# | Workload | GPU Speedup | Why |
# |---|---|---|
# | **1D element-wise** (add, SAXPY) | GPU *slower* | Memory-bound; NumPy (Accelerate) already optimal; copy overhead dominates |
# | **2D Laplacian** (single call) | ~0.8x | Still memory-bound at practical grid sizes |
# | **Mandelbrot** (4096x4096, iter=1000) | **~3000x** | Compute-bound: 100-1000 FLOPs per pixel, embarrassingly parallel |
# | **N-body** (N=8192) | **~900x** | Compute-bound: O(N) FLOPs per thread, GPU parallelizes across N particles |
# | **Multi-step diffusion** (C++ loop vs Python loop) | **4-8x** | Eliminates per-step Python-to-GPU copy overhead |
#
# **Key insight:** GPU acceleration shines when the compute-to-memory ratio is high. Simple element-wise operations (~1 FLOP per float loaded) can't overcome the overhead of dispatching work to the GPU. But for compute-heavy kernels (Mandelbrot, N-body), Metal provides **orders of magnitude** speedups. And for iterative algorithms, keeping data on GPU between steps (C++ dispatch loop) avoids the copy overhead that makes the Python-loop GPU approach slow.
