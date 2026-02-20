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

# # 2D Elastic Wave Propagation on Metal GPU
#
# This notebook demonstrates GPU-accelerated 2D elastic wave propagation using the
# velocity-stress staggered-grid finite-difference method (Virieux, 1986), running
# entirely on Apple Silicon via Metal compute shaders.
#
# **Physics:** The elastic wave equation is decomposed into:
# - **Hooke's Law** (stress update): relates strain rates to stress changes via elastic moduli ($\lambda$, $\mu$)
# - **Newton's Second Law** (velocity update): relates stress gradients to velocity changes via density ($\rho$)
#
# **Implementation:**
# - 2 Metal compute kernels: `stress_update` (9 buffers) and `velocity_update` (8 buffers)
# - Absorbing boundaries via Cerjan damping sponge (`apply_damping` kernel)
# - Source injection and receiver recording via direct CPU access to unified shared memory
# - All time-stepping runs in a C++ loop — data stays on GPU

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

ctx = metal.MetalContext()
ctx.load_library("build/05-WavePropagation/ops.metallib")
print(f"GPU device: {ctx.device_name}")


# ## Helper: Ricker Wavelet
#
# The Ricker wavelet (Mexican hat) is the standard source time function for seismic modelling.
# Peak frequency $f_0$ controls the dominant wavelength.


# +
def ricker_wavelet(nt, dt, f0, t0=None):
    """Ricker (Mexican hat) wavelet centered at t0 with peak frequency f0."""
    if t0 is None:
        t0 = 1.5 / f0
    t = np.arange(nt) * dt
    tau = t - t0
    w = (1.0 - 2.0 * (np.pi * f0 * tau) ** 2) * np.exp(-((np.pi * f0 * tau) ** 2))
    return w.astype(np.float32)


# Preview
f0_demo = 15.0
dt_demo = 0.0001
wav_demo = ricker_wavelet(2000, dt_demo, f0_demo)
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(np.arange(len(wav_demo)) * dt_demo * 1000, wav_demo, "k-", linewidth=1.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Ricker wavelet, f0 = {f0_demo} Hz")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
# -

# ## 1. Homogeneous Medium — P and S Waves
#
# An explosive point source in a homogeneous elastic medium produces:
# - **P-wave** (compressional): faster, arrives first, velocity = $v_p$
# - **S-wave** (shear): slower, velocity = $v_s$
#
# Both should propagate as expanding circles.

# +
# Grid parameters
nx, nz = 400, 400
dx, dz = 5.0, 5.0  # meters
f0 = 15.0  # peak frequency Hz

# Material properties
vp_val = 3000.0  # m/s
vs_val = 1700.0  # m/s
rho_val = 2200.0  # kg/m^3

# CFL stability condition: dt < min(dx,dz) / (sqrt(2) * vp_max)
dt = 0.8 * min(dx, dz) / (np.sqrt(2.0) * vp_val)
nt = 800
print(f"dt = {dt * 1e6:.1f} us, total time = {nt * dt * 1e3:.1f} ms")

wavelet = ricker_wavelet(nt, dt, f0)

# Homogeneous model
vp = np.full((nx, nz), vp_val, dtype=np.float32)
vs = np.full((nx, nz), vs_val, dtype=np.float32)
rho = np.full((nx, nz), rho_val, dtype=np.float32)

# Source at center
src_x, src_z = nx // 2, nz // 2

# Receivers: horizontal line at offset +50 grid points in x
recv_z = np.arange(20, nz - 20, 2, dtype=np.int32)
recv_x = np.full_like(recv_z, nx // 2 + 50)
print(f"Grid: {nx}x{nz}, {nt} time steps, {len(recv_z)} receivers")

# Run simulation
t0 = time.perf_counter()
seis_vx, seis_vz, snap_vx, snap_vz = metal.elastic_wave_propagate(
    ctx,
    vp,
    vs,
    rho,
    src_x,
    src_z,
    wavelet,
    recv_x,
    recv_z,
    dx,
    dz,
    dt,
    snapshot_interval=100,
    n_boundary=30,
)
elapsed = time.perf_counter() - t0
print(f"Completed in {elapsed:.2f}s ({elapsed / nt * 1e3:.2f} ms/step)")

# +
# Wavefield snapshots
n_show = min(snap_vx.shape[0], 4)
fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
if n_show == 1:
    axes = axes.reshape(2, 1)

vmax_vx = np.percentile(np.abs(snap_vx), 99.5)
vmax_vz = np.percentile(np.abs(snap_vz), 99.5)

extent = [0, nz * dz, nx * dx, 0]

for col in range(n_show):
    step_num = (col + 1) * 100
    t_ms = step_num * dt * 1e3

    axes[0, col].imshow(
        snap_vx[col], cmap="RdBu_r", vmin=-vmax_vx, vmax=vmax_vx, extent=extent, aspect="equal"
    )
    axes[0, col].set_title(f"vx — t = {t_ms:.0f} ms")

    axes[1, col].imshow(
        snap_vz[col], cmap="RdBu_r", vmin=-vmax_vz, vmax=vmax_vz, extent=extent, aspect="equal"
    )
    axes[1, col].set_title(f"vz — t = {t_ms:.0f} ms")

for ax in axes[:, 0]:
    ax.set_ylabel("x (m)")
for ax in axes[1, :]:
    ax.set_xlabel("z (m)")

fig.suptitle(
    f"Elastic Wave Propagation — Homogeneous Medium\nvp = {vp_val:.0f} m/s, vs = {vs_val:.0f} m/s",
    fontweight="bold",
)
fig.tight_layout()
plt.show()
# -

# ## 2. Seismograms
#
# Seismograms recorded at the receiver line. The moveout (time vs offset) reveals the P and S wave velocities.

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

t_axis = np.arange(nt) * dt * 1000  # ms
offsets = (recv_z - src_z) * dz  # meters from source

# Normalize and plot every 3rd trace
scale_vx = np.max(np.abs(seis_vx)) * 0.3 if np.max(np.abs(seis_vx)) > 0 else 1.0
scale_vz = np.max(np.abs(seis_vz)) * 0.3 if np.max(np.abs(seis_vz)) > 0 else 1.0

for i in range(0, seis_vx.shape[0], 3):
    ax1.plot(t_axis, seis_vx[i] / scale_vx + offsets[i], "k-", linewidth=0.3)
    ax2.plot(t_axis, seis_vz[i] / scale_vz + offsets[i], "k-", linewidth=0.3)

# Theoretical moveout lines
t_display = np.linspace(0, nt * dt * 1000, 100)
# P-wave: offset = vp * (t - t_src); S-wave: offset = vs * (t - t_src)
t_src_ms = 1.5 / f0 * 1000  # source delay in ms
for ax in (ax1, ax2):
    p_offset = vp_val * (t_display / 1000 - 1.5 / f0)
    s_offset = vs_val * (t_display / 1000 - 1.5 / f0)
    ax.plot(t_display, p_offset, "r--", linewidth=1, alpha=0.7, label=f"P ({vp_val:.0f} m/s)")
    ax.plot(t_display, s_offset, "b--", linewidth=1, alpha=0.7, label=f"S ({vs_val:.0f} m/s)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Offset (m)")
    ax.legend(loc="upper left")
    ax.set_ylim(offsets.min(), offsets.max())
    ax.set_xlim(0, nt * dt * 1000)
    ax.grid(True, alpha=0.2)

ax1.set_title("vx seismogram")
ax2.set_title("vz seismogram")
fig.suptitle("Seismograms — Homogeneous Medium", fontweight="bold")
fig.tight_layout()
plt.show()
# -

# ## 3. Layered Medium — Reflections
#
# A two-layer model with an impedance contrast produces reflected and transmitted waves at the interface.

# +
# Two-layer model: faster layer below
vp2 = np.full((nx, nz), 3000.0, dtype=np.float32)
vs2 = np.full((nx, nz), 1700.0, dtype=np.float32)
rho2 = np.full((nx, nz), 2200.0, dtype=np.float32)

layer_top = nx // 2 + 40  # interface below center
vp2[layer_top:, :] = 4500.0
vs2[layer_top:, :] = 2600.0
rho2[layer_top:, :] = 2800.0

# Impedance contrast
Z1 = 2200.0 * 3000.0
Z2 = 2800.0 * 4500.0
R = (Z2 - Z1) / (Z2 + Z1)
print(f"Layer 1: vp={3000}, vs={1700}, rho={2200}")
print(f"Layer 2: vp={4500}, vs={2600}, rho={2800}")
print(f"Reflection coefficient R = {R:.3f}")

# Source above interface
src_x2 = nx // 4

# Receivers: surface line
recv_x2 = np.full(len(recv_z), nx // 4 + 30, dtype=np.int32)

t0 = time.perf_counter()
seis_vx2, seis_vz2, snap_vx2, snap_vz2 = metal.elastic_wave_propagate(
    ctx,
    vp2,
    vs2,
    rho2,
    src_x2,
    nz // 2,
    wavelet,
    recv_x2,
    recv_z,
    dx,
    dz,
    dt * 0.75,
    snapshot_interval=150,
    n_boundary=30,
)
elapsed2 = time.perf_counter() - t0
print(f"Completed in {elapsed2:.2f}s")

# +
n_show2 = min(snap_vz2.shape[0], 4)
fig, axes = plt.subplots(1, n_show2, figsize=(4 * n_show2, 5))
if n_show2 == 1:
    axes = [axes]

vmax2 = np.percentile(np.abs(snap_vz2), 99.5)

for col in range(n_show2):
    step_num = (col + 1) * 100
    t_ms = step_num * dt * 1e3
    ax = axes[col]
    ax.imshow(snap_vz2[col], cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, extent=extent, aspect="equal")
    ax.axhline(y=layer_top * dx, color="lime", linewidth=1.5, linestyle="--", label="Interface")
    ax.set_title(f"t = {t_ms:.0f} ms")
    ax.set_xlabel("z (m)")

axes[0].set_ylabel("x (m)")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Layered Medium — vz snapshots (green = interface)", fontweight="bold")
fig.tight_layout()
plt.show()

# +
# Seismogram for layered medium
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

scale_vx2 = np.max(np.abs(seis_vx2)) * 0.025 if np.max(np.abs(seis_vx2)) > 0 else 1.0
scale_vz2 = np.max(np.abs(seis_vz2)) * 0.025 if np.max(np.abs(seis_vz2)) > 0 else 1.0

for i in range(0, seis_vx2.shape[0], 3):
    ax1.plot(t_axis, seis_vx2[i] / scale_vx2 + offsets[i], "k-", linewidth=0.3)
    ax2.plot(t_axis, seis_vz2[i] / scale_vz2 + offsets[i], "k-", linewidth=0.3)

for ax in (ax1, ax2):
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Offset (m)")
    ax.set_ylim(offsets.min(), offsets.max())
    ax.set_xlim(0, nt * dt * 1000)
    ax.grid(True, alpha=0.2)

ax1.set_title("vx seismogram")
ax2.set_title("vz seismogram")
fig.suptitle("Seismograms — Layered Medium (direct + reflected + transmitted)", fontweight="bold")
fig.tight_layout()
plt.show()


# -

# ## 4. GPU vs NumPy Benchmark
#
# We compare the Metal GPU elastic wave solver against a pure NumPy reference implementation at different grid sizes. The GPU solver keeps all data on GPU (unified memory) and dispatches kernels from a C++ loop. The NumPy solver uses vectorized stencil operations with Python loop over time steps.


def elastic_wave_numpy(vp, vs, rho, src_x, src_z, wavelet, recv_x, recv_z, dx, dz, dt):
    """Pure NumPy 2D elastic wave propagation (velocity-stress, staggered grid)."""
    nx, nz = vp.shape
    nt = len(wavelet)
    nrec = len(recv_x)

    # Material parameters
    lam2mu = rho * vp**2
    lam = rho * (vp**2 - 2.0 * vs**2)
    mu = rho * vs**2
    buoy = 1.0 / rho

    # Wavefields
    vx_f = np.zeros((nx, nz), dtype=np.float32)
    vz_f = np.zeros((nx, nz), dtype=np.float32)
    sxx_f = np.zeros((nx, nz), dtype=np.float32)
    szz_f = np.zeros((nx, nz), dtype=np.float32)
    sxz_f = np.zeros((nx, nz), dtype=np.float32)

    seis_vx = np.zeros((nrec, nt), dtype=np.float32)
    seis_vz = np.zeros((nrec, nt), dtype=np.float32)

    for t in range(nt):
        # Source injection
        sxx_f[src_x, src_z] += wavelet[t]
        szz_f[src_x, src_z] += wavelet[t]

        # Stress update — Hooke's Law
        dvx_dx = (vx_f[1:, :] - vx_f[:-1, :]) / dx
        dvz_dz = (vz_f[:, 1:] - vz_f[:, :-1]) / dz
        sxx_f[1:, 1:] += dt * (lam2mu[1:, 1:] * dvx_dx[:, 1:] + lam[1:, 1:] * dvz_dz[1:, :])
        szz_f[1:, 1:] += dt * (lam[1:, 1:] * dvx_dx[:, 1:] + lam2mu[1:, 1:] * dvz_dz[1:, :])
        sxz_f[:-1, :-1] += (
            dt
            * mu[:-1, :-1]
            * ((vx_f[:-1, 1:] - vx_f[:-1, :-1]) / dz + (vz_f[1:, :-1] - vz_f[:-1, :-1]) / dx)
        )

        # Velocity update — Newton's Law
        dsxx_dx = (sxx_f[1:, :] - sxx_f[:-1, :]) / dx
        dsxz_dz = (sxz_f[:, 1:] - sxz_f[:, :-1]) / dz
        vx_f[:-1, 1:] += dt * buoy[:-1, 1:] * (dsxx_dx[:, 1:] + dsxz_dz[:-1, :])

        dsxz_dx = (sxz_f[1:, :] - sxz_f[:-1, :]) / dx
        dszz_dz = (szz_f[:, 1:] - szz_f[:, :-1]) / dz
        vz_f[1:, :-1] += dt * buoy[1:, :-1] * (dsxz_dx[:, :-1] + dszz_dz[1:, :])

        # Record receivers
        for r in range(nrec):
            seis_vx[r, t] = vx_f[recv_x[r], recv_z[r]]
            seis_vz[r, t] = vz_f[recv_x[r], recv_z[r]]

    return seis_vx, seis_vz


# +
# Benchmark at different grid sizes
bench_configs = [
    (100, 100, 400),
    (200, 200, 400),
    (300, 300, 400),
    (400, 400, 400),
    (800, 800, 400),
    (1600, 1600, 400),
]

gpu_times_w = []
numpy_times_w = []

for gnx, gnz, gnt in bench_configs:
    g_dx, g_dz = 5.0, 5.0
    g_dt = 0.8 * min(g_dx, g_dz) / (np.sqrt(2.0) * 3000.0)
    g_vp = np.full((gnx, gnz), 3000.0, dtype=np.float32)
    g_vs = np.full((gnx, gnz), 1700.0, dtype=np.float32)
    g_rho = np.full((gnx, gnz), 2200.0, dtype=np.float32)
    g_wav = ricker_wavelet(gnt, g_dt, 15.0)
    g_rx = np.array([gnx // 2], dtype=np.int32)
    g_rz = np.array([gnz // 2], dtype=np.int32)

    # GPU
    t0 = time.perf_counter()
    metal.elastic_wave_propagate(
        ctx, g_vp, g_vs, g_rho, gnx // 4, gnz // 2, g_wav, g_rx, g_rz, g_dx, g_dz, g_dt, 0, 20
    )
    t_g = time.perf_counter() - t0
    gpu_times_w.append(t_g)

    # NumPy
    t0 = time.perf_counter()
    elastic_wave_numpy(g_vp, g_vs, g_rho, gnx // 4, gnz // 2, g_wav, g_rx, g_rz, g_dx, g_dz, g_dt)
    t_n = time.perf_counter() - t0
    numpy_times_w.append(t_n)

    print(
        f"  {gnx:>3d}x{gnz:<3d} {gnt:>4d} steps:  "
        f"GPU {t_g:.3f}s  NumPy {t_n:.3f}s  "
        f"speedup {t_n / t_g:.1f}x"
    )

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

labels = [f"{c[0]}x{c[1]}\n{c[2]} steps" for c in bench_configs]
x_pos = np.arange(len(labels))
bar_w = 0.35

ax1.bar(x_pos - bar_w / 2, gpu_times_w, bar_w, label="Metal GPU", color="#FF6B35")
ax1.bar(x_pos + bar_w / 2, numpy_times_w, bar_w, label="NumPy (CPU)", color="#004E89")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Time (seconds)")
ax1.set_title("Elastic Wave: Absolute Times", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

speedups_w = [n / g for n, g in zip(numpy_times_w, gpu_times_w)]
bars = ax2.bar(x_pos, speedups_w, color="#2D936C", width=0.5)
ax2.bar_label(bars, fmt="%.1fx", fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Speedup (NumPy / GPU)")
ax2.set_title("GPU Speedup", fontweight="bold")
ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Elastic Wave Benchmark — {ctx.device_name}", fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()
# -

# ## 5. Larger Grid — High-Resolution Simulation
#
# Let's push to a larger grid to see the GPU advantage grow.

# +
# High-resolution simulation: 800x800 grid
nx_hr, nz_hr = 800, 800
dx_hr, dz_hr = 5.0, 5.0
dt_hr = 0.8 * min(dx_hr, dz_hr) / (np.sqrt(2.0) * 4500.0)  # Use max vp
nt_hr = 4500

# Interesting model: gradient + two layers
vp_hr = np.full((nx_hr, nz_hr), 2500.0, dtype=np.float32)
vs_hr = np.full((nx_hr, nz_hr), 1400.0, dtype=np.float32)
rho_hr = np.full((nx_hr, nz_hr), 2000.0, dtype=np.float32)

# Layer 1: starts at 40%
l1 = int(nx_hr * 0.4)
vp_hr[l1:, :] = 3500.0
vs_hr[l1:, :] = 2000.0
rho_hr[l1:, :] = 2400.0

# Layer 2: starts at 70%
l2 = int(nx_hr * 0.7)
vp_hr[l2:, :] = 4500.0
vs_hr[l2:, :] = 2600.0
rho_hr[l2:, :] = 2800.0

wav_hr = ricker_wavelet(nt_hr, dt_hr, 12.0)
recv_z_hr = np.arange(40, nz_hr - 40, 4, dtype=np.int32)
recv_x_hr = np.full_like(recv_z_hr, 50)

print(f"Grid: {nx_hr}x{nz_hr}, {nt_hr} steps")
t0 = time.perf_counter()
seis_vx_hr, seis_vz_hr, snap_vx_hr, snap_vz_hr = metal.elastic_wave_propagate(
    ctx,
    vp_hr,
    vs_hr,
    rho_hr,
    80,
    nz_hr // 2,
    wav_hr,
    recv_x_hr,
    recv_z_hr,
    dx_hr,
    dz_hr,
    dt_hr,
    snapshot_interval=200,
    n_boundary=40,
)
elapsed_hr = time.perf_counter() - t0
print(f"Completed in {elapsed_hr:.2f}s ({elapsed_hr / nt_hr * 1e3:.2f} ms/step)")

# +
n_show_hr = min(snap_vz_hr.shape[0], 4)
fig, axes = plt.subplots(1, n_show_hr, figsize=(4 * n_show_hr, 5))
if n_show_hr == 1:
    axes = [axes]

vmax_hr = np.percentile(np.abs(snap_vz_hr), 99.5)
extent_hr = [0, nz_hr * dz_hr, nx_hr * dx_hr, 0]

for col in range(n_show_hr):
    step_num = (col + 1) * 600
    t_ms = step_num * dt_hr * 1e3
    ax = axes[col]
    ax.imshow(
        snap_vz_hr[col * 3],
        cmap="RdBu_r",
        vmin=-vmax_hr,
        vmax=vmax_hr,
        extent=extent_hr,
        aspect="equal",
    )
    ax.axhline(y=l1 * dx_hr, color="lime", linewidth=1, linestyle="--")
    ax.axhline(y=l2 * dx_hr, color="cyan", linewidth=1, linestyle="--")
    ax.set_title(f"t = {t_ms:.0f} ms")
    ax.set_xlabel("z (m)")

axes[0].set_ylabel("x (m)")
fig.suptitle(f"3-Layer Model — {nx_hr}x{nz_hr} grid, vz snapshots", fontweight="bold")
fig.tight_layout()
plt.show()

# +
# Seismogram for 3-layer model
fig, ax = plt.subplots(figsize=(10, 7))

t_axis_hr = np.arange(nt_hr) * dt_hr * 1000
offsets_hr = (recv_z_hr - nz_hr // 2) * dz_hr
scale_hr = np.max(np.abs(seis_vz_hr)) * 0.0025 if np.max(np.abs(seis_vz_hr)) > 0 else 1.0

for i in range(0, seis_vz_hr.shape[0], 2):
    ax.plot(t_axis_hr, seis_vz_hr[i] / scale_hr + offsets_hr[i], "k-", linewidth=0.3)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Offset (m)")
ax.set_title("vz Seismogram — 3-Layer Model", fontweight="bold")
# ax.set_xlim(0, nt_hr * dt_hr * 1000)
ax.grid(True, alpha=0.2)
fig.tight_layout()
plt.show()
# -

# ## Summary
#
# | Component | Metal GPU Kernel | Description |
# |---|---|---|
# | **Hooke's Law** | `stress_update` | Computes strain rates from velocity FD, updates sxx, szz, sxz |
# | **Newton's Law** | `velocity_update` | Computes stress divergence, updates vx, vz |
# | **Absorbing BC** | `apply_damping` | Cerjan sponge layer, element-wise damping |
# | **Source** | CPU (shared mem) | Direct write to sxx/szz at source position |
# | **Receivers** | CPU (shared mem) | Direct read from vx/vz at receiver positions |
#
# The entire simulation runs in a C++ time-stepping loop with data staying on GPU (unified shared memory). Source injection and receiver recording use zero-cost CPU access to the same memory.
