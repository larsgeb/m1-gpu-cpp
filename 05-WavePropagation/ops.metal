#include <metal_stdlib>
using namespace metal;

// =============================================================================
// 2D Elastic Wave Propagation — Velocity-Stress Staggered Grid (Virieux, 1986)
// =============================================================================
//
// Grid convention (row-major, matching existing 2D kernels):
//   index.x = i (row, x-direction)
//   index.y = j (column, z-direction)
//   flat index: idx = i * nz + j   where nz = grid.y
//
// Staggered grid locations:
//   sxx[i,j], szz[i,j]  : integer points (i, j)
//   sxz[i,j]            : half-half points (i+1/2, j+1/2)
//   vx[i,j]             : half-x points (i+1/2, j)
//   vz[i,j]             : half-z points (i, j+1/2)
//
// Material parameter grids (precomputed on host):
//   lam2mu[i,j] = rho * vp^2           at integer points
//   lam[i,j]    = rho * (vp^2 - 2*vs^2) at integer points
//   mu_xz[i,j]  = harmonic avg of mu    at half-half points
//   b_x[i,j]    = avg of 1/rho          at half-x points
//   b_z[i,j]    = avg of 1/rho          at half-z points
//
// params layout: [dt, dx, dz, nx_float, nz_float]

// =============================================================================
// Kernel 1: stress_update — Hooke's Law
// =============================================================================
// Computes strain rates from velocity finite differences and updates stress
// fields in-place.
//
// Normal stresses at (i, j), valid for i in [1, nx), j in [1, nz):
//   dvx_dx = (vx[i,j] - vx[i-1,j]) / dx
//   dvz_dz = (vz[i,j] - vz[i,j-1]) / dz
//   sxx[i,j] += dt * (lam2mu * dvx_dx + lam * dvz_dz)
//   szz[i,j] += dt * (lam * dvx_dx + lam2mu * dvz_dz)
//
// Shear stress at (i+1/2, j+1/2), valid for i in [0, nx-1), j in [0, nz-1):
//   dvx_dz = (vx[i,j+1] - vx[i,j]) / dz
//   dvz_dx = (vz[i+1,j] - vz[i,j]) / dx
//   sxz[i,j] += dt * mu_xz * (dvx_dz + dvz_dx)

kernel void stress_update(
    device const float* vx       [[buffer(0)]],
    device const float* vz       [[buffer(1)]],
    device float*       sxx      [[buffer(2)]],
    device float*       szz      [[buffer(3)]],
    device float*       sxz      [[buffer(4)]],
    device const float* lam2mu   [[buffer(5)]],
    device const float* lam      [[buffer(6)]],
    device const float* mu_xz    [[buffer(7)]],
    device const float* params   [[buffer(8)]],
    uint2 index                  [[thread_position_in_grid]],
    uint2 grid                   [[threads_per_grid]])
{
    int nx = int(params[3]);
    int nz = int(params[4]);
    int i = int(index.x);
    int j = int(index.y);

    if (i >= nx || j >= nz) return;

    float dt = params[0];
    float inv_dx = 1.0f / params[1];
    float inv_dz = 1.0f / params[2];
    int idx = i * nz + j;

    // Normal stress update at integer points (i, j)
    if (i >= 1 && j >= 1)
    {
        float dvx_dx = (vx[idx] - vx[idx - nz]) * inv_dx;
        float dvz_dz = (vz[idx] - vz[idx - 1])  * inv_dz;

        sxx[idx] += dt * (lam2mu[idx] * dvx_dx + lam[idx] * dvz_dz);
        szz[idx] += dt * (lam[idx] * dvx_dx + lam2mu[idx] * dvz_dz);
    }

    // Shear stress update at half-half points (i+1/2, j+1/2)
    if (i < nx - 1 && j < nz - 1)
    {
        float dvx_dz = (vx[idx + 1]  - vx[idx]) * inv_dz;
        float dvz_dx = (vz[idx + nz] - vz[idx]) * inv_dx;

        sxz[idx] += dt * mu_xz[idx] * (dvx_dz + dvz_dx);
    }
}

// =============================================================================
// Kernel 2: velocity_update — Newton's Second Law
// =============================================================================
// Computes stress divergence and updates velocity fields in-place.
//
// vx at (i+1/2, j), valid for i in [0, nx-1), j in [1, nz):
//   dsxx_dx = (sxx[i+1,j] - sxx[i,j]) / dx
//   dsxz_dz = (sxz[i,j] - sxz[i,j-1]) / dz
//   vx[i,j] += dt * b_x * (dsxx_dx + dsxz_dz)
//
// vz at (i, j+1/2), valid for i in [1, nx), j in [0, nz-1):
//   dsxz_dx = (sxz[i,j] - sxz[i-1,j]) / dx
//   dszz_dz = (szz[i,j+1] - szz[i,j]) / dz
//   vz[i,j] += dt * b_z * (dsxz_dx + dszz_dz)

kernel void velocity_update(
    device const float* sxx      [[buffer(0)]],
    device const float* szz      [[buffer(1)]],
    device const float* sxz      [[buffer(2)]],
    device float*       vx       [[buffer(3)]],
    device float*       vz       [[buffer(4)]],
    device const float* b_x      [[buffer(5)]],
    device const float* b_z      [[buffer(6)]],
    device const float* params   [[buffer(7)]],
    uint2 index                  [[thread_position_in_grid]],
    uint2 grid                   [[threads_per_grid]])
{
    int nx = int(params[3]);
    int nz = int(params[4]);
    int i = int(index.x);
    int j = int(index.y);

    if (i >= nx || j >= nz) return;

    float dt = params[0];
    float inv_dx = 1.0f / params[1];
    float inv_dz = 1.0f / params[2];
    int idx = i * nz + j;

    // vx update at half-x points (i+1/2, j)
    if (i < nx - 1 && j >= 1)
    {
        float dsxx_dx = (sxx[idx + nz] - sxx[idx]) * inv_dx;
        float dsxz_dz = (sxz[idx] - sxz[idx - 1])  * inv_dz;

        vx[idx] += dt * b_x[idx] * (dsxx_dx + dsxz_dz);
    }

    // vz update at half-z points (i, j+1/2)
    if (i >= 1 && j < nz - 1)
    {
        float dsxz_dx = (sxz[idx] - sxz[idx - nz]) * inv_dx;
        float dszz_dz = (szz[idx + 1] - szz[idx])  * inv_dz;

        vz[idx] += dt * b_z[idx] * (dsxz_dx + dszz_dz);
    }
}

// =============================================================================
// Kernel 3: apply_damping — absorbing boundary sponge layer
// =============================================================================
// Element-wise multiply: field[idx] *= damp[idx]
// Dispatched once per wavefield (5 times per time step).

kernel void apply_damping(
    device float*       field    [[buffer(0)]],
    device const float* damp     [[buffer(1)]],
    uint index                   [[thread_position_in_grid]])
{
    field[index] *= damp[index];
}
