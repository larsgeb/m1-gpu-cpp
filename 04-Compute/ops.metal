#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Mandelbrot set — compute-heavy: up to max_iter iterations per pixel
// =============================================================================
// params layout: [x_min, x_max, y_min, y_max, max_iter_float]
kernel void mandelbrot(
    device float* result         [[buffer(0)]],
    device const float* params   [[buffer(1)]],
    uint2 index                  [[thread_position_in_grid]],
    uint2 grid                   [[threads_per_grid]])
{
    float x_min = params[0];
    float x_max = params[1];
    float y_min = params[2];
    float y_max = params[3];
    int max_iter = int(params[4]);

    // Map pixel to complex plane
    float cx = x_min + (x_max - x_min) * float(index.x) / float(grid.x);
    float cy = y_min + (y_max - y_min) * float(index.y) / float(grid.y);

    float zx = 0.0f;
    float zy = 0.0f;
    int iter = 0;

    // z = z^2 + c, iterate until escape or max_iter
    while (zx * zx + zy * zy < 4.0f && iter < max_iter)
    {
        float tmp = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = tmp;
        iter++;
    }

    // Smooth coloring: fractional escape iteration
    float smooth_val;
    if (iter == max_iter)
    {
        smooth_val = float(max_iter);
    }
    else
    {
        float log_zn = log(zx * zx + zy * zy) / 2.0f;
        float nu = log(log_zn / log(2.0f)) / log(2.0f);
        smooth_val = float(iter) + 1.0f - nu;
    }

    int idx = index.x * grid.y + index.y;
    result[idx] = smooth_val;
}

// =============================================================================
// N-body gravitational forces — O(N) work per thread, O(N^2) total
// =============================================================================
// pos_mass: float4 per particle (x, y, z, mass)
// accelerations: float4 per particle (ax, ay, az, 0) — using float4 for alignment
// params: uint[1] = {N}
// fparams: float[1] = {softening_squared}
kernel void nbody_forces(
    device const float4* pos_mass     [[buffer(0)]],
    device float4* accelerations      [[buffer(1)]],
    device const uint* params         [[buffer(2)]],
    device const float* fparams       [[buffer(3)]],
    uint index                        [[thread_position_in_grid]])
{
    uint N = params[0];
    float eps2 = fparams[0]; // softening squared

    float4 my_pos = pos_mass[index];
    float3 acc = float3(0.0f);

    for (uint j = 0; j < N; j++)
    {
        float4 other = pos_mass[j];
        float3 r = other.xyz - my_pos.xyz;
        float dist_sq = dot(r, r) + eps2;
        float inv_dist = rsqrt(dist_sq);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        acc += r * (other.w * inv_dist3); // F = G*m/r^2 * r_hat = m/r^3 * r
    }

    accelerations[index] = float4(acc, 0.0f);
}

// =============================================================================
// N-body integration step (leapfrog) — updates positions and velocities in-place
// =============================================================================
// pos_mass: float4 per particle (x, y, z, mass) — position updated in place
// velocities: float4 per particle (vx, vy, vz, 0) — velocity updated in place
// accelerations: float4 per particle (ax, ay, az, 0) — from nbody_forces
// fparams: float[1] = {dt}
kernel void nbody_integrate(
    device float4* pos_mass           [[buffer(0)]],
    device float4* velocities         [[buffer(1)]],
    device const float4* accelerations [[buffer(2)]],
    device const float* fparams       [[buffer(3)]],
    uint index                        [[thread_position_in_grid]])
{
    float dt = fparams[0];
    float4 acc = accelerations[index];
    float4 vel = velocities[index];
    float4 pos = pos_mass[index];

    // Leapfrog: kick-drift
    vel.xyz += acc.xyz * dt;
    pos.xyz += vel.xyz * dt;

    velocities[index] = vel;
    pos_mass[index] = pos;
}
