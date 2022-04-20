#include <metal_stdlib>
using namespace metal;


kernel void add_arrays(device const float* X,
                       device const float* Y,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = X[index] + Y[index];
}


kernel void multiply_arrays(device const float* X,
                            device const float* Y,
                            device float* result,
                            uint index [[thread_position_in_grid]])
{
    result[index] = X[index] * Y[index];
}

kernel void saxpy(device const float* a,
                  device const float* X,
                  device const float* Y,
                  device float* result,
                  uint index [[thread_position_in_grid]])
{
    result[index] = (*a) * X[index] + Y[index];
}

