#include <metal_stdlib>
using namespace metal;


kernel void add_arrays(device const float* X [[buffer(0)]],
                       device const float* Y [[buffer(1)]],
                       device float* result  [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] + Y[index];
}


kernel void multiply_arrays(device const float* X [[buffer(0)]],
                            device const float* Y [[buffer(1)]],
                            device float* result  [[buffer(2)]],
                            uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] * Y[index];
}

kernel void saxpy(device const float* a [[buffer(0)]],
                  device const float* X [[buffer(1)]],
                  device const float* Y [[buffer(2)]],
                  device float* result  [[buffer(3)]],
                  uint index            [[thread_position_in_grid]])
{
    result[index] = (*a) * X[index] + Y[index];
}

