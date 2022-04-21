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

kernel void central_difference(
                  device const float* delta [[buffer(0)]],
                  device const float* X     [[buffer(1)]],
                  device float* result      [[buffer(2)]],
                  uint index                [[thread_position_in_grid]],
                  uint arrayLength          [[threads_per_grid]])
{
    if (index == 0)
    {
        result[index] = (X[index + 1] - X[index]) /  *delta;
    }
    else if (index == arrayLength - 1)
    {
        result[index] = (X[index] - X[index - 1]) /  *delta;
    }
    else
    {
        result[index] = (X[index + 1] - X[index - 1]) / (2 * *delta);
    }
}

