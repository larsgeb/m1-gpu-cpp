// Test: Verify 1D GPU kernels (add, multiply, saxpy, central_difference) match CPU
#include <iostream>
#include <cassert>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalOperations.hpp"
#include "CPUOperations.hpp"

static const size_t N = 10000; // Small array for fast tests

int main()
{
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    if (!device)
    {
        std::cerr << "FAIL: No Metal device found." << std::endl;
        return 1;
    }
    std::cout << "Running on " << device->name()->utf8String() << std::endl;

    size_t bufSize = N * sizeof(float);

    MTL::Buffer *a_MTL = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer *b_MTL = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer *c_MTL = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer *k_MTL = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    auto a = (float *)a_MTL->contents();
    auto b = (float *)b_MTL->contents();
    auto c = (float *)c_MTL->contents();
    auto k = (float *)k_MTL->contents();

    float *c_cpu = new float[N];

    generateRandomFloatData(a, N);
    generateRandomFloatData(b, N);
    *k = 2.5f;

    MetalOperations *ops = new MetalOperations(device);
    int failures = 0;

    // Test add_arrays
    setZeros(c, N);
    ops->addArrays(a_MTL, b_MTL, c_MTL, N);
    add(a, b, c_cpu, N);
    if (!equalArray(c, c_cpu, N))
    {
        std::cerr << "FAIL: add_arrays" << std::endl;
        failures++;
    }
    else
        std::cout << "PASS: add_arrays" << std::endl;

    // Test multiply_arrays
    setZeros(c, N);
    ops->multiplyArrays(a_MTL, b_MTL, c_MTL, N);
    multiply(a, b, c_cpu, N);
    if (!equalArray(c, c_cpu, N))
    {
        std::cerr << "FAIL: multiply_arrays" << std::endl;
        failures++;
    }
    else
        std::cout << "PASS: multiply_arrays" << std::endl;

    // Test saxpy
    setZeros(c, N);
    ops->saxpyArrays(k_MTL, a_MTL, b_MTL, c_MTL, N);
    saxpy(k, a, b, c_cpu, N);
    if (!equalArray(c, c_cpu, N))
    {
        std::cerr << "FAIL: saxpy" << std::endl;
        failures++;
    }
    else
        std::cout << "PASS: saxpy" << std::endl;

    // Test central_difference
    setZeros(c, N);
    *k = 0.01f;
    ops->central_difference(k_MTL, a_MTL, c_MTL, N);
    central_difference(k, a, c_cpu, N);
    if (!equalArray(c, c_cpu, N))
    {
        std::cerr << "FAIL: central_difference" << std::endl;
        failures++;
    }
    else
        std::cout << "PASS: central_difference" << std::endl;

    delete[] c_cpu;
    a_MTL->release();
    b_MTL->release();
    c_MTL->release();
    k_MTL->release();
    delete ops;
    device->release();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed." << std::endl;
        return 1;
    }
    std::cout << "All 1D operation tests passed." << std::endl;
    return 0;
}
