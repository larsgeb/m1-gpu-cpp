// Test: Verify 2D GPU kernels (quadratic2d, laplacian2d, laplacian2d9p) match CPU
#include <iostream>
#include <cassert>
#include <cmath>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalOperations.hpp"
#include "CPUOperations.hpp"

static const size_t ROWS = 64;
static const size_t COLS = 128;

// Relaxed comparison for multi-point stencils where FP rounding accumulates
static bool nearEqual(const float *x, const float *y, size_t n, float relTol)
{
    for (size_t i = 0; i < n; i++)
    {
        float diff = std::fabs(x[i] - y[i]);
        float scale = std::max(std::fabs(x[i]), std::fabs(y[i]));
        if (diff > relTol * scale && diff > 1e-6f)
        {
            std::cerr << "  mismatch at index " << i << ": gpu=" << x[i]
                      << " cpu=" << y[i] << " diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    if (!device)
    {
        std::cerr << "FAIL: No Metal device found." << std::endl;
        return 1;
    }
    std::cout << "Running on " << device->name()->utf8String() << std::endl;

    size_t totalElements = ROWS * COLS;
    size_t bufSize = totalElements * sizeof(float);

    MTL::Buffer *a_MTL = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer *b_MTL = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer *c_MTL = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);

    auto a = (float *)a_MTL->contents();
    auto b = (float *)b_MTL->contents();
    auto c = (float *)c_MTL->contents();

    float *c_cpu = new float[totalElements];

    generateRandomFloatData(a, totalElements);
    generateRandomFloatData(b, totalElements);

    MetalOperations *ops = new MetalOperations(device);
    int failures = 0;

    // Test quadratic2d
    setZeros(c, totalElements);
    setZeros(c_cpu, totalElements);
    ops->quadratic2d(a_MTL, b_MTL, c_MTL, ROWS, COLS);
    quadratic2d(a, b, c_cpu, ROWS, COLS);
    if (!equalArray(c, c_cpu, totalElements))
    {
        std::cerr << "FAIL: quadratic2d" << std::endl;
        failures++;
    }
    else
        std::cout << "PASS: quadratic2d" << std::endl;

    // Test laplacian2d
    setZeros(c, totalElements);
    setZeros(c_cpu, totalElements);
    ops->laplacian2d(a_MTL, c_MTL, ROWS, COLS);
    laplacian2d(a, c_cpu, ROWS, COLS);
    if (!equalArray(c, c_cpu, totalElements))
    {
        std::cerr << "FAIL: laplacian2d" << std::endl;
        failures++;
    }
    else
        std::cout << "PASS: laplacian2d" << std::endl;

    // Test laplacian2d9p (relaxed tolerance — 9-point stencil accumulates more FP rounding)
    setZeros(c, totalElements);
    setZeros(c_cpu, totalElements);
    ops->laplacian2d9p(a_MTL, c_MTL, ROWS, COLS);
    laplacian2d9p(a, c_cpu, ROWS, COLS);
    if (!nearEqual(c, c_cpu, totalElements, 1e-5f))
    {
        std::cerr << "FAIL: laplacian2d9p" << std::endl;
        failures++;
    }
    else
        std::cout << "PASS: laplacian2d9p" << std::endl;

    delete[] c_cpu;
    a_MTL->release();
    b_MTL->release();
    c_MTL->release();
    delete ops;
    device->release();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed." << std::endl;
        return 1;
    }
    std::cout << "All 2D operation tests passed." << std::endl;
    return 0;
}
