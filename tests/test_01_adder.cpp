// Test: Verify GPU array addition matches CPU computation
// Use a small array so the test runs on both real Apple Silicon and the
// CI paravirtual GPU (which can't handle the 108 M element sample size).
#define METAL_ADDER_ARRAY_LENGTH 10000

#include <iostream>
#include <cstdlib>
#include <cmath>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalAdder.hpp"

int main()
{
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    if (!device)
    {
        std::cerr << "FAIL: No Metal device found." << std::endl;
        return 1;
    }
    std::cout << "Running on " << device->name()->utf8String() << std::endl;

    MetalAdder *adder = new MetalAdder(device);

    // Run GPU addition
    adder->sendComputeCommand();

    // Verify against CPU
    float *a = (float *)adder->_mBufferA->contents();
    float *b = (float *)adder->_mBufferB->contents();
    float *result = (float *)adder->_mBufferResult->contents();

    int errors = 0;
    for (unsigned long i = 0; i < arrayLength; i++)
    {
        if (result[i] != (a[i] + b[i]))
        {
            if (errors < 5)
                std::cerr << "  mismatch at index " << i << ": got " << result[i]
                          << ", expected " << (a[i] + b[i]) << std::endl;
            errors++;
        }
    }

    delete adder;
    device->release();

    if (errors > 0)
    {
        std::cerr << "FAIL: " << errors << " mismatches out of " << arrayLength << std::endl;
        return 1;
    }
    std::cout << "PASS: GPU addition matches CPU (" << arrayLength << " elements)" << std::endl;
    return 0;
}
