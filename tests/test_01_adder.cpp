// Test: Verify GPU array addition matches CPU computation
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
    std::string deviceName = device->name()->utf8String();
    std::cout << "Running on " << deviceName << std::endl;

    // The MetalAdder sample uses 108 M element buffers (1.3 GB total).
    // The Apple Paravirtual device (used in CI virtual machines) cannot dispatch
    // a compute grid of this size; the kernel silently produces all-zero output.
    // Skip rather than fail so the CI result is informative, not misleading.
    if (deviceName.find("Paravirtual") != std::string::npos)
    {
        std::cout << "SKIP: Paravirtual device — test_01_adder requires real Apple Silicon "
                     "(108 M element dispatch not supported)." << std::endl;
        device->release();
        return 77; // CTest SKIP_RETURN_CODE
    }

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
