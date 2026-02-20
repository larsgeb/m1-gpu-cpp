// Verify the add_arrays GPU kernel from 01-MetalAdder.
// Loads default.metallib by explicit path so this test works on both real
// Apple Silicon and the CI paravirtual GPU, which rejects newDefaultLibrary()
// (that API searches for an app bundle and silently fails on the VM).

#include <cstdlib>
#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

static const unsigned int kN = 10000;

int main()
{
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    if (!device)
    {
        std::cerr << "FAIL: No Metal device found." << std::endl;
        return 1;
    }
    std::cout << "Running on " << device->name()->utf8String() << std::endl;

    NS::Error *error = nullptr;

    // Load by explicit path — works on both real and paravirtual GPUs.
    auto libPath = NS::String::string("default.metallib", NS::ASCIIStringEncoding);
    MTL::Library *lib = device->newLibrary(libPath, &error);
    if (!lib)
    {
        std::cerr << "FAIL: Could not load default.metallib: "
                  << (error ? error->description()->utf8String() : "unknown") << std::endl;
        device->release();
        return 1;
    }

    auto fnName = NS::String::string("add_arrays", NS::ASCIIStringEncoding);
    MTL::Function *fn = lib->newFunction(fnName);
    lib->release();
    if (!fn)
    {
        std::cerr << "FAIL: add_arrays not found in default.metallib" << std::endl;
        device->release();
        return 1;
    }

    MTL::ComputePipelineState *pso = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pso)
    {
        std::cerr << "FAIL: Could not create pipeline state" << std::endl;
        device->release();
        return 1;
    }

    MTL::CommandQueue *queue = device->newCommandQueue();

    // Allocate shared-memory buffers and fill inputs with random data.
    size_t nbytes = kN * sizeof(float);
    MTL::Buffer *bufA      = device->newBuffer(nbytes, MTL::ResourceStorageModeShared);
    MTL::Buffer *bufB      = device->newBuffer(nbytes, MTL::ResourceStorageModeShared);
    MTL::Buffer *bufResult = device->newBuffer(nbytes, MTL::ResourceStorageModeShared);

    float *a = (float *)bufA->contents();
    float *b = (float *)bufB->contents();
    for (unsigned int i = 0; i < kN; i++)
    {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // Dispatch the kernel.
    auto cmdBuf = queue->commandBuffer();
    auto enc    = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bufA,      0, 0);
    enc->setBuffer(bufB,      0, 1);
    enc->setBuffer(bufResult, 0, 2);
    NS::UInteger tgSize = pso->maxTotalThreadsPerThreadgroup();
    if (tgSize > kN) tgSize = kN;
    enc->dispatchThreads(MTL::Size::Make(kN, 1, 1), MTL::Size::Make(tgSize, 1, 1));
    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    // Verify against CPU reference.
    float *result = (float *)bufResult->contents();
    int errors = 0;
    for (unsigned int i = 0; i < kN; i++)
    {
        if (result[i] != (a[i] + b[i]))
        {
            if (errors < 5)
                std::cerr << "  mismatch at index " << i << ": got " << result[i]
                          << ", expected " << (a[i] + b[i]) << std::endl;
            errors++;
        }
    }

    pso->release();
    queue->release();
    bufA->release();
    bufB->release();
    bufResult->release();
    device->release();

    if (errors > 0)
    {
        std::cerr << "FAIL: " << errors << " mismatches out of " << kN << std::endl;
        return 1;
    }
    std::cout << "PASS: GPU addition matches CPU (" << kN << " elements)" << std::endl;
    return 0;
}
