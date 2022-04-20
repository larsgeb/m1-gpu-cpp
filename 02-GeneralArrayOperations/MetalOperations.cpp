

#include "MetalOperations.hpp"
#include <iostream>

MetalOperations::MetalOperations(MTL::Device *device)
{

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();

    if (defaultLibrary == nullptr)
    {
        std::cout << "Failed to find the default library." << std::endl;
        return;
    }

    auto str = NS::String::string("add_arrays", NS::ASCIIStringEncoding);
    MTL::Function *addFunction = defaultLibrary->newFunction(str);

    str = NS::String::string("multiply_arrays", NS::ASCIIStringEncoding);
    MTL::Function *multiplyFunction = defaultLibrary->newFunction(str);

    str = NS::String::string("saxpy", NS::ASCIIStringEncoding);
    MTL::Function *saxpyFunction = defaultLibrary->newFunction(str);

    if (addFunction == nullptr)
    {
        std::cout << "Failed to find the adder function." << std::endl;
        return;
    }
    if (multiplyFunction == nullptr)
    {
        std::cout << "Failed to find the multiply function." << std::endl;
        return;
    }
    if (saxpyFunction == nullptr)
    {
        std::cout << "Failed to find the saxpy function." << std::endl;
        return;
    }

    // Create a compute pipelines
    _mAddFunctionPSO = _mDevice->newComputePipelineState(addFunction, &error);
    _mMultiplyFunctionPSO = _mDevice->newComputePipelineState(multiplyFunction, &error);
    _mSaxpyFunctionPSO = _mDevice->newComputePipelineState(saxpyFunction, &error);

    if (_mAddFunctionPSO == nullptr)
    {
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return;
    }
    if (_mMultiplyFunctionPSO == nullptr)
    {
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return;
    }

    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }
}

void MetalOperations::addArrays(const MTL::Buffer *x_array,
                                const MTL::Buffer *y_array,
                                MTL::Buffer *r_array,
                                size_t arrayLength)
{
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mAddFunctionPSO);
    computeEncoder->setBuffer(x_array, 0, 0);
    computeEncoder->setBuffer(y_array, 0, 1);
    computeEncoder->setBuffer(r_array, 0, 2);

    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize = _mAddFunctionPSO->maxTotalThreadsPerThreadgroup();

    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalOperations::multiplyArrays(const MTL::Buffer *x_array,
                                     const MTL::Buffer *y_array,
                                     MTL::Buffer *r_array,
                                     size_t arrayLength)
{

    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mMultiplyFunctionPSO);
    computeEncoder->setBuffer(x_array, 0, 0);
    computeEncoder->setBuffer(y_array, 0, 1);
    computeEncoder->setBuffer(r_array, 0, 2);

    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize = _mMultiplyFunctionPSO->maxTotalThreadsPerThreadgroup();

    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalOperations::saxpyArrays(const MTL::Buffer *alpha,
                                  const MTL::Buffer *x_array,
                                  const MTL::Buffer *y_array,
                                  MTL::Buffer *r_array,
                                  size_t arrayLength)
{
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mSaxpyFunctionPSO);
    computeEncoder->setBuffer(alpha, 0, 0);
    computeEncoder->setBuffer(x_array, 0, 1);
    computeEncoder->setBuffer(y_array, 0, 2);
    computeEncoder->setBuffer(r_array, 0, 3);

    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize = _mSaxpyFunctionPSO->maxTotalThreadsPerThreadgroup();

    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}