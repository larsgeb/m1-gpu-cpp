/*
CPP translation of original Objective-C MetalAdder.h. Some stuff has been moved over
here from the cpp file. Source: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc

Original distribution license: LICENSE-original.txt.

Abstract:
A class to manage all of the Metal objects this app creates.
*/
#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#include "map"

class MetalOperations
{
public:
    MTL::Device *_mDevice;

    MetalOperations(MTL::Device *device);

    void Blocking1D(std::vector<MTL::Buffer *> buffers,
                    size_t arrayLength,
                    const char *method);

    void addArrays(MTL::Buffer *x_array,
                   MTL::Buffer *y_array,
                   MTL::Buffer *r_array,
                   size_t arrayLength);

    void addMultiply(MTL::Buffer *x_array,
                     MTL::Buffer *y_array,
                     MTL::Buffer *r_array,
                     size_t arrayLength);

    void multiplyArrays(MTL::Buffer *x_array,
                        MTL::Buffer *y_array,
                        MTL::Buffer *r_array,
                        size_t arrayLength);

    void saxpyArrays(MTL::Buffer *alpha,
                     MTL::Buffer *x_array,
                     MTL::Buffer *y_array,
                     MTL::Buffer *r_array,
                     size_t arrayLength);

    void central_difference(MTL::Buffer *delta,
                            MTL::Buffer *x_array,
                            MTL::Buffer *r_array,
                            size_t arrayLength);

private:
    std::map<std::string, MTL::Function *> functionMap;
    std::map<std::string, MTL::ComputePipelineState *> functionPipelineMap;

    // The command queue used to pass commands to the device.
    MTL::CommandQueue *_mCommandQueue;
};
