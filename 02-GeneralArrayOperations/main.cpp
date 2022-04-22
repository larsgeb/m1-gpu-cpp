// Lars Gebraad, 20th of April, 2022
//

#include <iostream>
#include <omp.h>
#include <assert.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalOperations.hpp"
#include "CPUOperations.hpp"

// Configuration -----------------------------------------------------------------------
// Amount of repeats for benchmarking
size_t repeats = 1000;
// Length of array to test kernels on
const unsigned int arrayLength = 1 << 26;
// end ---------------------------------------------------------------------------------

const unsigned int bufferSize = arrayLength * sizeof(float);

int main(int argc, char *argv[])
{

    // Set up objects and buffers ------------------------------------------------------

    MTL::Device *device = MTL::CreateSystemDefaultDevice();

    std::cout << "Running on " << device->name()->utf8String() << std::endl;
    std::cout << "Array size " << arrayLength << ", tests repeated " << repeats
              << " times" << std::endl
              << std::endl;

    // MTL buffers to hold data.
    MTL::Buffer *a_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer *b_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer *c_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer *k_MTL = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged); // Scalar

    // Get a C++-style reference to the buffer
    auto a_CPP = (float *)a_MTL->contents();
    auto b_CPP = (float *)b_MTL->contents();
    auto c_CPP = (float *)c_MTL->contents();
    auto k_CPP = (float *)k_MTL->contents();

    // Array to store CPU result on for verification of kernels
    auto c_VER = new float[arrayLength];

    generateRandomFloatData(a_CPP, arrayLength);
    generateRandomFloatData(b_CPP, arrayLength);
    setZeros(c_CPP, arrayLength);
    *k_CPP = 1.0f;

    // Create GPU object
    MetalOperations *arrayOps = new MetalOperations(device);

    // Verify all array operations -----------------------------------------------------
    arrayOps->addArrays(a_MTL, b_MTL, c_MTL, arrayLength);
    add(a_CPP, b_CPP, c_VER, arrayLength);
    assert(equalArray(c_CPP, c_VER, arrayLength));
    setZeros(c_CPP, arrayLength);
    std::cout << "Add result is equal to CPU code" << std::endl;

    arrayOps->multiplyArrays(a_MTL, b_MTL, c_MTL, arrayLength);
    multiply(a_CPP, b_CPP, c_VER, arrayLength);
    assert(equalArray(c_CPP, c_VER, arrayLength));
    setZeros(c_CPP, arrayLength);
    std::cout << "Multiply result is equal to CPU code" << std::endl;

    arrayOps->saxpyArrays(k_MTL, a_MTL, b_MTL, c_MTL, arrayLength);
    saxpy(k_CPP, a_CPP, b_CPP, c_VER, arrayLength);
    assert(equalArray(c_CPP, c_VER, arrayLength));
    setZeros(c_CPP, arrayLength);
    std::cout << "SAXPY result is equal to CPU code" << std::endl;

    arrayOps->central_difference(k_MTL, a_MTL, c_MTL, arrayLength);
    central_difference(k_CPP, a_CPP, c_VER, arrayLength);
    assert(equalArray(c_CPP, c_VER, arrayLength));
    setZeros(c_CPP, arrayLength);
    std::cout << "Central difference result is equal to CPU code" << std::endl
              << std::endl;

    // SAXPY benchmarking --------------------------------------------------------------
    std::cout << "Starting SAXPY benchmarking ..." << std::endl;

    float *durations = new float[repeats];
    float array_mean;
    float array_std;

    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        arrayOps->saxpyArrays(k_MTL, a_MTL, b_MTL, c_MTL, arrayLength);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = (stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    array_mean /= 1e3;
    array_std /= 1e3;
    std::cout << "Metal (GPU): \t\t"
              << array_mean << "ms +/- " << array_std << "ms" << std::endl;
    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        saxpy(k_CPP, a_CPP, b_CPP, c_VER, arrayLength);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = (stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    array_mean /= 1e3;
    array_std /= 1e3;
    std::cout << "Serial: \t\t"
              << array_mean << "ms +/- " << array_std << "ms" << std::endl;
    for (size_t threads = 2; threads < 15; threads++)
    {
        omp_set_num_threads(threads);
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            saxpy_openmp(k_CPP, a_CPP, b_CPP, c_VER, arrayLength);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = (stop - start).count();
            durations[repeat] = duration;
        }
        statistics(durations, repeats, array_mean, array_std);
        array_mean /= 1e3;
        array_std /= 1e3;
        std::cout << "OpenMP (" << omp_thread_count() << " cores): \t"
                  << array_mean << "ms +/- " << array_std << "ms" << std::endl;
    }

    // Central differencing benchmarking -----------------------------------------------
    std::cout << std::endl
              << "Starting central differencing benchmarking ..." << std::endl;

    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        arrayOps->central_difference(k_MTL, a_MTL, c_MTL, arrayLength);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = (stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    array_mean /= 1e3;
    array_std /= 1e3;
    std::cout << "Metal (GPU): \t\t"
              << array_mean << "ms +/- " << array_std << "ms" << std::endl;
    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        central_difference(k_CPP, a_CPP, c_VER, arrayLength);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = (stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    array_mean /= 1e3;
    array_std /= 1e3;
    std::cout << "Serial: \t\t"
              << array_mean << "ms +/- " << array_std << "ms" << std::endl;
    for (size_t threads = 2; threads < 15; threads++)
    {
        omp_set_num_threads(threads);
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            central_difference_openmp(k_CPP, a_CPP, c_VER, arrayLength);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = (stop - start).count();
            durations[repeat] = duration;
        }
        statistics(durations, repeats, array_mean, array_std);
        array_mean /= 1e3;
        array_std /= 1e3;
        std::cout << "OpenMP (" << omp_thread_count() << " cores): \t"
                  << array_mean << "ms +/- " << array_std << "ms" << std::endl;
    }
}
