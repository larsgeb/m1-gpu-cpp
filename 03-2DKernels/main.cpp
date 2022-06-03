// Lars Gebraad, 20th of April, 2022
//

#include <iostream>
#include <iomanip>
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

typedef std::chrono::microseconds time_unit;
auto unit_name = "microseconds";

// Configuration -----------------------------------------------------------------------
// Amount of repeats for benchmarking
size_t repeats = 100;
// Length of array to test kernels on

const unsigned int rows = 4000;
const unsigned int columns = 16000;
// end ---------------------------------------------------------------------------------

const unsigned int bufferSize = rows * columns * sizeof(float);

int main(int argc, char *argv[])
{

    // Set up objects and buffers ------------------------------------------------------

    MTL::Device *device = MTL::CreateSystemDefaultDevice();

    std::cout << "Running on " << device->name()->utf8String() << std::endl;
    std::cout << "Array rows: " << rows << ", columns: " << columns
              << ", tests repeated " << repeats << " times" << std::endl;

    // MTL buffers to hold data.
    MTL::Buffer *a_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer *b_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer *c_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    // Get a C++-style reference to the buffer
    auto a_CPP = (float *)a_MTL->contents();
    auto b_CPP = (float *)b_MTL->contents();
    auto c_CPP = (float *)c_MTL->contents();

    float *c_VER = new float[rows * columns];

    generateRandomFloatData(a_CPP, rows * columns);
    generateRandomFloatData(b_CPP, rows * columns);
    setZeros(c_CPP, rows * columns);

    // Create GPU object
    MetalOperations *arrayOps = new MetalOperations(device);

    arrayOps->quadratic2d(a_MTL, b_MTL, c_MTL, rows, columns);
    quadratic2d(a_CPP, b_CPP, c_VER, rows, columns);
    assert(equalArray(c_CPP, c_VER, rows * columns));
    std::cout << "2d function result is equal to CPU code" << std::endl;

    arrayOps->laplacian2d(a_MTL, c_MTL, rows, columns);
    laplacian2d(a_CPP, c_VER, rows, columns);
    assert(equalArray(c_CPP, c_VER, rows * columns));
    std::cout << "Laplacian result is equal to CPU code" << std::endl
              << std::endl;

    // 2d function benchmarking --------------------------------------------------------
    std::cout << "Starting 2d function benchmarking ..." << std::endl;

    float *durations = new float[repeats];
    float array_mean;
    float array_std;

    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        arrayOps->quadratic2d(a_MTL, b_MTL, c_MTL, rows, columns);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Metal (GPU): \t\t"
              << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        quadratic2d(a_CPP, b_CPP, c_VER, rows, columns);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Serial: \t\t"
              << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    for (size_t threads = 2; threads < 15; threads++)
    {
        omp_set_num_threads(threads);
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            quadratic2d_openmp(a_CPP, b_CPP, c_VER, rows, columns);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            durations[repeat] = duration;
        }
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "OpenMP (" << omp_thread_count() << " threads): \t"
                  << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    }

    // Laplacian benchmarking ----------------------------------------------------------
    std::cout << std::endl
              << "Starting Laplacian benchmarking ..." << std::endl;

    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        arrayOps->laplacian2d(a_MTL, c_MTL, rows, columns);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Metal (GPU): \t\t"
              << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        laplacian2d(a_CPP, c_VER, rows, columns);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Serial: \t\t"
              << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    for (size_t threads = 2; threads < 15; threads++)
    {
        omp_set_num_threads(threads);
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            laplacian2d_openmp(a_CPP, c_VER, rows, columns);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            durations[repeat] = duration;
        }
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "OpenMP (" << omp_thread_count() << " threads): \t"
                  << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    }

    // Laplacian 9 point benchmarking --------------------------------------------------
    std::cout << std::endl
              << "Starting Laplacian 9-point benchmarking ..." << std::endl;

    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        arrayOps->laplacian2d9p(a_MTL, c_MTL, rows, columns);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Metal (GPU): \t\t"
              << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        laplacian2d9p(a_CPP, c_VER, rows, columns);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Serial: \t\t"
              << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    for (size_t threads = 2; threads < 15; threads++)
    {
        omp_set_num_threads(threads);
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            laplacian2d9p_openmp(a_CPP, c_VER, rows, columns);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            durations[repeat] = duration;
        }
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "OpenMP (" << omp_thread_count() << " threads): \t"
                  << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
    }
}
