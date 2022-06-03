// Lars Gebraad, 11th of May, 2022
//

#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <assert.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "../02-GeneralArrayOperations/MetalOperations.hpp"
#include "../02-GeneralArrayOperations/CPUOperations.hpp"

typedef std::chrono::nanoseconds time_unit;
auto unit_name = "nanoseconds";

// Configuration -----------------------------------------------------------------------

int main(int argc, char *argv[])
{

    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MetalOperations *arrayOps = new MetalOperations(device);

    std::vector<int> threads_v = {2, 4, 8, 10};

    std::vector<int> arrayLengths = {16,
                                     128,
                                     1024,
                                     8192,
                                     65536,
                                     524288,
                                     4194304,
                                     33554432,
                                     268435456};

    for (auto &&arrayLength : arrayLengths)
    {

        std::cout << "Array length: " << arrayLength << std::endl;

        size_t repeats = 2 * 100 * (4 << (2 * 5)) / arrayLength;
        if (repeats < 50)
            repeats = 50;
        std::cout << "Repeats: " << repeats << std::endl;

        const unsigned int bufferSize = arrayLength * sizeof(float);

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

        // SAXPY benchmarking --------------------------------------------------------------
        std::cout << "Starting SAXPY benchmarking ..." << std::endl;

        std::ofstream filehandle;
        char filename[50];
        sprintf(filename, "results/runtimes_saxpy_%d.csv", arrayLength);
        filehandle.open(filename);

        float *durations = new float[repeats];
        float array_mean;
        float array_std;

        filehandle << "Metal";
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            arrayOps->saxpyArrays(k_MTL, a_MTL, b_MTL, c_MTL, arrayLength);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            filehandle << "," << duration;
            durations[repeat] = duration;
        }
        filehandle << std::endl;
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "Metal (GPU): \t\t"
                  << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;

        filehandle << "Serial";
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            saxpy(k_CPP, a_CPP, b_CPP, c_VER, arrayLength);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            filehandle << "," << duration;
            durations[repeat] = duration;
        }
        filehandle << std::endl;
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "Serial: \t\t"
                  << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;

        for (auto &&threads : threads_v)
        {
            omp_set_num_threads(threads);
            filehandle << "OpenMP" << threads;
            for (size_t repeat = 0; repeat < repeats; repeat++)
            {
                auto start = std::chrono::high_resolution_clock::now();
                saxpy_openmp(k_CPP, a_CPP, b_CPP, c_VER, arrayLength);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
                filehandle << "," << duration;
                durations[repeat] = duration;
            }
            filehandle << std::endl;
            statistics(durations, repeats, array_mean, array_std);
            std::cout << "OpenMP (" << omp_thread_count() << " threads): \t"
                      << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
        }
        filehandle.close();

        // Central differencing benchmarking -----------------------------------------------

        sprintf(filename, "results/runtimes_centraldif_%d.csv", arrayLength);
        filehandle.open(filename);

        std::cout
            << std::endl
            << "Starting central differencing benchmarking ..." << std::endl;

        filehandle << "Metal";
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            arrayOps->central_difference(k_MTL, a_MTL, c_MTL, arrayLength);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            filehandle << "," << duration;
            durations[repeat] = duration;
        }
        filehandle << std::endl;
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "Metal (GPU): \t\t"
                  << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;

        filehandle << "Serial";
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            central_difference(k_CPP, a_CPP, c_VER, arrayLength);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            filehandle << "," << duration;
            durations[repeat] = duration;
        }
        filehandle << std::endl;
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "Serial: \t\t"
                  << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;

        for (auto &&threads : threads_v)
        {
            omp_set_num_threads(threads);
            filehandle << "OpenMP" << threads;
            for (size_t repeat = 0; repeat < repeats; repeat++)
            {
                auto start = std::chrono::high_resolution_clock::now();
                central_difference_openmp(k_CPP, a_CPP, c_VER, arrayLength);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
                filehandle << "," << duration;
                durations[repeat] = duration;
            }
            filehandle << std::endl;
            statistics(durations, repeats, array_mean, array_std);
            std::cout << "OpenMP (" << omp_thread_count() << " threads): \t"
                      << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl;
        }
        filehandle.close();
        delete[] durations;
        delete[] c_VER;
    }
}
