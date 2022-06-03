// Lars Gebraad, 10th of May, 2022
//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <assert.h>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "../03-2DKernels/MetalOperations.hpp"
#include "../03-2DKernels/CPUOperations.hpp"

typedef std::chrono::nanoseconds time_unit;
auto unit_name = "nanoseconds";

// Configuration -----------------------------------------------------------------------
// Amount of repeats for benchmarking
// Length of array to test kernels on

int main(int argc, char *argv[])
{

    // Set up objects and buffers ------------------------------------------------------

    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MetalOperations *arrayOps = new MetalOperations(device);

    std::vector<int> threads_v = {2, 4, 8, 10};

    int rows_ = 4;
    for (size_t bitshift = 0; bitshift < 6; bitshift++)
    {
        rows_ = rows_ << 2;
        std::cout << "gridsize: " << rows_ << "x" << rows_ << std::endl;

        size_t repeats = 5 * 100 * (4 << (2 * 5)) / rows_;
        std::cout << "Repeats: " << repeats << std::endl;

        const unsigned int rows = rows_;
        const unsigned int columns = rows_;
        // end ---------------------------------------------------------------------------------

        const unsigned int bufferSize = rows * columns * sizeof(float);

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

        // 2d function benchmarking --------------------------------------------------------

        std::ofstream filehandle;
        char filename[50];
        sprintf(filename, "results/runtimes_function_%d.csv", rows_);
        filehandle.open(filename);

        std::cout << "Starting 2d function benchmarking ..." << std::endl;

        float *durations = new float[repeats];
        float array_mean;
        float array_std;

        filehandle << "Metal";
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            arrayOps->quadratic2d(a_MTL, b_MTL, c_MTL, rows, columns);
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
            quadratic2d(a_CPP, b_CPP, c_VER, rows, columns);
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
                quadratic2d_openmp(a_CPP, b_CPP, c_VER, rows, columns);
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

        // Laplacian benchmarking ----------------------------------------------------------

        // std::ofstream filehandle;
        // char filename[50];
        sprintf(filename, "results/runtimes_laplacian_%d.csv", rows_);
        filehandle.open(filename);

        std::cout << std::endl
                  << "Starting Laplacian benchmarking ..." << std::endl;
        filehandle << "Metal";
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            arrayOps->laplacian2d(a_MTL, c_MTL, rows, columns);
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
            laplacian2d(a_CPP, c_VER, rows, columns);
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
                laplacian2d_openmp(a_CPP, c_VER, rows, columns);
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
        // Laplacian 9 point benchmarking --------------------------------------------------

        // std::ofstream filehandle;
        // char filename[50];
        sprintf(filename, "results/runtimes_laplacian9p_%d.csv", rows_);
        filehandle.open(filename);

        std::cout << std::endl
                  << "Starting Laplacian 9-point benchmarking ..." << std::endl;
        filehandle << "Metal";
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            arrayOps->laplacian2d9p(a_MTL, c_MTL, rows, columns);
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
            laplacian2d9p(a_CPP, c_VER, rows, columns);
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
                laplacian2d9p_openmp(a_CPP, c_VER, rows, columns);
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
    }
}