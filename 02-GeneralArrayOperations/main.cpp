// Lars Gebraad, 20th of April, 2022
//

#include <iostream>
#include <omp.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalOperations.hpp"

const unsigned int arrayLength = 1 << 24;
const unsigned int bufferSize = arrayLength * sizeof(float);
void generateRandomFloatData(float *dataPtr, size_t arrayLength);

void add(const float *x, const float *y, float *r, size_t arrayLength);
void multiply(const float *x, const float *y, float *r, size_t arrayLength);
void saxpy(const float *a, const float *x, const float *y, float *r, size_t arrayLength);

void add_openmp(const float *x, const float *y, float *r, size_t arrayLength);
void multiply_openmp(const float *x, const float *y, float *r, size_t arrayLength);
void saxpy_openmp(const float *a, const float *x, const float *y, float *r, size_t arrayLength);

bool equalArray(const float *x, const float *y, size_t arrayLength);
void statistics(float *array, size_t length, float &array_mean, float &array_std);
int omp_thread_count();

int main(int argc, char *argv[])
{

    // Set up objects and buffers ------------------------------------------------------

    MTL::Device *device = MTL::CreateSystemDefaultDevice();

    std::cout << "Running on " << device->name()->utf8String() << std::endl
              << std::endl;

    // MTL buffers to hold data.
    // device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged)
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
    *k_CPP = 904.89547f;

    // Create GPU object
    MetalOperations *arrayOps = new MetalOperations(device);

    // Verify general array operations -------------------------------------------------
    arrayOps->addArrays(a_MTL, b_MTL, c_MTL, arrayLength);
    add(a_CPP, b_CPP, c_VER, arrayLength);
    if (equalArray(c_CPP, c_VER, arrayLength))
    {
        std::cout << "Add arrays works fine" << std::endl;
    }

    arrayOps->multiplyArrays(a_MTL, b_MTL, c_MTL, arrayLength);
    multiply(a_CPP, b_CPP, c_VER, arrayLength);
    if (equalArray(c_CPP, c_VER, arrayLength))
    {
        std::cout << "Multiply arrays works fine" << std::endl;
    }

    arrayOps->saxpyArrays(k_MTL, a_MTL, b_MTL, c_MTL, arrayLength);
    saxpy(k_CPP, a_CPP, b_CPP, c_VER, arrayLength);
    if (equalArray(c_CPP, c_VER, arrayLength))
    {
        std::cout << "SAXPY works fine" << std::endl
                  << std::endl;
    }

    std::cout << "Starting SAXPY benchmarking ..." << std::endl;

    size_t repeats = 1000;
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
}

void generateRandomFloatData(float *dataPtr, size_t arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}

void add(const float *x, const float *y, float *r, size_t arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        r[index] = x[index] + y[index];
    }
}

void multiply(const float *x, const float *y, float *r, size_t arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        r[index] = x[index] * y[index];
    }
}

void saxpy(const float *a, const float *x, const float *y, float *r, size_t arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        r[index] = *a * x[index] + y[index];
    }
}

void add_openmp(const float *x, const float *y, float *r, size_t arrayLength)
{
    unsigned long i;
#pragma omp parallel for default(none) private(i) shared(x, y, arrayLength, r)

    for (i = 0; i < arrayLength; i++)
    {
        r[i] = x[i] + y[i];
    }
}

void multiply_openmp(const float *x, const float *y, float *r, size_t arrayLength)
{
    unsigned long i;
#pragma omp parallel for default(none) private(i) shared(x, y, arrayLength, r)

    for (i = 0; i < arrayLength; i++)
    {
        r[i] = x[i] * y[i];
    }
}

void saxpy_openmp(const float *a, const float *x, const float *y, float *r, size_t arrayLength)
{
    unsigned long i;
#pragma omp parallel for default(none) private(i) shared(a, x, y, arrayLength, r)
    for (i = 0; i < arrayLength; i++)
    {
        r[i] = *a * x[i] * y[i];
    }
}

bool equalFloat(float a, float b, float epsilon)
{
    return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool equalArray(const float *x, const float *y, size_t arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        if (!equalFloat(x[index], y[index], std::numeric_limits<float>::epsilon()))
        {

            printf("Compute ERROR: index=%lu x=%e vs y=%e\n",
                   index, x[index], y[index]);
            return false;
        };
    }
    return true;
}

void statistics(float *array, size_t length, float &array_mean, float &array_std)
{
    // Compute mean and standard deviation serially, template function

    array_mean = 0;
    for (size_t repeat = 0; repeat < length; repeat++)
    {
        array_mean += array[repeat];
    }
    array_mean /= length;

    array_std = 0;
    for (size_t repeat = 0; repeat < length; repeat++)
    {
        array_std += pow(array_mean - array[repeat], 2.0);
    }
    array_std /= length;
    array_std = pow(array_std, 0.5);
}

int omp_thread_count()
{
    int n = 0;
#pragma omp parallel reduction(+ \
                               : n)
    n += 1;
    return n;
}
