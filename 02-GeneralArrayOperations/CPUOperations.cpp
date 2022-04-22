#include "CPUOperations.hpp"
#include <math.h>

void generateRandomFloatData(float *dataPtr, size_t arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}

void setZeros(float *dataPtr, size_t arrayLength)
{
    for (size_t i = 0; i < arrayLength; i++)
    {
        dataPtr[i] = 0;
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

void central_difference(const float *delta, const float *x, float *r, size_t arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        if (index == 0)
        {
            r[index] = (x[index + 1] - x[index]) / *delta;
        }
        else if (index == arrayLength - 1)
        {
            r[index] = (x[index] - x[index - 1]) / (*delta);
        }
        else
        {
            r[index] = (x[index + 1] - x[index - 1]) / (2 * *delta);
        }
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

void central_difference_openmp(const float *delta, const float *x, float *r, size_t arrayLength)
{
    unsigned long index;
#pragma omp parallel for default(none) private(index) shared(delta, x, arrayLength, r)

    for (index = 0; index < arrayLength; index++)
    {
        if (index == 0)
        {
            r[index] = (x[index + 1] - x[index]) / *delta;
        }
        else if (index == arrayLength - 1)
        {
            r[index] = (x[index] - x[index - 1]) / (*delta);
        }
        else
        {
            r[index] = (x[index + 1] - x[index - 1]) / (2 * *delta);
        }
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
