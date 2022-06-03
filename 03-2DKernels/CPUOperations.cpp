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
        if (!equalFloat(x[index], y[index], std::numeric_limits<float>::epsilon() * 2))
        {

            printf("Compute ERROR: index=%lu x=%e vs y=%e, epsilon=%e\n",
                   index, x[index], y[index], std::numeric_limits<float>::epsilon() * 2);
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

// 2D operations

int linear_IDX(int pos1, int pos2, int shape1, int shape2)
{
    return pos1 * shape2 + pos2;
}

void quadratic2d(const float *X, const float *Y, float *result, int shape1, int shape2)
{
    for (size_t i1 = 0; i1 < shape1; i1++)
    {
        for (size_t i2 = 0; i2 < shape2; i2++)
        {
            int idx = linear_IDX(i1, i2, shape1, shape2);
            result[idx] = X[idx] * X[idx] / (shape1 * shape1) + Y[idx] * Y[idx] / (shape2 * shape2);
        }
    }
}

void quadratic2d_openmp(const float *X, const float *Y, float *result, int shape1, int shape2)
{
#pragma omp parallel for collapse(2)
    for (size_t i1 = 0; i1 < shape1; i1++)
    {
        for (size_t i2 = 0; i2 < shape2; i2++)
        {
            int idx = linear_IDX(i1, i2, shape1, shape2);
            result[idx] = X[idx] * X[idx] / (shape1 * shape1) + Y[idx] * Y[idx] / (shape2 * shape2);
        }
    }
}

void laplacian2d(const float *X, float *result, int shape1, int shape2)
{
    for (size_t i1 = 1; i1 < shape1 - 1; i1++)
    {
        for (size_t i2 = 1; i2 < shape2 - 1; i2++)
        {
            int idx = linear_IDX(i1, i2, shape1, shape2);

            int idx_xm1 = linear_IDX(i1 - 1, i2, shape1, shape2);
            int idx_xp1 = linear_IDX(i1 + 1, i2, shape1, shape2);

            int idx_ym1 = linear_IDX(i1, i2 - 1, shape1, shape2);
            int idx_yp1 = linear_IDX(i1, i2 + 1, shape1, shape2);

            // Five-point stencil:
            result[idx] = X[idx_xm1] + X[idx_xp1] + X[idx_ym1] + X[idx_yp1] - 4 * X[idx];
        }
    }
}

void laplacian2d_openmp(const float *X, float *result, int shape1, int shape2)
{
#pragma omp parallel for collapse(2)
    for (size_t i1 = 1; i1 < shape1 - 1; i1++)
    {
        for (size_t i2 = 1; i2 < shape2 - 1; i2++)
        {
            int idx = linear_IDX(i1, i2, shape1, shape2);

            int idx_xm1 = linear_IDX(i1 - 1, i2, shape1, shape2);
            int idx_xp1 = linear_IDX(i1 + 1, i2, shape1, shape2);

            int idx_ym1 = linear_IDX(i1, i2 - 1, shape1, shape2);
            int idx_yp1 = linear_IDX(i1, i2 + 1, shape1, shape2);

            // Five-point stencil:
            result[idx] = X[idx_xm1] + X[idx_xp1] + X[idx_ym1] + X[idx_yp1] - 4 * X[idx];
        }
    }
}

void laplacian2d9p(const float *X, float *result, int shape1, int shape2)
{
    for (size_t i1 = 1; i1 < shape1 - 1; i1++)
    {
        for (size_t i2 = 1; i2 < shape2 - 1; i2++)
        {

            int idx = linear_IDX(i1, i2, shape1, shape2);

            int idx_xm1 = linear_IDX(i1 - 1, i2, shape1, shape2);
            int idx_xp1 = linear_IDX(i1 + 1, i2, shape1, shape2);

            int idx_ym1 = linear_IDX(i1, i2 - 1, shape1, shape2);
            int idx_yp1 = linear_IDX(i1, i2 + 1, shape1, shape2);

            int idx_xm1ym1 = linear_IDX(i1 - 1, i2 - 1, shape1, shape2);
            int idx_xp1yp1 = linear_IDX(i1 + 1, i2 + 1, shape1, shape2);

            int idx_xm1yp1 = linear_IDX(i1 - 1, i2 + 1, shape1, shape2);
            int idx_xp1ym1 = linear_IDX(i1 + 1, i2 - 1, shape1, shape2);

            // Five-point stencil:
            result[idx] =
                0.25 * (X[idx_xm1ym1] + X[idx_xp1yp1] + X[idx_xm1yp1] + X[idx_xp1ym1]) +
                0.5 * (X[idx_xm1] + X[idx_xp1] + X[idx_ym1] + X[idx_yp1]) -
                3 * X[idx];
        }
    }
}

void laplacian2d9p_openmp(const float *X, float *result, int shape1, int shape2)
{
#pragma omp parallel for collapse(2)
    for (size_t i1 = 1; i1 < shape1 - 1; i1++)
    {
        for (size_t i2 = 1; i2 < shape2 - 1; i2++)
        {
            int idx = linear_IDX(i1, i2, shape1, shape2);

            int idx_xm1 = linear_IDX(i1 - 1, i2, shape1, shape2);
            int idx_xp1 = linear_IDX(i1 + 1, i2, shape1, shape2);

            int idx_ym1 = linear_IDX(i1, i2 - 1, shape1, shape2);
            int idx_yp1 = linear_IDX(i1, i2 + 1, shape1, shape2);

            int idx_xm1ym1 = linear_IDX(i1 - 1, i2 - 1, shape1, shape2);
            int idx_xp1yp1 = linear_IDX(i1 + 1, i2 + 1, shape1, shape2);

            int idx_xm1yp1 = linear_IDX(i1 - 1, i2 + 1, shape1, shape2);
            int idx_xp1ym1 = linear_IDX(i1 + 1, i2 - 1, shape1, shape2);

            // Five-point stencil:
            result[idx] =
                0.25 * (X[idx_xm1ym1] + X[idx_xp1yp1] + X[idx_xm1yp1] + X[idx_xp1ym1]) +
                0.5 * (X[idx_xm1] + X[idx_xp1] + X[idx_ym1] + X[idx_yp1]) -
                3 * X[idx];
        }
    }
}

int omp_thread_count()
{
    int n = 0;
#pragma omp parallel reduction(+ \
                               : n)
    n += 1;
    return n;
}
