#pragma once

#include <iostream>
#include <omp.h>

void add(const float *x, const float *y, float *r, size_t arrayLength);
void multiply(const float *x, const float *y, float *r, size_t arrayLength);
void saxpy(const float *a, const float *x, const float *y, float *r, size_t arrayLength);
void central_difference(const float *delta, const float *x, float *r, size_t arrayLength);

void add_openmp(const float *x, const float *y, float *r, size_t arrayLength);
void multiply_openmp(const float *x, const float *y, float *r, size_t arrayLength);
void saxpy_openmp(const float *a, const float *x, const float *y, float *r, size_t arrayLength);
void central_difference_openmp(const float *delta, const float *x, float *r, size_t arrayLength);

bool equalArray(const float *x, const float *y, size_t arrayLength);
void statistics(float *array, size_t length, float &array_mean, float &array_std);

void generateRandomFloatData(float *dataPtr, size_t arrayLength);
void setZeros(float *dataPtr, size_t arrayLength);
int omp_thread_count();

int linear_IDX(int pos1, int pos2, int shape1, int shape2);
void quadratic2d(const float *X, const float *Y, float *result, int shape1, int shape2);
void quadratic2d_openmp(const float *X, const float *Y, float *result, int shape1, int shape2);
void laplacian2d(const float *X, float *result, int shape1, int shape2);
void laplacian2d_openmp(const float *X, float *result, int shape1, int shape2);
void laplacian2d9p(const float *X, float *result, int shape1, int shape2);
void laplacian2d9p_openmp(const float *X, float *result, int shape1, int shape2);