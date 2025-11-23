#pragma once
#include <cfloat>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

inline void __checkCudaErrors(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
      std::fprintf(stderr, "CUDA Runtime Error at %s:%d: %s: %s\n",
                   file, line, func, cudaGetErrorString(result));
      std::exit(1);
    }
  }

#define checkCudaErrors(val) __checkCudaErrors((val), #val, __FILE__, __LINE__)  

__device__ int get_best_centroid_index(const float* __restrict__ points,
                                const float* __restrict__ centroids,
                                int numCentroids, int dim, int i) {
    int best = 0;
    float bestDist = FLT_MAX;

    for (int k = 0; k < numCentroids; ++k) {
        float dist = 0.0f;
        int baseP = i * dim;
        int baseC = k * dim;

        for (int d = 0; d < dim; ++d) {
        float diff = points[baseP + d] - centroids[baseC + d];
        dist += diff * diff;
        }

        if (dist < bestDist) {
        bestDist = dist;
        best = k;
        }
    }

    return best;
}