#pragma once
#include <cfloat>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>

inline void __checkCudaErrors(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
      std::fprintf(stderr, "CUDA Runtime Error at %s:%d: %s: %s\n",
                   file, line, func, cudaGetErrorString(result));
      std::exit(1);
    }
  }

#define checkCudaErrors(val) __checkCudaErrors((val), #val, __FILE__, __LINE__)  

static __device__ inline int get_best_centroid_index(const float* __restrict__ points,
                                                     const float* __restrict__ centroids,
                                                     int numCentroids, int dim, int i) {
    int bestIndex = 0;
    float bestDist = FLT_MAX;
    const int pointBase = i * dim;
    for (int k = 0; k < numCentroids; ++k) {
        const int centroidBase = k * dim;
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = points[pointBase + d] - centroids[centroidBase + d];
            dist += diff * diff;
        }
        if (dist < bestDist) {
            bestDist = dist;
            bestIndex = k;
        }
    }
    return bestIndex;
}


