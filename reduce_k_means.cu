#include "cuda_runtime.h"
#include "helpers.cuh"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstring>

__global__ void assign_points_to_centroids(const float* __restrict__ points,
                                      const float* __restrict__ centroids,
                                      int* __restrict__ assignments,
                                      int numPoints, int numCentroids, int dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
      i < numPoints;
      i += blockDim.x * gridDim.x) {

    int best = get_best_centroid_index(points, centroids, numCentroids, dim, i);
    assignments[i] = best;

    __syncthreads();
  }
}

__global__ void compute_partial_sums(const float* __restrict__ points,
                                     const int* __restrict__ assignments,
                                     float* __restrict__ sums,
                                     int numPoints, int numCentroids, int dim) {
  int blockStart = (blockIdx.x * numPoints) / gridDim.x;
  int blockEnd   = ((blockIdx.x + 1) * numPoints) / gridDim.x;
  int tStart = blockStart + (threadIdx.x * (blockEnd - blockStart)) / blockDim.x;
  int tEnd   = blockStart + ((threadIdx.x + 1) * (blockEnd - blockStart)) / blockDim.x;

  int threadLinear = blockIdx.x * blockDim.x + threadIdx.x;
  int tileBase = threadLinear * numCentroids * dim;

  for (int t = 0; t < numCentroids * dim; ++t) {
    sums[tileBase + t] = 0.0f;
  }

  for (int i = tStart; i < tEnd; ++i) {
    int best = assignments[i];
    int pointBase = i * dim;
    int centroidBase = tileBase + best * dim;
    for (int d = 0; d < dim; ++d) {
      sums[centroidBase + d] += points[pointBase + d];
    }
  }
}

void run_reduce_kmeans(const std::vector<float>& h_points,
                       int N, int D, int K,
                       float threshold, int maxIters,
                       std::vector<float>& h_centroids) {

    float* d_points = nullptr;
    float* d_centroids = nullptr;
    float* d_sums = nullptr;
    int* d_counts = nullptr;
    int* d_assign = nullptr;
    int* d_prev_assign = nullptr;
    int* h_changed = nullptr;

    checkCudaErrors(cudaMalloc(&d_points, static_cast<size_t>(N) * D * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_centroids, static_cast<size_t>(K) * D * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_counts, static_cast<size_t>(K) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_assign, static_cast<size_t>(N) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_prev_assign, static_cast<size_t>(N) * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_points, h_points.data(), static_cast<size_t>(N) * D * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_centroids, h_centroids.data(), static_cast<size_t>(K) * D * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_prev_assign, -1, static_cast<size_t>(N) * sizeof(int)));

    int minGridSize = 0;
    int blockSize = 0;
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, assign_points_to_centroids, 0, 0));
    int gridSize = (N + blockSize - 1) / blockSize;

    dim3 grid(gridSize);
    dim3 block(blockSize);

    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Block size: " << blockSize << std::endl;

    assign_points_to_centroids<<<grid, block>>>(d_points, d_centroids, d_assign, N, K, D);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Allocate per-thread tiles: gridSize * blockSize * K * D
    {
      size_t tiles = static_cast<size_t>(gridSize) * static_cast<size_t>(blockSize);
      size_t sumsElems = tiles * static_cast<size_t>(K) * static_cast<size_t>(D);
      checkCudaErrors(cudaMalloc(&d_sums, sumsElems * sizeof(float)));
    }
    // Compute per-thread partial sums
    compute_partial_sums<<<grid, block>>>(d_points, d_assign, d_sums, N, K, D);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::vector<int> h_assignments(N);
    checkCudaErrors(cudaMemcpy(h_assignments.data(), d_assign, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        std::cout << "Assignments[" << i << "] = " << h_assignments[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_assign);
    cudaFree(d_prev_assign);
}