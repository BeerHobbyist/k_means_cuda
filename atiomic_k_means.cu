#include "cuda_runtime.h"
#include "helpers.cuh"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cfloat>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>

__global__ void assign_and_accumulate(const float* __restrict__ points,
                                      const float* __restrict__ centroids,
                                      int* __restrict__ assignments,
                                      float* __restrict__ sums,
                                      int* __restrict__ counts,
                                      const int* __restrict__ prevAssignments,
                                      int* __restrict__ numChanged,
                                      int numPoints, int numCentroids, int dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < numPoints;
       i += blockDim.x * gridDim.x) {

    int best = get_best_centroid_index(points, centroids, numCentroids, dim, i);
    int baseP = i * dim;
    int baseS = best * dim;

    assignments[i] = best;
    if (prevAssignments) {
      int prev = prevAssignments[i];
      if (prev != best) {
        atomicAdd(numChanged, 1);
      }
    }

    atomicAdd(&counts[best], 1);
    for (int d = 0; d < dim; ++d) {
      atomicAdd(&sums[baseS + d], points[baseP + d]);
    }
  }
}

__global__ void compute_new_centroids(const float* __restrict__ sums,
                                      const int* __restrict__ counts,
                                      float* __restrict__ centroids,
                                      int numCentroids, int dim) {
  for (int k = blockIdx.x; k < numCentroids; k += gridDim.x) {
    int t = threadIdx.x;
    int c = counts[k];
    for (int d = t; d < dim; d += blockDim.x) {
      float prev = centroids[k * dim + d];
      float next = (c > 0) ? (sums[k * dim + d] / static_cast<float>(c)) : prev;
      centroids[k * dim + d] = next;
    }
  }
}

void run_atomic_kmeans(const std::vector<float>& h_points,
                       int N, int D, int K,
                       float threshold, int maxIters,
                       std::vector<float>& h_centroids) {
  float* d_points = nullptr;
  float* d_centroids = nullptr;
  float* d_sums = nullptr;
  int* d_counts = nullptr;
  int* d_assign = nullptr;
  int* d_prev_assign = nullptr;
  int* d_changed = nullptr;

  checkCudaErrors(cudaMalloc(&d_points, static_cast<size_t>(N) * D * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_centroids, static_cast<size_t>(K) * D * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_sums, static_cast<size_t>(K) * D * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_counts, static_cast<size_t>(K) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_assign, static_cast<size_t>(N) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_prev_assign, static_cast<size_t>(N) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_changed, sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_points, h_points.data(), static_cast<size_t>(N) * D * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_centroids, h_centroids.data(), static_cast<size_t>(K) * D * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_prev_assign, -1, static_cast<size_t>(N) * sizeof(int)));

  int minGridSize = 0;
  int blockSize = 0;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, assign_and_accumulate, 0, 0));
  int gridSize = (N + blockSize - 1) / blockSize;
  dim3 grid(gridSize);
  dim3 block(blockSize);

  cudaDeviceProp prop{};
  checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
  int gridBlocksK = std::min(K, prop.maxGridSize[0]);
  dim3 gridK(gridBlocksK);
  dim3 blockDimUpdate(32);

  int hostChanged = 0;
  int it = 0;
  while (true) {
    checkCudaErrors(cudaMemset(d_sums, 0, static_cast<size_t>(K) * D * sizeof(float)));
    checkCudaErrors(cudaMemset(d_counts, 0, static_cast<size_t>(K) * sizeof(int)));
    checkCudaErrors(cudaMemset(d_changed, 0, sizeof(int)));

    assign_and_accumulate<<<grid, block>>>(d_points, d_centroids, d_assign, d_sums, d_counts, d_prev_assign, d_changed, N, K, D);
    checkCudaErrors(cudaGetLastError());

    compute_new_centroids<<<gridK, blockDimUpdate>>>(d_sums, d_counts, d_centroids, K, D);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&hostChanged, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    float changeRatio = static_cast<float>(hostChanged) / static_cast<float>(N);
    ++it;
    if (changeRatio <= threshold || it >= maxIters) {
      break;
    }

    std::swap(d_assign, d_prev_assign);
  }
  std::vector<int> h_assignments(N);

  checkCudaErrors(cudaMemcpy(h_centroids.data(), d_centroids, static_cast<size_t>(K) * D * sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << std::endl;

  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_sums);
  cudaFree(d_counts);
  cudaFree(d_assign);
  cudaFree(d_prev_assign);
  cudaFree(d_changed);
}


