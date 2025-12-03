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
                                      int* __restrict__ counts,
                                      int numPoints, int numCentroids, int dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
      i < numPoints;
      i += blockDim.x * gridDim.x) {

    int best = get_best_centroid_index(points, centroids, numCentroids, dim, i);
    assignments[i] = best;
    atomicAdd(&counts[best], 1);
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

__global__ void reduce_tiles_sum(const float* __restrict__ tiles,
                                 float* __restrict__ out,
                                 int numTiles, int numCentroids, int dim) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  // One block per element (k,d) across K*D elements
  int elementIndex = blockIdx.x; // 0 .. K*D-1
  int elementsPerTile = numCentroids * dim;

  float local = 0.0f;
  for (int t = tid; t < numTiles; t += blockDim.x) {
    int idx = t * elementsPerTile + elementIndex;
    local += tiles[idx];
  }
  sdata[tid] = local;
  __syncthreads();

  if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
  if (blockDim.x >= 512)  { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockDim.x >= 256)  { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockDim.x >= 128)  { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

  if (tid < 32) {
    volatile float* v = sdata;
    if (blockDim.x >= 64) v[tid] += v[tid + 32];
    if (blockDim.x >= 32) v[tid] += v[tid + 16];
    if (blockDim.x >= 16) v[tid] += v[tid + 8];
    if (blockDim.x >= 8)  v[tid] += v[tid + 4];
    if (blockDim.x >= 4)  v[tid] += v[tid + 2];
    if (blockDim.x >= 2)  v[tid] += v[tid + 1];
  }
  if (tid == 0) {
    out[elementIndex] = sdata[0];
  }
}

__global__ void compute_new_centroids_from_reduction(const float* __restrict__ sumsFinal,
                                                     const int* __restrict__ counts,
                                                     float* __restrict__ centroids,
                                                     int numCentroids, int dim) {
  for (int k = blockIdx.x; k < numCentroids; k += gridDim.x) {
    int t = threadIdx.x;
    int c = counts[k];
    for (int d = t; d < dim; d += blockDim.x) {
      float prev = centroids[k * dim + d];
      float next = (c > 0) ? (sumsFinal[k * dim + d] / static_cast<float>(c)) : prev;
      centroids[k * dim + d] = next;
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
    checkCudaErrors(cudaMemset(d_counts, 0, static_cast<size_t>(K) * sizeof(int)));

    int minGridSize = 0;
    int blockSize = 0;
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, assign_points_to_centroids, 0, 0));
    int gridSize = (N + blockSize - 1) / blockSize;

    dim3 grid(gridSize);
    dim3 block(blockSize);

    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Block size: " << blockSize << std::endl;

    assign_points_to_centroids<<<grid, block>>>(d_points, d_centroids, d_assign, d_counts, N, K, D);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    size_t tiles = static_cast<size_t>(gridSize) * static_cast<size_t>(blockSize);
    size_t sumsElems = tiles * static_cast<size_t>(K) * static_cast<size_t>(D);
    checkCudaErrors(cudaMalloc(&d_sums, sumsElems * sizeof(float)));

    compute_partial_sums<<<grid, block>>>(d_points, d_assign, d_sums, N, K, D);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Reduce per-thread tiles into a single K*D array
    float* d_sums_final = nullptr;
    checkCudaErrors(cudaMalloc(&d_sums_final, static_cast<size_t>(K) * static_cast<size_t>(D) * sizeof(float)));
    {
      int numTiles = gridSize * blockSize;
      int elements = K * D; // one block per element
      int redBlock = 256;
      dim3 redGrid(elements);
      dim3 redBlockDim(redBlock);
      size_t shm = static_cast<size_t>(redBlock) * sizeof(float);
      reduce_tiles_sum<<<redGrid, redBlockDim, shm>>>(d_sums, d_sums_final, numTiles, K, D);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    // Compute new centroids = sums_final / counts
    {
      int gridBlocksK = std::min(K, 65535); // safety cap if prop not used here
      dim3 gridK(gridBlocksK);
      dim3 blockDimUpdate(32);
      compute_new_centroids_from_reduction<<<gridK, blockDimUpdate>>>(d_sums_final, d_counts, d_centroids, K, D);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    std::vector<int> h_assignments(N);
    checkCudaErrors(cudaMemcpy(h_assignments.data(), d_assign, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float> h_sums_final(static_cast<size_t>(K) * static_cast<size_t>(D));
    checkCudaErrors(cudaMemcpy(h_sums_final.data(), d_sums_final, static_cast<size_t>(K) * static_cast<size_t>(D) * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_centroids.data(), d_centroids, static_cast<size_t>(K) * static_cast<size_t>(D) * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        std::cout << "Assignments[" << i << "] = " << h_assignments[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Reduced sums (first centroid):";
    for (int d = 0; d < D; ++d) {
      std::cout << " " << h_sums_final[d];
    }
    std::cout << std::endl;
    std::cout << "Centroids after reduction update (K=" << K << ", D=" << D << "):" << std::endl;
    for (int k = 0; k < K; ++k) {
      std::cout << "  c" << k << ":";
      for (int d = 0; d < D; ++d) {
        std::cout << " " << h_centroids[k * D + d];
      }
      std::cout << std::endl;
    }

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_sums_final);
    cudaFree(d_assign);
    cudaFree(d_prev_assign);
    cudaFree(d_counts);
}