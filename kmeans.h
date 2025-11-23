#pragma once
#include <vector>

void run_atomic_kmeans(const std::vector<float>& h_points,
                       int N, int D, int K,
                       float threshold, int maxIters,
                       std::vector<float>& h_centroids);

void run_reduce_kmeans(const std::vector<float>& h_points,
                       int N, int D, int K,
                       float threshold, int maxIters,
                       std::vector<float>& h_assignments);


