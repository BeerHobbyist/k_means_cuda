#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kmeans_cuda.h"
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>

struct Options {
  std::string inputFile;
  int K = -1;
  float threshold = 0.001f; // stop when changed_memberships / N <= threshold
};

static bool parseArgs(int argc, char** argv, Options& opts) {
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "-i") == 0 && a + 1 < argc) {
      opts.inputFile = argv[++a];
    } else if (std::strcmp(argv[a], "-n") == 0 && a + 1 < argc) {
      opts.K = std::atoi(argv[++a]);
    } else if (std::strcmp(argv[a], "-t") == 0 && a + 1 < argc) {
      opts.threshold = std::atof(argv[++a]);
    } else {
      std::cerr << "Unknown arg: " << argv[a] << "\n";
      return false;
    }
  }
  return true;
}

static bool loadAsciiPoints(const std::string& path, std::vector<float>& points, int& N, int& D) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  std::vector<float> data;
  int dim = -1;
  int count = 0;

  while (std::getline(in, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);
    int id;
    if (!(iss >> id)) continue;

    std::vector<float> coords;
    float v;

    while (iss >> v) coords.push_back(v);

    if (coords.empty()) continue;

    if (dim == -1) dim = static_cast<int>(coords.size());

    if (static_cast<int>(coords.size()) != dim) {
      std::cerr << "Inconsistent dimensionality in ASCII input\n";
      return false;
    }

    data.insert(data.end(), coords.begin(), coords.end());
    ++count;
  }

  if (dim <= 0 || count <= 0) return false;

  points.swap(data);
  N = count;
  D = dim;

  return true;
}

int main(int argc, char** argv) {
  Options opts;
  if (!parseArgs(argc, argv, opts)) return 1;

  // Defaults if no input file is provided
  int N = 10000;  // number of points
  int D = 2;      // dimension
  int K = opts.K > 1 ? opts.K : 3; // clusters
  const float threshold = (opts.threshold >= 0.0f) ? opts.threshold : 0.001f;
  const int maxIters = 1000; // safety cap

  std::vector<float> h_points;

  if (!opts.inputFile.empty()) {
    bool ok = loadAsciiPoints(opts.inputFile, h_points, N, D);
    if (!ok) {
      std::cerr << "Failed to load ASCII input file: " << opts.inputFile << "\n";

      return 1;
    }
  } else {
    cout << "No input file provided, returning 1" << endl;
    return 1;
  }

  if (K <= 1 || K > N) {
    std::cerr << "Invalid K: " << K << " for N=" << N << "\n";
    return 1;
  }

  std::vector<float> h_centroids(K * D);
  for (int k = 0; k < K; ++k) {
    for (int d = 0; d < D; ++d) {
      h_centroids[k * D + d] = h_points[k * D + d];
    }
  }

  run_atomic_kmeans(h_points, N, D, K, threshold, maxIters, h_centroids);

  std::cout << "Final centroids (K=" << K << ", D=" << D << "):\n";
  for (int k = 0; k < K; ++k) {
    std::cout << "  c" << k << ":";
    for (int d = 0; d < D; ++d) {
      std::cout << " " << h_centroids[k * D + d];
    }
    std::cout << "\n";
  }
  
  return 0;
}

