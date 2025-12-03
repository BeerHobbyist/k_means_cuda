#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>

struct Options {
  int numPoints = 100;
  int dims = 9;
  std::string output = "sample_2d_ascii.txt";
};

static void printUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " <n> <d> <output>\n"
      << "  n: number of points (int > 0)\n"
      << "  d: dimensions per point (int > 0)\n"
      << "  output: output file path\n";
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printUsage(argv[0]);
    return 1;
  }

  Options opt;
  opt.numPoints = std::atoi(argv[1]);
  opt.dims = std::atoi(argv[2]);
  opt.output = argv[3];

  if (opt.numPoints <= 0) {
    std::cerr << "numPoints must be > 0\n";
    return 1;
  }
  if (opt.dims <= 0) {
    std::cerr << "dims must be > 0\n";
    return 1;
  }

  std::srand(static_cast<unsigned>(std::time(nullptr)));

  std::ofstream out(opt.output);
  if (!out) {
    std::cerr << "Failed to open output file: " << opt.output << "\n";
    return 1;
  }
  out.setf(std::ios::fixed);
  out << std::setprecision(6);

  for (int i = 0; i < opt.numPoints; ++i) {
    out << (i + 1);
    for (int d = 0; d < opt.dims; ++d) {
      double u = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX); // [0,1]
      out << " " << u;
    }
    out << "\n";
  }

  return 0;
}


