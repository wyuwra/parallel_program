#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "matrix_dot.h"

volatile double g_uprof_sink = 0.0;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <naive|opt> <n> [repeat] [warmup]\n";
    std::cerr << "Example: " << prog << " naive 1024 200 5\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const int n = std::atoi(argv[2]);
    const int repeat = (argc >= 4) ? std::atoi(argv[3]) : 200;
    const int warmup = (argc >= 5) ? std::atoi(argv[4]) : 3;

    if (mode != "naive" && mode != "opt") {
        std::cerr << "Error: mode must be 'naive' or 'opt'.\n";
        print_usage(argv[0]);
        return 1;
    }
    if (n <= 0 || repeat <= 0 || warmup < 0) {
        std::cerr << "Error: n and repeat must be positive, warmup must be non-negative.\n";
        return 1;
    }

    std::vector<double> mat, vec, sum;
    init_matrix_case(mat, vec, n);

    for (int i = 0; i < warmup; ++i) {
        if (mode == "naive") {
            matrix_dot_naive(mat, vec, sum, n);
        } else {
            matrix_dot_cache_opt(mat, vec, sum, n);
        }
        if (!sum.empty()) {
            g_uprof_sink += sum[0];
        }
    }

    for (int i = 0; i < repeat; ++i) {
        if (mode == "naive") {
            matrix_dot_naive(mat, vec, sum, n);
        } else {
            matrix_dot_cache_opt(mat, vec, sum, n);
        }
        if (!sum.empty()) {
            g_uprof_sink += sum[0];
        }
    }

    if (g_uprof_sink == -1.0) {
        std::cout << g_uprof_sink << '\n';
    }
    return 0;
}
