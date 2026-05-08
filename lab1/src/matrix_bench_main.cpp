#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>

#include "matrix_dot.h"
#include "timer.h"

volatile double g_bench_sink = 0.0;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <naive|opt|both> <n> [target_seconds]\n";
    std::cerr << "Example: " << prog << " both 1024 0.3\n";
}

static double checksum(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s;
}

static void run_and_touch_naive(const std::vector<double>& mat,
                                const std::vector<double>& vec,
                                std::vector<double>& sum,
                                int n) {
    matrix_dot_naive(mat, vec, sum, n);
    if (!sum.empty()) g_bench_sink += sum[0];
}

static void run_and_touch_opt(const std::vector<double>& mat,
                              const std::vector<double>& vec,
                              std::vector<double>& sum,
                              int n) {
    matrix_dot_cache_opt(mat, vec, sum, n);
    if (!sum.empty()) g_bench_sink += sum[0];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const int n = std::atoi(argv[2]);
    const double target_seconds = (argc >= 4) ? std::atof(argv[3]) : 0.3;

    if (mode != "naive" && mode != "opt" && mode != "both") {
        std::cerr << "Error: mode must be naive, opt, or both.\n";
        return 1;
    }
    if (n <= 0 || target_seconds <= 0.0) {
        std::cerr << "Error: n and target_seconds must be positive.\n";
        return 1;
    }

    std::vector<double> mat, vec, ref, out;
    init_matrix_case(mat, vec, n);

    // 正确性检查：先跑两种算法
    matrix_dot_naive(mat, vec, ref, n);
    matrix_dot_cache_opt(mat, vec, out, n);

    if (!vectors_close(ref, out, 1e-9)) {
        std::cerr << "ERROR: naive and opt results do not match.\n";
        return 2;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Correctness check: PASS\n";
    std::cout << "n = " << n << "\n";

    if (mode == "naive" || mode == "both") {
        std::vector<double> sum;
        auto result = benchmark_ms([&]() {
            run_and_touch_naive(mat, vec, sum, n);
        }, target_seconds);

        std::cout << "[naive] avg_ms=" << result.avg_ms
                  << ", total_ms=" << result.total_ms
                  << ", repeat=" << result.repeat
                  << ", checksum=" << checksum(sum) << "\n";
    }

    if (mode == "opt" || mode == "both") {
        std::vector<double> sum;
        auto result = benchmark_ms([&]() {
            run_and_touch_opt(mat, vec, sum, n);
        }, target_seconds);

        std::cout << "[opt]   avg_ms=" << result.avg_ms
                  << ", total_ms=" << result.total_ms
                  << ", repeat=" << result.repeat
                  << ", checksum=" << checksum(sum) << "\n";
    }

    if (g_bench_sink == -1.0) {
        std::cout << g_bench_sink << "\n";
    }

    return 0;
}