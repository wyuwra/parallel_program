#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "sum_reduce.h"
#include "timer.h"

volatile double g_sum_sink = 0.0;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <naive|two_way|reduction|four_way|all> <n> [target_seconds]\n";
    std::cerr << "Example: " << prog << " all 1048576 0.3\n";
}

static bool nearly_equal(double a, double b, double eps = 1e-9) {
    double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= eps * scale;
}

static void run_and_touch_naive(const std::vector<double>& a, double& out) {
    out = sum_naive(a);
    g_sum_sink += out;
}

static void run_and_touch_two_way(const std::vector<double>& a, double& out) {
    out = sum_two_way(a);
    g_sum_sink += out;
}

static void run_and_touch_four_way(const std::vector<double>& a, double& out) {
    out = sum_four_way(a);
    g_sum_sink += out;
}

static void run_and_touch_reduction(const std::vector<double>& a, double& out) {
    std::vector<double> scratch = a;   // 每次显式复制，逻辑清楚
    out = sum_reduction_inplace(scratch);
    g_sum_sink += out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const int n = std::atoi(argv[2]);
    const double target_seconds = (argc >= 4) ? std::atof(argv[3]) : 0.3;

    if (mode != "naive" && mode != "two_way" &&
        mode != "reduction" && mode != "four_way" && mode != "all") {
        std::cerr << "Error: mode must be naive, two_way, reduction, four_way, or all.\n";
        return 1;
    }

    if (n <= 0 || target_seconds <= 0.0) {
        std::cerr << "Error: n and target_seconds must be positive.\n";
        return 1;
    }

    std::vector<double> a;
    init_sum_case(a, n);

    // 正确性检查：以 naive 作为参考
    const double ref = sum_naive(a);
    const double ans_two_way = sum_two_way(a);
    const double ans_reduction = sum_reduction(a);
    const double ans_four_way = sum_four_way(a);

    bool ok = nearly_equal(ref, ans_two_way) &&
              nearly_equal(ref, ans_reduction) &&
              nearly_equal(ref, ans_four_way);

    if (!ok) {
        std::cerr << "ERROR: sum results do not match within tolerance.\n";
        std::cerr << std::setprecision(17);
        std::cerr << "naive     = " << ref << "\n";
        std::cerr << "two_way   = " << ans_two_way << "\n";
        std::cerr << "reduction = " << ans_reduction << "\n";
        std::cerr << "four_way  = " << ans_four_way << "\n";
        return 2;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Correctness check: PASS\n";
    std::cout << "n = " << n << "\n";
    std::cout << std::setprecision(17);
    std::cout << "reference_sum = " << ref << "\n";
    std::cout << std::setprecision(6);

    if (mode == "naive" || mode == "all") {
        double out = 0.0;
        auto result = benchmark_ms([&]() {
            run_and_touch_naive(a, out);
        }, target_seconds);

        std::cout << "[naive]     avg_ms=" << result.avg_ms
                  << ", total_ms=" << result.total_ms
                  << ", repeat=" << result.repeat
                  << ", result=" << out << "\n";
    }

    if (mode == "two_way" || mode == "all") {
        double out = 0.0;
        auto result = benchmark_ms([&]() {
            run_and_touch_two_way(a, out);
        }, target_seconds);

        std::cout << "[two_way]   avg_ms=" << result.avg_ms
                  << ", total_ms=" << result.total_ms
                  << ", repeat=" << result.repeat
                  << ", result=" << out << "\n";
    }

    if (mode == "reduction" || mode == "all") {
        double out = 0.0;
        auto result = benchmark_ms([&]() {
            run_and_touch_reduction(a, out);
        }, target_seconds);

        std::cout << "[reduction] avg_ms=" << result.avg_ms
                  << ", total_ms=" << result.total_ms
                  << ", repeat=" << result.repeat
                  << ", result=" << out << "\n";
    }

    if (mode == "four_way" || mode == "all") {
        double out = 0.0;
        auto result = benchmark_ms([&]() {
            run_and_touch_four_way(a, out);
        }, target_seconds);

        std::cout << "[four_way]  avg_ms=" << result.avg_ms
                  << ", total_ms=" << result.total_ms
                  << ", repeat=" << result.repeat
                  << ", result=" << out << "\n";
    }

    if (g_sum_sink == -1.0) {
        std::cout << g_sum_sink << "\n";
    }

    return 0;
}