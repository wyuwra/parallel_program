#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "matrix_dot.h"
#include "sum_reduce.h"
#include "timer.h"

using namespace std;

volatile double g_sink = 0.0;

struct MatrixResultRow {
    int n;
    double naive_ms;
    double opt_ms;
    double speedup;
    int naive_repeat;
    int opt_repeat;
};

struct SumResultRow {
    int n;
    double naive_ms;
    double two_way_ms;
    double reduction_ms;
    double four_way_ms;
    double speedup_two_way;
    double speedup_reduction;
    double speedup_four_way;
    int naive_repeat;
    int two_way_repeat;
    int reduction_repeat;
    int four_way_repeat;
};

bool nearly_equal(double x, double y, double eps = 1e-9) {
    return std::fabs(x - y) <= eps;
}

void ensure_output_dirs() {
    std::filesystem::create_directories("result/csv");
}

void write_matrix_csv(const vector<MatrixResultRow>& rows) {
    ofstream fout("result/csv/matrix_results.csv");
    fout << "n,naive_ms,opt_ms,speedup,naive_repeat,opt_repeat\n";
    for (const auto& row : rows) {
        fout << row.n << ","
             << row.naive_ms << ","
             << row.opt_ms << ","
             << row.speedup << ","
             << row.naive_repeat << ","
             << row.opt_repeat << "\n";
    }
}

void write_sum_csv(const vector<SumResultRow>& rows) {
    ofstream fout("result/csv/sum_results.csv");
    fout << "n,naive_ms,two_way_ms,reduction_ms,four_way_ms,"
         << "speedup_two_way,speedup_reduction,speedup_four_way,"
         << "naive_repeat,two_way_repeat,reduction_repeat,four_way_repeat\n";

    for (const auto& row : rows) {
        fout << row.n << ","
             << row.naive_ms << ","
             << row.two_way_ms << ","
             << row.reduction_ms << ","
             << row.four_way_ms << ","
             << row.speedup_two_way << ","
             << row.speedup_reduction << ","
             << row.speedup_four_way << ","
             << row.naive_repeat << ","
             << row.two_way_repeat << ","
             << row.reduction_repeat << ","
             << row.four_way_repeat << "\n";
    }
}

bool run_correctness_check() {
    bool ok = true;

    {
        int n = 8;
        vector<double> mat, vec, out1, out2;
        init_matrix_case(mat, vec, n);

        matrix_dot_naive(mat, vec, out1, n);
        matrix_dot_cache_opt(mat, vec, out2, n);

        bool matrix_ok = vectors_close(out1, out2);
        cout << "Matrix correctness: " << (matrix_ok ? "PASS" : "FAIL") << "\n";
        ok = ok && matrix_ok;
    }

    {
        vector<double> a;
        init_sum_case(a, 16);

        double s1 = sum_naive(a);
        double s2 = sum_two_way(a);
        double s3 = sum_reduction(a);
        double s4 = sum_four_way(a);

        bool sum_ok = nearly_equal(s1, s2) &&
                      nearly_equal(s1, s3) &&
                      nearly_equal(s1, s4);

        cout << "Sum correctness: " << (sum_ok ? "PASS" : "FAIL") << "\n";
        ok = ok && sum_ok;
    }

    cout << "\n";
    return ok;
}

vector<MatrixResultRow> run_matrix_experiment() {
    vector<int> sizes = {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
    vector<MatrixResultRow> rows;

    cout << "===== MATRIX DOT EXPERIMENT =====\n";
    cout << "n,naive_ms,opt_ms,speedup\n";

    for (int n : sizes) {
        vector<double> mat, vec, out;
        init_matrix_case(mat, vec, n);

        // warm-up
        matrix_dot_naive(mat, vec, out, n);
        matrix_dot_cache_opt(mat, vec, out, n);

        BenchmarkResult naive_result = benchmark_ms([&]() {
            matrix_dot_naive(mat, vec, out, n);
            g_sink += out[0];
        });

        BenchmarkResult opt_result = benchmark_ms([&]() {
            matrix_dot_cache_opt(mat, vec, out, n);
            g_sink += out[0];
        });

        MatrixResultRow row;
        row.n = n;
        row.naive_ms = naive_result.avg_ms;
        row.opt_ms = opt_result.avg_ms;
        row.speedup = naive_result.avg_ms / opt_result.avg_ms;
        row.naive_repeat = naive_result.repeat;
        row.opt_repeat = opt_result.repeat;
        rows.push_back(row);

        cout << row.n << ","
             << row.naive_ms << ","
             << row.opt_ms << ","
             << row.speedup << "\n";
    }

    cout << "\n";
    return rows;
}

vector<SumResultRow> run_sum_experiment() {
    vector<int> sizes = {
        1 << 10, 1 << 12, 1 << 14, 1 << 16,
        1 << 18, 1 << 20, 1 << 22
    };

    vector<SumResultRow> rows;

    cout << "===== SUM EXPERIMENT =====\n";
    cout << "n,naive_ms,two_way_ms,reduction_ms,four_way_ms\n";

    for (int n : sizes) {
        vector<double> a;
        init_sum_case(a, n);

        // warm-up
        g_sink += sum_naive(a);
        g_sink += sum_two_way(a);
        g_sink += sum_reduction(a);
        g_sink += sum_four_way(a);

        BenchmarkResult naive_result = benchmark_ms([&]() {
            g_sink += sum_naive(a);
        });

        BenchmarkResult two_way_result = benchmark_ms([&]() {
            g_sink += sum_two_way(a);
        });

        BenchmarkResult reduction_result = benchmark_ms([&]() {
            g_sink += sum_reduction(a);
        });

        BenchmarkResult four_way_result = benchmark_ms([&]() {
            g_sink += sum_four_way(a);
        });

        SumResultRow row;
        row.n = n;
        row.naive_ms = naive_result.avg_ms;
        row.two_way_ms = two_way_result.avg_ms;
        row.reduction_ms = reduction_result.avg_ms;
        row.four_way_ms = four_way_result.avg_ms;
        row.speedup_two_way = naive_result.avg_ms / two_way_result.avg_ms;
        row.speedup_reduction = naive_result.avg_ms / reduction_result.avg_ms;
        row.speedup_four_way = naive_result.avg_ms / four_way_result.avg_ms;
        row.naive_repeat = naive_result.repeat;
        row.two_way_repeat = two_way_result.repeat;
        row.reduction_repeat = reduction_result.repeat;
        row.four_way_repeat = four_way_result.repeat;
        rows.push_back(row);

        cout << row.n << ","
             << row.naive_ms << ","
             << row.two_way_ms << ","
             << row.reduction_ms << ","
             << row.four_way_ms << "\n";
    }

    cout << "\n";
    return rows;
}

int main() {
    cout << fixed << setprecision(6);

    ensure_output_dirs();

    bool ok = run_correctness_check();
    if (!ok) {
        cerr << "Correctness check failed. Stop here and debug first.\n";
        return 1;
    }

    auto matrix_rows = run_matrix_experiment();
    auto sum_rows = run_sum_experiment();

    write_matrix_csv(matrix_rows);
    write_sum_csv(sum_rows);

    cout << "CSV files written to result/csv/\n";
    cerr << "Ignore sink value: " << g_sink << "\n";
    return 0;
}