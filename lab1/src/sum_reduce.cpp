#include "sum_reduce.h"

void init_sum_case(std::vector<double>& a, int n) {
    a.resize(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 100) * 0.1;
    }
}

double sum_naive(const std::vector<double>& a) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i];
    }
    return sum;
}

double sum_two_way(const std::vector<double>& a) {
    double sum1 = 0.0;
    double sum2 = 0.0;

    size_t i = 0;
    for (; i + 1 < a.size(); i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }

    if (i < a.size()) {
        sum1 += a[i];
    }

    return sum1 + sum2;
}

double sum_reduction_inplace(std::vector<double>& a) {
    if (a.empty()) {
        return 0.0;
    }

    int m = static_cast<int>(a.size());
    while (m > 1) {
        int new_m = 0;

        int i = 0;
        for (; i + 1 < m; i += 2) {
            a[new_m++] = a[i] + a[i + 1];
        }

        if (i < m) {
            a[new_m++] = a[i];
        }

        m = new_m;
    }

    return a[0];
}

double sum_reduction(std::vector<double> a) {
    return sum_reduction_inplace(a);
}

double sum_four_way(const std::vector<double>& a) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;

    size_t i = 0;
    for (; i + 3 < a.size(); i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }

    for (; i < a.size(); ++i) {
        s0 += a[i];
    }

    return s0 + s1 + s2 + s3;
}