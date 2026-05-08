#pragma once
#include <vector>

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

void init_sum_case(std::vector<double>& a, int n);

NOINLINE double sum_naive(const std::vector<double>& a);
NOINLINE double sum_two_way(const std::vector<double>& a);
NOINLINE double sum_reduction(std::vector<double> a);                 // 保留旧接口
NOINLINE double sum_reduction_inplace(std::vector<double>& scratch);  // 新增，更适合benchmark
NOINLINE double sum_four_way(const std::vector<double>& a);