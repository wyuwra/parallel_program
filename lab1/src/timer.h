#pragma once

#include<functional>

struct BenchmarkResult{
    double avg_ms; //单次耗时(ms)
    double total_ms;//总耗时(ms)
    int repeat;//重复次数
};

double now_seconds();
BenchmarkResult benchmark_ms(const std::function<void()>& func,double target_seconds=0.2);
