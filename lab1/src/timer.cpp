#include "timer.h"
#ifdef _WIN32
#include<windows.h>
#else
#include<sys/time.h>
#endif

double now_seconds(){
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static bool inited=false;
    if(!inited){
        QueryPerformanceFrequency(&freq);
        inited=true;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return static_cast<double>(counter.QuadPart)/static_cast<double>(freq.QuadPart);
#else
    timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec / 1e6;
#endif
}

BenchmarkResult benchmark_ms(const std::function<void()>& func,
                             double target_seconds) {
    int repeat = 1;
    double elapsed = 0.0;

    do {
        double t1 = now_seconds();
        for (int i = 0; i < repeat; ++i) {
            func();
        }
        double t2 = now_seconds();
        elapsed = t2 - t1;

        if (elapsed < target_seconds) {
            repeat *= 2;
        }
    } while (elapsed < target_seconds && repeat < (1 << 28));

    BenchmarkResult result;
    result.avg_ms = elapsed * 1000.0 / repeat;
    result.total_ms = elapsed * 1000.0;
    result.repeat = repeat;
    return result;
}