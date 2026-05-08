#include"matrix_dot.h"
#include<algorithm>
#include<cmath>

void init_matrix_case(
    std::vector<double>& mat,
    std::vector<double>& vec,
    int n) {
        mat.resize(static_cast<size_t>(n)*n);
        vec.resize(n);
        for(int i=0;i<n;++i){
            vec[i]=static_cast<double>(i+1);
            for(int j=0;j<n;++j){
                mat[static_cast<size_t>(i)*n+j]=static_cast<double>(i+j);
            }
        }
    }
    
void matrix_dot_naive(const std::vector<double>& mat,
                      const std::vector<double>& vec,
                      std::vector<double>& sum,
                      int n) {
    sum.resize(n);

    for (int col = 0; col < n; ++col) {
        sum[col] = 0.0;
        for (int row = 0; row < n; ++row) {
            sum[col] += mat[static_cast<size_t>(row) * n + col] * vec[row];
        }
    }
}

void matrix_dot_cache_opt(const std::vector<double>& mat,
                          const std::vector<double>& vec,
                          std::vector<double>& sum,
                          int n) {
    sum.assign(n, 0.0);

    for (int row = 0; row < n; ++row) {
        double v = vec[row];
        const size_t base = static_cast<size_t>(row) * n;
        for (int col = 0; col < n; ++col) {
            sum[col] += mat[base + col] * v;
        }
    }
}

bool vectors_close(const std::vector<double>& a,
                   const std::vector<double>& b,
                   double eps) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        double scale = std::max({1.0, std::fabs(a[i]), std::fabs(b[i])});
        if (std::fabs(a[i] - b[i]) > eps * scale) {
            return false;
        }
    }
    return true;
}