// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include<arm_neon.h>

// 辅助函数，计算向量的范数（平方和开根）
// static double vector_norm(const std::vector<double> &v)
// {
//     double sum = 0.0;
//     for (double x : v)
//         sum += x * x;//一次性是否可以计算多次
//     return std::sqrt(sum);
// }
//鲲鹏 920（AArch64）的 NEON 一次最多处理 2 个 double/4 个 float/16 个 int8，核心由128 位向量寄存器决定
//那么这里一次处理俩
// 使用 ARM NEON 计算 double 数组的平方和
static double sum_squares_neon(const double *data, int len)
{
    int i = 0;

    // float64x2_t 表示一次处理 2 个 double
    float64x2_t acc = vdupq_n_f64(0.0);

    // 主循环：每次处理 2 个 double
    for (; i + 1 < len; i += 2)
    {
        float64x2_t x = vld1q_f64(data + i);
        acc = vaddq_f64(acc, vmulq_f64(x, x));
    }

    // 把 SIMD 寄存器里的两个 double 取出来求和
    double tmp[2];
    vst1q_f64(tmp, acc);

    double sum = tmp[0] + tmp[1];

    // 处理剩余的 1 个元素
    for (; i < len; ++i)
    {
        sum += data[i] * data[i];
    }

    return sum;
}

// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    return std::sqrt(sum_squares_neon(v.data(), static_cast<int>(v.size())));
}

// 使用 ARM NEON 计算两个 double 数组的点积 //点积优化——用于构造householder变换和执行
static double dot_product_neon(const double *a, const double *b, int len)
{
    int i = 0;
    float64x2_t acc = vdupq_n_f64(0.0);

    for (; i + 1 < len; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        acc = vaddq_f64(acc, vmulq_f64(va, vb));
    }

    double tmp[2];
    vst1q_f64(tmp, acc);

    double sum = tmp[0] + tmp[1];

    for (; i < len; ++i)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

static void row_sub_scaled_neon4(double *row, const double *v, double scale, int len)
{
    int j = 0;
    float64x2_t scale_vec = vdupq_n_f64(scale);

    for (; j + 3 < len; j += 4)
    {
        float64x2_t row0 = vld1q_f64(row + j);
        float64x2_t v0 = vld1q_f64(v + j);

        float64x2_t row1 = vld1q_f64(row + j + 2);
        float64x2_t v1 = vld1q_f64(v + j + 2);

        row0 = vsubq_f64(row0, vmulq_f64(scale_vec, v0));
        row1 = vsubq_f64(row1, vmulq_f64(scale_vec, v1));

        vst1q_f64(row + j, row0);
        vst1q_f64(row + j + 2, row1);
    }

    for (; j < len; ++j)
    {
        row[j] -= scale * v[j];
    }
}

static void row_sub_scaled_neon8(double *row, const double *v, double scale, int len)
{
    int j = 0;
    float64x2_t scale_vec = vdupq_n_f64(scale);

    for (; j + 7 < len; j += 8)
    {
        float64x2_t row0 = vld1q_f64(row + j);
        float64x2_t v0 = vld1q_f64(v + j);

        float64x2_t row1 = vld1q_f64(row + j + 2);
        float64x2_t v1 = vld1q_f64(v + j + 2);

        float64x2_t row2 = vld1q_f64(row + j + 4);
        float64x2_t v2 = vld1q_f64(v + j + 4);

        float64x2_t row3 = vld1q_f64(row + j + 6);
        float64x2_t v3 = vld1q_f64(v + j + 6);

        row0 = vsubq_f64(row0, vmulq_f64(scale_vec, v0));
        row1 = vsubq_f64(row1, vmulq_f64(scale_vec, v1));
        row2 = vsubq_f64(row2, vmulq_f64(scale_vec, v2));
        row3 = vsubq_f64(row3, vmulq_f64(scale_vec, v3));

        vst1q_f64(row + j, row0);
        vst1q_f64(row + j + 2, row1);
        vst1q_f64(row + j + 4, row2);
        vst1q_f64(row + j + 6, row3);
    }

    for (; j < len; ++j)
    {
        row[j] -= scale * v[j];
    }
}

// 使用 ARM NEON 完成 row[j] -= scale * v[j]
//用来更新矩阵的行数据
static void row_sub_scaled_neon(double *row, const double *v, double scale, int len)
{
    int j = 0;
    float64x2_t scale_vec = vdupq_n_f64(scale);

    for (; j + 1 < len; j += 2)
    {
        float64x2_t row_vec = vld1q_f64(row + j);
        float64x2_t v_vec = vld1q_f64(v + j);

        row_vec = vsubq_f64(row_vec, vmulq_f64(scale_vec, v_vec));

        vst1q_f64(row + j, row_vec);
    }

    for (; j < len; ++j)
    {
        row[j] -= scale * v[j];
    }
}

static void row_add_scaled_neon(double *dst, const double *src, double scale, int len)
{
    int j = 0;
    float64x2_t scale_vec = vdupq_n_f64(scale);

    for (; j + 1 < len; j += 2)
    {
        float64x2_t dst_vec = vld1q_f64(dst + j);
        float64x2_t src_vec = vld1q_f64(src + j);

        dst_vec = vaddq_f64(dst_vec, vmulq_f64(scale_vec, src_vec));

        vst1q_f64(dst + j, dst_vec);
    }

    for (; j < len; ++j)
    {
        dst[j] += scale * src[j];
    }
}

// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            // sign(x[0])：此处规定 x[0]==0 时取 +1
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
            // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
            std::vector<double> v(x);
            v[0] += sigma; // v = x + sigma * e_1

            // 计算 v^T v
            double vTv = sum_squares_neon(v.data(), static_cast<int>(v.size()));

            // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                std::vector<double> w(n - k, 0.0);
                for (int j = 0; j < n - k; ++j)
                    for (int i = 0; i < m - k; ++i)
                        w[j] += v[i] * B.at(k + i, k + j);
                const int len_left = n - k;

                // for (int i = 0; i < m - k; ++i)
                // {
                //     const double scale = v[i];
                //     const double *Brow = &B.at(k + i, k);

                //     row_add_scaled_neon(w.data(), Brow, scale, len_left);
                // }//优化后反而更烂了

                for (int i = 0; i < m - k; ++i)
                {
                    double *Brow = &B.at(k + i, k);
                    const double scale = beta * v[i];
                    row_sub_scaled_neon(Brow, w.data(), scale, len_left);
                }

                // 累积 U：U_new = U_old * H_k
                // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                std::vector<double> wU(m, 0.0);
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m - k; ++j)
                        wU[i] += U.at(i, k + j) * v[j];
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m - k; ++j)
                        U.at(i, k + j) -= beta * wU[i] * v[j];
            }
        }

        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = sum_squares_neon(v.data(), static_cast<int>(v.size()));

                // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 注意：这里是从右侧作用 V_k
                    // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
                std::vector<double> w(m - k, 0.0);
                const int len = n - k - 1;

                for (int i = 0; i < m - k; ++i)
                {
                    const double *Brow = &B.at(k + i, k + 1);
                    w[i] = dot_product_neon(Brow, v.data(), len);
                }
                    for (int i = 0; i < m - k; ++i)
                {
                    double *Brow = &B.at(k + i, k + 1);
                    const double scale = beta * w[i];
                    //row_sub_scaled_neon(Brow, v.data(), scale, len);
                    //row_sub_scaled_neon4(Brow, v.data(), scale, len);
                    row_sub_scaled_neon8(Brow, v.data(), scale, len);
                }

                    // 累积 V：V_new = V_old * V_k
                    // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    std::vector<double> wV(n, 0.0);
for (int i = 0; i < n; ++i)
    for (int j = 0; j < n - k - 1; ++j)
        wV[i] += V.at(i, k + 1 + j) * v[j];

for (int i = 0; i < n; ++i)
    for (int j = 0; j < n - k - 1; ++j)
        V.at(i, k + 1 + j) -= beta * wV[i] * v[j];
                }
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}
