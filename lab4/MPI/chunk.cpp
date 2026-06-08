#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include<arm_neon.h>


namespace
{

    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };

        enum MpiTag
    {
        TAG_BLOCK_TASK = 200,
        TAG_BLOCK_DATA = 201,
        TAG_BLOCK_RESULT_INT = 202,
        TAG_BLOCK_RESULT_DOUBLE = 203,
        TAG_STOP = 204
    };

    static int g_mpi_rank = 0;
    static int g_mpi_size = 1;

    static double g_mpi_send_time = 0.0;
    static double g_mpi_recv_time = 0.0;
    static double g_mpi_worker_compute_time = 0.0;
    static int g_mpi_total_tasks = 0;

    static void mpi_worker_loop()
    {
        while (true)
        {
            int meta[4] = {0, 0, 0, 0};
            MPI_Status status;

            MPI_Recv(meta, 4, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP)
            {
                break;
            }

            if (status.MPI_TAG == TAG_BLOCK_TASK)
            {
                const int l = meta[0];
                const int r = meta[1];
                const int len = meta[2];
                const int n = meta[3];

                std::vector<double> super_values(len);

                if (len > 0)
                {
                    MPI_Recv(super_values.data(), len, MPI_DOUBLE,
                             0, TAG_BLOCK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                double max_super = 0.0;
                double sum_super = 0.0;

                const double t_compute_begin = MPI_Wtime();

                for (int i = 0; i < len; ++i)
                {
                    const double v = std::fabs(super_values[i]);
                    if (v > max_super)
                    {
                        max_super = v;
                    }
                    sum_super += v;
                }

                const double t_compute_end = MPI_Wtime();
                const double compute_time = t_compute_end - t_compute_begin;

                int result_int[5];
                result_int[0] = g_mpi_rank;
                result_int[1] = l;
                result_int[2] = r;
                result_int[3] = r - l + 1;
                result_int[4] = n;

                double result_double[3];
                result_double[0] = max_super;
                result_double[1] = sum_super;
                result_double[2] = compute_time;

                MPI_Send(result_int, 5, MPI_INT, 0,
                         TAG_BLOCK_RESULT_INT, MPI_COMM_WORLD);

                MPI_Send(result_double, 3, MPI_DOUBLE, 0,
                         TAG_BLOCK_RESULT_DOUBLE, MPI_COMM_WORLD);
            }
        }
    }

    static void mpi_send_stop_to_workers()
    {
        if (g_mpi_rank != 0 || g_mpi_size <= 1)
        {
            return;
        }

        int dummy[4] = {-1, -1, -1, -1};

        for (int rank = 1; rank < g_mpi_size; ++rank)
        {
            MPI_Send(dummy, 4, MPI_INT, rank, TAG_STOP, MPI_COMM_WORLD);
        }
    }

    static void finalize_mpi_at_exit()
    {
        int initialized = 0;
        int finalized = 0;

        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);

        if (initialized && !finalized)
        {
            if (g_mpi_rank == 0)
            {
                std::cerr << "[MPI Profiling] total block tasks = "
              << g_mpi_total_tasks << std::endl;

    std::cerr << "[MPI Profiling] master send time(ms) = "
              << g_mpi_send_time * 1000.0 << std::endl;

    std::cerr << "[MPI Profiling] master recv/wait time(ms) = "
              << g_mpi_recv_time * 1000.0 << std::endl;

    std::cerr << "[MPI Profiling] worker compute time sum(ms) = "
              << g_mpi_worker_compute_time * 1000.0 << std::endl;
                mpi_send_stop_to_workers();
            }

            MPI_Finalize();
        }
    }

    struct MpiBootstrap
    {
        MpiBootstrap()
        {
            int initialized = 0;
            MPI_Initialized(&initialized);

            if (!initialized)
            {
                MPI_Init(nullptr, nullptr);
            }

            MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);

            if (g_mpi_rank != 0)
            {
                std::cerr << "[MPI] worker rank " << g_mpi_rank
                          << " started, mpi_size = " << g_mpi_size
                          << ", enters worker loop before main." << std::endl;

                mpi_worker_loop();

                finalize_mpi_at_exit();

                std::_Exit(0);
            }

            std::cerr << "[MPI] master rank 0 started, mpi_size = "
                      << g_mpi_size << std::endl;

            std::atexit(finalize_mpi_at_exit);
        }
    };

    static MpiBootstrap g_mpi_bootstrap;

        static void mpi_send_block_task_to_worker(int worker_rank,
                                              const Matrix &B,
                                              const Block &blk,
                                              int n)
    {
        const int len = blk.r - blk.l;

        int meta[4];
        meta[0] = blk.l;
        meta[1] = blk.r;
        meta[2] = len;
        meta[3] = n;

        double t0 = MPI_Wtime();

        MPI_Send(meta, 4, MPI_INT, worker_rank,
                 TAG_BLOCK_TASK, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        g_mpi_send_time += (t1 - t0);

        if (len > 0)
        {
            std::vector<double> super_values(len);

            for (int k = blk.l; k < blk.r; ++k)
            {
                super_values[k - blk.l] = B.at(k, k + 1);
            }

            t0 = MPI_Wtime();

            MPI_Send(super_values.data(), len, MPI_DOUBLE, worker_rank,
                     TAG_BLOCK_DATA, MPI_COMM_WORLD);

            t1 = MPI_Wtime();
            g_mpi_send_time += (t1 - t0);
        }
    }

    static void mpi_master_dispatch_block_analysis(const Matrix &B,
                                                   const std::vector<Block> &blocks,
                                                   int n,
                                                   int iter)
    {
        if (g_mpi_rank != 0 || g_mpi_size <= 1)
        {
            return;
        }

        std::vector<Block> tasks;
        for (const auto &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                tasks.push_back(blk);
            }
        }

        if (tasks.empty())
        {
            return;
        }

        int next_task = 0;
        int active_workers = 0;

        for (int rank = 1; rank < g_mpi_size && next_task < static_cast<int>(tasks.size()); ++rank)
        {
            mpi_send_block_task_to_worker(rank, B, tasks[next_task], n);
            ++next_task;
            ++active_workers;
        }

        while (active_workers > 0)
        {
            int result_int[5] = {0, 0, 0, 0, 0};
            double result_double[3] = {0.0, 0.0, 0.0};
            MPI_Status status;

            double t0 = MPI_Wtime();

            MPI_Recv(result_int, 5, MPI_INT, MPI_ANY_SOURCE,
                     TAG_BLOCK_RESULT_INT, MPI_COMM_WORLD, &status);

            const int src = status.MPI_SOURCE;

            MPI_Recv(result_double, 3, MPI_DOUBLE, src,
                     TAG_BLOCK_RESULT_DOUBLE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double t1 = MPI_Wtime();
            g_mpi_recv_time += (t1 - t0);
            g_mpi_worker_compute_time += result_double[2];
            ++g_mpi_total_tasks;

            --active_workers;

            if (iter == 0)
            {
                std::cerr << "[MPI] block analyzed by worker "
                          << result_int[0]
                          << ", block = [" << result_int[1]
                          << ", " << result_int[2] << "]"
                          << ", block_size = " << result_int[3]
                          << ", n = " << result_int[4]
                          << ", max_super = " << result_double[0]
                          << ", sum_super = " << result_double[1]
                          << std::endl;
            }

            if (next_task < static_cast<int>(tasks.size()))
            {
                mpi_send_block_task_to_worker(src, B, tasks[next_task], n);
                ++next_task;
                ++active_workers;
            }
        }
    }

        static void mpi_send_chunk_task_to_worker(int worker_rank,
                                              const Matrix &B,
                                              int l,
                                              int r,
                                              int chunk_l,
                                              int chunk_r,
                                              int n)
    {
        // chunk_l / chunk_r 是超对角线下标范围，闭区间
        // 也就是发送 B[k][k+1], k = chunk_l ... chunk_r
        const int len = chunk_r - chunk_l + 1;

        int meta[4];
        meta[0] = chunk_l;
        meta[1] = chunk_r;
        meta[2] = len;
        meta[3] = n;

        double t0 = MPI_Wtime();

        MPI_Send(meta, 4, MPI_INT, worker_rank,
                 TAG_BLOCK_TASK, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        g_mpi_send_time += (t1 - t0);

        if (len > 0)
        {
            std::vector<double> super_values(len);

            for (int k = chunk_l; k <= chunk_r; ++k)
            {
                super_values[k - chunk_l] = B.at(k, k + 1);
            }

            t0 = MPI_Wtime();

            MPI_Send(super_values.data(), len, MPI_DOUBLE, worker_rank,
                     TAG_BLOCK_DATA, MPI_COMM_WORLD);

            t1 = MPI_Wtime();
            g_mpi_send_time += (t1 - t0);
        }
    }

        static void mpi_master_dispatch_chunk_analysis(const Matrix &B,
                                                   const std::vector<Block> &blocks,
                                                   int n,
                                                   int iter)
    {
        if (g_mpi_rank != 0 || g_mpi_size <= 1)
        {
            return;
        }

        struct ChunkTask
        {
            int block_l;
            int block_r;
            int chunk_l;
            int chunk_r;
        };

        std::vector<ChunkTask> tasks;

        const int worker_count = g_mpi_size - 1;
        const int base_chunk_size = 128;

        for (const auto &blk : blocks)
        {
            if (blk.r <= blk.l)
            {
                continue;
            }

            // 超对角线下标范围是 [blk.l, blk.r - 1]
            const int super_l = blk.l;
            const int super_r = blk.r - 1;
            const int super_len = super_r - super_l + 1;

            int chunk_size = base_chunk_size;

            // 如果 block 很小，就尽量切成 worker_count 份，便于观察多 worker
            if (super_len > 0 && super_len < base_chunk_size * worker_count)
            {
                chunk_size = std::max(1, (super_len + worker_count - 1) / worker_count);
            }

            for (int start = super_l; start <= super_r; start += chunk_size)
            {
                int end = std::min(start + chunk_size - 1, super_r);
                tasks.push_back({blk.l, blk.r, start, end});
            }
        }

        if (tasks.empty())
        {
            return;
        }

        int next_task = 0;
        int active_workers = 0;

        for (int rank = 1; rank < g_mpi_size && next_task < static_cast<int>(tasks.size()); ++rank)
        {
            const auto &task = tasks[next_task];

            mpi_send_chunk_task_to_worker(rank, B,
                                          task.block_l,
                                          task.block_r,
                                          task.chunk_l,
                                          task.chunk_r,
                                          n);

            ++next_task;
            ++active_workers;
        }

        double global_max_super = 0.0;
        double global_sum_super = 0.0;

        while (active_workers > 0)
        {
            int result_int[5] = {0, 0, 0, 0, 0};
            double result_double[3] = {0.0, 0.0, 0.0};
            MPI_Status status;

            double t0 = MPI_Wtime();

            MPI_Recv(result_int, 5, MPI_INT, MPI_ANY_SOURCE,
                     TAG_BLOCK_RESULT_INT, MPI_COMM_WORLD, &status);

            const int src = status.MPI_SOURCE;

            MPI_Recv(result_double, 3, MPI_DOUBLE, src,
                     TAG_BLOCK_RESULT_DOUBLE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double t1 = MPI_Wtime();

            g_mpi_recv_time += (t1 - t0);
            g_mpi_worker_compute_time += result_double[2];
            ++g_mpi_total_tasks;

            --active_workers;

            global_max_super = std::max(global_max_super, result_double[0]);
            global_sum_super += result_double[1];

            if (iter == 0)
            {
                std::cerr << "[MPI] chunk analyzed by worker "
                          << result_int[0]
                          << ", chunk = [" << result_int[1]
                          << ", " << result_int[2] << "]"
                          << ", chunk_size = " << result_int[3]
                          << ", n = " << result_int[4]
                          << ", max_super = " << result_double[0]
                          << ", sum_super = " << result_double[1]
                          << std::endl;
            }

            if (next_task < static_cast<int>(tasks.size()))
            {
                const auto &task = tasks[next_task];

                mpi_send_chunk_task_to_worker(src, B,
                                              task.block_l,
                                              task.block_r,
                                              task.chunk_l,
                                              task.chunk_r,
                                              n);

                ++next_task;
                ++active_workers;
            }
        }

        if (iter == 0)
        {
            std::cerr << "[MPI] chunk-analysis summary: "
                      << "tasks = " << tasks.size()
                      << ", n = " << n
                      << ", global_max_super = " << global_max_super
                      << ", global_sum_super = " << global_sum_super
                      << std::endl;
        }
    }

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            double a = M.at(r0, j);
            double b = M.at(r1, j);
            M.at(r0, j) = c * a + s * b;
            M.at(r1, j) = -s * a + c * b;
        }
    }
    //下面是apply_left_rows的SIMD优化
//     static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
// {
//     const int cols = M.cols();

//     double *row0 = &M.at(r0, 0);
//     double *row1 = &M.at(r1, 0);

//     int j = 0;

//     float64x2_t vc = vdupq_n_f64(c);
//     float64x2_t vs = vdupq_n_f64(s);
//     float64x2_t vneg_s = vdupq_n_f64(-s);

//     for (; j + 1 < cols; j += 2)
//     {
//         float64x2_t a = vld1q_f64(row0 + j);
//         float64x2_t b = vld1q_f64(row1 + j);

//         float64x2_t new0 = vaddq_f64(vmulq_f64(vc, a), vmulq_f64(vs, b));
//         float64x2_t new1 = vaddq_f64(vmulq_f64(vneg_s, a), vmulq_f64(vc, b));

//         vst1q_f64(row0 + j, new0);
//         vst1q_f64(row1 + j, new1);
//     }

//     for (; j < cols; ++j)
//     {
//         double a = row0[j];
//         double b = row1[j];
//         row0[j] = c * a + s * b;
//         row1[j] = -s * a + c * b;
//     }
// }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        for (int i = 0; i < M.rows(); ++i)
        {
            double a = M.at(i, c0);
            double b = M.at(i, c1);
            M.at(i, c0) = a * c - b * s;
            M.at(i, c1) = a * s + b * c;
        }
    }
    //优化下面
//     static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
// {
//     // 当前 GKH 调用中 c0 和 c1 通常是相邻列。
//     // 相邻时，M.at(i,c0) 和 M.at(i,c1) 在同一行内连续存储，
//     // 可以一次加载为 [a,b] 做 2 路 double SIMD。
//     if (c1 == c0 + 1)
//     {
//         const int rows = M.rows();

//         float64x2_t vc = vdupq_n_f64(c);

//         double sign_s_arr[2] = {-s, s};
//         float64x2_t vs = vld1q_f64(sign_s_arr);

//         for (int i = 0; i < rows; ++i)
//         {
//             double *ptr = &M.at(i, c0);

//             // ab = [a, b]
//             float64x2_t ab = vld1q_f64(ptr);

//             // ba = [b, a]
//             float64x2_t ba = vextq_f64(ab, ab, 1);

//             // result = [a,b]*[c,c] + [b,a]*[-s,s]
//             //        = [a*c - b*s, b*c + a*s]
//             float64x2_t result = vaddq_f64(
//                 vmulq_f64(ab, vc),
//                 vmulq_f64(ba, vs)
//             );

//             vst1q_f64(ptr, result);
//         }
//     }
//     else
//     {
//         // 保险起见：如果以后出现非相邻列调用，回退到原始串行版本
//         for (int i = 0; i < M.rows(); ++i)
//         {
//             double a = M.at(i, c0);
//             double b = M.at(i, c1);
//             M.at(i, c0) = a * c - b * s;
//             M.at(i, c1) = a * s + b * c;
//         }
//     }
// }

    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <- U * L^T。
        // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // 由于 L^T = [c -s; s c]，此处复用“右乘两列”接口并传入 -s。
        apply_right_cols(U, r0, r1, c, -s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols(B, l, l + 1, c, s);
        apply_right_cols(V, l, l + 1, c, s);

        // 首次左乘：消去 (l+1, l)。
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols(B, k, k + 1, c, s);
            apply_right_cols(V, k, k + 1, c, s);

            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols(B, i, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(U, B, V, k, tol))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

} // namespace

// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代（改进版）：
// - 输入输出满足 A = U * B * V^T 不变；
// - 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// - 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // 清理数值噪声，并优先处理 d_k≈0 的特殊情形。
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        // 根据超对角线断点拆分活动块
        // 这里子矩阵间是相互独立的，所以此处具有很大的并行潜力：你可以尝试多线程/多进程进行处理
        // 但根据算法，收集 Givens 旋转并更新 U/V 需要在每个块内顺序执行，所以这可能给并行带来麻烦。
        std::vector<Block> blocks = split_active_blocks(B, n, tol);

        if (iter == 0)
{
    mpi_master_dispatch_chunk_analysis(B, blocks, n, iter);
}

        // 若全部是 1x1 块，说明所有超对角都已收敛为 0。
        bool all_singletons = true;
        for (const auto &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                all_singletons = false;
                break;
            }
        }

        if (all_singletons)
        {
            converged = true;
            break;
        }

        // 从右到左处理每个非平凡块，减少末端块对前面块的干扰。
        for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
        {
            if (blocks[i].r > blocks[i].l)
            {
                one_block_step(U, B, V, blocks[i].l, blocks[i].r);
            }
        }
    }

    // 迭代结束后统一结构清理与标准化输出。
    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }
    make_nonnegative_and_sort(U, B, V);

    return converged;
}
