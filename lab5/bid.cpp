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
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(USE_CUDA_KERNELS) || defined(__CUDACC__)
#define SVD_USE_CUDA 1
#include <cuda_runtime.h>
#include <cublas_v2.h>
#else
#define SVD_USE_CUDA 0
#endif

// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double x : v)
        sum += x * x;
    return std::sqrt(sum);
}

#if SVD_USE_CUDA
static void check_cuda(cudaError_t status, const char *where)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

static void check_cublas(cublasStatus_t status, const char *where)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(where) + ": cuBLAS call failed");
    }
}

static std::vector<double> pack_matrix_row_major(const Matrix &M)
{
    std::vector<double> data(static_cast<size_t>(M.rows()) * M.cols());
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            data[static_cast<size_t>(i) * M.cols() + j] = M.at(i, j);
        }
    }
    return data;
}

static void unpack_matrix_row_major(Matrix &M, const std::vector<double> &data)
{
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            M.at(i, j) = data[static_cast<size_t>(i) * M.cols() + j];
        }
    }
}

class HouseholderBackend
{
public:
    HouseholderBackend()
    {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~HouseholderBackend()
    {
        if (handle_ != nullptr)
        {
            cublasDestroy(handle_);
        }
        cudaFree(d_B_);
        cudaFree(d_U_);
        cudaFree(d_V_);
        cudaFree(d_v_);
        cudaFree(d_w_);
    }

    void upload(Matrix &B, Matrix &U, Matrix &V)
    {
        B_ = &B;
        U_ = &U;
        V_ = &V;
        m_ = B.rows();
        n_ = B.cols();
        workspace_len_ = std::max(m_, n_);

        allocate(d_B_, static_cast<size_t>(B.rows()) * B.cols());
        allocate(d_U_, static_cast<size_t>(U.rows()) * U.cols());
        allocate(d_V_, static_cast<size_t>(V.rows()) * V.cols());
        allocate(d_v_, workspace_len_);
        allocate(d_w_, workspace_len_);

        upload_matrix(B, d_B_);
        upload_matrix(U, d_U_);
        upload_matrix(V, d_V_);
    }

    void download(Matrix &B, Matrix &U, Matrix &V)
    {
        download_matrix(B, d_B_);
        download_matrix(U, d_U_);
        download_matrix(V, d_V_);
    }

    void read_b_column_segment(const Matrix &, int row0, int col, int len, std::vector<double> &out)
    {
        out.assign(len, 0.0);
        check_cuda(cudaMemcpy2D(out.data(), sizeof(double),
                                d_B_ + static_cast<size_t>(row0) * n_ + col,
                                static_cast<size_t>(n_) * sizeof(double),
                                sizeof(double), len, cudaMemcpyDeviceToHost),
                   "cudaMemcpy2D D2H B column segment");
    }

    void read_b_row_segment(const Matrix &, int row, int col0, int len, std::vector<double> &out)
    {
        out.assign(len, 0.0);
        check_cuda(cudaMemcpy(out.data(), d_B_ + static_cast<size_t>(row) * n_ + col0,
                              len * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H B row segment");
    }

    void zero_b_column_below(Matrix &, int k)
    {
        int len = m_ - k - 1;
        if (len <= 0)
        {
            return;
        }
        check_cuda(cudaMemset2D(d_B_ + static_cast<size_t>(k + 1) * n_ + k,
                                static_cast<size_t>(n_) * sizeof(double),
                                0, sizeof(double), len),
                   "cudaMemset2D B column below");
    }

    void zero_b_row_right(Matrix &, int k)
    {
        int len = n_ - k - 2;
        if (len <= 0)
        {
            return;
        }
        check_cuda(cudaMemset(d_B_ + static_cast<size_t>(k) * n_ + k + 2,
                              0, len * sizeof(double)),
                   "cudaMemset B row right");
    }

    void copy_v_to_device(const std::vector<double> &v)
    {
        check_cuda(cudaMemcpy(d_v_, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D Householder vector");
    }

    double *device_for(Matrix &M)
    {
        if (&M == B_)
            return d_B_;
        if (&M == U_)
            return d_U_;
        if (&M == V_)
            return d_V_;
        throw std::runtime_error("unknown Matrix object for CUDA Householder backend");
    }

    double *v() { return d_v_; }
    double *w() { return d_w_; }
    cublasHandle_t handle() const { return handle_; }

private:
    static void allocate(double *&ptr, size_t count)
    {
        check_cuda(cudaMalloc(&ptr, count * sizeof(double)), "cudaMalloc persistent buffer");
    }

    static void upload_matrix(const Matrix &M, double *dst)
    {
        std::vector<double> data = pack_matrix_row_major(M);
        check_cuda(cudaMemcpy(dst, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D matrix upload");
    }

    static void download_matrix(Matrix &M, const double *src)
    {
        std::vector<double> data(static_cast<size_t>(M.rows()) * M.cols());
        check_cuda(cudaMemcpy(data.data(), src, data.size() * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H matrix download");
        unpack_matrix_row_major(M, data);
    }

    int m_ = 0;
    int n_ = 0;
    int workspace_len_ = 0;
    Matrix *B_ = nullptr;
    Matrix *U_ = nullptr;
    Matrix *V_ = nullptr;
    double *d_B_ = nullptr;
    double *d_U_ = nullptr;
    double *d_V_ = nullptr;
    double *d_v_ = nullptr;
    double *d_w_ = nullptr;
    cublasHandle_t handle_ = nullptr;
};

static void apply_householder_left(HouseholderBackend &backend, Matrix &M,
                                   int r0, int c0, int rows, int cols,
                                   const std::vector<double> &v, double beta)
{
    if (rows <= 0 || cols <= 0)
    {
        return;
    }

    backend.copy_v_to_device(v);
    double *d_A = backend.device_for(M) + static_cast<size_t>(r0) * M.cols() + c0;
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_beta = -beta;

    // Row-major M_sub is viewed as column-major M_sub^T with shape cols x rows.
    check_cublas(cublasDgemv(backend.handle(), CUBLAS_OP_N,
                             cols, rows,
                             &one, d_A, M.cols(),
                             backend.v(), 1,
                             &zero, backend.w(), 1),
                 "cublasDgemv left optimized");
    check_cublas(cublasDger(backend.handle(),
                            cols, rows,
                            &minus_beta,
                            backend.w(), 1,
                            backend.v(), 1,
                            d_A, M.cols()),
                 "cublasDger left optimized");
}

static void apply_householder_right(HouseholderBackend &backend, Matrix &M,
                                    int r0, int c0, int rows, int cols,
                                    const std::vector<double> &v, double beta)
{
    if (rows <= 0 || cols <= 0)
    {
        return;
    }

    backend.copy_v_to_device(v);
    double *d_A = backend.device_for(M) + static_cast<size_t>(r0) * M.cols() + c0;
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_beta = -beta;

    // Row-major M_sub is viewed as column-major M_sub^T with shape cols x rows.
    check_cublas(cublasDgemv(backend.handle(), CUBLAS_OP_T,
                             cols, rows,
                             &one, d_A, M.cols(),
                             backend.v(), 1,
                             &zero, backend.w(), 1),
                 "cublasDgemv right optimized");
    check_cublas(cublasDger(backend.handle(),
                            cols, rows,
                            &minus_beta,
                            backend.v(), 1,
                            backend.w(), 1,
                            d_A, M.cols()),
                 "cublasDger right optimized");
}

#if 0
static void check_cuda(cudaError_t status, const char *where)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

static constexpr int MAX_HOUSEHOLDER_V = 2048;
__constant__ double c_householder_v[MAX_HOUSEHOLDER_V];

__global__ static void gather_column_kernel(const double *A, int ld, int row0, int col, int len, double *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        out[idx] = A[(row0 + idx) * ld + col];
    }
}

__global__ static void zero_column_below_kernel(double *A, int ld, int col, int row_start, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        A[(row_start + idx) * ld + col] = 0.0;
    }
}

__global__ static void zero_row_right_kernel(double *A, int ld, int row, int col_start, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        A[row * ld + col_start + idx] = 0.0;
    }
}

__global__ static void householder_left_gemv_kernel(const double *M, int ld, int r0, int c0,
                                                    int rows, int cols, double *w)
{
    extern __shared__ double partial[];
    int col = blockIdx.x;
    int tid = threadIdx.x;
    double sum = 0.0;

    for (int i = tid; i < rows; i += blockDim.x)
    {
        sum += c_householder_v[i] * M[(r0 + i) * ld + c0 + col];
    }

    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        w[col] = partial[0];
    }
}

__global__ static void householder_left_update_kernel(double *M, int ld, int r0, int c0,
                                                      int rows, int cols, const double *w, double beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
    {
        M[(r0 + row) * ld + c0 + col] -= beta * c_householder_v[row] * w[col];
    }
}

__global__ static void householder_right_gemv_kernel(const double *M, int ld, int r0, int c0,
                                                     int rows, int cols, double *w)
{
    extern __shared__ double partial[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    double sum = 0.0;

    for (int j = tid; j < cols; j += blockDim.x)
    {
        sum += M[(r0 + row) * ld + c0 + j] * c_householder_v[j];
    }

    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        w[row] = partial[0];
    }
}

__global__ static void householder_right_update_kernel(double *M, int ld, int r0, int c0,
                                                       int rows, int cols, const double *w, double beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
    {
        M[(r0 + row) * ld + c0 + col] -= beta * w[row] * c_householder_v[col];
    }
}

static std::vector<double> pack_matrix_row_major(const Matrix &M)
{
    std::vector<double> data(static_cast<size_t>(M.rows()) * M.cols());
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            data[static_cast<size_t>(i) * M.cols() + j] = M.at(i, j);
        }
    }
    return data;
}

static void unpack_matrix_row_major(Matrix &M, const std::vector<double> &data)
{
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            M.at(i, j) = data[static_cast<size_t>(i) * M.cols() + j];
        }
    }
}

class HouseholderBackend
{
public:
    HouseholderBackend() = default;

    ~HouseholderBackend()
    {
        cudaFree(d_B_);
        cudaFree(d_U_);
        cudaFree(d_V_);
        cudaFree(d_w_);
    }

    void upload(Matrix &B, Matrix &U, Matrix &V)
    {
        B_ = &B;
        U_ = &U;
        V_ = &V;
        m_ = B.rows();
        n_ = B.cols();
        workspace_len_ = std::max(m_, n_);

        allocate_matrix(d_B_, static_cast<size_t>(B.rows()) * B.cols());
        allocate_matrix(d_U_, static_cast<size_t>(U.rows()) * U.cols());
        allocate_matrix(d_V_, static_cast<size_t>(V.rows()) * V.cols());
        allocate_matrix(d_w_, workspace_len_);

        upload_matrix(B, d_B_);
        upload_matrix(U, d_U_);
        upload_matrix(V, d_V_);
    }

    void download(Matrix &B, Matrix &U, Matrix &V)
    {
        download_matrix(B, d_B_);
        download_matrix(U, d_U_);
        download_matrix(V, d_V_);
    }

    double *device_for(Matrix &M)
    {
        if (&M == B_)
            return d_B_;
        if (&M == U_)
            return d_U_;
        if (&M == V_)
            return d_V_;
        throw std::runtime_error("unknown Matrix object for CUDA Householder backend");
    }

    double *workspace() { return d_w_; }

    void copy_v_to_constant(const std::vector<double> &v)
    {
        if (static_cast<int>(v.size()) > MAX_HOUSEHOLDER_V)
        {
            throw std::runtime_error("Householder vector is larger than constant memory buffer");
        }
        check_cuda(cudaMemcpyToSymbol(c_householder_v, v.data(), v.size() * sizeof(double)),
                   "cudaMemcpyToSymbol Householder vector");
    }

    void read_b_column_segment(const Matrix &, int row0, int col, int len, std::vector<double> &out)
    {
        out.assign(len, 0.0);
        const int threads = 256;
        const int blocks = (len + threads - 1) / threads;
        gather_column_kernel<<<blocks, threads>>>(d_B_, n_, row0, col, len, d_w_);
        check_cuda(cudaGetLastError(), "gather_column_kernel");
        check_cuda(cudaMemcpy(out.data(), d_w_, len * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H B column segment");
    }

    void read_b_row_segment(const Matrix &, int row, int col0, int len, std::vector<double> &out)
    {
        out.assign(len, 0.0);
        check_cuda(cudaMemcpy(out.data(), d_B_ + static_cast<size_t>(row) * n_ + col0,
                              len * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H B row segment");
    }

    void zero_b_column_below(Matrix &, int k)
    {
        int len = m_ - k - 1;
        if (len <= 0)
            return;
        const int threads = 256;
        const int blocks = (len + threads - 1) / threads;
        zero_column_below_kernel<<<blocks, threads>>>(d_B_, n_, k, k + 1, len);
        check_cuda(cudaGetLastError(), "zero_column_below_kernel");
    }

    void zero_b_row_right(Matrix &, int k)
    {
        int len = n_ - k - 2;
        if (len <= 0)
            return;
        const int threads = 256;
        const int blocks = (len + threads - 1) / threads;
        zero_row_right_kernel<<<blocks, threads>>>(d_B_, n_, k, k + 2, len);
        check_cuda(cudaGetLastError(), "zero_row_right_kernel");
    }

private:
    static void allocate_matrix(double *&ptr, size_t count)
    {
        check_cuda(cudaMalloc(&ptr, count * sizeof(double)), "cudaMalloc persistent buffer");
    }

    static void upload_matrix(const Matrix &M, double *dst)
    {
        std::vector<double> data = pack_matrix_row_major(M);
        check_cuda(cudaMemcpy(dst, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D matrix upload");
    }

    static void download_matrix(Matrix &M, const double *src)
    {
        std::vector<double> data(static_cast<size_t>(M.rows()) * M.cols());
        check_cuda(cudaMemcpy(data.data(), src, data.size() * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H matrix download");
        unpack_matrix_row_major(M, data);
    }

    int m_ = 0;
    int n_ = 0;
    int workspace_len_ = 0;
    Matrix *B_ = nullptr;
    Matrix *U_ = nullptr;
    Matrix *V_ = nullptr;
    double *d_B_ = nullptr;
    double *d_U_ = nullptr;
    double *d_V_ = nullptr;
    double *d_w_ = nullptr;
};

static void apply_householder_left(HouseholderBackend &backend, Matrix &M,
                                   int r0, int c0, int rows, int cols,
                                   const std::vector<double> &v, double beta)
{
    if (rows <= 0 || cols <= 0)
        return;

    backend.copy_v_to_constant(v);
    double *d_M = backend.device_for(M);
    constexpr int reduce_threads = 256;
    householder_left_gemv_kernel<<<cols, reduce_threads, reduce_threads * sizeof(double)>>>(
        d_M, M.cols(), r0, c0, rows, cols, backend.workspace());
    check_cuda(cudaGetLastError(), "householder_left_gemv_kernel");

    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    householder_left_update_kernel<<<grid, block>>>(
        d_M, M.cols(), r0, c0, rows, cols, backend.workspace(), beta);
    check_cuda(cudaGetLastError(), "householder_left_update_kernel");
}

static void apply_householder_right(HouseholderBackend &backend, Matrix &M,
                                    int r0, int c0, int rows, int cols,
                                    const std::vector<double> &v, double beta)
{
    if (rows <= 0 || cols <= 0)
        return;

    backend.copy_v_to_constant(v);
    double *d_M = backend.device_for(M);
    constexpr int reduce_threads = 256;
    householder_right_gemv_kernel<<<rows, reduce_threads, reduce_threads * sizeof(double)>>>(
        d_M, M.cols(), r0, c0, rows, cols, backend.workspace());
    check_cuda(cudaGetLastError(), "householder_right_gemv_kernel");

    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    householder_right_update_kernel<<<grid, block>>>(
        d_M, M.cols(), r0, c0, rows, cols, backend.workspace(), beta);
    check_cuda(cudaGetLastError(), "householder_right_update_kernel");
}
#endif
#else
class HouseholderBackend
{
public:
    void upload(Matrix &, Matrix &, Matrix &) {}
    void download(Matrix &, Matrix &, Matrix &) {}

    void read_b_column_segment(const Matrix &B, int row0, int col, int len, std::vector<double> &out)
    {
        out.resize(len);
        for (int i = 0; i < len; ++i)
        {
            out[i] = B.at(row0 + i, col);
        }
    }

    void read_b_row_segment(const Matrix &B, int row, int col0, int len, std::vector<double> &out)
    {
        out.resize(len);
        for (int j = 0; j < len; ++j)
        {
            out[j] = B.at(row, col0 + j);
        }
    }

    void zero_b_column_below(Matrix &B, int k)
    {
        for (int i = k + 1; i < B.rows(); ++i)
        {
            B.at(i, k) = 0.0;
        }
    }

    void zero_b_row_right(Matrix &B, int k)
    {
        for (int j = k + 2; j < B.cols(); ++j)
        {
            B.at(k, j) = 0.0;
        }
    }
};

static void apply_householder_left(HouseholderBackend &, Matrix &M,
                                   int r0, int c0, int rows, int cols,
                                   const std::vector<double> &v, double beta)
{
    std::vector<double> w(cols, 0.0);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            w[j] += v[i] * M.at(r0 + i, c0 + j);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            M.at(r0 + i, c0 + j) -= beta * v[i] * w[j];
}

static void apply_householder_right(HouseholderBackend &, Matrix &M,
                                    int r0, int c0, int rows, int cols,
                                    const std::vector<double> &v, double beta)
{
    std::vector<double> w(rows, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            w[i] += M.at(r0 + i, c0 + j) * v[j];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            M.at(r0 + i, c0 + j) -= beta * w[i] * v[j];
}
#endif

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
    HouseholderBackend householder_backend;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    householder_backend.upload(B, U, V);

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
        std::vector<double> x;
        householder_backend.read_b_column_segment(B, k, k, m - k, x);

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
            double vTv = 0.0;
            for (double vi : v)
                vTv += vi * vi;

            // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                apply_householder_left(householder_backend, B, k, k, m - k, n - k, v, beta);

                // 累积 U：U_new = U_old * H_k
                // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                apply_householder_right(householder_backend, U, 0, k, m, m - k, v, beta);
            }
        }

        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        householder_backend.zero_b_column_below(B, k);

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y;
            householder_backend.read_b_row_segment(B, k, k + 1, n - k - 1, y);

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = 0.0;
                for (double vi : v)
                    vTv += vi * vi;

                // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 注意：这里是从右侧作用 V_k
                    // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
                    apply_householder_right(householder_backend, B, k, k + 1, m - k, n - k - 1, v, beta);

                    // 累积 V：V_new = V_old * V_k
                    // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    apply_householder_right(householder_backend, V, 0, k + 1, n, n - k - 1, v, beta);
                }
            }

            // 强制置零
            householder_backend.zero_b_row_right(B, k);
        }
    }

    householder_backend.download(B, U, V);
    return B;
}
