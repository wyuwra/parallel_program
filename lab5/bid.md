# GPU 实验各版本关键代码记录

本文档用于写报告时摘录关键代码。当前仓库源码只保留当前工作版本；早期版本以关键代码片段和实现思路形式记录。

## 版本 0：CPU baseline

原始 `bidiagonalization.cpp` 中 Householder 更新由 CPU 双重循环完成。左 Householder 作用：

```cpp
std::vector<double> w(n - k, 0.0);
for (int j = 0; j < n - k; ++j)
    for (int i = 0; i < m - k; ++i)
        w[j] += v[i] * B.at(k + i, k + j);
for (int i = 0; i < m - k; ++i)
    for (int j = 0; j < n - k; ++j)
        B.at(k + i, k + j) -= beta * v[i] * w[j];
```

右 Householder 作用：

```cpp
std::vector<double> w(m - k, 0.0);
for (int i = 0; i < m - k; ++i)
    for (int j = 0; j < n - k - 1; ++j)
        w[i] += B.at(k + i, k + 1 + j) * v[j];
for (int i = 0; i < m - k; ++i)
    for (int j = 0; j < n - k - 1; ++j)
        B.at(k + i, k + 1 + j) -= beta * w[i] * v[j];
```

CPU baseline 的作用是提供正确性和性能对照。

## 版本 1：cuBLAS v1 简单替换版

第一版 GPU 实现将 TODO 中 GEMV/GER 替换为 cuBLAS，但每次 Householder 更新都会把子矩阵复制到 GPU，计算后再复制回 CPU。

核心处理方式：

```cpp
// A is a column-major copy of M_sub^T.
copy_transposed_submatrix_to_col_major(M, r0, c0, rows, cols, host_A);
cudaMemcpy(d_A, host_A.data(), host_A.size() * sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(d_v, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice);

// left: M_sub -= beta * v * (v^T M_sub)
cublasDgemv(handle, CUBLAS_OP_N,
            cols, rows,
            &one, d_A, cols,
            d_v, 1,
            &zero, d_w, 1);
cublasDger(handle,
           cols, rows,
           &minus_beta,
           d_w, 1,
           d_v, 1,
           d_A, cols);

cudaMemcpy(host_A.data(), d_A, host_A.size() * sizeof(double), cudaMemcpyDeviceToHost);
copy_col_major_transpose_back(M, r0, c0, rows, cols, host_A);
```

这一版解决了 cuBLAS 列主序问题：不直接把 `Matrix` 的行主序内存传给 cuBLAS，而是显式构造 `M_sub^T` 的列主序缓冲区。

问题：

- 每次更新都 `cudaMalloc/cudaFree`。
- 每次更新都拷贝完整子矩阵。
- Nsight Systems 显示 `cudaMemcpy` 占 CUDA API 时间约 `80.3%`，`cudaMalloc/cudaFree` 合计约 `17.6%`。
- 因此 v1 正确但性能较差。

## 版本 2：cuBLAS 优化版

当前代码版本是 cuBLAS 优化版。它仍然使用 `cublasDgemv` 和 `cublasDger`，但优化了数据组织和内存管理。

核心思路：

- `B`、`U`、`V` 在 `to_bidiagonal` 开始时上传到 GPU。
- 整个上二对角化阶段中 `B/U/V` 常驻 GPU。
- `d_v` 和 `d_w` 工作缓冲区只分配一次。
- 每一步只把构造 Householder 向量所需的列段或行段拷回 CPU。
- 结束时再把 `B/U/V` 下载回 CPU。

初始化和常驻 GPU 内存：

```cpp
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
```

读取 Householder 构造所需的列段和行段：

```cpp
void read_b_column_segment(const Matrix &, int row0, int col, int len, std::vector<double> &out)
{
    out.assign(len, 0.0);
    cudaMemcpy2D(out.data(), sizeof(double),
                 d_B_ + static_cast<size_t>(row0) * n_ + col,
                 static_cast<size_t>(n_) * sizeof(double),
                 sizeof(double), len, cudaMemcpyDeviceToHost);
}

void read_b_row_segment(const Matrix &, int row, int col0, int len, std::vector<double> &out)
{
    out.assign(len, 0.0);
    cudaMemcpy(out.data(), d_B_ + static_cast<size_t>(row) * n_ + col0,
               len * sizeof(double), cudaMemcpyDeviceToHost);
}
```

cuBLAS 左 Householder 更新：

```cpp
static void apply_householder_left(HouseholderBackend &backend, Matrix &M,
                                   int r0, int c0, int rows, int cols,
                                   const std::vector<double> &v, double beta)
{
    backend.copy_v_to_device(v);
    double *d_A = backend.device_for(M) + static_cast<size_t>(r0) * M.cols() + c0;

    // Row-major M_sub is viewed as column-major M_sub^T with shape cols x rows.
    cublasDgemv(backend.handle(), CUBLAS_OP_N,
                cols, rows,
                &one, d_A, M.cols(),
                backend.v(), 1,
                &zero, backend.w(), 1);
    cublasDger(backend.handle(),
               cols, rows,
               &minus_beta,
               backend.w(), 1,
               backend.v(), 1,
               d_A, M.cols());
}
```

cuBLAS 右 Householder 更新：

```cpp
static void apply_householder_right(HouseholderBackend &backend, Matrix &M,
                                    int r0, int c0, int rows, int cols,
                                    const std::vector<double> &v, double beta)
{
    backend.copy_v_to_device(v);
    double *d_A = backend.device_for(M) + static_cast<size_t>(r0) * M.cols() + c0;

    // Row-major M_sub is viewed as column-major M_sub^T with shape cols x rows.
    cublasDgemv(backend.handle(), CUBLAS_OP_T,
                cols, rows,
                &one, d_A, M.cols(),
                backend.v(), 1,
                &zero, backend.w(), 1);
    cublasDger(backend.handle(),
               cols, rows,
               &minus_beta,
               backend.v(), 1,
               backend.w(), 1,
               d_A, M.cols());
}
```

这一版的关键是利用行主序矩阵的转置视图：

```text
Matrix row-major M_sub
等价视为 cuBLAS column-major M_sub^T
```

因此：

- 左更新 `M_sub -= beta * v * (v^T M_sub)` 转换为对 `M_sub^T` 做 `w * v^T`。
- 右更新 `M_sub -= beta * (M_sub v) * v^T` 转换为对 `M_sub^T` 做 `v * w^T`。

当前 5 次正式测试平均结果：

```text
CPU baseline 上二对角化平均耗时: 2004.34 ms
cuBLAS 优化版上二对角化平均耗时: 345.754 ms
加速比: 5.797x
```

## 版本 3：手写 CUDA kernel 版

手写 CUDA kernel 是后续任务，应在 cuBLAS 优化版之后进行。之前已经临时实现并测试过一次，数据在 `data/gpu_final` 中；但当前源码已经切回 cuBLAS 优化版。

手写 kernel 版核心思路：

- 不再使用 cuBLAS。
- `B/U/V` 仍然常驻 GPU。
- Householder 向量 `v` 放入 `__constant__` memory。
- GEMV 使用自定义 reduction kernel。
- GER/update 使用二维线程块，每个线程负责矩阵中的一个元素。

关键代码形态如下：

```cpp
__constant__ double c_householder_v[MAX_HOUSEHOLDER_V];

__global__ void householder_left_gemv_kernel(const double *M, int ld,
                                             int r0, int c0,
                                             int rows, int cols,
                                             double *w)
{
    extern __shared__ double partial[];
    int col = blockIdx.x;
    int tid = threadIdx.x;
    double sum = 0.0;

    for (int i = tid; i < rows; i += blockDim.x)
        sum += c_householder_v[i] * M[(r0 + i) * ld + c0 + col];

    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        w[col] = partial[0];
}

__global__ void householder_left_update_kernel(double *M, int ld,
                                               int r0, int c0,
                                               int rows, int cols,
                                               const double *w,
                                               double beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
        M[(r0 + row) * ld + c0 + col] -= beta * c_householder_v[row] * w[col];
}
```

报告中可把该版本作为 cuBLAS 优化后的进一步尝试，重点说明：

- 使用 `__constant__` memory 存储 Householder 向量。
- GER/update 保证按矩阵元素并行更新。
- 不需要修改 GKH 迭代部分。
- 不需要在 GPU 上跑完整 SVD，只加速 `to_bidiagonal` 内部。
