# SVD GPU/cuBLAS 实验记录

## 实验目标

本次 GPU 编程进阶部分选择对矩阵 SVD 程序中的上二对角化阶段进行 GPU 化。根据实验指导要求，当前阶段只修改 `bidiagonalization.cpp`，将其中标注 `TODO(SIMD编程)` 的两处 Householder 更新中的 GEMV 和 GER 运算替换为 cuBLAS 调用。

原始程序流程保持不变：

1. `main.cpp` 构造测试矩阵。
2. `to_bidiagonal` 将矩阵化为上二对角形式，并累积 `U`、`V`。
3. `gkh_svd_from_bidiagonal` 在 CPU 上继续完成 GKH 迭代。
4. 程序检查重构误差、正交误差、对角结构、奇异值排序等指标。

## 当前代码实现

修改范围仅限 `bidiagonalization.cpp`。

当前实现加入了条件编译：

- 使用 `nvcc` 或定义 `USE_CUBLAS` 编译时，启用 cuBLAS 路径。
- 使用普通 `g++` 编译时，保留 CPU fallback，数学逻辑与原始循环一致。

新增的主要辅助函数：

- `apply_householder_left(...)`
  - 对应左侧 Householder 作用：
  - `M_sub -= beta * v * (v^T * M_sub)`
  - 用 cuBLAS 的 `cublasDgemv` 计算 `v^T * M_sub`
  - 用 cuBLAS 的 `cublasDger` 完成秩一更新

- `apply_householder_right(...)`
  - 对应右侧 Householder 作用：
  - `M_sub -= beta * (M_sub * v) * v^T`
  - 用 cuBLAS 的 `cublasDgemv` 计算 `M_sub * v`
  - 用 cuBLAS 的 `cublasDger` 完成秩一更新

由于 cuBLAS 默认使用列主序，而项目中的 `Matrix` 类使用行主序，当前实现没有直接把 `Matrix` 内存传给 cuBLAS。处理方式是：

1. 从 `Matrix` 中取出当前子矩阵。
2. 在 host 端构造该子矩阵转置后的列主序缓冲区，即保存 `M_sub^T`。
3. 将缓冲区拷贝到 GPU。
4. 调用 cuBLAS 的 GEMV/GER。
5. 将结果拷回 host。
6. 再写回原来的 `Matrix` 子矩阵位置。

这种方式优点是实现直接、正确性容易验证；缺点是每次 Householder 更新都会产生较多 host-device 数据传输和临时内存分配。

## 编译与运行

PowerShell 中可以使用如下命令加载 Visual Studio 2022 编译环境并编译 GPU 版本：

```powershell
$bat='C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat'
cmd /c "call `"$bat`" -arch=x64 && nvcc -x cu -std=c++17 -O2 -Xcompiler /utf-8 main.cpp bidiagonalization.cpp gkh.cpp -lcublas -o main_gpu.exe"
```

运行并保存结果：

```powershell
New-Item -ItemType Directory -Force data\gpu
.\main_gpu.exe 20260410 2>&1 | Out-File -Encoding utf8 data\gpu\gpu_20260410.txt
```

如果只用 CPU fallback 检查当前代码是否仍可普通编译：

```powershell
g++ -std=c++17 -O2 main.cpp bidiagonalization.cpp gkh.cpp -o main_cpu_check.exe
.\main_cpu_check.exe 20260410
```

## 已有数据

`data/cpu` 中已有 5 次原始 CPU 版本数据，均为 seed `20260410`，结果均通过 `5 / 5`。

1000x1000 随机矩阵上的 CPU baseline 摘要：

| 文件 | 上二对角化耗时(ms) | GKH 耗时(ms) |
| --- | ---: | ---: |
| `data/cpu/1.o` | 2034.24 | 4146.29 |
| `data/cpu/2.o` | 2017.43 | 5212.76 |
| `data/cpu/3.o` | 1934.61 | 5237.31 |
| `data/cpu/4.o` | 1946.83 | 5110.92 |
| `data/cpu/5.o` | 2088.61 | 5251.46 |
| 平均 | 2004.34 | 4991.75 |

`data/gpu/gpu_20260410.txt` 是早期单次测试数据，不纳入最终统计。当前正式使用 `data/gpu/1.txt` 到 `data/gpu/5.txt` 这 5 次 cuBLAS v1 测试数据，seed 均为 `20260410`，结果均通过 `5 / 5`。

1000x1000 随机矩阵上的当前 GPU cuBLAS v1 版本结果：

| 文件 | 上二对角化耗时(ms) | GKH 耗时(ms) |
| --- | ---: | ---: |
| `data/gpu/1.txt` | 12744.5 | 6305.75 |
| `data/gpu/2.txt` | 11685.5 | 6384.29 |
| `data/gpu/3.txt` | 11605.0 | 6275.23 |
| `data/gpu/4.txt` | 13052.1 | 6182.63 |
| `data/gpu/5.txt` | 12312.1 | 5463.31 |
| 平均 | 12279.84 | 6122.24 |

本实验主要关注上二对角化阶段，因此加速比只针对 `time bidiagonalization(ms)` 分析：

| 版本 | 上二对角化平均耗时(ms) | 相对 CPU baseline 加速比 |
| --- | ---: | ---: |
| CPU baseline | 2004.34 | 1.00x |
| GPU cuBLAS v1 | 12279.84 | 0.16x |

加速比计算方式：

```text
speedup = CPU baseline average / GPU cuBLAS v1 average
        = 2004.34 / 12279.84
        = 0.16x
```

当前 cuBLAS v1 版本相对 CPU baseline 没有加速，约为 CPU 速度的 `0.16x`，也可以理解为上二对角化阶段约慢 `6.13x`。这说明简单把 GEMV/GER 替换为 cuBLAS 并不足以获得性能提升，主要原因需要结合 profiling 分析。

## Profiling 结果

使用 Nsight Systems 对当前 cuBLAS v1 版本进行 profiling：

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe' profile --trace=cuda,cublas,wddm --stats=true --force-overwrite=true -o data\gpu\nsys_cublas_v1 .\main_gpu.exe 20260410
```

生成文件：

```text
data/gpu/nsys_cublas_v1.nsys-rep
data/gpu/nsys_cublas_v1.sqlite
```

Nsight Systems 的 CUDA API 汇总结果：

| CUDA API | 时间占比 | 总时间(ns) | 调用次数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `cudaMemcpy` | 80.3% | 5507138925 | 12270 | Host/Device 数据传输占主要开销 |
| `cudaFree` | 9.3% | 640402871 | 12290 | 每次更新后释放临时 GPU 内存 |
| `cudaMalloc` | 8.3% | 568414883 | 12285 | 每次更新前重新分配临时 GPU 内存 |
| `cudaLaunchKernel` | 1.9% | 130886991 | 8037 | cuBLAS 内部 kernel 调用 |

GPU 内存传输统计：

| 方向 | 调用次数 | 总传输量(MB) | 说明 |
| --- | ---: | ---: | --- |
| Host-to-Device | 8180 | 13353.348 | 每次 Householder 更新前上传子矩阵和向量 |
| Device-to-Host | 4090 | 13337.344 | 每次 Householder 更新后将子矩阵写回 CPU |

profiling 结论：

1. 当前版本的主要瓶颈不是 GPU 计算，而是 `cudaMemcpy`。
2. `cudaMemcpy` 占 CUDA API 时间约 `80.3%`，说明数据在 CPU 和 GPU 之间频繁往返。
3. `cudaMalloc/cudaFree` 合计约 `17.6%`，说明每次 Householder 更新都临时分配和释放 GPU 缓冲区，开销明显。
4. 真正的 kernel launch 时间占比只有约 `1.9%`，说明 cuBLAS 的计算本身不是当前主要瓶颈。
5. 当前实现验证了 cuBLAS 替换的正确性，但还不是高性能 GPU 实现。

## 当前 GPU 数据是否会用于报告

这五次 GPU 数据可以使用，但更适合作为“第一版 cuBLAS 替换实现”的阶段性数据，而不是最终性能优化结论。

它可以用于说明：

- cuBLAS 替换后的程序正确性通过。
- 简单替换 GEMV/GER 并不一定带来加速。
- 当前主要瓶颈很可能来自频繁的 `cudaMemcpy`、`cudaMalloc/cudaFree` 和小粒度 cuBLAS 调用。
- 这组数据可以作为后续 profiling 的动机。

如果最终报告需要展示性能提升，建议后续继续优化后再至少测 5 次 GPU 数据，与 `data/cpu` 的 5 次 baseline 做平均值对比。当前这五次 GPU 数据可以保留在报告的“初始 GPU 版本分析”或“性能瓶颈分析”部分。

## 后续优化方向

当前版本的数据路径是：

```text
CPU Matrix -> host 转置缓冲区 -> GPU -> cuBLAS -> host -> CPU Matrix
```

后续优化应尽量改为：

```text
CPU Matrix -> GPU
GPU 内部完成多轮 Householder 更新
GPU -> CPU Matrix
```

可考虑的优化点：

1. 复用 cuBLAS handle 和 GPU 工作缓冲区。
2. 减少每次 Householder 更新中的 `cudaMalloc/cudaFree`。
3. 让 `B`、`U`、`V` 尽量常驻 GPU 内存。
4. 使用 profiling 工具确认 `cudaMemcpy`、内存分配和 cuBLAS 调用各自占比。
5. 根据 profiling 结果决定是否需要自定义 kernel 处理部分更新。

## 当前 cuBLAS 优化版

当前代码已调整为 cuBLAS 优化版，仍然只修改 `bidiagonalization.cpp`。这一阶段还没有把 GEMV/GER 替换为手写 CUDA kernel；手写 kernel 属于后续实验任务。

cuBLAS 优化版核心变化：

1. `B`、`U`、`V` 在 `to_bidiagonal` 开始后上传到 GPU，并在整个上二对角化过程中常驻 GPU。
2. 循环中只将构造 Householder 向量所需的当前列段或行段从 GPU 拷回 CPU。
3. Householder 向量 `v` 复制到预分配的 GPU buffer。
4. 继续使用 cuBLAS 的 `cublasDgemv` 和 `cublasDger` 完成 GEMV/GER。
5. 每个矩阵只在开始时上传一次，在 `to_bidiagonal` 结束时下载一次。
6. 不再在每次 Householder 更新时执行大规模 host-device 子矩阵拷贝。
7. 不再在每次 Householder 更新中反复 `cudaMalloc/cudaFree`。
8. 针对 `Matrix` 行主序与 cuBLAS 列主序的差异，将 row-major 子矩阵视为 column-major 的转置视图调用 cuBLAS。

当前 cuBLAS 优化版对应实验指导中的 cuBLAS 阶段要求：

- GEMV/GER 使用 cuBLAS 调用。
- 处理了 cuBLAS 默认列主序与 `Matrix` 行主序的差异。
- 减少了 v1 中频繁的 `cudaMemcpy` 和 `cudaMalloc/cudaFree`。
- 不改动 GKH 迭代部分。
- 不在 GPU 上实现完整 SVD，只加速 `to_bidiagonal` 内部。

cuBLAS 优化版编译命令需要链接 `-lcublas`：

```powershell
$bat='C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat'
cmd /c "call `"$bat`" -arch=x64 && nvcc -x cu -std=c++17 -O2 -Xcompiler /utf-8 main.cpp bidiagonalization.cpp gkh.cpp -lcublas -o main_gpu_cublas_opt.exe"
```

cuBLAS 优化版首次测试结果，seed 为 `20260410`：

| 测试项 | 结果 |
| --- | ---: |
| 正确性 | `通过: 5 / 5` |
| 1000x1000 上二对角化耗时(ms) | 310.988 |
| 1000x1000 GKH 耗时(ms) | 7528.34 |
| 总上二对角化耗时(ms) | 644.316 |
| 总 GKH 耗时(ms) | 7528.42 |

相对于 CPU baseline 的 1000x1000 上二对角化平均耗时 `2004.34 ms`，本次 cuBLAS 优化版单次上二对角化加速比为：

```text
speedup = 2004.34 / 310.988 = 6.44x
```

该结果说明，在仍然使用 cuBLAS 的前提下，消除 v1 中频繁的大规模 `cudaMemcpy` 和 `cudaMalloc/cudaFree` 后，上二对角化阶段获得了明显加速。后续手写 CUDA kernel 阶段应在该 cuBLAS 优化版之后进行。

cuBLAS 优化版正式 5 次测试数据位于 `data/gpu_cublas_opt/1.txt` 到 `data/gpu_cublas_opt/5.txt`，均通过 `5 / 5`。

| 文件 | 1000x1000 上二对角化耗时(ms) | GKH 耗时(ms) |
| --- | ---: | ---: |
| `data/gpu_cublas_opt/1.txt` | 299.297 | 5589.88 |
| `data/gpu_cublas_opt/2.txt` | 421.552 | 5913.68 |
| `data/gpu_cublas_opt/3.txt` | 339.190 | 5832.42 |
| `data/gpu_cublas_opt/4.txt` | 325.254 | 5996.48 |
| `data/gpu_cublas_opt/5.txt` | 343.478 | 6265.02 |
| 平均 | 345.754 | 5919.496 |

正式 5 次平均加速比：

```text
speedup = CPU baseline average / GPU cuBLAS opt average
        = 2004.34 / 345.754
        = 5.797x
```

因此，在报告中建议以 `5.80x` 作为 cuBLAS 优化版相对 CPU baseline 的上二对角化加速比。

cuBLAS 优化版 Nsight Systems profiling：

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe' profile --trace=cuda,cublas,wddm --stats=true --force-overwrite=true -o data\gpu_cublas_opt\nsys_cublas_opt .\main_gpu_cublas_opt.exe 20260410
```

生成文件：

```text
data/gpu_cublas_opt/nsys_cublas_opt.nsys-rep
data/gpu_cublas_opt/nsys_cublas_opt.sqlite
```

CUDA API 汇总：

| CUDA API | 时间占比 | 总时间(ns) | 调用次数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `cudaFree` | 30.1% | 202662118 | 45 | 程序结束时释放少量持久 GPU 缓冲区 |
| `cudaMemcpy` | 26.3% | 177472824 | 5139 | 主要为 Householder 向量和少量矩阵上传/下载 |
| `cudaLaunchKernel` | 20.1% | 135587168 | 8037 | cuBLAS 内部 kernel 调用 |
| `cudaMemcpy2D` | 16.5% | 111503014 | 1029 | 读取列段用于构造 Householder 向量 |
| `cudaMemset2D` | 2.4% | 16071848 | 1026 | 清零下三角列元素 |
| `cudaMemset` | 2.0% | 13584942 | 1019 | 清零行右侧元素 |
| `cudaMalloc` | 0.4% | 2481916 | 40 | 只在初始化时分配持久缓冲区 |

GPU 内存传输统计：

| 方向 | 调用次数 | 总传输量(MB) |
| --- | ---: | ---: |
| Host-to-Device | 4105 | 40.010 |
| Device-to-Host | 2063 | 32.008 |
| memset | 2045 | 7.986 |

相比 cuBLAS v1，优化版将传输量从约 `13 GB + 13 GB` 降到约 `40 MB + 32 MB`，并将 `cudaMalloc` 调用次数从 `12285` 次降到 `40` 次。这说明 `B/U/V` 常驻 GPU 和复用 buffer 的优化有效。

## 手写 CUDA kernel profiling

手写 CUDA kernel 版本对应此前编译出的 `main_gpu_final.exe`，数据目录为 `data/gpu_final`。注意：当前源码已经切回 cuBLAS 优化版；`main_gpu_final.exe` 是之前手写 kernel 版本的可执行文件。

profiling 命令：

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe' profile --trace=cuda,wddm --stats=true --force-overwrite=true -o data\gpu_final\nsys_kernel_final .\main_gpu_final.exe 20260410
```

生成文件：

```text
data/gpu_final/nsys_kernel_final.nsys-rep
data/gpu_final/nsys_kernel_final.sqlite
```

CUDA API 汇总：

| CUDA API | 时间占比 | 总时间(ns) | 调用次数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `cudaLaunchKernel` | 35.5% | 270324222 | 11254 | 自定义 GEMV/GER/update kernel 调用 |
| `cudaMemcpy` | 26.2% | 199602873 | 2078 | 主要为矩阵上传/下载和少量行列段读取 |
| `cudaMalloc` | 20.6% | 156981798 | 20 | 初始化持久缓冲区 |
| `cudaMemcpyToSymbol` | 17.2% | 130908784 | 4090 | 每次 Householder 更新上传向量 `v` 到 constant memory |
| `cudaFree` | 0.3% | 2136756 | 20 | 程序结束释放缓冲区 |

自定义 kernel 时间汇总：

| Kernel | 时间占比 | 总时间(ns) | 调用次数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `householder_right_gemv_kernel` | 48.1% | 97466681 | 3064 | 右 Householder 的 GEMV |
| `householder_right_update_kernel` | 33.6% | 68009107 | 3064 | 右 Householder 的 GER/update |
| `householder_left_gemv_kernel` | 11.9% | 24070953 | 1026 | 左 Householder 的 GEMV |
| `householder_left_update_kernel` | 4.8% | 9783452 | 1026 | 左 Householder 的 GER/update |
| `gather_column_kernel` | 0.6% | 1189338 | 1029 | 读取列段 |
| `zero_column_below_kernel` | 0.6% | 1122223 | 1026 | 清零列下方元素 |
| `zero_row_right_kernel` | 0.5% | 932876 | 1019 | 清零行右侧元素 |

GPU 内存传输统计：

| 方向 | 调用次数 | 总传输量(MB) |
| --- | ---: | ---: |
| Host-to-Device | 4105 | 40.010 |
| Device-to-Host | 2063 | 32.008 |

手写 kernel 版的瓶颈从 v1 的大规模数据传输转移到了自定义 kernel 计算和 `cudaMemcpyToSymbol`。其中右 Householder 的 GEMV/update 占自定义 kernel 时间最多，是后续继续优化的重点。
