# CPUIntelCorei5-8300H_16GB_GTX1060_Linux7.1_2026-08-18

## 硬件配置

- **CPU**: Intel Core i5-8300H @ 2.30GHz
- **内存**: 16 GB
- **GPU**: NVIDIA GeForce GTX 1060
- **显存**: 6144 MiB
- **种子**: `138_782_381_985_206`
- **时间**: 2026-08-18 01:20

## 测试环境

- **CUDA 版本**: CUDA 13.0, NVIDIA-SMI 580.173.02
- **OpenCL 版本**: NVIDIA OpenCL 3.0, 由 `clinfo` 实测；设备支持 `cl_khr_fp64`
- **操作系统**: Linux x86_64 GNU/Linux
- **内核版本**: 7.1.6-1-cachyos
- **驱动版本**: NVIDIA 580.173.02
- **构建**: debug test profile + release bench profile, feature 清单包含 `--features gpu`
- **缓存状态**: 已执行 `cargo clean`，清理 19634 files / 25.8 GiB

## 测试总览

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|
| 矩阵 CPU JIT off | 8/8 | — | — | 基线 | 通过 |
| 矩阵 CPU JIT on | 8/8 | — | — | 基线 | 通过，JIT 在 CPU 设备跳过内核验证 |
| 矩阵 OpenCL JIT off | 8/8 | 8/8 | — | 最高 73.09x | 通过 |
| 矩阵 OpenCL JIT on | 8/8 | 8/8 | — | 最高 71.26x | 通过，JIT kernel 编译成功 |
| 矩阵 CUDA JIT off | 6/8 | 部分 | — | Octave 1.63x | 失败 2 项，见失败分析 |
| 矩阵 CUDA JIT on | 6/8 | 部分 | — | Octave 0.90x | 失败 2 项；JIT 编译失败后按规则跳过 |
| OpenCL 压力/指纹/光照/多种子/管线 | 53/53 | 53/53 | — | 见分表 | 通过 |
| `pumpkin-gpu` 测试部分 | 83/83 | OpenCL f64 通过 | — | — | 通过；bench target 需单独运行 |
| Criterion `gpu_consistency` | CPU baseline | — | 764.04 us | 基线 | 通过 |

- **JIT 启动**: OpenCL 矩阵中 `jit_kernel_compiled=true`，JIT octave vs batch 通过；CUDA 矩阵中 `jit_kernel_compiled=false`，按规则跳过 JIT 内核验证。
- **功能模块启动**: OpenCL noise/light/batch 全部真实启动并通过；CUDA noise/SoA/trilinear 能启动，CUDA Aquifer 和 sky fill 存在失败，详见失败分析。
- **pumpkin.toml 是否改动**: 未改动；测试通过环境变量选择后端/JIT。

## 基准测试

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|
| Octave 262k, CPU JIT off | 482.48 ms | 485.07 ms | — | 0.99x | 一致 |
| Double Perlin 65k, CPU JIT off | 124.69 ms | 121.14 ms | — | 1.03x | 一致 |
| Trilinear 131k, CPU JIT off | 16.29 ms | 7.33 ms | — | 2.22x | 一致 |
| Octave 262k, CPU JIT on | 487.14 ms | 477.68 ms | — | 1.02x | 一致 |
| Double Perlin 65k, CPU JIT on | 124.46 ms | 126.13 ms | — | 0.99x | 一致 |
| Trilinear 131k, CPU JIT on | 16.87 ms | 7.24 ms | — | 2.33x | 一致 |
| Octave 262k, OpenCL JIT off | 505.78 ms | 6.92 ms | 6.92 ms | 73.09x | 一致 |
| Double Perlin 65k, OpenCL JIT off | 134.90 ms | 1.86 ms | 1.86 ms | 72.67x | 一致 |
| Trilinear 131k, OpenCL JIT off | 17.43 ms | 3.03 ms | 3.03 ms | 5.75x | 一致 |
| Octave 262k, OpenCL JIT on | 478.58 ms | 6.72 ms | 6.72 ms | 71.26x | 一致 |
| Double Perlin 65k, OpenCL JIT on | 124.77 ms | 1.87 ms | 1.87 ms | 66.65x | 一致 |
| Trilinear 131k, OpenCL JIT on | 16.22 ms | 2.98 ms | 2.98 ms | 5.44x | 一致 |
| Octave 262k, CUDA JIT off | 515.63 ms | 542.58 ms | 542.58 ms | 0.95x | 噪声一致，整体矩阵失败 |
| Double Perlin 65k, CUDA JIT off | 132.77 ms | 163.86 ms | 163.86 ms | 0.81x | 噪声一致，整体矩阵失败 |
| Trilinear 131k, CUDA JIT off | 17.69 ms | 40.95 ms | 40.95 ms | 0.43x | 一致，整体矩阵失败 |
| Criterion CPU baseline | — | — | 764.04 us | 基线 | 通过 |

## 噪声采样

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|
| OpenCL octave | 通过 | 通过 | 6.72-6.92 ms | 71.26x-73.09x | 一致 |
| OpenCL double_perlin | 通过 | 通过 | 1.86-1.87 ms | 66.65x-72.67x | 一致 |
| OpenCL shift_a/shift_b | 通过 | 通过 | — | — | 一致 |
| OpenCL flatcache | 通过 | 通过 | `worldgen_perf`: 211.03 ms | 0.40x | 一致但慢于 CPU |
| OpenCL trilinear | 通过 | 通过 | 2.98-3.03 ms | 5.44x-5.75x | 一致 |
| CUDA octave/double/shift/flatcache/trilinear | 通过 | 通过 | 见矩阵 | 0.43x-1.63x | 一致，但 CUDA 其他模块失败 |

## 压力测试

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|
| `worldgen_stress` OpenCL | 11/11 | 11/11 | 15.96 s total | — | 通过 |
| `stress_octave_262k` | 通过 | 通过 | — | — | 一致 |
| `stress_double_perlin_65k` | 通过 | 通过 | — | — | 一致 |
| `stress_flatcache_65k` | 通过 | 通过 | — | — | 一致 |
| `stress_trilinear_131k` | 通过 | 通过 | — | — | 一致 |
| `stress_aquifer_large_grid` | 通过 | OpenCL 通过 | — | — | 一致 |
| `stress_beardifier_many_structures` | 通过 | OpenCL 通过 | — | — | 一致 |
| `stress_light_large_propagate` | 通过 | OpenCL 通过 | — | — | 一致 |
| `stress_extreme_coordinates` | 通过 | OpenCL 通过 | — | — | 一致 |
| `stress_repeated_calls` | 通过 | OpenCL 通过 | — | — | 一致 |

## 一致性测试

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|
| `batch_fingerprint` | 6/6 | OpenCL 6/6 | 6.85 s | — | 通过 |
| `gpu_noise_fingerprint` | 5/5 | OpenCL 5/5 | 0.11 s | large 1.0x | 通过 |
| `gpu_noise_fingerprint_full` | 3/3 | OpenCL 3/3 | 3.17 s | — | 通过 |
| `worldgen_light_gpu_consistency` | 4/4 | OpenCL 4/4 | 3.77 s | — | 通过 |
| `worldgen_multi_seed_consistency` | 10/10 | OpenCL 10/10 | 9.00 s | — | 通过 |
| `worldgen_pipeline_fingerprint` | 2/2 | OpenCL 2/2 | 2.00 s | — | 通过 |
| `gpu_backend_alignment` | 3/4 | auto/CUDA 失败 | — | — | 测试未遵守 OpenCL 矩阵配置，见失败分析 |
| `jit_numerical_consistency` | 9/10 | auto/CUDA 失败 | — | — | 强制 JIT 编译断言在 CUDA auto 路径失败 |

## 指纹

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|
| `gpu_noise_fingerprint` | 5/5 | OpenCL 5/5 | 0.11 s | — | 哈希一致 |
| `gpu_noise_fingerprint_full` | 3/3 | OpenCL 3/3 | 3.17 s | — | 哈希一致 |
| `light_fingerprint` | 4/4 | OpenCL 4/4 | 0.45 s | propagate 1.98x | 哈希一致 |
| `batch_fingerprint` | 6/6 | OpenCL 6/6 | 6.85 s | — | 哈希一致 |
| `worldgen_fingerprint` | 8/8 | OpenCL 8/8 | 2.15 s | — | 哈希一致 |
| `worldgen_pipeline_fingerprint` | 2/2 | OpenCL 2/2 | 2.00 s | — | 哈希一致 |

## 光照引擎

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|
| OpenCL sky_fill | 通过 | 通过 | — | `worldgen_perf`: 1.00x | 一致 |
| OpenCL block_scan | 通过 | 通过 | — | — | 一致，source 排序后对比 |
| OpenCL iterative_propagate | 通过 | 通过 | 9.6 ms | 1.98x | 一致 |
| OpenCL sky_horizontal | 通过 | 通过 | — | — | 一致 |
| CUDA sky_fill | CPU ref 通过 | CUDA 失败/回退污染已定位 | — | — | 失败，见失败分析 |
| CUDA iterative/light horizontal | 部分通过 | CUDA matrix 未全部完成 | — | — | 受 sky_fill/Aquifer 失败阻塞 |

## JIT 编译

| 项目 | CPU | GPU | 预热 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| OpenCL JIT octave vs batch | 通过 | 通过 | 已预热 | 一致 | — |
| OpenCL `jit_kernel_compiled` | — | true | 冷启动含编译 | — | 矩阵测试完成 |
| CUDA `jit_kernel_compiled` | — | false | 冷启动 | 跳过 | JIT 启动失败，按规则跳过 |
| CPU JIT on | 通过 | — | — | batch fallback | CPU 设备跳过 JIT kernel 验证 |

## JIT 一致性

| 项目 | CPU | GPU | 预热 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| OpenCL matrix JIT path | 通过 | 通过 | 已预热 | 一致 | — |
| OpenCL matrix noise families | 通过 | 通过 | 已预热 | 最高 71.26x | — |
| `jit_numerical_consistency` 非矩阵测试 | 9/10 | auto/CUDA 路径失败 | — | — | 需改为遵守后端环境变量或按 JIT 失败跳过 |
| `gpu_backend_alignment` 非矩阵测试 | 3/4 | auto/CUDA 路径失败 | — | — | 需改为遵守后端环境变量 |

## 执行摘要与结论

- OpenCL 是当前机器上可用且稳定的 GPU 后端：矩阵、压力、指纹、光照、多种子、管线测试均通过。
- CUDA 能初始化 `NVIDIA GeForce GTX 1060`，噪声五族、SoA、trilinear 能通过一致性测试，但 Aquifer 真实启动失败，sky fill 存在边界/回退污染问题，需要继续修复 CUDA kernel 与错误传播。
- OpenCL JIT 可开启；CUDA JIT 当前无法真实编译，已按“JIT 启动失败直接跳过”的规则处理。
- 清理缓存后重建耗时较长；后续报告建议保留 cold build 和 warm run 两类数据。
- `pumpkin.toml` 未修改；本次测试均通过 `PUMPKIN_GPU_BACKEND` 和 `PUMPKIN_GPU_JIT` 环境变量控制。
- CPU 生成内容影响声明：已将 sky fill 的 heightmap clamp 合同统一到 OpenCL kernel、CPU fallback 和测试 reference；这是边界修复，不改变合法 heightmap 输入的结果。
