# GPU 模块基准测试报告

> 命名规范: CPU+内存+GPU+系统+时间(见 `benchmarks/README.md`)
> 本报告为清理缓存后的复验运行(2026-08-13 第三次执行),数据以本次实测为准。

## 硬件配置

- **CPU**: Intel(R) Xeon(R) W-3345 @ 3.00GHz
- **内存**: 32 GB(34,358,743,040 字节)
- **GPU**: NVIDIA Tesla T10(Turing)
- **显存**: 16,384 MiB
- **种子**: `138_782_381_985_206`(全部测试固定种子)
- **时间**: 2026-08-13

## 测试环境

- **CUDA 版本**: 13.3(`nvidia-smi` 报告 CUDA UMD Version: 13.3)
- **OpenCL 版本**: OpenCL 3.0(NVIDIA 驱动 610.47 提供;NVIDIA 465+ 驱动为 OpenCL 3.0,未在运行时单独查询)
- **操作系统**: Windows 10 专业工作站版 64 位
- **内核版本**: 10.0.19045(build 19045)
- **驱动版本**: 610.47(KMD 610.47)
- **构建**: debug profile,`--features gpu`
- **缓存状态**: 执行前 `cargo clean -p pumpkin-world -p pumpkin-gpu -p pumpkin-config`(移除 2169 文件 / 3.0GiB),全部测试二进制重建后运行

---

## 测试总览 CPU vs GPU

| 项目 | CPU | GPU | 结果 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| 六组合全路径矩阵(cpu/cuda/opencl × JIT 开/关) | ✅ | ✅ | **48/48 通过** | — | — |
| `pumpkin-gpu` 全套(38 单元 + 40 集成) | ✅ | ✅ | **78/78 通过** | — | — |
| `pumpkin-world` 单元(纯 CPU 世界生成) | ✅ | — | **152/152 通过** | — | — |
| `worldgen_perf` 基准 6 组合 | ✅ | ✅ | **48/48 通过** | 见基准表 | 见基准表 |
| 18 个一致性/指纹/压力/光照/JIT/特性集成套件 | ✅ | ✅ | **120/120 通过** | — | — |
| JIT 启动 | — | ✅ CUDA/OpenCL 全部正常 | **无跳过** | — | — |
| 功能模块启动 | ✅ | ✅ | **无启动失败** | — | — |
| **合计** | — | — | **446/446 通过,0 失败** | — | — |

---

## 基准测试

> 稳态数据(预热后,排除首次内核编译;来源 `matrix_perf`,FNV-1a 哈希逐位一致性断言全部通过)

| 项目 | CPU | GPU | 结果 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| 八度噪声 262k | 301.9ms | CUDA 5.61ms / OpenCL 4.38ms | ✅ 一致 | **53.8x / 69.3x** | 5.6 / 4.4ms |
| 双 Perlin 65k | 71.6ms | CUDA 1.72ms / OpenCL 1.88ms | ✅ 一致 | **41.5x / 41.8x** | 1.7 / 1.9ms |
| 三线性插值 131k | 7.5ms | CUDA 2.58ms / OpenCL 2.90ms | ✅ 一致 | 2.9x / 3.1x | 2.6 / 2.9ms |
| 八度噪声 262k(CPU 后端批处理) | 278.6ms | —(CPU 路径 275.0ms) | ✅ 一致 | 1.01x | 275.0ms |

---

## 噪声采样

> 首次调用延迟(全新进程,含内核编译;来源 `worldgen_perf`)

| 项目 | CPU | GPU | 结果 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| 八度噪声 262k(6 八度) | 306.4ms | CUDA 4.6ms / OpenCL 4.4ms | ✅ 逐位一致 | 67.0x / 63.3x | 4.6 / 4.4ms |
| 双 Perlin 65k | 83.7ms | CUDA 1.6ms / OpenCL 1.9ms | ✅ 逐位一致 | 51.0x / 38.1x | 1.6 / 1.9ms |
| flatcache 65k | 52.1ms | CUDA 1.3ms / OpenCL 1.4ms | ✅ 逐位一致 | 41.7x / 32.6x | 1.3 / 1.4ms |
| surface 预计算 256 | 0.30ms | CUDA 1.1ms / OpenCL 1.2ms | ✅ 一致 | 0.3x(小负载 GPU 无收益) | 1.1ms |
| aquifer 16k | 487.7ms | CUDA 628.1ms† / OpenCL 233.2ms† | ✅ 一致 | — | 见注 |
| beardifier 16k | 0.19ms‡ | CUDA 569.1ms† / OpenCL 190.5ms† | ✅ 一致 | — | 见注 |

† 含首次 NVRTC(~0.5s)/OpenCL(~0.2s)内核编译,稳态为毫秒级(预热后,见矩阵测试模式);
‡ CPU 0.19ms 因 16k 随机点大多在结构包围盒外(早退);GPU 首调被编译主导,非 GPU 慢。

---

## 压力测试

| 项目 | CPU | GPU | 结果 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| stress_octave_262k(26 万点) | ✅ | ✅ | 通过,输出有限 | — | — |
| stress_double_perlin_65k | ✅ | ✅ | 通过 | — | — |
| stress_flatcache_65k | ✅ | ✅ | 通过 | — | — |
| stress_trilinear_131k | ✅ | ✅ | 通过 | — | — |
| stress_aquifer_large_grid(大网格) | ✅ | ✅ | 通过 | — | — |
| stress_beardifier_many_structures(多结构) | ✅ | ✅ | 通过 | — | — |
| stress_light_large_propagate(18×18×384) | ✅ | ✅ | 通过 | — | — |
| stress_extreme_coordinates(极端坐标) | ✅ | ✅ | 通过 | — | — |
| stress_edge_sizes(边界尺寸) | ✅ | ✅ | 通过 | — | — |
| stress_many_cell_cache_specs | ✅ | ✅ | 通过 | — | — |
| stress_repeated_calls(重复调用无泄漏) | ✅ | ✅ | 通过 | — | — |

---

## 一致性测试

> 全部为 CPU vs GPU 哈希/逐位断言(矩阵 6 组合 + 专项套件)

| 项目 | CPU | GPU | 结果 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| 八度噪声(全部家族) | ✅ | ✅ CUDA/OpenCL | 逐位一致 | — | — |
| 双 Perlin / shift_a / shift_b | ✅ | ✅ | 逐位一致 | — | — |
| flatcache 预计算 | ✅ | ✅ | 逐位一致 | — | — |
| 三线性插值 | ✅ | ✅ | 逐位一致 | — | — |
| aquifer(标准/tiled/水分支) | ✅ | ✅ | 逐位一致 | — | — |
| beardifier(vanilla 等价) | ✅ | ✅ | 逐位一致 | — | — |
| cell_cache vanilla 路径 | ✅ | ✅ | 逐位一致 | — | — |
| SoA 布局变体 | ✅ | ✅ | 逐位一致 | — | — |
| 多种子一致性 | ✅ | ✅ | 10/10 通过 | — | — |
| 管线指纹稳定性 | ✅ | ✅ | 2/2 通过 | — | — |

---

## 指纹

| 项目 | CPU | GPU | 结果 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| `gpu_noise_fingerprint`(5) | ✅ | ✅ | 5/5 通过 | — | — |
| `gpu_noise_fingerprint_full`(3) | ✅ | ✅ | 3/3 通过 | — | — |
| `light_fingerprint`(4) | ✅ | ✅ | 4/4 通过 | — | — |
| `batch_fingerprint`(6) | ✅ | ✅ | 6/6 通过 | — | — |
| `worldgen_fingerprint`(8) | ✅ | ✅ | 8/8 通过 | — | — |
| `worldgen_pipeline_fingerprint`(2) | ✅ | ✅ | 2/2 通过 | — | — |

---

## 光照引擎

| 项目 | CPU | GPU | 结果 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| sky_light_fill(98k u8) | ✅ | ✅ | 一致 | 1.0x(负载太小) | 0.8ms |
| block_light_scan | ✅ | ✅ | 值一致(源索引顺序不同,不影响收敛) | — | — |
| iterative_propagate(含 persistent) | ✅ | ✅ | 与 CPU 收敛结果一致 | — | — |
| sky_light_horizontal(收敛检查步长=4) | ✅ | ✅ | 不动点一致 | 同步开销降为 1/4 | — |
| persistent kernel(cooperative launch) | ✅ | ✅ T10 实测 | `light_persistent_consistency` 1/1 | — | — |

---

## JIT 编译

| 项目 | CPU | GPU | 预热 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| CUDA JIT 特化 octave 262k | 295.4ms | 4.10ms | 已预热 | **72.0x**(比 batch 5.61ms 快 27%) | 4.1ms |
| CUDA JIT 特化 double 65k | 71.4ms | 1.59ms | 已预热 | 44.8x(比 batch 1.72ms 快 8%) | 1.6ms |
| OpenCL JIT 特化 octave 262k | 282.5ms | 4.48ms | 已预热 | 63.1x(≈ batch 4.38ms) | 4.5ms |
| OpenCL JIT 特化 double 65k | 71.9ms | 1.74ms | 已预热 | 41.2x(比 batch 1.88ms 快 8%) | 1.7ms |
| CUDA JIT 首次编译 | — | — | 冷启动 | — | ~15-20ms(octave/flatcache) |
| OpenCL JIT 首次编译 | — | — | 冷启动 | — | ~3-6ms 特化开销(double/flatcache 首调 3.2-4.0ms) |

**结论**: 本次运行 CUDA 开 JIT 稳态 +8~27%;OpenCL 开 JIT 与 batch 打平(±8%)。注意历史运行中 OpenCL JIT double_perlin 曾出现 29-47% 劣化(2.4-2.7ms vs 1.8ms),**OpenCL JIT 性能存在运行间方差**,建议 OpenCL 侧保持默认关闭、按部署环境实测后决定。

---

## JIT 一致性

| 项目 | CPU | GPU | 预热 | 性能 | 延迟 |
|------|-----|-----|------|------|------|
| `jit_numerical_consistency`(10) | ✅ | ✅ | — | 10/10 通过 | — |
| `gpu_backend_alignment`(4,含 JIT parity) | ✅ | ✅ | — | 4/4 通过 | — |
| `matrix_jit_path`(6 组合) | ✅ | ✅ | — | 6/6 通过(CPU 后端跳过 JIT 验证,回退 batch) | — |
| JIT 五族逐位一致(octave/double/shift_a/shift_b/flatcache) | ✅ | ✅ CUDA/OpenCL | — | 哈希断言通过 | — |
| JIT 名称碰撞回归(同八度不同种子) | ✅ | ✅ | — | 通过(指纹命名) | — |

---

## 执行摘要与结论

| 结论项 | 结果 |
|--------|------|
| 测试总通过率 | ✅ **446/446,0 失败**(矩阵 48 + 基准 48 + 集成套件 120 + gpu 78 + world 单元 152) |
| JIT 启动失败 | ✅ **无**——CUDA/OpenCL 全部正常,无需跳过 |
| 功能模块启动失败 | ✅ **无**——无需失败分析 md |
| `pumpkin.toml` 修改 | 无需修改(模板默认值已正确;测试经环境变量选择后端/JIT) |
| CPU 生成内容影响 | 无影响——纯 CPU 单元 152/152、全部一致性断言逐位通过 |

**遗留建议**(不阻塞):
1. 生产启动预热:初始化后以 64 点小批量触发全部内核编译,避免首块区块生成卡顿(NVRTC ~0.5s / OpenCL ~0.2s);
2. CUDA 开 JIT 收益稳定(+8~27%),OpenCL JIT 存在运行间方差,按部署环境实测决定(保守起见保持默认关);
3. 小负载(light_sky_fill/surface/beardifier 早退)GPU 无收益,可设最小批量阈值走 CPU 直通;
4. aquifer 4-NN 排序网络优化(涉及平局语义逐位审计,低优先级)。
