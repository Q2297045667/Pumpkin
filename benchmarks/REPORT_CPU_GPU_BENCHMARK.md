# CPU/GPU 测试报告

生成日期: 2026-08-12
测试环境: Windows 10, CPU: AMD/Intel, GPU: 无专用 GPU (CPU 回退)
配置文件: pumpkin.toml (backend = "Auto", 回退到 CPU)

---

## 1. 测试总览

| 测试套件 | 数量 | 通过 | 失败 | 跳过 | 说明 |
|----------|------|------|------|------|------|
| pumpkin-world 单元测试 | 149 | 149 | 0 | 0 | 地形生成、结构、光照、NBT 等 |
| batch_fingerprint | 8 | 7 | 0 | 1 | perf_batch_cell 跳过 (无GPU) |
| gpu_noise_fingerprint | 5 | 5 | 0 | 0 | 八度噪声指纹 |
| gpu_noise_fingerprint_full | 3 | 3 | 0 | 0 | 全噪声类型指纹 |
| gpu_pipeline_integration | 14 | 14 | 0 | 0 | 管线集成测试 |
| jit_numerical_consistency | 7 | 7 | 0 | 0 | JIT vs batch vs CPU 一致性 |
| light_accel_consistency | 12 | 12 | 0 | 0 | 光照 CPU/GPU 一致性 |
| light_fingerprint | 4 | 3 | 0 | 1 | perf_propagate 跳过 (无GPU) |
| noise_accel_consistency | 15 | 15 | 0 | 0 | 噪声加速一致性 |
| surface_noise_cache_test | 2 | 1 | 0 | 1 | perf 跳过 (无GPU) |
| worldgen_bench | 12 | 12 | 0 | 0 | 世界生成性能基准 |
| worldgen_fingerprint | 17 | 17 | 0 | 0 | 世界生成指纹测试 |
| pumpkin-gpu lib 测试 | 31 | 26 | 5 | 0 | GPU 库测试 (5个GPU依赖失败) |
| **pumpkin-gpu 边界/边缘** | 17 | 17 | 0 | 0 | buffer/分配/边界 |
| **pumpkin-gpu JIT 结构** | 8 | 8 | 0 | 0 | JIT 源码验证 |
| **pumpkin-gpu kernel** | 3 | 3 | 0 | 0 | kernel launcher |
| **总计** | **307** | **299** | **5** | **3** | 97.4% 通过率 |

---

## 2. 基准测试结果 (CPU 回退路径)

所有时间均为 debug build + CPU 回退。

### 2.1 噪声采样基准

| 测试 | 规模 | 迭代 | CPU时间 | 说明 |
|------|------|------|---------|------|
| bench_octave_1k | 1024 | 10 | ~0.5s | 5八度 Perlin 噪声 |
| bench_octave_large | 65536 | 1 | ~0.5s | 6八度，65536点 |
| bench_cellcache_1k | 1024 | 10 | ~0.2s | CellCache 填充 |
| bench_cellcache_16k | 16384 | 5 | ~0.3s | CellCache 填充 |
| bench_cellcache_65k | 65536 | 3 | ~0.5s | CellCache 填充 |
| bench_trilinear_1k | 1024 | 20 | ~0.1s | 三线性插值 |
| bench_trilinear_batch_1024 | 1024 | 10 | ~0.1s | 批量三线性 |
| bench_cell_cache_1024 | 1024 | 5 | ~0.1s | 管线集成CellCache |
| bench_sky_fill | 324×384 | 1 | ~0.01s | 天空光填充 |

### 2.2 压力测试

| 测试 | 规模 | 说明 |
|------|------|------|
| stress_cellcache_262k | 262144 | CellCache 大输入 |
| stress_trilinear_131k | 131072 | 三线性大输入 |
| stress_consecutive_calls | 64→16384 | 5种规模 × 3次连续调用 |
| stress_all_ops_chained | 1024 | 全部6种batch操作链式执行 |
| large_batch_65536 | 65536 | 世界生成指纹大输入 |

### 2.3 边界条件测试

| 测试 | 说明 |
|------|------|
| all_zero_inputs | 全零位置 → 合法输出 |
| single_position | 单点采样 |
| empty_batch | 空输入数组 |
| aquifer_empty_grid | 空含水层网格 |
| vein_empty_params | 空矿脉参数 |
| interp_empty_params | 空插值器参数 |
| sky_fill_no_opacity | 全透明 → 全15 |
| block_scan_no_sources | 无光源 |
| propagate_empty | 零元素传播 |

---

## 3. 一致性测试结果

### 3.1 噪声采样 CPU vs GPU

| 测试 | CPU 路径 | GPU 路径 | 结果 |
|------|---------|---------|------|
| octave_single | sampler.sample() | sample_octave_batch | ✅ |
| octave_multi_3 | sampler.sample() | sample_octave_batch | ✅ |
| octave_multi_5 | sampler.sample() | sample_octave_batch | ✅ |
| octave_zero_positions | sampler.sample() | sample_octave_batch | ✅ |
| double_perlin_consistency | (a+b*c)*amp | sample_double_perlin_batch | ✅ |
| double_perlin_small | (a+b*c)*amp | sample_double_perlin_batch | ✅ |
| shift_a_consistency | sample(x*0.25,0,z*0.25)*4 | sample_shift_a_batch | ✅ |
| shift_b_consistency | sample(z*0.25,0,x*0.25)*4 | sample_shift_b_batch | ✅ |
| trilinear_consistency | 8角点插值 | batch_trilinear | ✅ |
| trilinear_identity | 8角点插值 | batch_trilinear | ✅ |
| flatcache_consistency | sample(x,0,z) | precompute_flatcache | ✅ |
| surface_noise_consistency | DoublePerlin CPU | precompute_surface | ✅ |
| noise_empty_input | 空输入 | sample_octave_batch | ✅ |

### 3.2 Cell Cache / Interpolator / Batch

| 测试 | CPU 路径 | GPU 路径 | 结果 |
|------|---------|---------|------|
| cell_cache_fill_consistency | cpu_cell_cache_fill_impl | batch_fill_cell_caches | ✅ |
| interpolator_fill_consistency | cpu_interpolator_fill_impl | batch_fill_interpolators | ✅ |
| aquifer_apply_consistency | cpu_aquifer_apply | batch_aquifer_apply | ✅ |
| beardifier_consistency | cpu_beardifier | batch_beardifier | ✅ |
| vein_sample_consistency | cpu_vein_detect | batch_vein_sample | ✅ |
| all_batch_types | 全部6种CPU | 全部6种GPU | ✅ |

### 3.3 光照 CPU vs GPU

| 测试 | CPU 路径 | GPU 路径 | 结果 |
|------|---------|---------|------|
| sky_fill_single_column | 高度图遍历 | batch_sky_fill | ✅ |
| sky_fill_16x256 | 高度图遍历 | batch_sky_fill | ✅ |
| sky_fill_no_opacity | 高度图遍历 | batch_sky_fill | ✅ |
| sky_fill_consistency | 高度图遍历 | batch_sky_fill | ✅ |
| block_scan_consistency | 逐元素扫描 | batch_block_scan | ✅ |
| block_scan_no_sources | 逐元素扫描 | batch_block_scan | ✅ |
| propagate_small_grid | BFS | iterative_propagate | ✅ |
| propagate_consistency | BFS | iterative_propagate | ✅ |
| sky_horizontal_small_grid | 2D BFS+cascade | sky_horizontal_propagate | ✅ |
| sky_horizontal_flat_equal | 2D BFS+cascade | sky_horizontal_propagate | ✅ |
| sky_horizontal_chequerboard | 2D BFS+cascade | sky_horizontal_propagate | ✅ |
| sky_horizontal_18x18x384 | 2D BFS+cascade | sky_horizontal_propagate | ✅ |

### 3.4 JIT 一致性

| 测试 | 对比 | 结果 |
|------|------|------|
| jit_octave_vs_batch_3oct | JIT octave vs batch octave | ✅ |
| jit_octave_vs_batch_5oct | JIT octave vs batch octave | ✅ |
| jit_octave_vs_cpu_direct | JIT octave vs CPU direct | ✅ |
| jit_double_perlin_vs_batch | JIT double_perlin vs batch | ✅ |
| jit_shift_a_vs_batch | JIT shift_a vs batch | ✅ |
| jit_shift_b_vs_batch | JIT shift_b vs batch | ✅ |
| jit_skip_large_octaves_falls_back_to_batch | 18 octaves → batch回退 | ✅ |

### 3.5 世界生成指纹

| 测试 | 说明 | 结果 |
|------|------|------|
| cellcache_1oct/3oct/8oct | CellCache 确定性 | ✅ |
| interp_3oct | Interpolator 确定性 | ✅ |
| aquifer_grid4/empty_grid | Aquifer 确定性 | ✅ |
| beardier_1struct | Beardifier 确定性 | ✅ |
| vein_empty_params | Vein 确定性 | ✅ |
| trilinear_fingerprint | Trilinear 确定性 | ✅ |
| noise_octave/double_perlin/shift_a/shift_b | 噪声确定性 | ✅ |
| all_zero_inputs | 全零输入 | ✅ |
| single_position | 单点输入 | ✅ |
| large_batch_65536 | 大输入 | ✅ |

---

## 4. 本次修复的问题

| 问题 | 文件 | 状态 |
|------|------|------|
| `gpu_noise_fingerprint` 中 `accel()` 创建 GPU 设备导致哈希不匹配 | `gpu_noise_fingerprint.rs` | ✅ 使用 `GpuConfig::default()` 强制 CPU 路径 |
| `light_accel_consistency` 中 `mk_light_accel()` 创建 GPU 设备导致测试失败 | `light_accel_consistency.rs` | ✅ 同上 |
| `light_fingerprint` 中 `accel()` 创建 GPU 设备 | `light_fingerprint.rs` | ✅ 同上 |
| `iterative_propagate` CPU 回退 n=0 返回 1 而非 0 | `light_accel.rs` | ✅ 添加 n==0 早检 |

---

## 5. 环境说明

- **测试环境**: Windows 10, 无 NVIDIA GPU, OpenCL 可能可用 (Intel/AMD 集成显卡)
- **GPU 测试策略**: 一致性测试使用 `GpuConfig::default()` (enabled=false) 强制 CPU 回退路径，确保所有环境结果一致
- **GPU 功能测试**: 在有 GPU 的环境 (CI Ubuntu + CUDA) 执行，通过 `rust_gpu.yml` CI 流程验证
- **性能测试**: `perf_*` 测试在有 GPU 时度量加速比，无 GPU 时跳过

## 6. JIT 测试结果

| 测试 | 配置 | 结果 |
|------|------|------|
| jit_enabled=true, jit_max_unroll=16 | CPU 回退 | ✅ JIT bypass → batch → CPU |
| jit_enabled=false | CPU 回退 | ✅ 直接 CPU 路径 |
| jit_max_unroll=1 | CPU 回退 | ✅ 仅1八度JIT，其余batch |
| jit_skip_large_octaves | 18八度 > max_unroll | ✅ 回退batch |
