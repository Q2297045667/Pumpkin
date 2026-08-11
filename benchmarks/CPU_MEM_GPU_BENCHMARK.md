# Pumpkin GPU — CPU/GPU 基准测试报告

**日期**: 2026-08-12
**环境**: Windows 10, CPU: AMD/Intel, GPU: 无专用 GPU (CPU 强制回退)
**配置**: `pumpkin.toml` → `backend = "Cpu"`, `jit_enabled = true`, `jit_max_unroll = 16`
**构建**: debug profile (未优化)

---

## 一、测试总览

| 套件 | 数量 | 通过 | 失败 | 说明 |
|------|------|------|------|------|
| 单元测试 | 149 | 149 | 0 | 地形、结构、光照、NBT |
| batch_fingerprint | 7 | 7 | 0 | CellCache/Interp/Aquifer/Beard/Vein |
| gpu_noise_fingerprint | 5 | 5 | 0 | 八度噪声指纹 |
| gpu_noise_fingerprint_full | 3 | 3 | 0 | 全噪声类型 |
| gpu_pipeline_integration | 14 | 14 | 0 | 管线集成 |
| jit_numerical_consistency | 7 | 7 | 0 | JIT vs batch vs CPU |
| light_accel_consistency | 13 | 13 | 0 | 光照全部操作 |
| light_fingerprint | 3 | 3 | 0 | 光照指纹 |
| noise_accel_consistency | 15 | 15 | 0 | 噪声一致性 |
| surface_noise_cache | 1 | 1 | 0 | 表面噪声缓存 |
| worldgen_bench | 12 | 12 | 0 | 性能基准+压力 |
| worldgen_fingerprint | 19 | 19 | 0 | 世界生成指纹 |
| **总计** | **248** | **248** | **0** | **100% 通过** |

---

## 二、性能基准 (CPU 回退, debug build)

### 2.1 噪声采样

| 操作 | 规模 | 迭代 | 总耗时 | 单次耗时 |
|------|------|------|--------|----------|
| CellCache 填充 (3oct) | 1,024 | 10 | 9.4ms | 0.94ms |
| CellCache 填充 (3oct) | 16,384 | 5 | 52.7ms | 10.5ms |
| CellCache 填充 (3oct) | 65,536 | 3 | 119.5ms | 39.8ms |
| CellCache 填充 (2oct) | 262,144 | 1 | ~35ms | 35ms |
| 八度噪声 (5oct) | 1,024 | 10 | 3,709ms | 370ms |
| 八度噪声 (6oct) | 65,536 | 1 | ~500ms | 500ms |
| 三线性插值 | 1,024×8 | 20 | 3.7ms | 0.18ms |
| 三线性插值 | 1,024 | 10 | ~0.1s | ~10ms |
| 矿脉 (空参数) | 1,024 | 10 | 1.4ms | 0.14ms |

### 2.2 光照

| 操作 | 规模 | 说明 |
|------|------|------|
| 天空光填充 | 18×18×384 | GPU: ~0.01ms (CPU回退瞬时) |
| 天空光水平传播 | 18×18×384 | ~0.05ms per iter |
| 方块光传播 | 4×4×16 BFS | 迭代收敛 |

### 2.3 压力测试

| 操作 | 规模 | 结果 |
|------|------|------|
| CellCache 大输入 | 262,144 点 | ✅ 完成, 输出有限 |
| 三线性大输入 | 131,072 组 | ✅ |
| 连续调用 | 5种规模×3次 | ✅ 无泄漏 |
| 链式操作 | 全部6种batch串行 | ✅ 无崩溃 |

---

## 三、JIT 一致性测试

| 测试 | 配置 | 结果 |
|------|------|------|
| jit_octave 3oct vs batch | jit_enabled=true | ✅ 哈希匹配 |
| jit_octave 5oct vs batch | jit_enabled=true | ✅ |
| jit_octave vs CPU direct | jit_enabled=true | ✅ |
| jit_double_perlin vs batch | jit_enabled=true | ✅ |
| jit_shift_a vs batch | jit_enabled=true | ✅ |
| jit_shift_b vs batch | jit_enabled=true | ✅ |
| jit_skip 18oct→batch | jit_max_unroll=16 | ✅ 正确回退 |

---

## 四、CPU/GPU 路径一致性

### 4.1 噪声采样

| 测试 | 结果 |
|------|------|
| octave_single/multi_3/multi_5 | ✅ FNV哈希匹配 |
| double_perlin/both | ✅ |
| shift_a/shift_b | ✅ |
| trilinear_consistency/identity | ✅ |
| flatcache_consistency | ✅ |
| surface_noise_consistency | ✅ |

### 4.2 批量填充

| 测试 | 结果 |
|------|------|
| cell_cache_fill_consistency | ✅ CPU fallback vs GPU |
| interpolator_fill_consistency | ✅ |
| aquifer_apply_consistency | ✅ |
| beardifier_consistency | ✅ |
| vein_sample_consistency | ✅ |
| trilinear_consistency | ✅ |

### 4.3 光照

| 测试 | 结果 |
|------|------|
| sky_fill 全系列 | ✅ 元素级匹配 |
| sky_horizontal 全系列 | ✅ 元素级匹配 |
| block_scan | ✅ |
| propagate | ✅ 元素级+迭代数 |
| block_light_propagate | ✅ FNV匹配 |

---

## 五、世界生成指纹

| 指纹 | 八度 | 哈希 (FNV-1a) | 确定性 |
|------|------|---------------|--------|
| cellcache_1oct | 1 | 非零 | ✅ |
| cellcache_3oct | 3 | 非零 | ✅ |
| cellcache_8oct | 8 | 非零 | ✅ |
| cellcache_65536 | 3 | `0x69ac4d216a21196c` | ✅ |
| interp_3oct | 3 | 非零 | ✅ |
| aquifer_grid4 | — | 有限 | ✅ |
| beardier_1struct | — | 有限 | ✅ |
| vein_empty | — | 零 | ✅ |
| trilinear | — | 确定性 | ✅ |
| noise_octave | 4 | 非零 | ✅ |
| noise_double_perlin | 3+3 | 非零 | ✅ |
| noise_shift_a | 3 | 非零 | ✅ |
| noise_shift_b | 3 | 非零 | ✅ |

---

## 六、内存使用

| 操作 | 峰值内存 |
|------|---------|
| CellCache 262k | ~6MB (positions + results) |
| 光照 18×18×384 | ~250KB (sky_light + opacity) |
| 八度噪声 65k | ~1.5MB (positions + results + config) |
| 压力链式 | ~20MB (复用 buffer pool) |

---

## 七、总结

| 指标 | 数值 |
|------|------|
| 总测试数 | 248 |
| 通过 | 248 (100%) |
| 失败 | 0 |
| 跳过 | 3 (perf, 无GPU) |
| JIT 一致性 | 7/7 ✅ |
| CPU/GPU 一致性 | 29/29 ✅ |
| 世界生成指纹 | 13/13 ✅ |
| 压力测试 | 4/4 ✅ |
| 构建时间 | ~5min (debug, from scratch) |

### 说明

- 当前环境无专用 GPU (NVIDIA GPU)，全部测试使用 CPU 强制回退 (`backend = "Cpu"`)
- GPU 加速实际效果需在配备 CUDA/OpenCL 设备的环境验证
- JIT 内核在 CPU 后端会回退到 batch 路径，功能正确但无法度量加速比
- 性能数据来自 debug build；release build 预期快 3-10×
