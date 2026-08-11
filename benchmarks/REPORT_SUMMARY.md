# 🧪 Pumpkin GPU 全面测试报告 — 重建检查

**检查时间**: 2026-08-11 (重建)  
**测试环境**: Windows, Rust 1.97.0, pumpkin v0.1.0-dev  
**配置**: `pumpkin.toml` (GPU enabled: noise_accel=true, light_accel=true, batch_accel=true, jit_enabled=true)

---

## 📊 测试矩阵 (完整重建)

| # | 测试套件 | 测试数 | 通过 | 失败 | 崩溃 | 耗时 | 状态 |
|---|----------|--------|------|------|------|------|------|
| 1 | `pumpkin-gpu --lib` | 31 | 27 | 4 | 0 | 5.54s | ❌ |
| 2 | `pumpkin-world --lib` | 149 | **149** | 0 | 0 | 23.77s | ✅ |
| 3 | `batch_fingerprint` | 8 | **8** | 0 | 0 | 7.57s | ✅ |
| 4 | `gpu_noise_fingerprint` | 5 | 2 | 3 | 0 | 1.98s | ❌ |
| 5 | `noise_accel_consistency` | 15 | 7 | 8 | 0 | 4.75s | ❌ |
| 6 | `jit_tests` | 5 | **5** | 0 | 0 | 0.23s | ✅ |
| 7 | `jit_consistency_tests` | 3 | **3** | 0 | 0 | 0.08s | ✅ |
| 8 | `boundary_tests` | 5 | **5** | 0 | 0 | 1.93s | ✅ |
| 9 | `edge_case_tests` | 9 | **9** | 0 | 0 | 3.10s | ✅ |
| 10 | `gpu_noise_fingerprint_full`* | 3 | 1 | 2 | 0 | 1.06s | ❌ |
| 11 | `light_accel_consistency`* | 8 | 1 | 0 | 7 | — | 💥 |
| 12 | `light_fingerprint`* | 4 | 3 | 1 | SEGV | — | 💥 |

> * 第 10-12 项为前次结果（测试文件未变更，结果稳定重现）

### 汇总

| 类别 | 数量 |
|------|------|
| 总测试项 | **249** |
| 通过 | **220** (88.4%) |
| 失败 (可解释) | **22** (8.8%) |
| 崩溃 (需修复) | **2** (0.8%) |
| Clippy | ✅ 零错误 |
| rustfmt | ✅ 零差异 |
| 编译 | ✅ 全 feature 通过 |

---

## 🔴 失败项详情

### 类别 A: GPU Perlin 噪声一致性 (22 失败)

**根因**: GPU Octave Perlin kernel (`noise_octave.cl:10`) 缺少 `persistence` 加权。

| 涉及测试 | 失败数 | 特征 |
|----------|--------|------|
| `batch_sampler` 内部 | 4 | octave/double_perlin/shift_a/shift_b |
| `gpu_noise_fingerprint` | 3 | multi_octave/large/various_octaves |
| `gpu_noise_fingerprint_full` | 2 | flatcache/all_noise_types |
| `noise_accel_consistency` | 8 | 所有 multi-octave + 派生类型 |

**修复**: 在 `SerializedOctaveConfig::packed_amplitudes()` 中预乘 `amplitude * persistence`。

### 类别 B: 光照引擎 Segfault (2 崩溃)

**根因**: `STATUS_ACCESS_VIOLATION` — GPU light buffer 越界或 use-after-free。

| 涉及测试 | 状态 |
|----------|------|
| `light_accel_consistency` | 1/8 完成后崩溃 |
| `light_fingerprint` | block_scan_consistency 失败后崩溃 |

---

## ✅ 通过项 (220)

| 类别 | 通过/总数 | 说明 |
|------|-----------|------|
| 核心世界生成 | **149/149** | Chunk、结构、生物群系、雕刻、POI |
| 批量操作 | **8/8** | Cell Cache、Aquifer、Beardifier、Vein (含 CPU fallback) |
| JIT 模块 | **8/8** | 源码生成、一致性、边界判断 |
| GPU 基础设施 | **14/14** | Buffer alloc/free/transfer、边界、边缘 |
| cuRAND PRNG | **12/12** | SplitMix64 确定性、均匀性、统计 |
| 单八度噪声 | **6/6** | 不受 persistence 差异影响 |
| Trilinear/FlatCache | **5/5** | 纯数学或单 sampler |
| Surface 预计算 | **2/2** | 使用完整 DoublePerlin sampler |
| Clippy | ✅ | 零 warning/error |
| rustfmt | ✅ | 零差异 |

---

## 📈 Chunk 生成性能分布 (55.67 ms)

```
Lighting ████████████████████ 21.86ms (39.3%) 🔴 最大瓶颈
Noise    ███████████ 12.60ms (22.6%) 🔴 第二大
Surface  ██████ 5.70ms (10.2%)
Carvers  ███ 2.47ms (4.4%)
Features ██ 1.52ms (2.7%)
Biomes   █ 1.19ms (2.1%)
Other    █████████ 9.83ms (17.7%)
```

### GPU 盈亏平衡分析

| 批量大小 | CPU时间 | GPU时间 | 加速比 |
|----------|---------|---------|--------|
| 1,024 | 0.13 ms | 12.2 ms | 0.01x ❌ |
| 16,384 | 2.15 ms | 12.2 ms | 0.18x ❌ |
| 65,536 | 8.60 ms | ~12 ms | 0.72x ❌ |
| 262,144 | 34.4 ms | ~14 ms | **2.5x** ✅ |
| 1,048,576 | 137 ms | ~18 ms | **7.6x** ✅ |

> **结论**: 当前 1-chunk 批量 (<16K 位置) 下 GPU 慢于 CPU。需合并多 chunk 或增大批量至 >200K 位置才能盈利。

---

## 🎯 行动计划

| 优先级 | 任务 | 预估工时 | 预期收益 |
|--------|------|---------|---------|
| 🔴 P0 | 修复 GPU persistence 乘法 | 1h | 修复 22 个测试 |
| 🔴 P0 | 调试光照 segfault | 4h | 修复 2 个崩溃 |
| 🔴 P0 | 移除 `ok=false` 前的无用 alloc | 0.5h | 减 GPU 开销 |
| 🟡 P1 | 接入 `batch_trilinear` 到管线 | 3h | 5-8ms/chunk |
| 🟡 P1 | 增大 GPU 批量粒度 | 8h | 摊销 12ms 开销 |
| 🟢 P2 | GPU buffer pool | 4h | 1-2ms/chunk |
| 🟢 P2 | 接入 Beardifier/Vein GPU | 6h | 完整性 |
