# 📦 批量操作 GPU 加速测试报告 — CPU vs GPU

**日期**: 2026-08-11  
**测试文件**: `crates/pumpkin-world/tests/batch_fingerprint.rs`  
**测试状态**: **8/8 全部通过** ✅

---

## 一、测试结果

| 测试名 | 状态 | 说明 |
|--------|------|------|
| `cell_cache_fill_consistency` | ✅ | Cell Cache GPU 填充与 CPU 零填充哈希一致 |
| `interpolator_fill_consistency` | ✅ | 插值器 GPU 填充一致性 |
| `aquifer_apply_consistency` | ✅ | 含水层 GPU 判定与 CPU 参考一致 |
| `beardifier_consistency` | ✅ | Beardifier GPU 与 CPU 参考一致 |
| `vein_sample_consistency` | ✅ | 矿脉 GPU 采样与 CPU 全零一致 |
| `all_batch_types` | ✅ | 全类型的综合一致性 |
| `empty_batch` | ✅ | 空输入边界测试 |
| `perf_batch_cell` | ✅ | 性能测试 (功能正确性验证) |

---

## 二、各功能状态详解

### 2.1 Cell Cache 填充

| 维度 | 详情 |
|------|------|
| **GPU kernel** | `cell_cache_fill_f64` — 已完整实现并编译 |
| **GPU launch** | `GpuCellBatchSampler::batch_fill_cell_caches` — 完整参数准备 ✅ |
| **管线集成** | `sample_block_state()` 传入 `CellFillParams` — ⚠️ 使用空配置 |
| **CPU fallback** | `cpu_cell_cache_fill_impl` — 完整的 Perlin 噪声求值 |

**当前状态**: GPU kernel 和 launch code 已就绪，但管线中传入了空 `perlin_configs`，导致 GPU 路径检测到空配置后回退 CPU。真正的 DAG 上下文驱动的 `CellFillParams` 填充尚未完成。

### 2.2 插值器填充

| 维度 | 详情 |
|------|------|
| **GPU kernel** | `interpolator_fill_f64` — 完整实现 |
| **GPU launch** | `GpuCellBatchSampler::batch_fill_interpolators` — 完整参数 ✅ |
| **管线集成** | ❌ 未被任何调用方使用 |

**当前状态**: GPU 路径已完整实现但未接入管线。`fill_interpolator_buffers` 主路径完全使用 CPU 递归 DAG 遍历。

### 2.3 含水层判定

| 维度 | 详情 |
|------|------|
| **GPU kernel** | `aquifer_batch_f64` + `aquifer_batch_tiled_f64` |
| **GPU launch** | `GpuAquiferBatchSampler::batch_aquifer_apply` — ✅ |
| **管线集成** | ✅ 通过 `ChainedBlockStateSampler` 调用 |
| **CPU fallback** | `cpu_aquifer_apply` — 4-NN 搜索 |

**当前状态**: **唯一完整接入管线的批量 GPU 功能**。8/8 测试通过。

### 2.4 Beardifier

| 维度 | 详情 |
|------|------|
| **GPU kernel** | `beardifier_batch_f64` — 已实现 |
| **GPU launch** | `GpuBeardifierBatchSampler::batch_beardifier` — 硬编码 `ok=false` |
| **管线集成** | Beardifier `fill()` 方法中有 `#[cfg(feature="gpu")]` 路径 |
| **CPU fallback** | `cpu_beardifier` — 结构/连接点遍历 |

**当前状态**: GPU kernel 存在但被硬编码禁用 (`let ok = false`)。禁用前仍分配 GPU buffer 造成浪费。

### 2.5 矿脉采样

| 维度 | 详情 |
|------|------|
| **GPU kernel** | `vein_batch_f64` — 已实现 |
| **GPU launch** | `GpuVeinBatchSampler::batch_vein_sample` — 硬编码 `ok=false` |
| **管线集成** | ❌ 未接入 |
| **CPU fallback** | `cpu_vein_detect` — 三重 Perlin + 概率判定 |

**当前状态**: GPU kernel 被硬编码禁用。CPU fallback 提供完整矿脉检测功能。

---

## 三、测试设计分析

### 当前测试的局限性

1. **`cell_cache_fill_consistency`**: CPU 参考路径使用 `results.fill(0.0)` (零填充)，而非实际 Perlin 计算。测试验证的是"GPU+fallback 最终结果 = 零"，而非 GPU kernel 的正确性。

2. **`vein_sample_consistency`**: 同上，CPU 参考使用全零结果。

3. **`interpolator_fill_consistency`**: CPU 参考使用零填充。

4. **所有测试使用空 `perlin_configs`**: `CellFillParams { perlin_configs: vec![], ... }`。

### 建议增强

```rust
// 添加真实验证测试
#[test]
fn cell_cache_with_real_noise_params() {
    // 使用有效的 perlin_configs 和 num_octaves
    let params = CellFillParams {
        perlin_configs: vec![/* 真实的 perlin 配置 */],
        num_octaves: vec![3],  // 3 个八度
        sampler_types: vec![0],
    };
    // 对比 GPU 路径和 CPU 参考 (使用 `cpu_cell_cache_fill_impl`)
}
```

---

## 四、性能

| 操作 | 时间 | 说明 |
|------|------|------|
| CPU 零填充 (4096位置) | 0.015 ms | memset 基准 |
| GPU batch_fill_cell_caches (4096位置) | 397.97 ms | 含 kernel 编译+首次启动 |

> GPU 首次启动包含 kernel JIT 编译 (~300ms)，后续调用约为 ~12ms。

---

## 五、结论

1. **Aquifer 是唯一完整接入的批量 GPU 功能** — 通过 `ChainedBlockStateSampler` 自动调用
2. **Cell Cache/Interpolator GPU kernel 已完整但管线未接入** — 传入空配置
3. **Beardifier/Vein GPU kernel 被硬编码禁用** — 需修复 `ok=false`
4. **测试需要增强** — 当前验证零填充而非实际 GPU kernel 正确性
