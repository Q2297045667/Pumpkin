# 💡 光照引擎 GPU 加速测试报告

**日期**: 2026-08-11  
**测试文件**:
- `crates/pumpkin-world/tests/light_accel_consistency.rs`
- `crates/pumpkin-world/tests/light_fingerprint.rs`
- `crates/pumpkin-world/src/light_accel.rs`
- `crates/pumpkin-world/src/lighting/engine.rs`

---

## 一、测试结果

### 1.1 `light_accel_consistency` — 💥 崩溃

| 指标 | 值 |
|------|-----|
| 声明测试数 | 8 |
| 完成测试 | 1 (`block_scan_no_sources` ✅) |
| 崩溃 | 7 个测试未完成 |
| 退出码 | `0xc0000005` (STATUS_ACCESS_VIOLATION) |

崩溃发生在 `batch_block_scan` 或 `iterative_propagate` 测试中。

### 1.2 `light_fingerprint` — 💥 崩溃

| 指标 | 值 |
|------|-----|
| 总测试 | 4 |
| 通过 | 3 |
| 失败 | 1 (`block_scan_consistency`) |
| 崩溃 | 测试 harness cleanup 阶段 segfault |

```
sources mismatch: GPU vs CPU light block scan sources list
然后 SIGSEGV on cleanup
```

---

## 二、光照 GPU 加速架构

### 已实现的 GPU 路径

```
LightEngine::initialize_light()
├── try_gpu_sky_fill()           ← 256列并行天空光衰减
│   └── LightAccelerator::batch_sky_fill()
│       └── GpuLightSampler::batch_sky_fill()
│           └── GPU kernel: sky_light_fill_u8
│
├── sky_horizontal_propagate()   ← GPU天空光后CPU水平传播
│
├── block_light.propagate_light() ← CPU BFS (未GPU化)
│
└── clear()                      ← CPU
```

### 已实现但可能有 Bug 的路径

```
LightAccelerator::batch_block_scan()    ← GPU 方块光扫描
LightAccelerator::iterative_propagate() ← GPU 迭代传播
```

---

## 三、崩溃分析

### 可能原因

| # | 怀疑点 | 可能性 | 说明 |
|---|--------|--------|------|
| 1 | GPU buffer 越界访问 | **高** | `batch_block_scan` 或 `iterative_propagate` 中的 buffer 尺寸计算错误 |
| 2 | Use-after-free | **高** | 测试 cleanup 时释放已在 kernel 中使用的 buffer |
| 3 | OpenCL/CUDA 驱动问题 | 中 | 无 GPU 硬件时 `GpuLightSampler` 的 CPU fallback 路径可能有 bug |
| 4 | 光照数据布局不匹配 | 中 | GPU kernel 期望的数据布局与上传的不一致 |
| 5 | `GpuBuffer` Drop 顺序问题 | 中 | `free()` 被调用顺序导致 dangling pointer |

### 调试建议

```bash
# 1. 获取堆栈跟踪
RUST_BACKTRACE=full cargo test --features gpu -p pumpkin-world \
    --test light_accel_consistency sky_fill_single_column -- --nocapture

# 2. 逐个测试隔离运行
cargo test --features gpu -p pumpkin-world --test light_accel_consistency \
    sky_fill_single_column -- --nocapture
cargo test --features gpu -p pumpkin-world --test light_accel_consistency \
    sky_fill_16x256 -- --nocapture
# ... 逐个定位崩溃点

# 3. 检查 buffer 管理模式
# 查看 light.rs 中 batch_block_scan 的 alloc/free 配对
```

---

## 四、光照 GPU 覆盖状态

| 步骤 | GPU状态 | CPU回退 |
|------|---------|---------|
| 天空光初始填充 (256列) | ✅ 已实现 | ✅ 可用 |
| 天空光水平传播 | ❌ CPU-only | ✅ |
| 方块光初始扫描 | ⚠️ 有GPU路径但疑似有bug | ✅ 可用 |
| 方块光 BFS 传播 | ⚠️ 有GPU路径但疑似有bug | ✅ 可用 |
| 减光队列 | ❌ CPU-only | ✅ |

**总体覆盖**: ~50% 步骤有 GPU 路径，但 2 个路径有运行时 bug。

---

## 五、性能影响

光照阶段占总生成时间 **39.3%** (21.86ms / 55.67ms)，是最大的单一瓶颈。

| 优化项 | 预估收益 | 风险 |
|--------|---------|------|
| 修复现有 GPU block_scan/propagate | 5-15% | 低 (修复bug) |
| GPU 化天空光水平传播 | 3-8% | 中 |
| GPU 化减光队列 | 2-5% | 高 |

---

## 六、修复建议

1. **P0**: 在调试器下复现 `light_accel_consistency` 崩溃，获取堆栈
2. **P0**: 隔离测试每个光照 GPU 方法，定位具体崩溃点
3. **P1**: 修复 `block_scan_consistency` — 光源列表不一致
4. **P1**: 为 GPU buffer 添加 RAII 管理 (`GpuBufferSet` 已在 `batch_sampler.rs` 中实现，可复用于 `light.rs`)
5. **P2**: 将 `light.rs` 的 manual alloc/free 改为使用 `GpuBufferSet`
