# GPU 优化建议报告 — 实例代码与收益分析

生成日期: 2026-08-12

---

## 建议 1: Aquifer GPU 管线集成

**优先级**: ⭐ 最高  
**预估加速**: 10–50×（取决于含水层网格密度）  
**难度**: 低  
**状态**: GPU kernel `aquifer_batch_f64` 已完全实现并通过所有测试，CPU 回退路径已验证

### 当前状态

```rust
// 当前 WorldAquiferSampler::apply_internal 仅使用 CPU 路径
let result = cpu_aquifer_apply(positions, densities, packed_grid, fluid_level, barrier_scale);
```

### 目标实现

```rust
// crates/pumpkin-world/src/generation/noise/aquifer_sampler.rs
impl WorldAquiferSampler {
    pub fn apply_internal(
        &mut self,
        positions: &[f64],
        densities: &[f64],
        packed_grid: &[i64],
        fluid_level: f64,
        barrier_scale: f64,
        batch_accel: Option<&BatchAccelerator>,
    ) -> AquiferBatchResult {
        // GPU 加速路径
        if let Some(accel) = batch_accel {
            if accel.is_active() {
                return accel.batch_aquifer_apply(
                    positions, densities, packed_grid, fluid_level, barrier_scale,
                );
            }
        }
        // CPU 回退
        cpu_aquifer_apply(positions, densities, packed_grid, fluid_level, barrier_scale)
    }
}
```

**收益**: 每个区块生成中 Aquifer 操作从 CPU 密集循环迁移到 GPU 并行 4-NN 搜索。

---

## 建议 2: 合并 Cell Cache 125→1 次 GPU 调用

**优先级**: ⭐ 高  
**预估加速**: 延迟降低 80–95%  
**难度**: 低  
**理由**: 当前每次 `batch_fill_cell_caches` 都是完整的上传→kernel→下载→释放周期。PCIe 传输开销和 kernel launch 开销主导总延迟。

### 目标实现

```rust
// crates/pumpkin-world/src/generation/proto_chunk.rs (或 noise_router)
// 收集所有 125 个位置的坐标和参数，单次批量处理

fn fill_all_cell_caches_batched(
    batch: &BatchAccelerator,
    all_positions: &[Vec<f64>; 125], // 125 次独立的坐标数组
    all_params: &[CellFillParams; 125],
    all_results: &mut [Vec<f64>; 125],
) {
    // 1. 拼接所有位置: flat_positions = concat(all_positions)
    // 2. 拼接参数: 将所有 octave 配置追加到单个 component_stack
    // 3. 生成 cell_indices: 每个位置指向其所属的 stack 配置
    // 4. 单次 GPU 调用
    let flat_pos = concat_3d(all_positions);
    let (component_stack, cell_indices) = flatten_params(all_params);
    batch.batch_fill_cell_caches_merged(&flat_pos, &component_stack, &cell_indices, &mut flat_results);
    // 5. 拆分结果回 125 个数组
    split_results(flat_results, all_results);
}
```

**注意**: 需要扩展 `GpuCellBatchSampler::batch_fill_cell_caches` 以接受 `cell_indices` 参数（kernel 已支持但 Rust 接口未暴露）。

**收益**: 消除 125 次 PCIe 传输 + kernel launch 开销，单次大 kernel launch 利用率更高。

---

## 建议 3: 移除 `light.rs` 中的立即同步

**优先级**: ⭐ 中  
**预估加速**: 流水线延迟降低 5–15%  
**难度**: 低

### 当前代码

```rust
// crates/pumpkin-gpu/src/light.rs:59
l.launch(/* ... */)?;
l.synchronize()?; // ← 立即同步，阻塞流水线
self.device.copy_from_device(&d_sl, sky_light)?;
// copy_from_device 在 CUDA 默认流上已有隐式同步
```

### 目标代码

```rust
l.launch(/* ... */)?;
// 移除 synchronize() — copy_from_device 在默认流中隐式同步
self.device.copy_from_device(&d_sl, sky_light)?;
```

**同样适用于**: `batch_block_scan` (L120)、`iterative_propagate` (L200, L240)。

**收益**: CUDA 驱动可在 kernel 执行期间并行准备 CPU 端缓冲区，消除 GPU 空泡。

---

## 建议 4: 延迟编译 persistent kernel

**优先级**: 低  
**预估收益**: CUDA 启动时间减少约 200ms  
**难度**: 低

### 当前代码

```rust
// crates/pumpkin-gpu/src/compile.rs
// all_cuda_kernel_sources() 无条件包含 light_propagate_persistent.cu
pub(crate) fn all_cuda_kernel_sources() -> Vec<CompiledKernel> {
    vec![
        // ...
        CompiledKernel {
            name: "light_propagate_u8_persistent".into(),
            source: include_str!("../../kernels/cuda/light_propagate_persistent.cu").into(),
        },
    ]
}
```

### 目标代码

```rust
pub(crate) fn all_cuda_kernel_sources(persistent_enabled: bool) -> Vec<CompiledKernel> {
    let mut kernels = vec![
        // ... 常规 kernel ...
    ];
    if persistent_enabled {
        kernels.push(CompiledKernel {
            name: "light_propagate_u8_persistent".into(),
            source: include_str!("../../kernels/cuda/light_propagate_persistent.cu").into(),
        });
    }
    kernels
}
```

**收益**: 当不需要 persistent kernel 时跳过其 PTX 编译（PTX 编译是 CUDA 启动中最慢的步骤之一）。

---

## 建议 5: JIT 特化 kernel 使用 `--fmad=true` + `--opt-level=3`

**优先级**: 低  
**预估收益**: JIT kernel 额外 10–20%  
**难度**: 低  
**理由**: JIT kernel 的八度数 ≤ 16，循环完全展开，`fmad` 融合乘加对确定性无影响（编译器已在编译时解析所有常量）。

### 目标实现

```rust
// crates/pumpkin-gpu/src/compile.rs — CudaKernelCompiler::compile_jit_kernel
fn build_compile_opts(&self, for_jit: bool) -> Vec<String> {
    let mut opts = vec![
        "--use_fast_math".into(),
        "--restrict".into(),
    ];
    if for_jit {
        opts.push("--fmad=true".into());
        opts.push("--opt-level=3".into());
    } else {
        // 常规 kernel 保持精度优先
        opts.push("--fmad=false".into());
        opts.push("--opt-level=2".into());
    }
    opts
}
```

**收益**: JIT 特化的 small-octave kernel 获得完整的编译器优化，包括融合乘加指令和激进内联。

---

## 建议 6: 提取 `gen_perm_table` 到共享位置

**优先级**: 低  
**收益**: 消除代码重复，降低维护风险  
**难度**: 低

### 当前状态

```rust
// crates/pumpkin-gpu/src/noise/batch_cell.rs (L894-904)
fn gen_perm_table(seed: u64, octave: usize) -> [u8; 256] { /* ... */ }

// crates/pumpkin-world/src/batch_accel.rs (L629-639)
fn gen_perm_table(seed: u64, octave: usize) -> [u8; 256] { /* ... */ }
```

两个函数完全相同但分属不同 crate（`pumpkin-gpu` 和 `pumpkin-world`），均为私有函数。

### 方案

```rust
// 选项 A: 提取到 pumpkin-util
// crates/pumpkin-util/src/noise/perm.rs
pub fn gen_deterministic_permutation_table(seed: u64, octave: usize) -> [u8; 256] {
    let mut perm = [0u8; 256];
    for (i, p) in perm.iter_mut().enumerate() {
        let h = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(octave as u64)
            .wrapping_add(i as u64);
        *p = (h ^ (h >> 24)) as u8;
    }
    perm
}

// 选项 B: 保持两份独立（两个 crate 都不希望公开此函数）
// 收益低（仅 16 行代码重复），建议保持现状
```

**建议**: 保持现状。两个函数虽相同但功能简单（16 行），且分属不同 crate 的私有作用域。

---

## 建议 7: `batch_sampler.rs` 中的 `GpuBufferSet` 抽象复用

**优先级**: 低  
**收益**: 减少 `batch_cell.rs` 中的手动缓冲区管理代码  
**难度**: 低

### 当前状态

`GpuBufferSet` (batch_sampler.rs) 封装了 f64/u8 双缓冲的分配/上传/下载/释放管理，但 `GpuCellBatchSampler` (batch_cell.rs) 使用 `alloc_f64_pooled` 等手动方法。

### 方案

将 `GpuBufferSet` 提取为 `crate::common::buffer_pool` 模块，使 `batch_cell.rs` 也使用同一套缓冲池 API。

```rust
// crates/pumpkin-gpu/src/common/buffer_pool.rs
pub(crate) struct GpuBufferSet {
    f64_bufs: Vec<GpuBuffer<f64>>,
    u8_bufs: Vec<GpuBuffer<u8>>,
}

impl GpuBufferSet {
    pub fn alloc_f64(&mut self, device: &GpuDevice, len: usize) -> Result<&GpuBuffer<f64>, DeviceError> { /* ... */ }
    pub fn upload_f64(&mut self, device: &GpuDevice, idx: usize, data: &[f64]) -> Result<(), DeviceError> { /* ... */ }
    // ...
}
```

**收益**: 统一缓冲池管理，减少 `batch_cell.rs` 中约 80 行重复代码。

---

## 已完成的优化（之前会话）

| # | 优化项 | 状态 |
|---|--------|------|
| 1 | `persistence` 预乘修复 | ✅ |
| 2 | Cell/Interpolator 缓冲池集成 | ✅ |
| 3 | Beard kernel GPU 持久化 (108KB) | ✅ |
| 4 | ShiftA/ShiftB JIT 特化 | ✅ |
| 5 | DoublePerlin JIT 特化 | ✅ |
| 6 | CUDA PTX flags (`--use_fast_math`, `--restrict`) | ✅ |
| 7 | Try-launch sync 移除 | ✅ |
| 8 | 懒编译注册表 | ✅ |
| 9 | Cell cache 合并 (125→1 调用) | ⚠️ Kernel 支持但 Rust 接口未暴露 |
| 10 | FlatCache GPU 集成 | ✅ |
| 11 | `precompute_surface` CPU 回退修复 | ✅ |
| 12 | `BatchAccelerator` 设备缓存 | ✅ |
| 13 | BatchAccelerator 采样器持久化 | ✅ (本会话修复) |
| 14 | JIT `jit_enabled=false` 门控 | ✅ |

---

## 预估总收益（如果实现所有建议）

| 优化项 | 预估提升 | 场景 |
|--------|---------|------|
| Aquifer GPU 集成 | 10–50× | 含水层密集维度（Overworld） |
| Cell Cache 调用合并 | 5–10× | 125 次独立调用的场景 |
| 移除立即同步 | 5–15% | 所有 GPU 操作 |
| 延迟编译 | 启动 -200ms | CUDA 初始化 |
| JIT fmad + opt=3 | 10–20% | JIT kernel 执行 |
| 缓冲池抽象复用 | ~50 行代码减少 | 维护性提升 |

**综合预估**: GPU 路径整体可实现 5–50× 加速 vs CPU 路径（取决于操作类型和数据规模）。
