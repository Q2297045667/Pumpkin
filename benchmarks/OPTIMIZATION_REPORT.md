# 🚀 Pumpkin GPU — 优化建议报告

**生成时间**: 2026-08-12  
**分析范围**: 全项目 CPU vs GPU 生成逻辑对比

---

## 一、GPU 缺失功能总览

| 功能 | CPU | GPU | 优先级 | 说明 |
|------|:---:|:---:|:------:|------|
| `barrier_noise` | ✅ | ❌ | 低 | Aquifer 内部调用，已有底层 Perlin 间接加速 |
| `fluid_level_*` | ✅ | ❌ | 低 | 同上 |
| `lava_noise` | ✅ | ❌ | 低 | 同上 |
| `erosion`/`depth` | ✅ | ❌ | 低 | Biome 采样路径 |
| Aquifer 采样 | ✅ | ⚠️ | **高** | Kernel 已存在，`apply_internal` 未接入 |
| Biome 采样 | ✅ | ❌ | 中 | `MultiNoiseSampler` 纯 CPU |
| 表面 GPU 填充 | ✅ | ⚠️ | 中 | `CachedSurfaceNoise` 结构就绪，kernel 未接入 |
| 光照增量更新 | ✅ | ❌ | 低 | 单点更新不适合 GPU |
| CacheOnce/Cache2D | ✅ | ❌ | 低 | 低频使用 |

**最大缺口**: Aquifer GPU 接入（kernel `aquifer_batch_f64` 已实现，缺桥接代码）

---

## 二、可安全删除的未使用代码

| # | 文件 | 行号 | 内容 | 原因 |
|---|------|------|------|------|
| 1 | `pumpkin-gpu/src/logging.rs` | L41-50 | `pub fn log_available_devices()` | 零调用 |
| 2 | `pumpkin-gpu/src/common/layout.rs` | L26-35 | `pub fn aos2d_to_soa()` | 零调用 |
| 3 | `pumpkin-gpu/src/common/layout.rs` | L39-48 | `pub fn soa_to_aos3d()` | 零调用 |
| 4 | `pumpkin-gpu/src/jit.rs` | L111-113 | `pub fn should_jit_specialize()` | 仅测试使用，内部已有等效检查 |

---

## 三、代码重复与精简建议

### 3.1 三线性插值三重实现

| 位置 | 函数 |
|------|------|
| `batch_sampler.rs:866` | `cpu_trilinear` |
| `batch_accel.rs:805` | `cpu_trilinear_impl` |
| `noise_accel.rs:153` | 内联代码 |

**建议**: 统一到 `pumpkin-util::math::trilinear_interpolate`

### 3.2 `gen_perm_table` 双份实现

| 位置 | 用途 |
|------|------|
| `batch_cell.rs:856` | GPU 端 Perlin 置换表 |
| `batch_accel.rs:563` | CPU 回退置换表 |

**建议**: 提取到 `pumpkin-util::noise::perlin::gen_perm_table`

### 3.3 Buffer 池死代码

`GpuCellBatchSampler` 声明了 `perm_pool` + `f64_pool` 但从未使用。
**建议**: 在 `batch_fill_cell_caches` / `batch_fill_interpolators` 中接入。

---

## 四、优化方案（按优先级）

### 🔴 P0-1: 延迟编译 — 移除启动时全量编译

**文件**: `crates/pumpkin-gpu/src/cuda/kernel.rs` + `opencl/kernel.rs`

**当前**: `compile_all()` 启动时编译全部 16-17 个 kernel，NVRTC 耗时 2-10s
**优化**: 移除 `compile_all()`，依赖已有的 `try_compile_kernel_on_demand` 按需编译

**实例代码**:
```rust
// cuda/kernel.rs — 删除 compile_all() 调用
pub fn init(...) {
    // 旧: compiler.compile_all(ctx, flags)?;
    // 新: 延迟编译，首次 launch 时自动触发
    self.compiler = Some(compiler); // 仅保存编译器，不预编译
}
```
**收益**: 启动时间 **-2~10s**，首次使用单个 kernel 仅 50-300ms

---

### 🔴 P0-2: Beard Kernel GPU 持久化

**文件**: `crates/pumpkin-gpu/src/noise/batch_cell.rs`

**当前**: 每次 `batch_beardifier` 调用都 alloc 108KB + upload + free
**优化**: 使用全局 `OnceLock<GpuBuffer<f64>>` 持久化

**实例代码**:
```rust
// 全局持久化 buffer
static BEARD_KERNEL_GPU_BUF: OnceLock<Mutex<Option<GpuBuffer<f64>>>> = OnceLock::new();

fn get_or_upload_beard_kernel(device: &GpuDevice) -> Result<&GpuBuffer<f64>, DeviceError> {
    let lock = BEARD_KERNEL_GPU_BUF.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap();
    if guard.is_none() {
        let mut buf = device.alloc_f64(13824)?;
        device.copy_to_device(&mut buf, get_beard_kernel_gpu())?;
        *guard = Some(buf);
    }
    Ok(guard.as_ref().unwrap())
}
```
**收益**: 消除每次 chunk 108KB GPU alloc+upload+free（~0.2-0.5ms × N 次）

---

### 🔴 P0-3: CUDA `--use_fast_math` + `--restrict`

**文件**: `crates/pumpkin-gpu/src/compile.rs`

**当前**: 仅有 `--fmad=true` + `--opt-level=3`
**优化**: 添加 `--use_fast_math` 和 `--restrict`

**实例代码**:
```rust
fn build_compile_opts(&self, flags: &[String]) -> CompileOptions {
    let mut opts = CompileOptions::default();
    // ... 架构设置 ...
    opts.options.push("--fmad=true".into());
    opts.options.push("--opt-level=3".into());
    opts.options.push("--use_fast_math".into());   // 新增
    opts.options.push("--restrict".into());         // 新增
    // ...
}
```
**收益**: 1.1×–1.3× kernel 执行加速（噪声不需要 bit-identical 精度）

---

### 🟡 P1-1: ShiftA/ShiftB JIT 特化

**文件**: `crates/pumpkin-gpu/src/jit.rs` + `batch_sampler.rs`

**当前**: 仅 OctavePerlin 和 DoublePerlin 有 JIT
**优化**: 为 ShiftA/ShiftB 添加 JIT（同构模板，约 30 行）

**实例代码**:
```rust
// jit.rs 新增
pub fn specialize_shift(sampler_type: &str, config: &SerializedOctaveConfig, max_unroll: usize) 
    -> Option<JitSpecializedKernel> 
{
    let m = config.num_octaves();
    if m > max_unroll { return None; }
    // ... 复用 specialize_octave_perlin 的代码生成逻辑
    // 区别：输入为 2D (xz)，kernel 参数少 1 个
}

// batch_sampler.rs 新增
pub fn sample_shift_a_jit(&mut self, sampler: &OctavePerlinNoiseSampler, 
    xz: &[f64], results: &mut [f64]) -> Result<(), DeviceError> 
{
    // 与 sample_octave_jit 同构，输入维度不同
}
```
**收益**: 1.5×–2× kernel 加速

---

### 🟡 P1-2: BatchAccelerator 持久化 Sampler 实例

**文件**: `crates/pumpkin-world/src/batch_accel.rs`

**当前**: `with_gpu` 每次创建新 sampler → NoiseCache/池丢失
**优化**: 持有持久化 sampler 实例

**实例代码**:
```rust
pub struct BatchAccelerator {
    config: GpuConfig,
    #[cfg(feature = "gpu")]
    device: Mutex<Option<GpuDevice>>,
    #[cfg(feature = "gpu")]
    cell_sampler: Mutex<Option<GpuCellBatchSampler>>,   // 新增
    #[cfg(feature = "gpu")]
    noise_sampler: Mutex<Option<GpuNoiseSampler>>,      // 新增
}

impl BatchAccelerator {
    fn get_or_init_cell_sampler(&self) -> Option<MutexGuard<GpuCellBatchSampler>> {
        // 惰性初始化 + 复用
    }
}
```
**收益**: 消除每次调用的 NoiseCache 重建 + buffer 池积累

---

### 🟢 P2-1: 融合 `fill_interpolator_buffers` 循环

**文件**: `crates/pumpkin-world/src/generation/noise/mod.rs`

**当前**: `sample_density` 循环中 8-32 次独立 kernel launch
**优化**: 收集所有 cell_z 位置，合并为 1 次 GPU 调用

**实例代码**:
```rust
fn sample_density_combined(&mut self, start: bool, current_x: i32) {
    // 收集所有 cell_z 位置
    let all_positions = (0..=self.horizontal_cell_count)
        .flat_map(|cell_z| { /* 收集该 cell_z 的所有 y 坐标 */ })
        .collect::<Vec<f64>>();
    // 单次 GPU 调用
    accel.batch_fill_interpolators(&all_positions, &params, &mut all_results);
    // 分发结果到各 cell_z 的 buffer
}
```
**收益**: 减少 8-32 次 → 1 次 kernel launch

---

### 🟢 P2-2: 接入现有 Buffer 池

**文件**: `crates/pumpkin-gpu/src/noise/batch_cell.rs`

**当前**: `perm_pool` + `f64_pool` 已声明但从未使用
**优化**: 在 `batch_fill_cell_caches` / `batch_fill_interpolators` 中接入

**实例代码**:
```rust
// 替换:
let d_perms = self.device.alloc_u8(total_octaves * 256)?;
// 为:
let d_perms = self.alloc_u8_pooled(total_octaves * 256)?;

// 在 return 前:
self.free_u8_pooled(total_octaves * 256, d_perms);
```
**收益**: 消除同尺寸 buffer 的重复 alloc/free（~20-40% 减少）

---

## 五、暂未实现功能的可行性分析

| 功能 | 可行性 | 方案 |
|------|:------:|------|
| **Aquifer GPU** | ✅ 高 | `apply_internal` 中收集位置 → `batch_aquifer_apply`，kernel+回退已就绪 |
| **Biome GPU** | ⚠️ 中 | 使用 `flatcache_precompute_f64` 2D 批量采样 |
| **表面 GPU 填充** | ✅ 高 | `CachedSurfaceNoise` 已就绪，接入 `NoiseAccelerator` |
| **天空光水平传播 GPU** | ⚠️ 中 | 需新 2D BFS kernel |
| **方块光增量更新 GPU** | ❌ 低 | 增量队列去重 + 部分区域重算，GPU 复杂度高 |
| **DAG 中间组合 GPU** | ❌ 低 | DAG 拓扑动态变化，不适合 GPU 批量 |
| **结构生成 GPU** | ❌ 低 | 离散 NBT 模板放置，不适合 GPU |

---

## 六、收益汇总

| 优化 | 类型 | 预期收益 | 难度 | 代码量 |
|------|:----:|---------|:----:|:------:|
| 延迟编译 | 启动 | -2~10s 启动时间 | 极低 | 删 2 行 |
| Beard kernel 持久化 | 运行时 | ~0.2-0.5ms/chunk | 极低 | +15 行 |
| CUDA fast_math | 运行时 | 1.1-1.3× kernel | 极低 | +2 行 |
| ShiftA/B JIT | 运行时 | 1.5-2× kernel | 低 | +30 行 |
| Sampler 持久化 | 运行时 | -20% alloc 开销 | 中 | +40 行 |
| Interpolator 融合 | 运行时 | 8-32→1 launch | 中 | +50 行 |
| Buffer 池接入 | 运行时 | -20-40% alloc | 低 | +10 行 |

**累计预期收益**: 启动时间 10s → <1s，运行时 chunk 生成 2×–5× 加速
