# Pumpkin 项目 CPU/GPU 生成逻辑全面分析报告

生成日期: 2026-08-12

---

## 一、测试覆盖缺口 & 需补全的测试

### 1.1 严重缺口 (需立即补全)

#### ❌ JIT 数值正确性测试

**现状**: `jit_tests.rs` 和 `jit_consistency_tests.rs` 仅验证源码结构（名称、循环展开、常量内联），从未执行 JIT kernel 并对比结果。

**建议新增**: `tests/jit_numerical_consistency.rs`

```rust
#[cfg(feature = "gpu")]
mod jit_numerical {
    use pumpkin_gpu::{GpuDevice, jit};
    use pumpkin_gpu::noise::cache::SerializedOctaveConfig;
    use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

    /// 验证 JIT octave kernel 输出 == CPU 直接采样
    #[test]
    fn jit_octave_vs_cpu() {
        let sampler = mk_sampler(&[0, 1, 2, 3]);
        let config = SerializedOctaveConfig::from_sampler(&sampler);
        let jit_kernel = jit::specialize_octave_perlin(&config, 16)
            .expect("4 octaves should JIT-specialize");

        let n = 512;
        let pos = mk_pos(n);
        let mut cpu = vec![0.0; n];
        let mut gpu = vec![0.0; n];

        // CPU reference
        for i in 0..n {
            cpu[i] = sampler.sample(pos[i*3], pos[i*3+1], pos[i*3+2]);
        }

        // GPU JIT
        let mut device = GpuDevice::from_config(&GpuConfig::default());
        device.compile_jit_kernel(&jit_kernel).unwrap();
        // ... launch kernel, compare hash
        assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "JIT octave mismatch");
    }

    /// 验证 JIT double_perlin kernel 输出 == CPU 直接采样
    #[test]
    fn jit_double_perlin_vs_cpu() { /* 同上模式 */ }

    /// 验证 JIT shift_a kernel 输出 == CPU 直接采样
    #[test]
    fn jit_shift_a_vs_cpu() { /* 同上模式 */ }

    /// 验证 JIT shift_b kernel 输出 == CPU 直接采样
    #[test]
    fn jit_shift_b_vs_cpu() { /* 同上模式 */ }
}
```

#### ❌ CellCache/Interpolator CPU 参考验证

**现状**: `batch_fingerprint.rs` 仅验证 GPU 确定性（两次调用哈希相同），无独立 CPU 基准。

**建议新增**: 在 `worldgen_fingerprint.rs` 中增加:

```rust
/// CellCache GPU 输出 vs CPU reference (DAG walk)
#[test]
fn cellcache_gpu_vs_cpu_reference() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let params = extract_cell_params(&sampler);
    let n = 512;
    let pos = mk_pos3d(n);

    let mut gpu_res = vec![0.0; n];
    accel().batch_fill_cell_caches(&pos, &params, &mut gpu_res);

    // CPU reference: 直接调用 Perlin 采样器对每个位置求值
    let mut cpu_res = vec![0.0; n];
    for i in 0..n {
        cpu_res[i] = sampler.sample(pos[i*3], pos[i*3+1], pos[i*3+2]);
    }

    assert_eq!(f64_hash(&cpu_res), f64_hash(&gpu_res), "CellCache CPU/GPU mismatch");
}

/// Interpolator GPU 输出 vs CPU reference
#[test]
fn interpolator_gpu_vs_cpu_reference() { /* 同上模式 */ }

/// Vein GPU 输出 vs CPU reference (非空参数)
#[test]
fn vein_gpu_vs_cpu_reference() {
    let params = VeinParams {
        toggle_config: vec![1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ridged_config: vec![1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        gap_config:    vec![1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    };
    // ... 非平凡 vein 参数，GPU vs CPU 对比
}
```

#### ❌ 方块光传播测试

**现状**: 仅测试天空光 (`sky_fill`、`sky_horizontal`、`block_scan`)，无方块光 BFS 传播完整测试。

**建议新增**: `tests/block_light_fingerprint.rs`

```rust
/// 方块光传播：CPU BFS vs GPU iterative_propagate
#[test]
fn block_light_propagate_consistency() {
    let n = 4096;
    let max_iters = 32;
    // 构建 3D 网格的光源、不透明度、邻居索引
    // CPU BFS 传播
    // GPU iterative_propagate
    // 对比元素级一致性 + 迭代次数
}
```

### 1.2 中等缺口

| 缺口 | 建议 |
|------|------|
| Aquifer 大网格边缘情况 | 添加 64×64×384 网格测试，验证 4-NN 在大数据集上的正确性 |
| Beardifier 零结构/仅连接点 | 添加 `beardier_empty_structures`、`beardier_junctions_only` |
| 多采样器类型 CellCache | 测试 `sampler_types: vec![0, 1, 2]` 多类型混合 |
| 光照移除 (decrease) | 测试 block placement 导致的光照衰减 |
| 多线程安全 | 添加 `rayon` 并发调用 `BatchAccelerator` 的测试 |
| GPU 不可用回退 | 强制 `GpuConfig { enabled: false }`，验证所有 API 不 panic |

### 1.3 压力测试缺口

| 缺口 | 建议 |
|------|------|
| OOM 恢复 | 分配超大 buffer (接近 VRAM 上限)，验证错误处理不崩溃 |
| 连续调用稳定性 | 1000 次 `batch_fill_cell_caches` 连续调用，验证无内存泄漏 |
| 极值坐标 | 位置坐标在 `±1e308` 范围内，验证 GPU kernel 不产生 NaN |
| 零位置 | 所有位置为 `(0,0,0)`，验证输出确定性 |

---

## 二、GPU 模块缺失功能

### 2.1 ⚠️ 高影响

| 功能 | 状态 | 详情 |
|------|------|------|
| **`light_propagate_u8_persistent` OpenCL** | ❌ 缺失 | CUDA 有 persistent kernel，OpenCL 无。所有 OpenCL 光照传播走迭代路径，3-10× 慢 |
| **`precompute_surface` 绕过 JIT** | ⚠️ 未接入 | 直接调用 `sample_double_perlin_batch` 而非 `_jit`，浪费 1.5-2.5× 潜在加速 |
| **`backfill_noise_cache` 纯 CPU** | ⚠️ 未接入 | 虽是 `#[cfg(gpu)]`，但调用 `sampler.sample()` 逐点 CPU 计算，应用 GPU 批量路径 |

### 2.2 ⚠️ 低影响

| 功能 | 状态 | 详情 |
|------|------|------|
| `batch_fill_cell_caches_indexed` 未暴露 | 未使用 | GPU kernel 支持 `cell_indices` 多采样器选择，但 `BatchAccelerator` 无对应 API |
| `compile_kernel_by_name` OpenCL 存根 | 存根 | 仅记录日志，不实际编译 — 延迟加载未完全实现 |
| `sky_horizontal_propagate` 无 persistent 变体 | 缺失 | 仅有迭代 kernel，每次迭代需 `copy_from_device` 检查收敛 |

---

## 三、死代码 & 可安全移除的代码

### 3.1 🔴 可安全移除

| 文件 | 行号 | 项目 | 理由 |
|------|------|------|------|
| `common/buffer.rs` | 97-101 | `GpuBuffer::backend_type()` | 零调用者 |
| `opencl/kernel.rs` | 90-94 | `OpenClKernelLauncher::queue_at()` | 零调用者 |
| `cuda/mod.rs` | 28-29 | `CudaBackend.ctx` 字段 | 存储但从不读取 |
| `cuda/mod.rs` | 36-37 | `CudaBackend.use_curand` 字段 | 存储但从不读取 |

### 3.2 🟡 待评估

| 文件 | 项目 | 建议 |
|------|------|------|
| `cuda/curand.rs` | `CuRandGenerator` 完整模块 | 仅在 `#[cfg(test)]` 中使用，生产代码零调用。谨慎：若计划集成到光线追踪噪声中则保留 |
| `common/buffer.rs` | `CudaSliceHolder<T>` + 方法 | CUDA stub，注释注明"待 CUDA 硬件完成后激活"。可保留 |

### 3.3 重复代码

| 重复项 | 位置 | 行数 | 建议 |
|--------|------|------|------|
| `gen_perm_table` | `batch_cell.rs:877` = `batch_accel.rs:629` | 16 行 ×2 | 提取到 `pumpkin-util` 或共用模块。收益：消除确定性置换生成的不一致风险 |
| CPU 三线性插值 | `batch_sampler.rs:953` ≈ `batch_accel.rs:871` ≈ `noise_accel.rs:159` | ~15 行 ×3 | 统一为 `cpu_trilinear_impl` 单一实现。注意：`pumpkin_util::math::lerp3` 因浮点评估顺序不同不可统一 |
| CPU 光照传播 | `light.rs:257` ≈ `light_accel.rs:122` | ~30 行 ×2 | pumpkin-world 应调用 pumpkin-gpu 版本而非复制 |
| 双 Perlin CPU 回退 | `noise_accel.rs:98` ≈ `batch_accel.rs:262` | ~8 行 ×2 | 共享常量 `c = 1.0181268882175227` 和公式 |

---

## 四、优化建议 (附实例代码 & 收益预估)

### 4.1 ⭐ `precompute_surface` 接入 JIT 路径

**文件**: `noise_accel.rs:232-246`
**预估收益**: 1.5-2.5× (最热噪声路径)
**原因**: 当前绕过 JIT 直接调用 batch kernel，JIT 可烘焙 8-16 个八度的振幅/频率/原点为编译时常量，消除循环开销和间接访存。

**实例代码**:
```rust
// noise_accel.rs precompute_surface — 当前代码 (~L234)
inner.sample_double_perlin_batch(surface_a, surface_b, surface_amp, &pos_3d, &mut surf)

// 优化后: 先尝试 JIT
if inner.sample_double_perlin_jit(surface_a, surface_b, surface_amp, &pos_3d, &mut surf).is_err() {
    inner.sample_double_perlin_batch(surface_a, surface_b, surface_amp, &pos_3d, &mut surf)?;
}
```

### 4.2 ⭐ `backfill_noise_cache` 接入 GPU

**文件**: `chunk_noise_router.rs:676-713`
**预估收益**: 消除 CPU 噪声重复计算 (cell/interpolator 填充后已 GPU 计算的值，CPU 再次计算)
**实例代码**:
```rust
// 当前: 逐点 CPU 调用 sampler.sample()
for info in &samplers {
    for idx in 0..n {
        let value = info.sampler.sample(x, y, z);
        cache_entries.insert((info.sampler_id, ix, iy, iz), value);
    }
}

// 优化后: 调用 NoiseAccelerator::fill_noise_cache
if let Some(mut noise_accel) = crate::gpu::get_noise_accel() {
    for info in &samplers {
        noise_accel.fill_noise_cache(info.sampler_id, &info.sampler, positions);
    }
    return;
}
// CPU fallback 保持不变
```

### 4.3 🟡 移除死字段减少内存

| 字段 | 文件 | 节省 |
|------|------|------|
| `CudaBackend.ctx` (Arc<CudaContext>) | `cuda/mod.rs:28` | ~8 bytes (Arc 指针) |
| `CudaBackend.use_curand` (bool) | `cuda/mod.rs:37` | 1 byte (已对齐到 word) |
| `GpuBuffer::backend_type()` | `buffer.rs:97` | ~10 行代码 |

### 4.4 🟡 提取共享工具函数减少维护风险

```rust
// pumpkin-gpu 或 pumpkin-util 中的新函数
pub fn gen_perm_table(seed: u64, octave: usize) -> [u8; 256] {
    let mut perm = [0u8; 256];
    for (i, p) in perm.iter_mut().enumerate() {
        let h = seed.wrapping_mul(6364136223846793005)
            .wrapping_add(octave as u64)
            .wrapping_add(i as u64);
        *p = (h ^ (h >> 24)) as u8;
    }
    perm
}
```
**收益**: 消除 `batch_cell.rs` 和 `batch_accel.rs` 中的 16 行重复，确保GPU/CPU回退使用**相同**的置换表。

### 4.5 🟢 条件编译排除 CUDA-only 路径

```rust
// compile.rs — 按需注册 persistent kernel
pub(crate) fn all_cuda_kernel_sources(persistent_enabled: bool) -> Vec<CompiledKernel> {
    let mut kernels = vec![/* 常规 kernel */];
    if persistent_enabled {
        kernels.push(CompiledKernel {
            name: "light_propagate_u8_persistent".into(),
            source: kernels_light::LIGHT_PROPAGATE_PERSISTENT_CU.into(),
        });
    }
    kernels
}
```
**收益**: 当配置 `persistent_kernels = false` 时，启动时间减少 ~200ms (跳过 PTX 编译)。

---

## 五、未实现功能 & 可行性方案

### 5.1 Biome GPU 加速

| 维度 | 评估 |
|------|------|
| **可行性** | ⚠️ 中 — `MultiNoiseSampler` 涉 7 维噪声参数 + biome 规则匹配 |
| **方案** | GPU 加速噪声采样部分（7 维 → 批量求值），biome 选择逻辑仍留 CPU |
| **预估加速** | 2-5× (噪声采样占 biome 查询的 60-80% 时间) |
| **复杂度** | 高 — 需新建 7 维 kernel，接入 `MultiNoiseSampler::sample` |

**详细方案**:
1. 创建 `multi_noise_sample_f64` kernel：接受 7×N 个噪声参数 + N 个位置
2. 在 `MultiNoiseSampler` 中批量采样 7 个维度
3. biome 选择 (二分搜索 parameter space) 留 CPU

### 5.2 矿脉批量真正集成

| 维度 | 评估 |
|------|------|
| **可行性** | ✅ 高 — kernel 已完成，`batch_vein_sample` 已通过测试 |
| **方案** | `OreVeinSampler::sample` 改为接受位置数组，调用 `BatchAccelerator::batch_vein_sample` |
| **预估加速** | 10-50× (矿脉采样是 per-block 操作，批量可消除 kernel launch 开销) |
| **复杂度** | 中 — 需重构 `OreVeinSampler` 接口 |

### 5.3 OpenCL 多队列流水线

| 维度 | 评估 |
|------|------|
| **可行性** | ⚠️ 中 — 配置中 `pipeline_queues` 参数已就绪 |
| **方案** | `OpenClBackend` 管理多个 `CommandQueue`，交替提交 kernel/copy |
| **预估加速** | 10-20% 吞吐 (kernel 执行 + 数据传输重叠) |
| **复杂度** | 中 — 需重构 `OpenClBackend::queue()` 为轮询选择器 |

### 5.4 Noise Cache 回填到 `OctavePerlinNoiseSampler`

| 维度 | 评估 |
|------|------|
| **可行性** | ✅ 高 — `fill_noise_cache` 已实现，`set_noise_cache` API 已存在 |
| **方案** | `OctavePerlinNoiseSampler::sample` 检查 `NOISE_CACHE` 命中 |
| **预估加速** | 2-5× 对重复采样场景 |
| **复杂度** | 低 — 在 sample() 开头添加缓存查询 |

### 5.5 OpenCL `light_propagate_u8_persistent`

| 维度 | 评估 |
|------|------|
| **可行性** | ❌ 不可行 — OpenCL 缺少 CUDA cooperative groups 等价物 |
| **替代方案** | 使用多迭代 kernel + 减少 `copy_from_device` 频率 (每 N 次迭代读一次) |
| **预估加速** | 2-3× vs 当前每次迭代同步 |

---

## 六、总结 & 优先级排序

| 优先级 | 类别 | 任务 | 预估收益 |
|--------|------|------|---------|
| 🔴 P0 | 测试 | 补全 JIT 数值正确性测试 | 避免 JIT kernel 静默产生错误结果 |
| 🔴 P0 | 测试 | 补全 CellCache/Interpolator CPU 参考验证 | 确保 GPU 批量填充与 CPU DAG 一致 |
| 🔴 P0 | 优化 | `precompute_surface` 接入 JIT | 1.5-2.5× 最热路径加速 |
| 🟡 P1 | 死代码 | 移除 4 个可安全清除的项 | 减少 ~20 行，提高代码清晰度 |
| 🟡 P1 | 重复代码 | 提取 `gen_perm_table` | 消除确定性风险 |
| 🟡 P1 | 优化 | `backfill_noise_cache` 接入 GPU | 消除冗余 CPU 计算 |
| 🟡 P1 | 功能 | OpenCL persistent kernel 替代方案 | 3-10× 光照传播加速 |
| 🟢 P2 | 测试 | 方块光传播 + 光照移除 + 多线程安全 | 覆盖缺失路径 |
| 🟢 P2 | 功能 | 矿脉批量集成 | 10-50× per-block |
| 🟢 P2 | 功能 | Biome GPU 加速 | 2-5× |
| 🟢 P3 | 重复 | 统一 CPU 三线性/光照传播 | 减少维护负担 |
