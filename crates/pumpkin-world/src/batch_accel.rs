//! 批量加速接口 — Cell Cache（vanilla 语义）、Aquifer、Beardifier、Trilinear 的 GPU 批量处理。
#![allow(clippy::doc_markdown)]

use pumpkin_config::gpu::GpuConfig;
#[cfg(feature = "gpu")]
use pumpkin_gpu::{
    GpuDevice,
    noise::{
        GpuNoiseSampler,
        batch_cell::{
            AquiferBatchResult, BeardifierJunctionData, BeardifierStructureData,
            GpuAquiferBatchSampler, GpuBeardifierBatchSampler,
        },
    },
};
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

/// CellCache 批量填充规格（vanilla `Noise` 组件语义：DoublePerlin + NoiseData 缩放）。
///
/// 与 vanilla `Noise.compute` 逐位对应：
/// `value = dbl.sample(x * xz_scale, y * y_scale, z * xz_scale)`。
pub struct CellCacheFillSpec<'a> {
    pub first: &'a OctavePerlinNoiseSampler,
    pub second: &'a OctavePerlinNoiseSampler,
    pub amplitude: f64,
    pub xz_scale: f64,
    pub y_scale: f64,
}

/// 批量加速器 — 为 Cell Cache（vanilla 语义）、Aquifer、Beardifier、Trilinear 提供 GPU 批量采样。
pub struct BatchAccelerator {
    config: GpuConfig,
    /// 缓存的 GPU 设备（懒初始化），通过 Mutex 包装以提供 Sync。
    #[cfg(feature = "gpu")]
    cached_device: std::sync::Mutex<Option<GpuDevice>>,
    /// 持久化 Noise 采样器。
    #[cfg(feature = "gpu")]
    noise_sampler: std::sync::Mutex<Option<GpuNoiseSampler>>,
    /// 持久化 Beardifier 采样器（含 beard kernel GPU buffer）。
    #[cfg(feature = "gpu")]
    beardifier_sampler: std::sync::Mutex<Option<GpuBeardifierBatchSampler>>,
    /// 持久化 Aquifer 采样器。
    #[cfg(feature = "gpu")]
    aquifer_sampler: std::sync::Mutex<Option<GpuAquiferBatchSampler>>,
}

impl BatchAccelerator {
    /// 从 GPU 配置创建批量加速器。
    ///
    /// 仅在配置同时满足 `enabled` 和
    /// `(noise_acceleration || batch_acceleration || jit_enabled)` 时
    /// 才会在后续调用中尝试初始化 GPU 设备。
    #[must_use]
    pub fn new(config: &GpuConfig) -> Self {
        Self {
            config: config.clone(),
            #[cfg(feature = "gpu")]
            cached_device: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            noise_sampler: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            beardifier_sampler: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            aquifer_sampler: std::sync::Mutex::new(None),
        }
    }

    /// 是否激活 — 即配置满足批量加速的基本条件。
    ///
    /// 注意：返回 `true` 仅表示配置层面允许尝试 GPU 加速，
    /// 实际运行时仍可能因驱动缺失等原因回退到 CPU。
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.config.enabled
            && (self.config.noise_acceleration
                || self.config.batch_acceleration
                || self.config.jit_enabled)
    }

    /// 懒初始化或获取设备（持久化复用）。
    #[cfg(feature = "gpu")]
    fn ensure_device(&self) -> Option<GpuDevice> {
        let mut guard = self.cached_device.lock().ok()?;
        if guard.is_none() && self.is_active() {
            let device = GpuDevice::from_config(&self.config);
            if device.device_type() != pumpkin_gpu::DeviceType::Cpu {
                // 需要给每个 sampler 独立设备实例，因此每次调用都新建
                // GpuDevice 内部使用 Arc 共享资源，轻量
                let device2 = GpuDevice::from_config(&self.config);
                *guard = Some(device);
                return Some(device2);
            }
        }
        guard.as_ref().and_then(|_| {
            let d = GpuDevice::from_config(&self.config);
            (d.device_type() != pumpkin_gpu::DeviceType::Cpu).then_some(d)
        })
    }

    /// 懒初始化 Noise 采样器并执行操作。
    #[cfg(feature = "gpu")]
    fn with_noise_sampler<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut GpuNoiseSampler) -> R,
    {
        let mut guard = self.noise_sampler.lock().ok()?;
        if guard.is_none() {
            let device = self.ensure_device()?;
            *guard = Some(GpuNoiseSampler::new(device));
        }
        guard.as_mut().map(f)
    }

    /// 懒初始化 Beardifier 采样器并执行操作。
    #[cfg(feature = "gpu")]
    fn with_beardifier_sampler<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut GpuBeardifierBatchSampler) -> R,
    {
        let mut guard = self.beardifier_sampler.lock().ok()?;
        if guard.is_none() {
            let device = self.ensure_device()?;
            *guard = Some(GpuBeardifierBatchSampler::new(device));
        }
        guard.as_mut().map(f)
    }

    /// 懒初始化 Aquifer 采样器并执行操作。
    #[cfg(feature = "gpu")]
    fn with_aquifer_sampler<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut GpuAquiferBatchSampler) -> R,
    {
        let mut guard = self.aquifer_sampler.lock().ok()?;
        if guard.is_none() {
            let device = self.ensure_device()?;
            *guard = Some(GpuAquiferBatchSampler::new(device));
        }
        guard.as_mut().map(f)
    }

    // --------------------------------------------------------------------------
    // Cell Cache（vanilla 语义）
    // --------------------------------------------------------------------------

    /// 批量填充 Cell Cache（vanilla `Noise` 语义：每个 cache 一组 DoublePerlin）。
    ///
    /// `results` 布局为 `[cache_index][position]`，长度 = `(positions.len() / 3) * specs.len()`。
    /// GPU 路径对每个 cache 启动一次 `double_perlin_sample_f64` kernel（已与 CPU 逐位一致），
    /// 不可用时回退到 CPU 上的 `DoublePerlinNoiseSampler` 等价计算。
    pub fn batch_fill_cell_caches_vanilla(
        &self,
        positions: &[f64],
        specs: &[CellCacheFillSpec<'_>],
        results: &mut [f64],
    ) {
        let n = positions.len() / 3;
        debug_assert_eq!(positions.len(), n * 3);
        debug_assert_eq!(results.len(), n * specs.len());

        for (cache_index, spec) in specs.iter().enumerate() {
            let out = &mut results[cache_index * n..(cache_index + 1) * n];

            // 应用 NoiseData 缩放（与 vanilla `Noise.compute` 一致）
            let mut scaled = Vec::with_capacity(positions.len());
            for i in 0..n {
                scaled.push(positions[i * 3] * spec.xz_scale);
                scaled.push(positions[i * 3 + 1] * spec.y_scale);
                scaled.push(positions[i * 3 + 2] * spec.xz_scale);
            }

            #[cfg(feature = "gpu")]
            {
                // 优先 JIT 特化路径（配置开启时八度参数烘焙为常量）。
                // `sample_double_perlin_jit` 内部在 JIT 不可用时自动回退到标准 batch kernel。
                let jit_ok = self.with_noise_sampler(|sampler| {
                    sampler
                        .sample_double_perlin_jit(
                            spec.first,
                            spec.second,
                            spec.amplitude,
                            &scaled,
                            out,
                        )
                        .is_ok()
                }) == Some(true);
                if jit_ok {
                    continue;
                }
                // JIT 完全不可用（如设备为 CPU）→ 标准 batch kernel。
                if self.with_noise_sampler(|sampler| {
                    sampler
                        .sample_double_perlin_batch(
                            spec.first,
                            spec.second,
                            spec.amplitude,
                            &scaled,
                            out,
                        )
                        .is_ok()
                }) == Some(true)
                {
                    continue;
                }
            }

            // CPU fallback：与 `DoublePerlinNoiseSampler::sample` 逐位一致
            let c = 1.0181268882175227f64;
            for (i, res) in out.iter_mut().enumerate() {
                let x = scaled[i * 3];
                let y = scaled[i * 3 + 1];
                let z = scaled[i * 3 + 2];
                *res = (spec.first.sample(x, y, z) + spec.second.sample(x * c, y * c, z * c))
                    * spec.amplitude;
            }
        }
    }

    // --------------------------------------------------------------------------
    // Aquifer
    // --------------------------------------------------------------------------

    /// 批量含水层判定。
    ///
    /// GPU 路径通过 `aquifer_batch_f64` kernel 并行计算，
    /// 失败或不可用时回退到 4-NN 搜索。
    #[allow(clippy::must_use_candidate)]
    pub fn batch_aquifer_apply(
        &self,
        positions: &[f64],
        densities: &[f64],
        packed_grid: &[i64],
        fluid_level: f64,
        barrier_scale: f64,
    ) -> AquiferBatchResult {
        #[cfg(feature = "gpu")]
        {
            if let Some(Ok(aquifer_result)) = self.with_aquifer_sampler(|sampler| {
                sampler.batch_aquifer_apply(
                    positions,
                    densities,
                    packed_grid,
                    fluid_level,
                    barrier_scale,
                )
            }) {
                return aquifer_result;
            }
        }
        // CPU fallback: 4-NN 搜索
        cpu_aquifer_apply(
            positions,
            densities,
            packed_grid,
            fluid_level,
            barrier_scale,
        )
    }

    // --------------------------------------------------------------------------
    // Beardifier
    // --------------------------------------------------------------------------

    /// 批量 Beardifier 地形适应计算（与 vanilla `Beardifier::sample` 逐位一致）。
    ///
    /// `affected_box` 为 `[min_x, min_y, min_z, max_x, max_y, max_z]`（含边界），
    /// 盒外位置输出 0。
    /// GPU 路径通过 `beardifier_batch_f64` kernel 并行计算，
    /// 失败或不可用时回退到 CPU 上的 vanilla 等价计算。
    pub fn batch_beardifier(
        &self,
        positions: &[f64],
        structures: &[BeardifierStructureData],
        junctions: &[BeardifierJunctionData],
        affected_box: [i32; 6],
        results: &mut [f64],
    ) {
        #[cfg(feature = "gpu")]
        if self.with_beardifier_sampler(|sampler| {
            sampler
                .batch_beardifier(positions, structures, junctions, affected_box, results)
                .is_ok()
        }) == Some(true)
        {
            return;
        }
        // CPU fallback
        cpu_beardifier(positions, structures, junctions, affected_box, results);
    }

    // --------------------------------------------------------------------------
    // Trilinear Interpolation
    // --------------------------------------------------------------------------

    /// 批量三线性插值。
    ///
    /// 对 n 组 8 角点 + 3 delta 执行三线性插值。
    /// GPU 路径通过 `trilinear_interpolate_f64` kernel 并行计算，
    /// 失败或不可用时回退到 CPU 路径。
    pub fn batch_trilinear(&self, corners: &[f64], deltas: &[f64], results: &mut [f64]) {
        #[cfg(feature = "gpu")]
        if self
            .with_noise_sampler(|sampler| sampler.batch_trilinear(corners, deltas, results).is_ok())
            == Some(true)
        {
            return;
        }
        // CPU fallback: 标准三线性插值
        tracing::debug!("GPU trilinear failed — using CPU fallback");
        cpu_trilinear_impl(corners, deltas, results);
    }
}

// ============================================================================
// CPU Fallbacks
// ============================================================================

/// CPU 回退：含水层 4-NN 搜索。
///
/// 对每个查询位置在打包网格中查找 4 个最近邻，
/// 计算屏障密度并确定流体状态。
fn cpu_aquifer_apply(
    positions: &[f64],
    densities: &[f64],
    packed_grid: &[i64],
    fluid_level: f64,
    barrier_scale: f64,
) -> AquiferBatchResult {
    let n = densities.len();
    let mut block_ids = vec![0i32; n];
    let mut fluid_updates = vec![0u8; n];

    let m = packed_grid.len() / 4;
    if m < 4 {
        return AquiferBatchResult {
            block_ids,
            fluid_updates,
        };
    }

    // 预提取网格位置和密度
    let grid_positions: Vec<[f64; 3]> = (0..m)
        .map(|i| {
            [
                f64::from_bits(packed_grid[i * 4] as u64),
                f64::from_bits(packed_grid[i * 4 + 1] as u64),
                f64::from_bits(packed_grid[i * 4 + 2] as u64),
            ]
        })
        .collect();
    let grid_densities: Vec<f64> = (0..m)
        .map(|i| f64::from_bits(packed_grid[i * 4 + 3] as u64))
        .collect();

    for i in 0..n {
        let qx = positions[i * 3];
        let qy = positions[i * 3 + 1];
        let qz = positions[i * 3 + 2];
        let q_density = densities[i];

        // 4-NN 线性搜索
        let mut best_idx = [0usize; 4];
        let mut best_dist = [f64::INFINITY; 4];

        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            let dx = qx - grid_positions[j][0];
            let dy = qy - grid_positions[j][1];
            let dz = qz - grid_positions[j][2];
            let dist = dx * dx + dy * dy + dz * dz;

            for k in 0..4 {
                if dist < best_dist[k] {
                    for kk in (k + 1..4).rev() {
                        best_idx[kk] = best_idx[kk - 1];
                        best_dist[kk] = best_dist[kk - 1];
                    }
                    best_idx[k] = j;
                    best_dist[k] = dist;
                    break;
                }
            }
        }

        let barrier_sum: f64 = best_idx.iter().map(|&idx| grid_densities[idx]).sum();
        let barrier_density = barrier_sum / 4.0;
        let effective = q_density + barrier_density * barrier_scale;

        if effective > 0.0 {
            block_ids[i] = 1; // 石头
            fluid_updates[i] = 0;
        } else if qy < fluid_level {
            block_ids[i] = 2; // 水
            fluid_updates[i] = 1;
        } else {
            block_ids[i] = 0; // 空气
            fluid_updates[i] = 0;
        }
    }

    AquiferBatchResult {
        block_ids,
        fluid_updates,
    }
}

/// CPU 回退：Beardifier 遍历结构与连接点。
///
/// 简化为对每个位置计算到结构包围盒的距离贡献。
/// 与 vanilla `Beardifier::get_beard_contribution` 逐位一致的参考实现。
///
/// 核表值 `exp(-(dx² + (dy+0.5)² + dz²)/16)` 与 vanilla 的预计算 24³ 表逐位一致
/// （同一公式、同一 IEEE 运算），因此无需拷贝表。
fn cpu_beard_contrib(dx: i32, dy: i32, dz: i32, y_to_ground: i32) -> f64 {
    let xi = dx + 12;
    let yi = dy + 12;
    let zi = dz + 12;
    if (0..24).contains(&xi) && (0..24).contains(&yi) && (0..24).contains(&zi) {
        let dy_off = f64::from(y_to_ground) + 0.5;
        let dsq = f64::from(dx).powi(2) + dy_off.powi(2) + f64::from(dz).powi(2);
        let value = -dy_off * (dsq / 2.0).sqrt().recip() / 2.0;
        let kdsq = f64::from(dx).powi(2) + (f64::from(dy) + 0.5).powi(2) + f64::from(dz).powi(2);
        value * std::f64::consts::E.powf(-kdsq / 16.0)
    } else {
        0.0
    }
}

/// 与 vanilla `Beardifier::get_bury_contribution` 逐位一致。
fn cpu_bury_contrib(dx: f64, dy: f64, dz: f64) -> f64 {
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    if distance < 0.0 {
        1.0
    } else if distance > 6.0 {
        0.0
    } else {
        1.0 - distance / 6.0
    }
}

/// CPU 回退：vanilla `Beardifier::sample` 的逐位等价批量实现。
fn cpu_beardifier(
    positions: &[f64],
    structures: &[BeardifierStructureData],
    junctions: &[BeardifierJunctionData],
    affected_box: [i32; 6],
    results: &mut [f64],
) {
    let [aminx, aminy, aminz, amaxx, amaxy, amaxz] = affected_box;
    for i in 0..results.len() {
        let x = positions[i * 3] as i32;
        let y = positions[i * 3 + 1] as i32;
        let z = positions[i * 3 + 2] as i32;

        if x < aminx || x > amaxx || y < aminy || y > amaxy || z < aminz || z > amaxz {
            results[i] = 0.0;
            continue;
        }

        let mut weight = 0.0;

        for s in structures {
            let bminx = s.box_min_x;
            let bminy = s.box_min_y;
            let bminz = s.box_min_z;
            let bmaxx = s.box_max_x;
            let bmaxy = s.box_max_y;
            let bmaxz = s.box_max_z;

            let dx = 0.max((bminx - x).max(x - bmaxx));
            let dz = 0.max((bminz - z).max(z - bmaxz));
            let ground_y = bminy + s.ground_delta;
            let dy_to_ground = y - ground_y;

            let dy = match s.adaptation {
                0 => 0,                                    // None
                1 | 3 => dy_to_ground,                     // BeardThin / Bury
                2 => 0.max((ground_y - y).max(y - bmaxy)), // BeardBox
                _ => 0.max((bminy - y).max(y - bmaxy)),    // Encapsulate
            };

            let contrib = match s.adaptation {
                0 => 0.0,
                3 => cpu_bury_contrib(f64::from(dx), f64::from(dy) / 2.0, f64::from(dz)),
                1 | 2 => cpu_beard_contrib(dx, dy, dz, dy_to_ground) * 0.8,
                _ => {
                    cpu_bury_contrib(
                        f64::from(dx) / 2.0,
                        f64::from(dy) / 2.0,
                        f64::from(dz) / 2.0,
                    ) * 0.8
                }
            };
            weight += contrib;
        }

        for j in junctions {
            let jdx = x - j.x;
            let jdy = y - j.ground_y;
            let jdz = z - j.z;
            weight += cpu_beard_contrib(jdx, jdy, jdz, jdy) * 0.4;
        }

        results[i] = weight;
    }
}

// ============================================================================
// Trilinear CPU fallback
// ============================================================================

/// CPU fallback: 标准三线性插值（与 GPU `trilinear_interpolate_f64` kernel 等价）。
///
/// 对 n 组 8 角点 + 3 delta 执行标准三线性插值。
fn cpu_trilinear_impl(corners: &[f64], deltas: &[f64], results: &mut [f64]) {
    let n = results.len();
    for i in 0..n {
        let b = i * 8;
        let dx = deltas[i * 3];
        let dy = deltas[i * 3 + 1];
        let dz = deltas[i * 3 + 2];
        results[i] = corners[b] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + corners[b + 1] * dx * (1.0 - dy) * (1.0 - dz)
            + corners[b + 2] * (1.0 - dx) * dy * (1.0 - dz)
            + corners[b + 3] * dx * dy * (1.0 - dz)
            + corners[b + 4] * (1.0 - dx) * (1.0 - dy) * dz
            + corners[b + 5] * dx * (1.0 - dy) * dz
            + corners[b + 6] * (1.0 - dx) * dy * dz
            + corners[b + 7] * dx * dy * dz;
    }
}
