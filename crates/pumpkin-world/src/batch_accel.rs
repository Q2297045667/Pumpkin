//! 批量加速接口 — Cell Cache、Aquifer、Beardifier、Vein 的 GPU 批量处理。
#![allow(clippy::doc_markdown)]

use pumpkin_config::gpu::GpuConfig;
#[cfg(feature = "gpu")]
use pumpkin_gpu::{
    GpuDevice,
    noise::{
        GpuNoiseSampler,
        batch_cell::{
            AquiferBatchResult, BeardifierJunctionData, BeardifierStructureData, CellFillParams,
            GpuAquiferBatchSampler, GpuBeardifierBatchSampler, GpuCellBatchSampler,
            GpuVeinBatchSampler, VeinParams,
        },
    },
};

/// 批量加速器 — 为 Cell Cache、Aquifer、Beardifier、Vein 提供 GPU 批量采样。
pub struct BatchAccelerator {
    config: GpuConfig,
    /// 缓存的 GPU 设备（懒初始化），通过 Mutex 包装以提供 Sync。
    #[cfg(feature = "gpu")]
    cached_device: std::sync::Mutex<Option<GpuDevice>>,
    /// 持久化 Cell Cache 采样器（复用 NoiseCache + buffer 池）。
    #[cfg(feature = "gpu")]
    cell_sampler: std::sync::Mutex<Option<GpuCellBatchSampler>>,
    /// 持久化 Noise 采样器。
    #[cfg(feature = "gpu")]
    noise_sampler: std::sync::Mutex<Option<GpuNoiseSampler>>,
    /// 持久化 Beardifier 采样器（含 beard kernel GPU buffer）。
    #[cfg(feature = "gpu")]
    beardifier_sampler: std::sync::Mutex<Option<GpuBeardifierBatchSampler>>,
    /// 持久化 Vein 采样器。
    #[cfg(feature = "gpu")]
    vein_sampler: std::sync::Mutex<Option<GpuVeinBatchSampler>>,
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
            cell_sampler: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            noise_sampler: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            beardifier_sampler: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            vein_sampler: std::sync::Mutex::new(None),
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

    /// 懒初始化 Cell Cache 采样器并执行操作。
    #[cfg(feature = "gpu")]
    fn with_cell_sampler<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut GpuCellBatchSampler) -> R,
    {
        let mut guard = self.cell_sampler.lock().ok()?;
        if guard.is_none() {
            let device = self.ensure_device()?;
            *guard = Some(GpuCellBatchSampler::new(device));
        }
        guard.as_mut().map(f)
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

    /// 懒初始化 Vein 采样器并执行操作。
    #[cfg(feature = "gpu")]
    fn with_vein_sampler<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut GpuVeinBatchSampler) -> R,
    {
        let mut guard = self.vein_sampler.lock().ok()?;
        if guard.is_none() {
            let device = self.ensure_device()?;
            *guard = Some(GpuVeinBatchSampler::new(device));
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
    // Cell Cache
    // --------------------------------------------------------------------------

    /// 批量填充 Cell Cache。
    ///
    /// GPU 路径通过 `cell_cache_fill_f64` kernel 并行计算，
    /// 失败或不可用时回退到零填充。
    pub fn batch_fill_cell_caches(
        &self,
        positions: &[f64],
        params: &CellFillParams,
        results: &mut [f64],
    ) {
        #[cfg(feature = "gpu")]
        if self.with_cell_sampler(|sampler| {
            sampler
                .batch_fill_cell_caches(positions, params, results)
                .is_ok()
        }) == Some(true)
        {
            return;
        }
        // CPU fallback
        tracing::debug!("GPU cell cache fill failed — using CPU fallback");
        cpu_cell_cache_fill_impl(positions, params, results);
    }

    /// 批量填充插值器缓冲区。
    ///
    /// GPU 路径通过 `interpolator_fill_f64` kernel 并行计算，
    /// 失败或不可用时回退到零填充。
    pub fn batch_fill_interpolators(
        &self,
        positions: &[f64],
        params: &CellFillParams,
        results: &mut [f64],
    ) {
        #[cfg(feature = "gpu")]
        if self.with_cell_sampler(|sampler| {
            sampler
                .batch_fill_interpolators(positions, params, results)
                .is_ok()
        }) == Some(true)
        {
            return;
        }
        // CPU fallback
        tracing::debug!("GPU interpolator fill failed — using CPU fallback");
        cpu_interpolator_fill_impl(positions, params, results);
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

    /// 批量 Beardifier 地形适应计算。
    ///
    /// GPU 路径通过 `beardifier_batch_f64` kernel 并行计算，
    /// 失败或不可用时回退到 CPU 遍历结构与连接点。
    pub fn batch_beardifier(
        &self,
        positions: &[f64],
        structures: &[BeardifierStructureData],
        junctions: &[BeardifierJunctionData],
        results: &mut [f64],
    ) {
        #[cfg(feature = "gpu")]
        if self.with_beardifier_sampler(|sampler| {
            sampler
                .batch_beardifier(positions, structures, junctions, results)
                .is_ok()
        }) == Some(true)
        {
            return;
        }
        // CPU fallback
        cpu_beardifier(positions, structures, junctions, results);
    }

    // --------------------------------------------------------------------------
    // Vein
    // --------------------------------------------------------------------------

    /// 批量矿脉判定。
    ///
    /// GPU 路径通过 `vein_batch_f64` kernel 并行计算，
    /// 失败或不可用时回退到默认值（无矿脉）。
    pub fn batch_vein_sample(&self, positions: &[f64], params: &VeinParams, results: &mut [i32]) {
        #[cfg(feature = "gpu")]
        if self.with_vein_sampler(|sampler| {
            sampler
                .batch_vein_sample(positions, params, results)
                .is_ok()
        }) == Some(true)
        {
            return;
        }
        // CPU fallback
        tracing::debug!("GPU vein sample failed — using CPU fallback");
        cpu_vein_detect(positions, params, results);
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
fn cpu_beardifier(
    positions: &[f64],
    structures: &[BeardifierStructureData],
    junctions: &[BeardifierJunctionData],
    results: &mut [f64],
) {
    for i in 0..results.len() {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        let mut beard = 0.0;

        // 结构贡献：距离反比衰减
        for s in structures {
            let cx = s.center_x;
            let cy = s.center_y;
            let cz = s.center_z;
            let rx = s.radius_x + 1.0;
            let ry = s.radius_y + 1.0;
            let rz = s.radius_z + 1.0;

            if rx <= 0.0 || ry <= 0.0 || rz <= 0.0 {
                continue;
            }

            let dx = (x - cx) / rx;
            let dy = (y - cy) / ry;
            let dz = (z - cz) / rz;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            // 仅包围盒内（归一化距离 < 1）才贡献
            if dist_sq < 1.0 {
                let contrib = (1.0 - dist_sq.sqrt()).max(0.0);
                let y_factor = if s.ground_delta_y > 0.0 {
                    ((y - s.min_y) / s.ground_delta_y).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                beard += contrib * y_factor * 0.5;
            }
        }

        // 连接点贡献：固定半径高斯衰减
        for j in junctions {
            let dx = x - f64::from(j.x);
            let dy = y - f64::from(j.ground_y);
            let dz = z - f64::from(j.z);
            let jr: f64 = 12.0;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let norm = dist_sq / (jr * jr);
            if norm < 1.0 {
                beard += (1.0 - norm) * 0.25;
            }
        }

        results[i] = beard;
    }
}

/// CPU 矿脉检测回退：使用 toggle/ridged/gap 三重 perlin 噪声进行矿脉判定。
///
/// 算法与 GPU kernel `vein_batch_f64` 一致，参考 `OreVeinSampler::sample` 逻辑：
/// 1. 对每个位置计算三段 perlin 噪声
/// 2. 根据 toggle 符号选择矿脉类型（铜/铁）
/// 3. Y 轴边界检查 + 概率判定 → 矿石 / 粗矿 / 围岩 / 无矿脉
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn cpu_vein_detect(positions: &[f64], params: &VeinParams, results: &mut [i32]) {
    // 矿石类型定义（与 OreVeinSampler 对应）
    #[derive(Clone, Copy)]
    struct VeinTypeCpu {
        min_y: i32,
        max_y: i32,
    }
    const COPPER: VeinTypeCpu = VeinTypeCpu {
        min_y: 0,
        max_y: 50,
    };
    const IRON: VeinTypeCpu = VeinTypeCpu {
        min_y: -60,
        max_y: -8,
    };

    let n = results.len();
    if n == 0 {
        return;
    }

    // 从 VeinParams 解析八度数
    let octaves_toggle = params.toggle_config.len() / 8;
    let octaves_ridged = params.ridged_config.len() / 8;
    let octaves_gap = params.gap_config.len() / 8;

    if octaves_toggle == 0 || octaves_ridged == 0 || octaves_gap == 0 {
        results.fill(0);
        return;
    }

    // 对每个位置执行矿脉判定
    for idx in 0..n {
        let x = positions[idx * 3];
        let y = positions[idx * 3 + 1];
        let z = positions[idx * 3 + 2];

        // 1. 计算 toggle 噪声
        let mut toggle = 0.0f64;
        for o in 0..octaves_toggle {
            let po = o * 8;
            let amp = params.toggle_config[po];
            let lac = params.toggle_config[po + 1];
            let org_x = params.toggle_config[po + 2];
            let org_y = params.toggle_config[po + 3];
            let org_z = params.toggle_config[po + 4];
            let perm = gen_perm_table(0x546F67676C65, o); // "Toggle" seed
            toggle += amp * sample_perlin(&perm, org_x + x * lac, org_y + y * lac, org_z + z * lac);
        }

        // 2. 根据 toggle 符号选择矿脉类型
        let vein_type: VeinTypeCpu = if toggle > 0.0 { COPPER } else { IRON };
        let block_y = y as i32;
        let max_to_y = vein_type.max_y - block_y;
        let y_to_min = block_y - vein_type.min_y;

        // Y 轴边界检查
        if max_to_y < 0 || y_to_min < 0 {
            results[idx] = 0;
            continue;
        }

        // 边界衰减
        let closest_to_bound = max_to_y.min(y_to_min) as f64;
        let mapped_diff = pumpkin_util::math::clamped_map(closest_to_bound, 0.0, 20.0, -0.2, 0.0);
        let abs_toggle = toggle.abs();

        if abs_toggle + mapped_diff < 0.4 {
            results[idx] = 0;
            continue;
        }

        // 3. 计算 ridged 噪声
        let mut ridged = 0.0f64;
        for o in 0..octaves_ridged {
            let po = o * 8;
            let amp = params.ridged_config[po];
            let lac = params.ridged_config[po + 1];
            let org_x = params.ridged_config[po + 2];
            let org_y = params.ridged_config[po + 3];
            let org_z = params.ridged_config[po + 4];
            let perm = gen_perm_table(0x526964676564, o); // "Ridged" seed
            let sample = sample_perlin(&perm, org_x + x * lac, org_y + y * lac, org_z + z * lac);
            ridged += amp * (1.0 - sample.abs());
        }

        // ridged 检查（对应 random.next_f32() <= 0.7 && ridged < 0）
        if ridged >= 0.0 {
            results[idx] = 0;
            continue;
        }

        // 4. 计算 gap 噪声
        let mut gap = 0.0f64;
        for o in 0..octaves_gap {
            let po = o * 8;
            let amp = params.gap_config[po];
            let lac = params.gap_config[po + 1];
            let org_x = params.gap_config[po + 2];
            let org_y = params.gap_config[po + 3];
            let org_z = params.gap_config[po + 4];
            let perm = gen_perm_table(0x476170, o); // "Gap" seed
            gap += amp * sample_perlin(&perm, org_x + x * lac, org_y + y * lac, org_z + z * lac);
        }

        // 概率判定
        let clamped_sample = pumpkin_util::math::clamped_map(abs_toggle, 0.4, 0.6, 0.1, 0.3);
        let pseudo_rand = ((x * 12.9898 + y * 78.233 + z * 45.164).sin() * 43758.5453)
            .fract()
            .abs();

        if pseudo_rand < clamped_sample && gap > -0.3 {
            // 矿石 / 粗矿判定
            let pseudo_rand2 = ((x * 39.346 + y * 11.745 + z * 92.11).sin() * 37523.422)
                .fract()
                .abs();
            if pseudo_rand2 < 0.02 {
                results[idx] = 2; // 粗矿
            } else {
                results[idx] = 1; // 矿石
            }
        } else {
            results[idx] = 3; // 围岩
        }
    }
}

// ============================================================================
// 共享 Perlin 噪声工具
// ============================================================================

/// 生成确定性置换表（每个 octave 一个 256 字节表）。
// Duplicate of pumpkin-gpu/src/noise/batch_cell.rs:gen_perm_table — keep in sync
fn gen_perm_table(seed: u64, octave: usize) -> [u8; 256] {
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

/// 简化版 3D Perlin 噪声采样器（梯度哈希 + 三线性插值）。
fn sample_perlin(perm: &[u8; 256], x: f64, y: f64, z: f64) -> f64 {
    let xi = (x.floor() as i32) & 255;
    let yi = (y.floor() as i32) & 255;
    let zi = (z.floor() as i32) & 255;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0);
    let v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0);
    let w = zf * zf * zf * (zf * (zf * 6.0 - 15.0) + 10.0);

    let a = perm[xi as usize] as usize + yi as usize;
    let aa = perm[a & 255] as usize + zi as usize;
    let ab = perm[(a + 1) & 255] as usize + zi as usize;
    let b = perm[(xi + 1) as usize & 255] as usize + yi as usize;
    let ba = perm[b & 255] as usize + zi as usize;
    let bb = perm[(b + 1) & 255] as usize + zi as usize;

    let g000 = grad_perlin(perm[aa & 255] as usize, xf, yf, zf);
    let g100 = grad_perlin(perm[ba & 255] as usize, xf - 1.0, yf, zf);
    let g010 = grad_perlin(perm[ab & 255] as usize, xf, yf - 1.0, zf);
    let g110 = grad_perlin(perm[bb & 255] as usize, xf - 1.0, yf - 1.0, zf);
    let g001 = grad_perlin(perm[(aa + 1) & 255] as usize, xf, yf, zf - 1.0);
    let g101 = grad_perlin(perm[(ba + 1) & 255] as usize, xf - 1.0, yf, zf - 1.0);
    let g011 = grad_perlin(perm[(ab + 1) & 255] as usize, xf, yf - 1.0, zf - 1.0);
    let g111 = grad_perlin(perm[(bb + 1) & 255] as usize, xf - 1.0, yf - 1.0, zf - 1.0);

    pumpkin_util::math::lerp3(g000, g100, g010, g110, g001, g101, g011, g111, u, v, w)
}

/// Perlin 梯度函数
fn grad_perlin(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if (h & 1) == 0 { u } else { -u }) + (if (h & 2) == 0 { v } else { -v })
}

// ============================================================================
// Cell Cache / Interpolator CPU fallback 实现
// ============================================================================

/// CPU fallback for cell cache fill.
///
/// Parses `CellFillParams` and evaluates perlin noise for each position.
/// Encoding of perlin_configs:
///   For sampler s at offset base_s (cumulative sum of sizes):
///     - 1 f64: num_octaves
///     - num_octaves f64: amplitudes
///     - num_octaves f64: lacunarities
///     - num_octaves × 3 f64: origins (x, y, z per octave)
///   Total per sampler: 1 + num_octaves * 5 f64 values.
fn cpu_cell_cache_fill_impl(positions: &[f64], params: &CellFillParams, results: &mut [f64]) {
    let n = results.len();
    if n == 0 {
        return;
    }

    let total_octaves: i32 = params.num_octaves.iter().sum();
    let expected_config_len = params.num_octaves.len() + (total_octaves * 5) as usize;

    // 配置数据不足 → 零填充
    if params.perlin_configs.is_empty()
        || total_octaves == 0
        || params.perlin_configs.len() < expected_config_len
    {
        results.fill(0.0);
        return;
    }

    // 构建采样器偏移表
    let mut sampler_offsets: Vec<usize> = Vec::with_capacity(params.num_octaves.len());
    let mut offset = 0usize;
    for &no in &params.num_octaves {
        sampler_offsets.push(offset);
        let size = 1 + (no * 5) as usize; // 1 num_octaves + 5 per octave
        offset += size;
    }

    // 为每个采样器生成置换表
    let mut sampler_perms: Vec<Vec<[u8; 256]>> = Vec::with_capacity(params.num_octaves.len());
    for (s_idx, &no) in params.num_octaves.iter().enumerate() {
        let perms: Vec<[u8; 256]> = (0..no as usize)
            .map(|o| gen_perm_table(0x4365_6C6Cu64.wrapping_add(s_idx as u64), o))
            .collect();
        sampler_perms.push(perms);
    }

    for idx in 0..n {
        let x = positions[idx * 3];
        let y = positions[idx * 3 + 1];
        let z = positions[idx * 3 + 2];

        // 使用第一个采样器（与 GPU kernel 的 cell_indices[0] 行为一致）
        let s_idx = 0usize;
        if s_idx >= sampler_offsets.len() {
            results[idx] = 0.0;
            continue;
        }

        let base = sampler_offsets[s_idx];
        let num_octaves = params.perlin_configs[base] as i32;
        if num_octaves <= 0 {
            results[idx] = 0.0;
            continue;
        }

        let amps_start = base + 1;
        let lacs_start = amps_start + num_octaves as usize;
        let orgs_start = lacs_start + num_octaves as usize;

        let mut sum = 0.0f64;
        for o in 0..num_octaves as usize {
            let amp = params.perlin_configs[amps_start + o];
            let lac = params.perlin_configs[lacs_start + o];
            let org_x = params.perlin_configs[orgs_start + o * 3];
            let org_y = params.perlin_configs[orgs_start + o * 3 + 1];
            let org_z = params.perlin_configs[orgs_start + o * 3 + 2];

            if o < sampler_perms[s_idx].len() {
                let perm = &sampler_perms[s_idx][o];
                sum += amp * sample_perlin(perm, org_x + x * lac, org_y + y * lac, org_z + z * lac);
            }
        }
        results[idx] = sum;
    }
}

/// CPU fallback for interpolator fill.
///
/// Parses `CellFillParams` and evaluates perlin noise for each position.
/// Encoding of perlin_configs (8 doubles per octave):
///   [amp, lac, org_x, org_y, org_z, xz_scale, y_scale, _reserved]
/// Concatenated for all octaves of all samplers.
fn cpu_interpolator_fill_impl(positions: &[f64], params: &CellFillParams, results: &mut [f64]) {
    let n = results.len();
    if n == 0 {
        return;
    }

    let total_octaves: i32 = params.num_octaves.iter().sum();
    let expected_config_len = (total_octaves * 8) as usize;

    // 配置数据不足 → 零填充
    if params.perlin_configs.is_empty()
        || total_octaves == 0
        || params.perlin_configs.len() < expected_config_len
    {
        results.fill(0.0);
        return;
    }

    // 为每个采样器生成置换表
    let mut sampler_perms: Vec<Vec<[u8; 256]>> = Vec::with_capacity(params.num_octaves.len());
    for (s_idx, &no) in params.num_octaves.iter().enumerate() {
        let perms: Vec<[u8; 256]> = (0..no as usize)
            .map(|o| gen_perm_table(0x496E_7465_7270u64.wrapping_add(s_idx as u64), o))
            .collect();
        sampler_perms.push(perms);
    }

    // 计算每个采样器的起始偏移
    let mut sampler_offsets: Vec<usize> = Vec::with_capacity(params.num_octaves.len());
    let mut offset = 0usize;
    for &no in &params.num_octaves {
        sampler_offsets.push(offset);
        offset += (no * 8) as usize;
    }

    for idx in 0..n {
        let x = positions[idx * 3];
        let y = positions[idx * 3 + 1];
        let z = positions[idx * 3 + 2];

        // 使用第一个采样器
        let s_idx = 0usize;
        if s_idx >= sampler_offsets.len() {
            results[idx] = 0.0;
            continue;
        }

        let base = sampler_offsets[s_idx];
        let num_octaves = params.num_octaves[s_idx] as usize;
        if num_octaves == 0 {
            results[idx] = 0.0;
            continue;
        }

        let mut sum = 0.0f64;
        for o in 0..num_octaves {
            let bo = base + o * 8;
            let amp = params.perlin_configs[bo];
            let lac = params.perlin_configs[bo + 1];
            let org_x = params.perlin_configs[bo + 2];
            let org_y = params.perlin_configs[bo + 3];
            let org_z = params.perlin_configs[bo + 4];
            let xz_scale = params.perlin_configs[bo + 5];
            let y_scale = params.perlin_configs[bo + 6];

            if o < sampler_perms[s_idx].len() {
                let perm = &sampler_perms[s_idx][o];
                sum += amp
                    * sample_perlin(
                        perm,
                        org_x + x * xz_scale * lac,
                        org_y + y * y_scale * lac,
                        org_z + z * xz_scale * lac,
                    );
            }
        }
        results[idx] = sum;
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
