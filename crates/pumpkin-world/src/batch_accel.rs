//! 批量加速接口 — Cell Cache、Aquifer、Beardifier、Vein 的 GPU 批量处理。
#![allow(clippy::doc_markdown)]

use pumpkin_config::gpu::GpuConfig;
#[cfg(feature = "gpu")]
use pumpkin_gpu::{
    GpuDevice,
    noise::batch_cell::{
        AquiferBatchResult, BeardifierJunctionData, BeardifierStructureData, CellFillParams,
        GpuAquiferBatchSampler, GpuBeardifierBatchSampler, GpuCellBatchSampler,
        GpuVeinBatchSampler, VeinParams,
    },
};

/// 批量加速器 — 为 Cell Cache、Aquifer、Beardifier、Vein 提供 GPU 批量采样，
/// GPU 不可用时自动回退到 CPU。
pub struct BatchAccelerator {
    config: GpuConfig,
}

impl BatchAccelerator {
    /// 从 GPU 配置创建批量加速器。
    ///
    /// 仅在配置同时满足 `enabled` 和
    /// `(noise_acceleration || surface_acceleration || jit_enabled)` 时
    /// 才会在后续调用中尝试初始化 GPU 设备。
    #[must_use]
    pub fn new(config: &GpuConfig) -> Self {
        Self {
            config: config.clone(),
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
                || self.config.surface_acceleration
                || self.config.jit_enabled)
    }

    /// 尝试创建 GPU 设备。
    ///
    /// 仅在配置激活且探测到非 CPU 后端时返回 `Some`。
    #[cfg(feature = "gpu")]
    fn make_device(&self) -> Option<GpuDevice> {
        if !self.is_active() {
            return None;
        }
        let device = GpuDevice::from_config(&self.config);
        (device.device_type() != pumpkin_gpu::DeviceType::Cpu).then_some(device)
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
        if let Some(device) = self.make_device() {
            let mut sampler = GpuCellBatchSampler::new(device);
            if sampler
                .batch_fill_cell_caches(positions, params, results)
                .is_ok()
            {
                return;
            }
        }
        // CPU fallback: 零填充
        results.fill(0.0);
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
        if let Some(device) = self.make_device() {
            let mut sampler = GpuCellBatchSampler::new(device);
            if sampler
                .batch_fill_interpolators(positions, params, results)
                .is_ok()
            {
                return;
            }
        }
        // CPU fallback: 零填充
        results.fill(0.0);
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
        if let Some(device) = self.make_device() {
            let mut sampler = GpuAquiferBatchSampler::new(device);
            if let Ok(result) = sampler.batch_aquifer_apply(
                positions,
                densities,
                packed_grid,
                fluid_level,
                barrier_scale,
            ) {
                return result;
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
        if let Some(device) = self.make_device() {
            let mut sampler = GpuBeardifierBatchSampler::new(device);
            if sampler
                .batch_beardifier(positions, structures, junctions, results)
                .is_ok()
            {
                return;
            }
        }
        // CPU fallback: 遍历结构/连接点
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
        if let Some(device) = self.make_device() {
            let mut sampler = GpuVeinBatchSampler::new(device);
            if sampler
                .batch_vein_sample(positions, params, results)
                .is_ok()
            {
                return;
            }
        }
        // CPU fallback: 无矿脉
        results.fill(0);
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
            let cx = f64::from(s.min_x + s.max_x) * 0.5;
            let cy = f64::from(s.min_y + s.max_y) * 0.5;
            let cz = f64::from(s.min_z + s.max_z) * 0.5;
            let rx = f64::from(s.max_x - s.min_x).abs() * 0.5 + 1.0;
            let ry = f64::from(s.max_y - s.min_y).abs() * 0.5 + 1.0;
            let rz = f64::from(s.max_z - s.min_z).abs() * 0.5 + 1.0;

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
                let y_factor = if s.ground_level_delta > 0 {
                    ((y - f64::from(s.min_y)) / f64::from(s.ground_level_delta)).clamp(0.0, 1.0)
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
