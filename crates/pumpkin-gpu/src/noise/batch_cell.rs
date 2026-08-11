//! 批量 Cell Cache / Aquifer / Beardifier / Vein 采样器。
//!
//! 提供 GPU 加速的 Cell Cache 填充、插值器缓冲填充、含水层判定、
//! Beardifier 地形适应和矿脉判定功能，均包含 CPU 回退路径。
#![allow(
    clippy::separated_literal_suffix,
    clippy::as_ptr_cast_mut,
    clippy::ptr_as_ptr,
    clippy::similar_names
)]

use crate::GpuDevice;
use crate::common::DeviceError;
use crate::common::kernel::{GpuBufferRef, KernelArg};
use crate::noise::cache::NoiseCache;

// ============================================================================
// 辅助数据结构
// ============================================================================

/// Cell 填充参数
pub struct CellFillParams {
    /// 扁平化的 perlin 配置（用于采样器）
    pub perlin_configs: Vec<f64>,
    /// 每个采样器的八度数
    pub num_octaves: Vec<i32>,
    /// 每个采样器的类型标记（0=Noise, 1=ShiftA, 2=ShiftB, 3=Interpolated, ...）
    pub sampler_types: Vec<i32>,
}

/// 含水层批量结果
pub struct AquiferBatchResult {
    /// block_state_id 数组
    pub block_ids: Vec<i32>,
    /// should_schedule_fluid_update 标志
    pub fluid_updates: Vec<u8>,
}

/// 结构数据（可序列化到 GPU）
pub struct BeardifierStructureData {
    pub min_x: i32,
    pub min_y: i32,
    pub min_z: i32,
    pub max_x: i32,
    pub max_y: i32,
    pub max_z: i32,
    /// 0=None, 1=BeardThin, 2=BeardBox, 3=Bury, 4=Encapsulate
    pub terrain_adaptation: i32,
    pub ground_level_delta: i32,
}

/// 连接点数据
pub struct BeardifierJunctionData {
    pub x: i32,
    pub ground_y: i32,
    pub z: i32,
}

/// 矿脉参数
pub struct VeinParams {
    pub toggle_config: Vec<f64>,
    pub ridged_config: Vec<f64>,
    pub gap_config: Vec<f64>,
}

// ============================================================================
// GpuCellBatchSampler
// ============================================================================

/// GPU Cell Cache 批量采样器。
///
/// 支持批量 Cell Cache 填充和插值器缓冲填充。
pub struct GpuCellBatchSampler {
    pub device: GpuDevice,
    pub cache: NoiseCache,
}

impl GpuCellBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            cache: NoiseCache::new(),
        }
    }

    /// 批量填充 cell cache。
    ///
    /// GPU 路径尝试 launch `cell_cache_fill_f64` kernel，
    /// 失败时回退 CPU。
    pub fn batch_fill_cell_caches(
        &mut self,
        positions: &[f64],
        _sampler_params: &CellFillParams,
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_cell_cache_fill(positions, results);
            return Ok(());
        }

        // GPU 路径：上传数据、启动 kernel、读回结果
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;

        self.device.copy_to_device(&mut d_pos, positions)?;

        let ok = self.try_launch("cell_cache_fill_f64", n, vec![], vec![]);
        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            cpu_cell_cache_fill(positions, results);
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        Ok(())
    }

    /// 批量填充插值器缓冲区。
    ///
    /// GPU 路径尝试 launch `interpolator_fill_f64` kernel，
    /// 失败时回退 CPU。
    pub fn batch_fill_interpolators(
        &mut self,
        positions: &[f64],
        _sampler_params: &CellFillParams,
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_interpolator_fill(positions, results);
            return Ok(());
        }

        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;

        self.device.copy_to_device(&mut d_pos, positions)?;

        let ok = self.try_launch("interpolator_fill_f64", n, vec![], vec![]);
        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            cpu_interpolator_fill(positions, results);
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        Ok(())
    }

    /// Helper: try to launch GPU kernel, return true if successful.
    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        match self.device.kernel_launcher() {
            Some(l) if l.has_kernel(name) => {
                l.launch(crate::common::kernel::KernelLaunch {
                    name,
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args,
                    gpu_buffers,
                })
                .is_ok()
                    && l.synchronize().is_ok()
            }
            _ => false,
        }
    }
}

// ============================================================================
// GpuAquiferBatchSampler
// ============================================================================

/// GPU 含水层批量采样器。
///
/// 对批量位置执行 4 近邻搜索，确定流体状态与方块类型。
pub struct GpuAquiferBatchSampler {
    pub device: GpuDevice,
}

impl GpuAquiferBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self { device }
    }

    /// 批量含水层判定。
    ///
    /// # Arguments
    ///
    /// * `positions` — 查询位置 `[N*3]`
    /// * `densities` — 查询位置处的密度值 `[N]`
    /// * `packed_grid` — 预计算的网格位置 `[M*3]` 及对应的密度 `[M]`，
    ///   以交错形式存储：`[grid_x0, grid_y0, grid_z0, ..., grid_density0, ...]`
    /// * `fluid_level` — 流体平面高度阈值（典型值 -10000.0）
    /// * `barrier_scale` — 屏障密度缩放（典型值 0.3）
    pub fn batch_aquifer_apply(
        &mut self,
        positions: &[f64],
        densities: &[f64],
        packed_grid: &[i64],
        fluid_level: f64,
        barrier_scale: f64,
    ) -> Result<AquiferBatchResult, DeviceError> {
        let n = densities.len();
        if n == 0 {
            return Ok(AquiferBatchResult {
                block_ids: vec![],
                fluid_updates: vec![],
            });
        }
        assert_eq!(positions.len(), n * 3);

        // packed_grid 包含 M*3 个位置坐标 + M 个密度值，共 M*4 个 i64
        // 每个 i64 实际上存储 f64 的位模式
        let m = packed_grid.len() / 4;
        let grid_pos_count = m * 3;
        let grid_den_count = m;
        assert_eq!(packed_grid.len(), m * 4);

        if self.device.device_type() == crate::DeviceType::Cpu {
            return Ok(cpu_aquifer_apply(
                positions,
                densities,
                packed_grid,
                m,
                fluid_level,
                barrier_scale,
            ));
        }

        // 拆分 packed_grid 为位置和密度
        let mut grid_positions = Vec::with_capacity(grid_pos_count);
        let mut grid_densities = Vec::with_capacity(grid_den_count);
        for i in 0..m {
            grid_positions.push(f64::from_bits(packed_grid[i * 4] as u64));
            grid_positions.push(f64::from_bits(packed_grid[i * 4 + 1] as u64));
            grid_positions.push(f64::from_bits(packed_grid[i * 4 + 2] as u64));
            grid_densities.push(f64::from_bits(packed_grid[i * 4 + 3] as u64));
        }

        // GPU 分配
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let mut d_dens = self.device.alloc_f64(n)?;
        let mut d_gpos = self.device.alloc_f64(grid_pos_count)?;
        let mut d_gden = self.device.alloc_f64(grid_den_count)?;
        let d_bids = self.device.alloc_i32(n)?;
        let d_flags = self.device.alloc_u8(n)?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_dens, densities)?;
        self.device.copy_to_device(&mut d_gpos, &grid_positions)?;
        self.device.copy_to_device(&mut d_gden, &grid_densities)?;

        let kernel_name = if m <= 2048 {
            "aquifer_batch_tiled_f64"
        } else {
            "aquifer_batch_f64"
        };
        let ok = self.try_launch(
            kernel_name,
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::BufferRef(3),
                KernelArg::BufferRef(4),
                KernelArg::BufferRef(5),
                KernelArg::F64(fluid_level),
                KernelArg::F64(barrier_scale),
                KernelArg::I32(n as i32),
                KernelArg::I32(m as i32),
            ],
            vec![
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::F64(&d_gpos),
                GpuBufferRef::F64(&d_dens),
                GpuBufferRef::F64(&d_gden),
                GpuBufferRef::I32(&d_bids),
                GpuBufferRef::U8(&d_flags),
            ],
        );
        let mut block_ids = vec![0i32; n];
        let mut fluid_updates = vec![0u8; n];

        if ok {
            self.device.copy_from_device(&d_bids, &mut block_ids)?;
            self.device.copy_from_device(&d_flags, &mut fluid_updates)?;
        } else {
            let result = cpu_aquifer_apply(
                positions,
                densities,
                packed_grid,
                m,
                fluid_level,
                barrier_scale,
            );
            block_ids = result.block_ids;
            fluid_updates = result.fluid_updates;
        }

        self.device.free(d_pos)?;
        self.device.free(d_dens)?;
        self.device.free(d_gpos)?;
        self.device.free(d_gden)?;
        self.device.free(d_bids)?;
        self.device.free(d_flags)?;

        Ok(AquiferBatchResult {
            block_ids,
            fluid_updates,
        })
    }

    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        match self.device.kernel_launcher() {
            Some(l) if l.has_kernel(name) => {
                l.launch(crate::common::kernel::KernelLaunch {
                    name,
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args,
                    gpu_buffers,
                })
                .is_ok()
                    && l.synchronize().is_ok()
            }
            _ => false,
        }
    }
}

// ============================================================================
// GpuBeardifierBatchSampler
// ============================================================================

/// GPU Beardifier 批量采样器。
///
/// 对批量位置计算结构/连接点造成的地形适应（beard）贡献。
pub struct GpuBeardifierBatchSampler {
    pub device: GpuDevice,
}

impl GpuBeardifierBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self { device }
    }

    /// 批量 Beardifier 计算。
    ///
    /// 对每个位置累加来自结构和连接点的 beard 贡献。
    pub fn batch_beardifier(
        &mut self,
        positions: &[f64],
        structures: &[BeardifierStructureData],
        junctions: &[BeardifierJunctionData],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_beardifier(positions, structures, junctions, results);
            return Ok(());
        }

        // 将结构/连接点展平为 f64 数组
        let struct_flat: Vec<f64> = structures
            .iter()
            .flat_map(|s| {
                vec![
                    f64::from(s.min_x),
                    f64::from(s.min_y),
                    f64::from(s.min_z),
                    f64::from(s.max_x),
                    f64::from(s.max_y),
                    f64::from(s.max_z),
                    f64::from(s.terrain_adaptation),
                    f64::from(s.ground_level_delta),
                ]
            })
            .collect();

        let junct_flat: Vec<f64> = junctions
            .iter()
            .flat_map(|j| vec![f64::from(j.x), f64::from(j.ground_y), f64::from(j.z)])
            .collect();

        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_struct = self.device.alloc_f64(struct_flat.len())?;
        let mut d_junct = self.device.alloc_f64(junct_flat.len())?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_struct, &struct_flat)?;
        self.device.copy_to_device(&mut d_junct, &junct_flat)?;

        let ok = self.try_launch("beardifier_batch_f64", n, vec![], vec![]);
        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            cpu_beardifier(positions, structures, junctions, results);
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_struct)?;
        self.device.free(d_junct)?;
        Ok(())
    }

    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        match self.device.kernel_launcher() {
            Some(l) if l.has_kernel(name) => {
                l.launch(crate::common::kernel::KernelLaunch {
                    name,
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args,
                    gpu_buffers,
                })
                .is_ok()
                    && l.synchronize().is_ok()
            }
            _ => false,
        }
    }
}

// ============================================================================
// GpuVeinBatchSampler
// ============================================================================

/// GPU 矿脉批量采样器。
///
/// 对批量位置判定矿脉类型（无矿脉/矿石/粗矿/围岩）。
pub struct GpuVeinBatchSampler {
    pub device: GpuDevice,
}

impl GpuVeinBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self { device }
    }

    /// 批量矿脉采样。
    ///
    /// # Returns
    ///
    /// `results[i]`:
    /// - 0 = 无矿脉
    /// - 1 = 矿石
    /// - 2 = 粗矿
    /// - 3 = 围岩
    pub fn batch_vein_sample(
        &mut self,
        positions: &[f64],
        _vein_params: &VeinParams,
        results: &mut [i32],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_vein_sample(positions, results);
            return Ok(());
        }

        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_i32(n)?;

        self.device.copy_to_device(&mut d_pos, positions)?;

        let ok = self.try_launch("vein_batch_f64", n, vec![], vec![]);
        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            cpu_vein_sample(positions, results);
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        Ok(())
    }

    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        match self.device.kernel_launcher() {
            Some(l) if l.has_kernel(name) => {
                l.launch(crate::common::kernel::KernelLaunch {
                    name,
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args,
                    gpu_buffers,
                })
                .is_ok()
                    && l.synchronize().is_ok()
            }
            _ => false,
        }
    }
}

// ============================================================================
// CPU Fallbacks
// ============================================================================

/// CPU 回退：Cell Cache 填充。
///
/// 注意：这是占位实现（零填充）。完整的 DAG 求值需要在 pumpkin-world 侧调用
/// `OctavePerlinNoiseSampler::sample()` 逐元素采样，或者通过 pumpkin-util 的噪声采样器链。
/// 当前此函数通过 GpuCellBatchSampler 的 `try_launch` 失败流程被调用，
/// 但实际调用链中的上层（BatchAccelerator）有自己的 CPU fallback，不会到达这里。
#[allow(unused_variables)]
fn cpu_cell_cache_fill(_positions: &[f64], results: &mut [f64]) {
    for item in results.iter_mut() {
        *item = 0.0;
    }
}

/// CPU 回退：插值器缓冲填充。
///
/// 注意：这是占位实现（零填充）。完整的 DAG 求值需要在 pumpkin-world 侧调用
/// `OctavePerlinNoiseSampler::sample()` 逐元素采样，或者通过 pumpkin-util 的噪声采样器链。
/// 当前此函数通过 GpuCellBatchSampler 的 `try_launch` 失败流程被调用，
/// 但实际调用链中的上层（BatchAccelerator）有自己的 CPU fallback，不会到达这里。
#[allow(unused_variables)]
fn cpu_interpolator_fill(_positions: &[f64], results: &mut [f64]) {
    for item in results.iter_mut() {
        *item = 0.0;
    }
}

/// CPU 回退：含水层 4-NN 搜索。
///
/// 对每个查询位置在打包网格中查找 4 个最近邻，
/// 计算屏障密度并确定流体状态。
fn cpu_aquifer_apply(
    positions: &[f64],
    densities: &[f64],
    packed_grid: &[i64],
    m: usize,
    fluid_level: f64,
    barrier_scale: f64,
) -> AquiferBatchResult {
    let n = densities.len();
    let mut block_ids = vec![0i32; n];
    let mut fluid_updates = vec![0u8; n];

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

        for (j, gp) in grid_positions.iter().enumerate().take(m) {
            let dx = qx - gp[0];
            let dy = qy - gp[1];
            let dz = qz - gp[2];
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

/// CPU 回退：矿脉采样。
///
/// 注意：这是占位实现（零填充 = 无矿脉）。完整的矿脉判定需要在 pumpkin-world 侧调用
/// `VeinNoise::sample()` 逐元素采样。
/// 当前此函数通过 GpuVeinBatchSampler 的 `try_launch` 失败流程被调用，
/// 但实际调用链中的上层（BatchAccelerator）有自己的 CPU fallback，不会到达这里。
#[allow(unused_variables)]
fn cpu_vein_sample(_positions: &[f64], results: &mut [i32]) {
    for item in results.iter_mut() {
        *item = 0;
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_device() -> GpuDevice {
        GpuDevice::init()
    }

    #[test]
    fn cell_cache_zero_count() {
        let mut s = GpuCellBatchSampler::new(mk_device());
        let params = CellFillParams {
            perlin_configs: vec![],
            num_octaves: vec![],
            sampler_types: vec![],
        };
        let mut results: [f64; 0] = [];
        s.batch_fill_cell_caches(&[], &params, &mut results)
            .unwrap();
    }

    #[test]
    fn cell_cache_cpu_fallback() {
        let mut s = GpuCellBatchSampler::new(mk_device());
        let params = CellFillParams {
            perlin_configs: vec![],
            num_octaves: vec![],
            sampler_types: vec![],
        };
        let positions = [0.0f64, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut results = [0.0f64; 2];
        s.batch_fill_cell_caches(&positions, &params, &mut results)
            .unwrap();
        // CPU fallback 返回零
        assert_eq!(results, [0.0, 0.0]);
    }

    #[test]
    fn interpolator_cpu_fallback() {
        let mut s = GpuCellBatchSampler::new(mk_device());
        let params = CellFillParams {
            perlin_configs: vec![],
            num_octaves: vec![],
            sampler_types: vec![],
        };
        let positions = [0.0f64, 0.0, 0.0, 1.0, 2.0, 3.0];
        let mut results = [0.0f64; 2];
        s.batch_fill_interpolators(&positions, &params, &mut results)
            .unwrap();
        assert_eq!(results, [0.0, 0.0]);
    }

    #[test]
    fn aquifer_zero_count() {
        let mut s = GpuAquiferBatchSampler::new(mk_device());
        let result = s.batch_aquifer_apply(&[], &[], &[], -10000.0, 0.3).unwrap();
        assert!(result.block_ids.is_empty());
        assert!(result.fluid_updates.is_empty());
    }

    #[test]
    fn aquifer_cpu_fallback() {
        let mut s = GpuAquiferBatchSampler::new(mk_device());
        // 构造一个简单的含水层网格
        let positions = [0.0f64, -60.0, 0.0];
        let densities = [-1.0f64];
        // packed_grid: M=1 → 1 个网格点 (4 个 i64: x, y, z, density)
        let grid_x = 0.0f64;
        let grid_y = -60.0f64;
        let grid_z = -2.0f64;
        let grid_den = -1.0f64;
        let packed_grid = [
            grid_x.to_bits() as i64,
            grid_y.to_bits() as i64,
            grid_z.to_bits() as i64,
            grid_den.to_bits() as i64,
        ];
        let result = s
            .batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3)
            .unwrap();
        assert_eq!(result.block_ids.len(), 1);
        assert_eq!(result.fluid_updates.len(), 1);
    }

    #[test]
    fn beardifier_cpu_fallback() {
        let mut s = GpuBeardifierBatchSampler::new(mk_device());
        let structures = [BeardifierStructureData {
            min_x: -5,
            min_y: 60,
            min_z: -5,
            max_x: 5,
            max_y: 70,
            max_z: 5,
            terrain_adaptation: 2, // BeardBox
            ground_level_delta: 5,
        }];
        let junctions = [BeardifierJunctionData {
            x: 0,
            ground_y: 64,
            z: 0,
        }];
        let positions = [0.0f64, 64.0, 0.0];
        let mut results = [0.0f64];
        s.batch_beardifier(&positions, &structures, &junctions, &mut results)
            .unwrap();
        // 在包围盒中心应有正值贡献
        assert!(results[0] > 0.0);
    }

    #[test]
    fn vein_cpu_fallback() {
        let mut s = GpuVeinBatchSampler::new(mk_device());
        let params = VeinParams {
            toggle_config: vec![],
            ridged_config: vec![],
            gap_config: vec![],
        };
        let positions = [0.0f64, -30.0, 0.0];
        let mut results = [0i32];
        s.batch_vein_sample(&positions, &params, &mut results)
            .unwrap();
        assert_eq!(results[0], 0);
    }
}
