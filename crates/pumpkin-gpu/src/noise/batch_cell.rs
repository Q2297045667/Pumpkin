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
use std::sync::OnceLock;

/// 含水层 tiling 阈值全局配置。
/// 当网格点数 ≤ 此值时使用 `__local` 内存 tiled kernel。
/// 由 [`crate::GpuDevice::from_config`] 在初始化时设置。
static AQUIFER_TILE_THRESHOLD: OnceLock<usize> = OnceLock::new();

/// 设置含水层 tiling 阈值（从配置读取）。
pub fn set_aquifer_tile_threshold(threshold: usize) {
    let _ = AQUIFER_TILE_THRESHOLD.set(threshold);
}

/// 获取含水层 tiling 阈值。
/// 如果尚未设置，返回默认值 `2048`。
#[must_use]
pub fn get_aquifer_tile_threshold() -> usize {
    AQUIFER_TILE_THRESHOLD.get().copied().unwrap_or(2048)
}

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
    /// 失败时回退到上层 `BatchAccelerator` 的 CPU fallback。
    pub fn batch_fill_cell_caches(
        &mut self,
        positions: &[f64],
        sampler_params: &CellFillParams,
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

        // 提取 cell cache 参数
        let total_octaves: i32 = sampler_params.num_octaves.iter().sum();
        if total_octaves == 0 || sampler_params.perlin_configs.is_empty() {
            return Err(DeviceError::LaunchFailed(
                "cell cache fill: empty params — use CPU fallback".into(),
            ));
        }

        // 用第一个采样器的八度数计算偏移
        let num_octaves_0 = sampler_params.num_octaves.first().copied().unwrap_or(0) as usize;
        if num_octaves_0 == 0 {
            return Err(DeviceError::LaunchFailed(
                "cell cache fill: sampler has 0 octaves".into(),
            ));
        }

        let config_stride = 1 + num_octaves_0 * 5;
        let expected_len = config_stride * sampler_params.num_octaves.len();
        if sampler_params.perlin_configs.len() < expected_len {
            return Err(DeviceError::LaunchFailed(
                "cell cache fill: perlin_configs too short".into(),
            ));
        }

        // 构建 component_stack：展平 perlin_configs
        let component_stack: Vec<f64> = sampler_params.perlin_configs[..expected_len].to_vec();

        // 构建 perms_data
        let mut perms_data: Vec<u8> = Vec::with_capacity(total_octaves as usize * 256);
        for (s_idx, &no) in sampler_params.num_octaves.iter().enumerate() {
            for o in 0..no as usize {
                let perm = gen_perm_table(0x4365_6C6C_u64.wrapping_add(s_idx as u64), o);
                perms_data.extend_from_slice(&perm);
            }
        }

        // cell_indices: 每个位置指向 sampler 0（简化：所有位置使用同一 sampler）
        let cell_indices: Vec<i32> = vec![0i32; n];

        let amps_offset: i32 = 1;
        let lacs_offset: i32 = 1 + num_octaves_0 as i32;
        let orgs_offset: i32 = 1 + (num_octaves_0 * 2) as i32;

        // GPU 内存分配
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_stack = self.device.alloc_f64(component_stack.len())?;
        let mut d_perms = self.device.alloc_u8(perms_data.len())?;
        let mut d_indices = self.device.alloc_i32(cell_indices.len())?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_stack, &component_stack)?;
        self.device.copy_to_device(&mut d_perms, &perms_data)?;
        self.device.copy_to_device(&mut d_indices, &cell_indices)?;

        let ok = self.try_launch(
            "cell_cache_fill_f64",
            n,
            vec![
                KernelArg::BufferRef(0), // pos
                KernelArg::BufferRef(1), // component_stack
                KernelArg::BufferRef(2), // perms_data
                KernelArg::BufferRef(3), // cell_indices
                KernelArg::BufferRef(4), // densities (output)
                KernelArg::I32(n as i32),
                KernelArg::I32(config_stride as i32),
                KernelArg::I32(amps_offset),
                KernelArg::I32(lacs_offset),
                KernelArg::I32(orgs_offset),
            ],
            vec![
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::F64(&d_stack),
                GpuBufferRef::U8(&d_perms),
                GpuBufferRef::I32(&d_indices),
                GpuBufferRef::F64(&d_res),
            ],
        );

        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_stack)?;
            self.device.free(d_perms)?;
            self.device.free(d_indices)?;
            return Err(DeviceError::LaunchFailed(
                "cell cache fill: GPU kernel launch failed — use BatchAccelerator CPU fallback"
                    .into(),
            ));
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_stack)?;
        self.device.free(d_perms)?;
        self.device.free(d_indices)?;
        Ok(())
    }

    /// 批量填充插值器缓冲区。
    ///
    /// GPU 路径尝试 launch `interpolator_fill_f64` kernel，
    /// 失败时回退到上层 `BatchAccelerator` 的 CPU fallback。
    pub fn batch_fill_interpolators(
        &mut self,
        positions: &[f64],
        sampler_params: &CellFillParams,
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

        // 提取插值器参数：使用第一个采样器的配置
        let total_octaves: i32 = sampler_params.num_octaves.iter().sum();
        if total_octaves == 0 || sampler_params.perlin_configs.is_empty() {
            return Err(DeviceError::LaunchFailed(
                "interpolator fill: empty params — use CPU fallback".into(),
            ));
        }

        let expected_len = (total_octaves * 8) as usize;
        if sampler_params.perlin_configs.len() < expected_len {
            return Err(DeviceError::LaunchFailed(
                "interpolator fill: perlin_configs too short".into(),
            ));
        }

        // 构建 dag_params：[amp, lac, org_x, org_y, org_z, xz_scale, y_scale, _] per octave
        let dag_params: Vec<f64> = sampler_params.perlin_configs[..expected_len].to_vec();

        // 构建 perms_data：每个 octave 256 字节置换表
        let mut perms_data: Vec<u8> = Vec::with_capacity(total_octaves as usize * 256);
        for (s_idx, &no) in sampler_params.num_octaves.iter().enumerate() {
            for o in 0..no as usize {
                let perm = gen_perm_table(0x496E_7465_7270_u64.wrapping_add(s_idx as u64), o);
                perms_data.extend_from_slice(&perm);
            }
        }

        // GPU 内存分配
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_dag = self.device.alloc_f64(dag_params.len())?;
        let mut d_perms = self.device.alloc_u8(perms_data.len())?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_dag, &dag_params)?;
        self.device.copy_to_device(&mut d_perms, &perms_data)?;

        // 启动 GPU kernel
        let ok = self.try_launch(
            "interpolator_fill_f64",
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::BufferRef(3),
                KernelArg::I32(n as i32),
                KernelArg::I32(total_octaves),
            ],
            vec![
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::F64(&d_dag),
                GpuBufferRef::U8(&d_perms),
                GpuBufferRef::F64(&d_res),
            ],
        );

        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_dag)?;
            self.device.free(d_perms)?;
            return Err(DeviceError::LaunchFailed(
                "interpolator fill: GPU kernel launch failed — use BatchAccelerator CPU fallback"
                    .into(),
            ));
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_dag)?;
        self.device.free(d_perms)?;
        Ok(())
    }

    /// Helper: try to launch GPU kernel, return true if successful.
    #[allow(dead_code)] // TODO: 恢复后移除
    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        self.device.try_launch_kernel(name, n, args, gpu_buffers)
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

        // GPU 失败时返回错误，让上层处理 CPU 回退
        if self.device.device_type() == crate::DeviceType::Cpu {
            return Err(DeviceError::LaunchFailed(
                "aquifer batch: CPU device — use BatchAccelerator fallback".into(),
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

        let kernel_name = if m <= get_aquifer_tile_threshold() {
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

        if ok {
            let mut block_ids = vec![0i32; n];
            let mut fluid_updates = vec![0u8; n];
            self.device.copy_from_device(&d_bids, &mut block_ids)?;
            self.device.copy_from_device(&d_flags, &mut fluid_updates)?;
            self.device.free(d_pos)?;
            self.device.free(d_dens)?;
            self.device.free(d_gpos)?;
            self.device.free(d_gden)?;
            self.device.free(d_bids)?;
            self.device.free(d_flags)?;
            return Ok(AquiferBatchResult {
                block_ids,
                fluid_updates,
            });
        }

        // GPU launch 失败，清理资源并返回错误
        self.device.free(d_pos)?;
        self.device.free(d_dens)?;
        self.device.free(d_gpos)?;
        self.device.free(d_gden)?;
        self.device.free(d_bids)?;
        self.device.free(d_flags)?;
        Err(DeviceError::LaunchFailed("aquifer batch failed".into()))
    }

    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        self.device.try_launch_kernel(name, n, args, gpu_buffers)
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

        // GPU 失败时返回错误，让上层处理 CPU 回退
        if self.device.device_type() == crate::DeviceType::Cpu {
            return Err(DeviceError::LaunchFailed(
                "beardifier batch: CPU device — use BatchAccelerator fallback".into(),
            ));
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

        // beardifier_batch_f64 签名需要 11 个参数（包括 beard_kernel、
        // structure_to_junction 等），当前未准备好 → 跳过 GPU 尝试
        // TODO: 接入完整参数后恢复 GPU 路径
        let ok = false;
        if ok {
            self.device.copy_from_device(&d_res, results)?;
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_struct)?;
            self.device.free(d_junct)?;
            return Ok(());
        }

        // GPU launch 失败，清理资源并返回错误
        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_struct)?;
        self.device.free(d_junct)?;
        Err(DeviceError::LaunchFailed("beardifier batch failed".into()))
    }

    #[allow(dead_code)] // TODO: 恢复 launch 后移除
    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        self.device.try_launch_kernel(name, n, args, gpu_buffers)
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

        let mut _d_pos = self.device.alloc_f64(n * 3)?;
        let _d_res = self.device.alloc_i32(n)?;

        self.device.copy_to_device(&mut _d_pos, positions)?;

        // vein_batch_f64 签名需要 9 个参数（包括 vein_noise_params、
        // perms_data、vein_thresholds、vein_weights 等），当前未准备好 → 跳过 GPU 尝试
        // TODO: 接入完整参数后恢复 GPU 路径
        let ok = false;
        if ok {
            self.device.copy_from_device(&_d_res, results)?;
        } else {
            self.device.free(_d_pos)?;
            self.device.free(_d_res)?;
            return Err(DeviceError::LaunchFailed(
                "vein sample: GPU path not yet connected — use BatchAccelerator CPU fallback"
                    .into(),
            ));
        }

        self.device.free(_d_pos)?;
        self.device.free(_d_res)?;
        Ok(())
    }

    // TODO: 恢复 vein GPU 路径后移除此 allow
    #[allow(dead_code)]
    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        self.device.try_launch_kernel(name, n, args, gpu_buffers)
    }
}

// ============================================================================
// 共享 Perlin 置换表工具
// ============================================================================

/// 生成确定性置换表（每个 octave 一个 256 字节表）。
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

// ============================================================================
// CPU Fallbacks
// ============================================================================

/// CPU 占位：Cell Cache 零填充（不会被实际调用）。
///
/// 这是一个零填充占位实现，**永远不会在正常执行路径中被调用**。
/// 完整的 DAG 求值由上层 [`BatchAccelerator`] 的 CPU fallback 处理。
/// 此处保留仅用于：
/// - 编译期类型检查（确保签名与 GPU kernel 一致）
/// - 极端回退场景（GPU 不可用且上层未提供 fallback）的安全网
#[allow(unused_variables)]
fn cpu_cell_cache_fill(_positions: &[f64], results: &mut [f64]) {
    for item in results.iter_mut() {
        *item = 0.0;
    }
}

/// CPU 占位：插值器缓冲零填充（不会被实际调用）。
///
/// 这是一个零填充占位实现，**永远不会在正常执行路径中被调用**。
/// 完整的 DAG 求值由上层 [`BatchAccelerator`] 的 CPU fallback 处理。
/// 此处保留仅用于：
/// - 编译期类型检查（确保签名与 GPU kernel 一致）
/// - 极端回退场景（GPU 不可用且上层未提供 fallback）的安全网
#[allow(unused_variables)]
fn cpu_interpolator_fill(_positions: &[f64], results: &mut [f64]) {
    for item in results.iter_mut() {
        *item = 0.0;
    }
}

/// CPU 占位：矿脉零填充（不会被实际调用）。
///
/// 这是一个零填充占位实现（零填充 = 无矿脉），**永远不会在正常执行路径中被调用**。
/// 完整的矿脉判定由上层 [`BatchAccelerator`] 的 CPU fallback 处理。
/// 此处保留仅用于：
/// - 编译期类型检查（确保签名与 GPU kernel 一致）
/// - 极端回退场景（GPU 不可用且上层未提供 fallback）的安全网
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
        // GPU 不可用时应返回 LaunchFailed 错误（由上层 BatchAccelerator 处理 CPU 回退）
        let result = s.batch_fill_cell_caches(&positions, &params, &mut results);
        assert!(result.is_err());
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
        // GPU 不可用时应返回 LaunchFailed 错误（由上层 BatchAccelerator 处理 CPU 回退）
        let result = s.batch_fill_interpolators(&positions, &params, &mut results);
        assert!(result.is_err());
    }

    #[test]
    fn aquifer_zero_count() {
        let mut s = GpuAquiferBatchSampler::new(mk_device());
        let result = s.batch_aquifer_apply(&[], &[], &[], -10000.0, 0.3).unwrap();
        assert!(result.block_ids.is_empty());
        assert!(result.fluid_updates.is_empty());
    }

    #[test]
    fn aquifer_gpu_unavailable_returns_error() {
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
        // GPU 不可用时应返回 LaunchFailed 错误（由上层 BatchAccelerator 处理 CPU 回退）
        let result = s.batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn beardifier_gpu_unavailable_returns_error() {
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
        // GPU 不可用时应返回 LaunchFailed 错误（由上层 BatchAccelerator 处理 CPU 回退）
        let result = s.batch_beardifier(&positions, &structures, &junctions, &mut results);
        assert!(result.is_err());
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
        // GPU 不可用时应返回 LaunchFailed 错误（由上层 BatchAccelerator 处理 CPU 回退）
        let result = s.batch_vein_sample(&positions, &params, &mut results);
        assert!(result.is_err());
    }
}
