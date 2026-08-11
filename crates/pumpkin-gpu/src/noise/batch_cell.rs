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
#[derive(Clone)]
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

/// 结构数据（可序列化到 GPU）。
///
/// 编码为 9 个 f64 值，与 `beardifier_batch_f64` kernel 的 structures 布局一致：
///   `[center_x, center_y, center_z, radius_x, radius_y, radius_z,
///     min_y, ground_delta_y, max_y]`
pub struct BeardifierStructureData {
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub radius_x: f64,
    pub radius_y: f64,
    pub radius_z: f64,
    pub min_y: f64,
    pub ground_delta_y: f64,
    pub max_y: f64,
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
/// 内部维护持久化 buffer 池以减少重复分配。
pub struct GpuCellBatchSampler {
    pub device: GpuDevice,
    pub cache: NoiseCache,
    /// 持久化 perm buffer 缓存
    perm_pool: std::collections::HashMap<usize, crate::GpuBuffer<u8>>,
    /// 持久化 f64 buffer 缓存
    f64_pool: std::collections::HashMap<usize, crate::GpuBuffer<f64>>,
    /// 持久化 i32 buffer 缓存
    i32_pool: std::collections::HashMap<usize, crate::GpuBuffer<i32>>,
}

impl GpuCellBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            cache: NoiseCache::default(),
            perm_pool: std::collections::HashMap::new(),
            f64_pool: std::collections::HashMap::new(),
            i32_pool: std::collections::HashMap::new(),
        }
    }

    /// 从 u8 buffer 池中分配或复用。
    fn alloc_u8_pooled(&mut self, len: usize) -> Result<crate::GpuBuffer<u8>, DeviceError> {
        if let Some(buf) = self.perm_pool.remove(&len) {
            Ok(buf)
        } else {
            self.device.alloc_u8(len)
        }
    }

    /// 归还 u8 buffer 到池中。
    fn free_u8_pooled(&mut self, len: usize, buf: crate::GpuBuffer<u8>) {
        self.perm_pool.entry(len).or_insert(buf);
    }

    /// 从 f64 buffer 池中分配或复用。
    fn alloc_f64_pooled(&mut self, len: usize) -> Result<crate::GpuBuffer<f64>, DeviceError> {
        if let Some(buf) = self.f64_pool.remove(&len) {
            Ok(buf)
        } else {
            self.device.alloc_f64(len)
        }
    }

    /// 归还 f64 buffer 到池中。
    fn free_f64_pooled(&mut self, len: usize, buf: crate::GpuBuffer<f64>) {
        self.f64_pool.entry(len).or_insert(buf);
    }

    /// 从 i32 buffer 池中分配或复用。
    fn alloc_i32_pooled(&mut self, len: usize) -> Result<crate::GpuBuffer<i32>, DeviceError> {
        if let Some(buf) = self.i32_pool.remove(&len) {
            Ok(buf)
        } else {
            self.device.alloc_i32(len)
        }
    }

    /// 归还 i32 buffer 到池中。
    fn free_i32_pooled(&mut self, len: usize, buf: crate::GpuBuffer<i32>) {
        self.i32_pool.entry(len).or_insert(buf);
    }

    /// 批量填充 cell cache — 支持自定义 cell_indices。
    ///
    /// 与 `batch_fill_cell_caches` 的区别：接受预构建的 `cell_indices`，
    /// 允许不同位置组使用不同的 sampler 配置。用于合并多次调用为单次 GPU launch。
    pub fn batch_fill_cell_caches_indexed(
        &mut self,
        positions: &[f64],
        sampler_params: &CellFillParams,
        cell_indices: &[i32],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);
        assert_eq!(cell_indices.len(), n);

        if self.device.device_type() == crate::DeviceType::Cpu {
            return Err(DeviceError::LaunchFailed(
                "CPU device — use BatchAccelerator fallback".into(),
            ));
        }

        let total_octaves: i32 = sampler_params.num_octaves.iter().sum();
        if total_octaves == 0 || sampler_params.perlin_configs.is_empty() {
            return Err(DeviceError::LaunchFailed(
                "cell cache fill: empty params — use CPU fallback".into(),
            ));
        }

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

        let component_stack: Vec<f64> = sampler_params.perlin_configs[..expected_len].to_vec();

        let mut perms_data: Vec<u8> = Vec::with_capacity(total_octaves as usize * 256);
        for (s_idx, &no) in sampler_params.num_octaves.iter().enumerate() {
            for o in 0..no as usize {
                let perm = gen_perm_table(0x4365_6C6C_u64.wrapping_add(s_idx as u64), o);
                perms_data.extend_from_slice(&perm);
            }
        }

        let amps_offset: i32 = 1;
        let lacs_offset: i32 = 1 + num_octaves_0 as i32;
        let orgs_offset: i32 = 1 + (num_octaves_0 * 2) as i32;

        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_stack = self.alloc_f64_pooled(component_stack.len())?;
        let mut d_perms = self.alloc_u8_pooled(perms_data.len())?;
        let mut d_indices = self.alloc_i32_pooled(cell_indices.len())?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_stack, &component_stack)?;
        self.device.copy_to_device(&mut d_perms, &perms_data)?;
        self.device.copy_to_device(&mut d_indices, cell_indices)?;

        let ok = self.try_launch(
            "cell_cache_fill_f64",
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::BufferRef(3),
                KernelArg::BufferRef(4),
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
            self.free_f64_pooled(component_stack.len(), d_stack);
            self.free_u8_pooled(perms_data.len(), d_perms);
            self.free_i32_pooled(cell_indices.len(), d_indices);
            return Err(DeviceError::LaunchFailed(
                "cell cache fill: GPU kernel launch failed".into(),
            ));
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.free_f64_pooled(component_stack.len(), d_stack);
        self.free_u8_pooled(perms_data.len(), d_perms);
        self.free_i32_pooled(cell_indices.len(), d_indices);
        Ok(())
    }

    /// 批量填充 cell cache（默认所有位置使用 sampler 0）。
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
        let cell_indices: Vec<i32> = vec![0i32; n];
        self.batch_fill_cell_caches_indexed(positions, sampler_params, &cell_indices, results)
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
            return Err(DeviceError::LaunchFailed(
                "CPU device — use BatchAccelerator fallback".into(),
            ));
        }

        // 提取 interpolator fill 参数
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

        // GPU 内存分配（pos/res 按需，dag/perms 池化）
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_dag = self.alloc_f64_pooled(dag_params.len())?;
        let mut d_perms = self.alloc_u8_pooled(perms_data.len())?;

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
            self.free_f64_pooled(dag_params.len(), d_dag);
            self.free_u8_pooled(perms_data.len(), d_perms);
            return Err(DeviceError::LaunchFailed(
                "interpolator fill: GPU kernel launch failed — use BatchAccelerator CPU fallback"
                    .into(),
            ));
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.free_f64_pooled(dag_params.len(), d_dag);
        self.free_u8_pooled(perms_data.len(), d_perms);
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

/// 预计算的 24³ beard kernel 缓存（静态数据，全局复用）。
static BEARD_KERNEL_GPU: std::sync::OnceLock<Box<[f64]>> = std::sync::OnceLock::new();

fn get_beard_kernel_gpu() -> &'static [f64] {
    BEARD_KERNEL_GPU.get_or_init(|| {
        const KS: usize = 24;
        const KV: usize = KS * KS * KS;
        let mut kernel = vec![0.0f64; KV].into_boxed_slice();
        let ksh = KS as f64 * 0.5;
        for zi in 0..KS {
            for xi in 0..KS {
                for yi in 0..KS {
                    let dx = xi as f64 - ksh;
                    let dy = yi as f64 - ksh + 0.5;
                    let dz = zi as f64 - ksh;
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    kernel[xi * KS * KS + yi * KS + zi] = (-dist_sq / 16.0).exp();
                }
            }
        }
        kernel
    })
}

/// GPU Beardifier 批量采样器。
///
/// 对批量位置计算结构/连接点造成的地形适应（beard）贡献。
pub struct GpuBeardifierBatchSampler {
    pub device: GpuDevice,
    /// 持久化 beard kernel GPU buffer（108KB，首次上传后复用）
    beard_kernel_buf: Option<crate::GpuBuffer<f64>>,
}

impl GpuBeardifierBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            beard_kernel_buf: None,
        }
    }

    /// 批量 Beardifier 计算。
    ///
    /// 对每个位置累加来自结构和连接点的 beard 贡献。
    #[allow(clippy::too_many_lines)]
    pub fn batch_beardifier(
        &mut self,
        positions: &[f64],
        structures: &[BeardifierStructureData],
        junctions: &[BeardifierJunctionData],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        const KERNEL_SIZE: usize = 24;
        const KERNEL_VOLUME: usize = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE; // 13824

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

        // 构建预计算的 beard kernel (24³三线性采样核) — GPU 持久化缓存
        let beard_kernel = get_beard_kernel_gpu();
        if self.beard_kernel_buf.is_none() {
            let mut buf = self.device.alloc_f64(KERNEL_VOLUME)?;
            self.device.copy_to_device(&mut buf, beard_kernel)?;
            self.beard_kernel_buf = Some(buf);
        }
        // SAFETY: `beard_kernel_buf` was set to `Some` immediately above if it was `None`.
        let d_kernel = self.beard_kernel_buf.as_ref().ok_or_else(|| {
            DeviceError::LaunchFailed("beardifier: kernel buffer not initialized".into())
        })?;

        // 扁平化结构数据（9 doubles per structure）
        let struct_flat: Vec<f64> = structures
            .iter()
            .flat_map(|s| {
                vec![
                    s.center_x,
                    s.center_y,
                    s.center_z,
                    s.radius_x,
                    s.radius_y,
                    s.radius_z,
                    s.min_y,
                    s.ground_delta_y,
                    s.max_y,
                ]
            })
            .collect();

        let junct_flat: Vec<f64> = junctions
            .iter()
            .flat_map(|j| vec![f64::from(j.x), f64::from(j.ground_y), f64::from(j.z)])
            .collect();

        // structure_to_junction — kernel 声明但未使用，传递零填充占位数组
        let struct_to_junction: Vec<i32> = vec![0i32; structures.len()];

        // GPU 内存分配（kernel 已缓存，其余按需）
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_struct = self.device.alloc_f64(struct_flat.len())?;
        let mut d_junct = self.device.alloc_f64(junct_flat.len())?;
        let mut d_stoj = self.device.alloc_i32(struct_to_junction.len())?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_struct, &struct_flat)?;
        self.device.copy_to_device(&mut d_junct, &junct_flat)?;
        self.device
            .copy_to_device(&mut d_stoj, &struct_to_junction)?;

        let ok = self.try_launch(
            "beardifier_batch_f64",
            n,
            vec![
                KernelArg::BufferRef(0), // pos
                KernelArg::BufferRef(1), // beard_kernel
                KernelArg::BufferRef(2), // structures
                KernelArg::BufferRef(3), // junctions
                KernelArg::BufferRef(4), // structure_to_junction
                KernelArg::BufferRef(5), // beard_values (output)
                KernelArg::I32(n as i32),
                KernelArg::I32(structures.len() as i32),
                KernelArg::I32(junctions.len() as i32),
                KernelArg::I32(KERNEL_SIZE as i32),
                KernelArg::F64(1.0 / KERNEL_SIZE as f64),
            ],
            vec![
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::F64(d_kernel),
                GpuBufferRef::F64(&d_struct),
                GpuBufferRef::F64(&d_junct),
                GpuBufferRef::I32(&d_stoj),
                GpuBufferRef::F64(&d_res),
            ],
        );

        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_struct)?;
            self.device.free(d_junct)?;
            self.device.free(d_stoj)?;
            return Err(DeviceError::LaunchFailed("beardifier batch failed".into()));
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_struct)?;
        self.device.free(d_junct)?;
        self.device.free(d_stoj)?;
        Ok(())
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
    #[allow(clippy::too_many_lines)]
    pub fn batch_vein_sample(
        &mut self,
        positions: &[f64],
        vein_params: &VeinParams,
        results: &mut [i32],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            return Err(DeviceError::LaunchFailed(
                "CPU device — use BatchAccelerator fallback".into(),
            ));
        }

        // 提取矿脉参数
        let octaves_per_vein = (vein_params.toggle_config.len() / 8) as i32;
        if octaves_per_vein == 0 {
            return Err(DeviceError::LaunchFailed(
                "vein sample: empty toggle_config".into(),
            ));
        }

        let num_veins: i32 = 1; // Minecraft 使用单一矿脉噪声集
        let total_octaves = (num_veins * 3 * octaves_per_vein) as usize;
        let expected_toggle = (octaves_per_vein * 8) as usize;

        if vein_params.toggle_config.len() < expected_toggle
            || vein_params.ridged_config.len() < expected_toggle
            || vein_params.gap_config.len() < expected_toggle
        {
            return Err(DeviceError::LaunchFailed(
                "vein sample: configs too short".into(),
            ));
        }

        // 展平矿脉噪声参数：toggle + ridged + gap，每段 8 doubles/octave
        let mut vein_noise_flat = Vec::with_capacity(total_octaves * 8);
        vein_noise_flat.extend_from_slice(&vein_params.toggle_config[..expected_toggle]);
        vein_noise_flat.extend_from_slice(&vein_params.ridged_config[..expected_toggle]);
        vein_noise_flat.extend_from_slice(&vein_params.gap_config[..expected_toggle]);

        // 构建 perms_data：每个 octave 256 字节
        let mut perms_data: Vec<u8> = Vec::with_capacity(total_octaves * 256);
        for v in 0..num_veins as usize {
            for _seg in 0..3usize {
                for o in 0..octaves_per_vein as usize {
                    let perm = gen_perm_table(
                        0x7665_696E5F6Eu64
                            .wrapping_add(v as u64)
                            .wrapping_add(o as u64),
                        o,
                    );
                    perms_data.extend_from_slice(&perm);
                }
            }
        }

        // 阈值和权重（与 OreveinSampler / cpu_vein_detect 一致）
        let vein_thresholds: Vec<f64> = vec![
            0.0,  // toggle threshold (> 0 → Copper, < 0 → Iron)
            0.0,  // ridged threshold (must be < 0)
            -0.3, // gap threshold (must be > -0.3)
        ];
        let vein_weights: Vec<f64> = vec![1.0];

        // GPU 内存分配
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_i32(n)?;
        let mut d_noise = self.device.alloc_f64(vein_noise_flat.len())?;
        let mut d_perms = self.device.alloc_u8(perms_data.len())?;
        let mut d_thresh = self.device.alloc_f64(vein_thresholds.len())?;
        let mut d_weights = self.device.alloc_f64(vein_weights.len())?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_noise, &vein_noise_flat)?;
        self.device.copy_to_device(&mut d_perms, &perms_data)?;
        self.device
            .copy_to_device(&mut d_thresh, &vein_thresholds)?;
        self.device.copy_to_device(&mut d_weights, &vein_weights)?;

        let ok = self.try_launch(
            "vein_batch_f64",
            n,
            vec![
                KernelArg::BufferRef(0), // pos
                KernelArg::BufferRef(1), // vein_noise_params
                KernelArg::BufferRef(2), // perms_data
                KernelArg::BufferRef(3), // vein_thresholds
                KernelArg::BufferRef(4), // vein_weights
                KernelArg::BufferRef(5), // vein_types (output)
                KernelArg::I32(n as i32),
                KernelArg::I32(num_veins),
                KernelArg::I32(octaves_per_vein),
            ],
            vec![
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::F64(&d_noise),
                GpuBufferRef::U8(&d_perms),
                GpuBufferRef::F64(&d_thresh),
                GpuBufferRef::F64(&d_weights),
                GpuBufferRef::I32(&d_res),
            ],
        );

        if ok {
            self.device.copy_from_device(&d_res, results)?;
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_noise)?;
            self.device.free(d_perms)?;
            self.device.free(d_thresh)?;
            self.device.free(d_weights)?;
            return Ok(());
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_noise)?;
        self.device.free(d_perms)?;
        self.device.free(d_thresh)?;
        self.device.free(d_weights)?;
        Err(DeviceError::LaunchFailed("vein batch failed".into()))
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
            center_x: 0.0,
            center_y: 65.0,
            center_z: 0.0,
            radius_x: 5.0,
            radius_y: 5.0,
            radius_z: 5.0,
            min_y: 60.0,
            ground_delta_y: 5.0,
            max_y: 70.0,
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
