//! 批量 Aquifer / Beardifier 采样器。
//!
//! 提供 GPU 加速的含水层判定与 Beardifier 地形适应，均包含 CPU 回退路径。
//! （Cell Cache / Interpolator 填充已改用 vanilla 语义的 DoublePerlin 规格路径，
//! 见 `pumpkin-world::batch_accel::batch_fill_cell_caches_vanilla`。）
#![allow(
    clippy::separated_literal_suffix,
    clippy::as_ptr_cast_mut,
    clippy::ptr_as_ptr,
    clippy::similar_names
)]

use crate::GpuDevice;
use crate::common::DeviceError;
use crate::common::kernel::{GpuBufferRef, KernelArg};
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

/// 含水层批量结果
pub struct AquiferBatchResult {
    /// block_state_id 数组
    pub block_ids: Vec<i32>,
    /// should_schedule_fluid_update 标志
    pub fluid_updates: Vec<u8>,
}

/// 结构数据（可序列化到 GPU）。
///
/// 与 vanilla `Beardifier::sample` 的包围盒语义一致：
/// 编码为 8 个 f64，与 `beardifier_batch_f64` kernel 的 structures 布局一致：
///   `[box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z,
///     adaptation, ground_delta]`
pub struct BeardifierStructureData {
    pub box_min_x: i32,
    pub box_min_y: i32,
    pub box_min_z: i32,
    pub box_max_x: i32,
    pub box_max_y: i32,
    pub box_max_z: i32,
    /// 地形适应类型：0=None 1=BeardThin 2=BeardBox 3=Bury 4=Encapsulate
    pub adaptation: i32,
    /// 地面高度差（`ground_level_delta`）
    pub ground_delta: i32,
}

/// 连接点数据
pub struct BeardifierJunctionData {
    pub x: i32,
    pub ground_y: i32,
    pub z: i32,
}

// ============================================================================
// GpuAquiferBatchSampler
// ============================================================================

/// GPU 含水层批量采样器。
pub struct GpuAquiferBatchSampler {
    pub device: GpuDevice,
    /// 持久化 buffer 池（按长度复用 f64/u8/i32 buffer）。
    buffer_pool: crate::common::GpuBufferPool,
}

impl GpuAquiferBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            buffer_pool: crate::common::GpuBufferPool::new(),
        }
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

        // 从缓冲池分配（复用跨调用缓冲区）
        let mut d_pos = self.buffer_pool.take_f64(&self.device, n * 3)?;
        let mut d_dens = self.buffer_pool.take_f64(&self.device, n)?;
        let mut d_gpos = self.buffer_pool.take_f64(&self.device, grid_pos_count)?;
        let mut d_gden = self.buffer_pool.take_f64(&self.device, grid_den_count)?;
        let d_bids = self.buffer_pool.take_i32(&self.device, n)?;
        let d_flags = self.buffer_pool.take_u8(&self.device, n)?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_dens, densities)?;
        self.device.copy_to_device(&mut d_gpos, &grid_positions)?;
        self.device.copy_to_device(&mut d_gden, &grid_densities)?;

        let kernel_name = if m <= get_aquifer_tile_threshold() {
            "aquifer_batch_tiled_f64"
        } else {
            "aquifer_batch_f64"
        };
        // tiled kernel 的尾部 __local / extern __shared__ 参数大小（字节）：
        // tile_positions [M*3] f64 + tile_densities [M] f64
        let local_mem_bytes = if kernel_name == "aquifer_batch_tiled_f64" {
            vec![m * 3 * size_of::<f64>(), m * size_of::<f64>()]
        } else {
            Vec::new()
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
            local_mem_bytes,
        );

        if ok {
            let mut block_ids = vec![0i32; n];
            let mut fluid_updates = vec![0u8; n];
            self.device.copy_from_device(&d_bids, &mut block_ids)?;
            self.device.copy_from_device(&d_flags, &mut fluid_updates)?;
            self.buffer_pool.put_f64(d_pos);
            self.buffer_pool.put_f64(d_dens);
            self.buffer_pool.put_f64(d_gpos);
            self.buffer_pool.put_f64(d_gden);
            self.buffer_pool.put_i32(d_bids);
            self.buffer_pool.put_u8(d_flags);
            return Ok(AquiferBatchResult {
                block_ids,
                fluid_updates,
            });
        }

        // GPU launch 失败，清理资源并返回错误
        self.buffer_pool.put_f64(d_pos);
        self.buffer_pool.put_f64(d_dens);
        self.buffer_pool.put_f64(d_gpos);
        self.buffer_pool.put_f64(d_gden);
        self.buffer_pool.put_i32(d_bids);
        self.buffer_pool.put_u8(d_flags);
        Err(DeviceError::LaunchFailed("aquifer batch failed".into()))
    }

    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
        local_mem_bytes: Vec<usize>,
    ) -> bool {
        self.device
            .try_launch_kernel(name, n, args, gpu_buffers, local_mem_bytes)
    }
}

// ============================================================================
// GpuBeardifierBatchSampler
// ============================================================================

/// 预计算的 24³ beard kernel 缓存（静态数据，全局复用）。
///
/// 布局与 vanilla `Beardifier::get_beard_kernel` 一致（zi-major：`zi*576 + xi*24 + yi`），
/// 值也与 vanilla 相同：`exp(-(dx² + (dy+0.5)² + dz²)/16)`。
static BEARD_KERNEL_GPU: std::sync::OnceLock<Box<[f64]>> = std::sync::OnceLock::new();

fn get_beard_kernel_gpu() -> &'static [f64] {
    BEARD_KERNEL_GPU.get_or_init(|| {
        const KS: i32 = 24;
        const KR: i32 = 12;
        const KV: usize = (KS * KS * KS) as usize;
        let mut kernel = vec![0.0f64; KV].into_boxed_slice();
        for zi in 0..KS {
            for xi in 0..KS {
                for yi in 0..KS {
                    let dx = xi - KR;
                    let dy = (yi - KR) as f64 + 0.5;
                    let dz = zi - KR;
                    let dist_sq = (dx as f64).powi(2) + dy.powi(2) + (dz as f64).powi(2);
                    kernel[(zi * 24 * 24 + xi * 24 + yi) as usize] =
                        std::f64::consts::E.powf(-dist_sq / 16.0);
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
    /// 持久化 buffer 池（按长度复用 f64 buffer）。
    buffer_pool: crate::common::GpuBufferPool,
}

impl GpuBeardifierBatchSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            beard_kernel_buf: None,
            buffer_pool: crate::common::GpuBufferPool::new(),
        }
    }

    /// 批量 Beardifier 计算（与 vanilla `Beardifier::sample` 逐位一致）。
    ///
    /// `affected_box` 为 `[min_x, min_y, min_z, max_x, max_y, max_z]`（包含边界），
    /// 盒子外的位置直接输出 0（与 vanilla 一致）。
    #[allow(clippy::too_many_lines)]
    pub fn batch_beardifier(
        &mut self,
        positions: &[f64],
        structures: &[BeardifierStructureData],
        junctions: &[BeardifierJunctionData],
        affected_box: [i32; 6],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        const KERNEL_VOLUME: usize = 24 * 24 * 24; // 13824

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

        // 构建预计算的 beard kernel（vanilla 24³ 表，zi-major 布局）— GPU 持久化缓存
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

        // 扁平化结构数据（8 f64 per structure：包围盒 + adaptation + ground_delta）
        let struct_flat: Vec<f64> = structures
            .iter()
            .flat_map(|s| {
                vec![
                    f64::from(s.box_min_x),
                    f64::from(s.box_min_y),
                    f64::from(s.box_min_z),
                    f64::from(s.box_max_x),
                    f64::from(s.box_max_y),
                    f64::from(s.box_max_z),
                    f64::from(s.adaptation),
                    f64::from(s.ground_delta),
                ]
            })
            .collect();

        let junct_flat: Vec<f64> = junctions
            .iter()
            .flat_map(|j| vec![f64::from(j.x), f64::from(j.ground_y), f64::from(j.z)])
            .collect();

        let affected_flat: Vec<f64> = affected_box.iter().map(|&v| f64::from(v)).collect();

        // 从缓冲池分配（kernel 已缓存，其余按需复用）
        let mut d_pos = self.buffer_pool.take_f64(&self.device, n * 3)?;
        let d_res = self.buffer_pool.take_f64(&self.device, n)?;
        let mut d_struct = self.buffer_pool.take_f64(&self.device, struct_flat.len())?;
        let mut d_junct = self.buffer_pool.take_f64(&self.device, junct_flat.len())?;
        let mut d_affected = self.buffer_pool.take_f64(&self.device, 6)?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device.copy_to_device(&mut d_struct, &struct_flat)?;
        self.device.copy_to_device(&mut d_junct, &junct_flat)?;
        self.device
            .copy_to_device(&mut d_affected, &affected_flat)?;

        let ok = self.try_launch(
            "beardifier_batch_f64",
            n,
            vec![
                KernelArg::BufferRef(0), // pos
                KernelArg::BufferRef(1), // beard_kernel
                KernelArg::BufferRef(2), // structures
                KernelArg::BufferRef(3), // junctions
                KernelArg::BufferRef(4), // affected box
                KernelArg::BufferRef(5), // beard_values (output)
                KernelArg::I32(n as i32),
                KernelArg::I32(structures.len() as i32),
                KernelArg::I32(junctions.len() as i32),
            ],
            vec![
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::F64(d_kernel),
                GpuBufferRef::F64(&d_struct),
                GpuBufferRef::F64(&d_junct),
                GpuBufferRef::F64(&d_affected),
                GpuBufferRef::F64(&d_res),
            ],
        );

        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            self.buffer_pool.put_f64(d_pos);
            self.buffer_pool.put_f64(d_res);
            self.buffer_pool.put_f64(d_struct);
            self.buffer_pool.put_f64(d_junct);
            self.buffer_pool.put_f64(d_affected);
            return Err(DeviceError::LaunchFailed("beardifier batch failed".into()));
        }

        self.buffer_pool.put_f64(d_pos);
        self.buffer_pool.put_f64(d_res);
        self.buffer_pool.put_f64(d_struct);
        self.buffer_pool.put_f64(d_junct);
        self.buffer_pool.put_f64(d_affected);
        Ok(())
    }

    fn try_launch(
        &self,
        name: &str,
        n: usize,
        args: Vec<KernelArg<'_>>,
        gpu_buffers: Vec<GpuBufferRef<'_>>,
    ) -> bool {
        self.device
            .try_launch_kernel(name, n, args, gpu_buffers, Vec::new())
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

    /// 强制 CPU 后端（不依赖本机是否有 GPU）。
    fn mk_cpu_device() -> GpuDevice {
        GpuDevice::from_config(&pumpkin_config::gpu::GpuConfig {
            enabled: true,
            batch_acceleration: true,
            backend: pumpkin_config::gpu::GpuBackend::Cpu,
            ..Default::default()
        })
    }

    #[test]
    fn aquifer_zero_count() {
        let mut s = GpuAquiferBatchSampler::new(mk_device());
        let result = s.batch_aquifer_apply(&[], &[], &[], -10000.0, 0.3).unwrap();
        assert!(result.block_ids.is_empty());
        assert!(result.fluid_updates.is_empty());
    }

    #[test]
    fn aquifer_cpu_device_returns_error() {
        let mut s = GpuAquiferBatchSampler::new(mk_cpu_device());
        // 构造一个简单的含水层网格
        let positions = [0.0f64, -60.0, 0.0];
        let densities = [-1.0f64];
        let packed_grid = [
            0.0f64.to_bits() as i64,
            (-60.0f64).to_bits() as i64,
            (-2.0f64).to_bits() as i64,
            (-1.0f64).to_bits() as i64,
        ];
        // CPU 后端应返回错误（由上层 BatchAccelerator 处理 CPU 回退）
        let result = s.batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn beardifier_gpu_unavailable_returns_error() {
        let mut s = GpuBeardifierBatchSampler::new(mk_device());
        let structures = [BeardifierStructureData {
            box_min_x: -5,
            box_min_y: 60,
            box_min_z: -5,
            box_max_x: 5,
            box_max_y: 70,
            box_max_z: 5,
            adaptation: 1, // BeardThin
            ground_delta: 5,
        }];
        let junctions = [BeardifierJunctionData {
            x: 0,
            ground_y: 64,
            z: 0,
        }];
        let positions = [0.0f64, 64.0, 0.0];
        let mut results = [0.0f64];
        let result = s.batch_beardifier(
            &positions,
            &structures,
            &junctions,
            [-10, 50, -10, 10, 80, 10],
            &mut results,
        );
        // GPU 不可用（CPU 后端）时必须返回错误，由上层 BatchAccelerator 处理 CPU 回退；
        // 真 GPU 环境下 kernel 正常执行、返回 Ok 是合法行为。
        if s.device.device_type() == crate::DeviceType::Cpu {
            assert!(result.is_err());
        }
    }

    #[test]
    fn aquifer_tile_threshold_default() {
        // 尚未设置时返回默认值 2048
        assert_eq!(get_aquifer_tile_threshold(), 2048);
    }
}
