//! GPU 噪声批量采样器 — 全类型支持。
//!
//! 支持：Octave, `DoublePerlin`, ShiftA/B, `ShiftedNoise`, `InterpolatedNoise`,
//! `VeinNoise`, `AquiferDensity`。
#![allow(
    clippy::separated_literal_suffix,
    clippy::as_ptr_cast_mut,
    clippy::ptr_as_ptr,
    clippy::too_many_lines
)]

use crate::GpuDevice;
use crate::common::DeviceError;
use crate::common::GpuBuffer;
use crate::common::kernel::GpuBufferRef;
use crate::common::kernel::KernelArg;
use crate::noise::cache::{NoiseCache, SerializedOctaveConfig};

#[cfg(feature = "pumpkin-util")]
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

/// SoA 布局是否启用 — 由 `from_config()` 通过 `set_soa_layout()` 注入。
static SOA_LAYOUT_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 设置 SoA 布局开关（在初始化时调用一次）。
pub fn set_soa_layout(enabled: bool) {
    SOA_LAYOUT_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// 读取当前 SoA 布局开关。
pub fn use_soa_layout() -> bool {
    SOA_LAYOUT_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// GPU buffer 生命周期管理助手。
///
/// 管理一组 GPU buffer 的分配、上传、下载和释放，
/// 消除了每个方法中重复的逐 buffer 管理代码。
struct GpuBufferSet {
    f64_bufs: Vec<GpuBuffer<f64>>,
    u8_bufs: Vec<GpuBuffer<u8>>,
}

impl GpuBufferSet {
    fn new() -> Self {
        Self {
            f64_bufs: Vec::new(),
            u8_bufs: Vec::new(),
        }
    }

    /// 分配一个 f64 buffer，返回索引。
    fn alloc_f64(&mut self, device: &GpuDevice, len: usize) -> Result<usize, DeviceError> {
        let buf = device.alloc_f64(len)?;
        let idx = self.f64_bufs.len();
        self.f64_bufs.push(buf);
        Ok(idx)
    }

    /// 分配一个 u8 buffer，返回索引。
    fn alloc_u8(&mut self, device: &GpuDevice, len: usize) -> Result<usize, DeviceError> {
        let buf = device.alloc_u8(len)?;
        let idx = self.u8_bufs.len();
        self.u8_bufs.push(buf);
        Ok(idx)
    }

    /// 按索引获取 f64 buffer 引用。
    fn f64_ref(&self, idx: usize) -> &GpuBuffer<f64> {
        &self.f64_bufs[idx]
    }

    /// 按索引获取 u8 buffer 引用。
    fn u8_ref(&self, idx: usize) -> &GpuBuffer<u8> {
        &self.u8_bufs[idx]
    }

    /// 上传 f64 数据到指定索引的 buffer。
    fn upload_f64(
        &mut self,
        device: &GpuDevice,
        idx: usize,
        data: &[f64],
    ) -> Result<(), DeviceError> {
        device.copy_to_device(&mut self.f64_bufs[idx], data)
    }

    /// 上传 u8 数据到指定索引的 buffer。
    fn upload_u8(
        &mut self,
        device: &GpuDevice,
        idx: usize,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        device.copy_to_device(&mut self.u8_bufs[idx], data)
    }

    /// 从指定索引的 f64 buffer 下载数据。
    fn download_f64(
        &self,
        device: &GpuDevice,
        idx: usize,
        data: &mut [f64],
    ) -> Result<(), DeviceError> {
        device.copy_from_device(&self.f64_bufs[idx], data)
    }

    /// 分配并上传一份完整的八度配置。
    /// 返回 `(perm, amp, lac, org)` 四个 buffer 的索引。
    fn load_octave_config(
        &mut self,
        device: &GpuDevice,
        config: &SerializedOctaveConfig,
    ) -> Result<(usize, usize, usize, usize), DeviceError> {
        let m = config.num_octaves();
        let perm = self.alloc_u8(device, m * 256)?;
        let amp = self.alloc_f64(device, m)?;
        let lac = self.alloc_f64(device, m)?;
        let org = self.alloc_f64(device, m * 3)?;
        self.upload_u8(device, perm, &config.packed_permutations())?;
        self.upload_f64(device, amp, &config.packed_amplitudes())?;
        self.upload_f64(device, lac, &config.packed_lacunarities())?;
        self.upload_f64(device, org, &config.packed_origins())?;
        Ok((perm, amp, lac, org))
    }

    /// 一次性释放所有 buffer。
    fn free_all(self, device: &GpuDevice) -> Result<(), DeviceError> {
        for buf in self.f64_bufs {
            device.free(buf)?;
        }
        for buf in self.u8_bufs {
            device.free(buf)?;
        }
        Ok(())
    }
}

/// GPU 噪声批量采样器。
pub struct GpuNoiseSampler {
    pub device: GpuDevice,
    pub cache: NoiseCache,
    pub double_cache: NoiseCache,
}

impl GpuNoiseSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            cache: NoiseCache::new(),
            double_cache: NoiseCache::new(),
        }
    }

    // ========== Octave Perlin ==========

    #[cfg(feature = "pumpkin-util")]
    pub fn sample_octave_batch(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_octave_batch(sampler, positions, results);
            return Ok(());
        }

        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let config = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);

        let m = config.num_octaves();
        let mut bufs = GpuBufferSet::new();
        let res_idx = bufs.alloc_f64(&self.device, n)?;
        let (perm_idx, amp_idx, lac_idx, org_idx) =
            bufs.load_octave_config(&self.device, &config)?;

        // SoA 路径：当启用 soa_layout 且数据量足够大时，使用独立 X/Y/Z 数组
        let use_soa = use_soa_layout() && n >= 64;
        if use_soa {
            let (x, y, z) = crate::common::layout::aos3d_to_soa(positions);
            let x_idx = bufs.alloc_f64(&self.device, n)?;
            let y_idx = bufs.alloc_f64(&self.device, n)?;
            let z_idx = bufs.alloc_f64(&self.device, n)?;
            bufs.upload_f64(&self.device, x_idx, &x)?;
            bufs.upload_f64(&self.device, y_idx, &y)?;
            bufs.upload_f64(&self.device, z_idx, &z)?;

            let ok = self.try_launch(
                "octave_perlin_sample_soa_f64",
                n,
                vec![
                    KernelArg::BufferRef(0),
                    KernelArg::BufferRef(1),
                    KernelArg::BufferRef(2),
                    KernelArg::BufferRef(3),
                    KernelArg::BufferRef(4),
                    KernelArg::BufferRef(5),
                    KernelArg::BufferRef(6),
                    KernelArg::BufferRef(7),
                    KernelArg::I32(n as i32),
                    KernelArg::I32(m as i32),
                ],
                vec![
                    GpuBufferRef::F64(bufs.f64_ref(x_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(y_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(z_idx)),
                    GpuBufferRef::U8(bufs.u8_ref(perm_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(amp_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(lac_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(org_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(res_idx)),
                ],
            );
            if ok {
                bufs.download_f64(&self.device, res_idx, results)?;
            } else {
                cpu_octave_batch(sampler, positions, results);
            }
        } else {
            // 标准 AoS 路径
            let pos_idx = bufs.alloc_f64(&self.device, n * 3)?;
            bufs.upload_f64(&self.device, pos_idx, positions)?;

            let ok = self.try_launch(
                "octave_perlin_sample_f64",
                n,
                vec![
                    KernelArg::BufferRef(0),
                    KernelArg::BufferRef(1),
                    KernelArg::BufferRef(2),
                    KernelArg::BufferRef(3),
                    KernelArg::BufferRef(4),
                    KernelArg::BufferRef(5),
                    KernelArg::I32(n as i32),
                    KernelArg::I32(m as i32),
                ],
                vec![
                    GpuBufferRef::F64(bufs.f64_ref(pos_idx)),
                    GpuBufferRef::U8(bufs.u8_ref(perm_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(amp_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(lac_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(org_idx)),
                    GpuBufferRef::F64(bufs.f64_ref(res_idx)),
                ],
            );
            if ok {
                bufs.download_f64(&self.device, res_idx, results)?;
            } else {
                cpu_octave_batch(sampler, positions, results);
            }
        }

        bufs.free_all(&self.device)?;
        Ok(())
    }

    /// 使用 JIT 特化 kernel 进行八度 Perlin 批量采样。
    ///
    /// 仅在八度数 ≤ 16 时使用 JIT 特化。
    /// 如果 JIT kernel 不可用，回退到标准 kernel 或 CPU 路径。
    #[cfg(feature = "pumpkin-util")]
    pub fn sample_octave_jit(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_octave_batch(sampler, positions, results);
            return Ok(());
        }

        // 获取采样器配置
        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let config = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);

        // 尝试 JIT 特化
        let max_unroll = crate::jit::get_jit_max_unroll();
        if let Some(jit_kernel) = crate::jit::specialize_octave_perlin(&config, max_unroll) {
            let m = config.num_octaves();

            // 分配缓冲区
            let mut d_pos = self.device.alloc_f64(n * 3)?;
            let d_res = self.device.alloc_f64(n)?;
            let mut d_perm = self.device.alloc_u8(m * 256)?;

            self.device.copy_to_device(&mut d_pos, positions)?;
            self.device
                .copy_to_device(&mut d_perm, &config.packed_permutations())?;

            // 确保 JIT kernel 已编译
            if !self
                .device
                .kernel_launcher()
                .is_some_and(|l| l.has_kernel(&jit_kernel.name))
            {
                if let Err(e) = self.device.compile_jit_kernel(&jit_kernel) {
                    tracing::debug!("JIT compile failed for '{}': {e}", jit_kernel.name);
                }
            }

            // 尝试 JIT launch
            let ok = self.try_launch(
                &jit_kernel.name,
                n,
                vec![
                    KernelArg::BufferRef(0), // pos
                    KernelArg::BufferRef(1), // perms
                    KernelArg::BufferRef(2), // res
                    KernelArg::I32(n as i32),
                ],
                vec![
                    GpuBufferRef::F64(&d_pos),
                    GpuBufferRef::U8(&d_perm),
                    GpuBufferRef::F64(&d_res),
                ],
            );
            if ok {
                self.device.copy_from_device(&d_res, results)?;
                self.device.free(d_pos)?;
                self.device.free(d_res)?;
                self.device.free(d_perm)?;
                return Ok(());
            }

            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_perm)?;
        }

        // JIT 不可用，回退到标准路径
        self.sample_octave_batch(sampler, positions, results)
    }

    // ========== Double Perlin ==========

    /// 使用 JIT 特化 kernel 进行双 Perlin 批量采样。
    ///
    /// 八度数均 ≤ max_unroll 时生成专用 kernel，
    /// 将两组振幅/间隙/原点硬编码为常量，展开两个循环。
    #[cfg(feature = "pumpkin-util")]
    pub fn sample_double_perlin_jit(
        &mut self,
        first: &OctavePerlinNoiseSampler,
        second: &OctavePerlinNoiseSampler,
        amplitude: f64,
        positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_double_perlin_batch(first, second, amplitude, positions, results);
            return Ok(());
        }

        let k1 = std::ptr::from_ref(first) as u64;
        let k2 = std::ptr::from_ref(second) as u64;

        let g1 = self.cache.get_or_insert(k1, first);
        let c1 = g1
            .get(&k1)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(first));
        drop(g1);
        let g2 = self.double_cache.get_or_insert(k2, second);
        let c2 = g2
            .get(&k2)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(second));
        drop(g2);

        let max_unroll = crate::jit::get_jit_max_unroll();
        if let Some(jit_kernel) =
            crate::jit::specialize_double_perlin(&c1, &c2, amplitude, max_unroll)
        {
            let m1 = c1.num_octaves();
            let m2 = c2.num_octaves();

            let mut d_pos = self.device.alloc_f64(n * 3)?;
            let d_res = self.device.alloc_f64(n)?;
            let mut d_perm1 = self.device.alloc_u8(m1 * 256)?;
            let mut d_perm2 = self.device.alloc_u8(m2 * 256)?;

            self.device.copy_to_device(&mut d_pos, positions)?;
            self.device
                .copy_to_device(&mut d_perm1, &c1.packed_permutations())?;
            self.device
                .copy_to_device(&mut d_perm2, &c2.packed_permutations())?;

            if !self
                .device
                .kernel_launcher()
                .is_some_and(|l| l.has_kernel(&jit_kernel.name))
            {
                let _ = self.device.compile_jit_kernel(&jit_kernel);
            }

            let ok = self.try_launch(
                &jit_kernel.name,
                n,
                vec![
                    KernelArg::BufferRef(0),
                    KernelArg::BufferRef(1),
                    KernelArg::BufferRef(2),
                    KernelArg::BufferRef(3),
                    KernelArg::I32(n as i32),
                ],
                vec![
                    GpuBufferRef::F64(&d_pos),
                    GpuBufferRef::U8(&d_perm1),
                    GpuBufferRef::U8(&d_perm2),
                    GpuBufferRef::F64(&d_res),
                ],
            );
            if ok {
                self.device.copy_from_device(&d_res, results)?;
                self.device.free(d_pos)?;
                self.device.free(d_res)?;
                self.device.free(d_perm1)?;
                self.device.free(d_perm2)?;
                return Ok(());
            }
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_perm1)?;
            self.device.free(d_perm2)?;
        }

        // 回退标准路径
        self.sample_double_perlin_batch(first, second, amplitude, positions, results)
    }

    #[cfg(feature = "pumpkin-util")]
    pub fn sample_double_perlin_batch(
        &mut self,
        first: &OctavePerlinNoiseSampler,
        second: &OctavePerlinNoiseSampler,
        amplitude: f64,
        positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(positions.len(), n * 3);

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_double_perlin_batch(first, second, amplitude, positions, results);
            return Ok(());
        }

        let k1 = std::ptr::from_ref(first) as u64;
        let k2 = std::ptr::from_ref(second) as u64;

        let g1 = self.cache.get_or_insert(k1, first);
        let c1 = g1
            .get(&k1)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(first));
        drop(g1);
        let g2 = self.double_cache.get_or_insert(k2, second);
        let c2 = g2
            .get(&k2)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(second));
        drop(g2);

        let m1 = c1.num_octaves();
        let m2 = c2.num_octaves();

        let mut bufs = GpuBufferSet::new();
        let pos_idx = bufs.alloc_f64(&self.device, n * 3)?;
        let res_idx = bufs.alloc_f64(&self.device, n)?;
        let (p1_idx, a1_idx, l1_idx, o1_idx) = bufs.load_octave_config(&self.device, &c1)?;
        let (p2_idx, a2_idx, l2_idx, o2_idx) = bufs.load_octave_config(&self.device, &c2)?;

        bufs.upload_f64(&self.device, pos_idx, positions)?;

        let ok = self.try_launch(
            "double_perlin_sample_f64",
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::BufferRef(3),
                KernelArg::BufferRef(4),
                KernelArg::BufferRef(5),
                KernelArg::BufferRef(6),
                KernelArg::BufferRef(7),
                KernelArg::BufferRef(8),
                KernelArg::F64(amplitude),
                KernelArg::BufferRef(9),
                KernelArg::I32(n as i32),
                KernelArg::I32(m1 as i32),
                KernelArg::I32(m2 as i32),
            ],
            vec![
                GpuBufferRef::F64(bufs.f64_ref(pos_idx)),
                GpuBufferRef::U8(bufs.u8_ref(p1_idx)),
                GpuBufferRef::F64(bufs.f64_ref(a1_idx)),
                GpuBufferRef::F64(bufs.f64_ref(l1_idx)),
                GpuBufferRef::F64(bufs.f64_ref(o1_idx)),
                GpuBufferRef::U8(bufs.u8_ref(p2_idx)),
                GpuBufferRef::F64(bufs.f64_ref(a2_idx)),
                GpuBufferRef::F64(bufs.f64_ref(l2_idx)),
                GpuBufferRef::F64(bufs.f64_ref(o2_idx)),
                GpuBufferRef::F64(bufs.f64_ref(res_idx)),
            ],
        );
        if ok {
            bufs.download_f64(&self.device, res_idx, results)?;
        } else {
            cpu_double_perlin_batch(first, second, amplitude, positions, results);
        }

        bufs.free_all(&self.device)?;
        Ok(())
    }

    // ========== ShiftA / ShiftB ==========

    #[cfg(feature = "pumpkin-util")]
    pub fn sample_shift_a_batch(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        xz_positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(xz_positions.len(), n * 2);
        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_shift_a_batch(sampler, xz_positions, results);
            return Ok(());
        }
        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let config = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);
        let m = config.num_octaves();

        let mut bufs = GpuBufferSet::new();
        let pos_idx = bufs.alloc_f64(&self.device, n * 2)?;
        let res_idx = bufs.alloc_f64(&self.device, n)?;
        let (perm_idx, amp_idx, lac_idx, org_idx) =
            bufs.load_octave_config(&self.device, &config)?;
        bufs.upload_f64(&self.device, pos_idx, xz_positions)?;

        let ok = self.try_launch(
            "shift_a_sample_f64",
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::BufferRef(3),
                KernelArg::BufferRef(4),
                KernelArg::BufferRef(5),
                KernelArg::I32(n as i32),
                KernelArg::I32(m as i32),
            ],
            vec![
                GpuBufferRef::F64(bufs.f64_ref(pos_idx)),
                GpuBufferRef::U8(bufs.u8_ref(perm_idx)),
                GpuBufferRef::F64(bufs.f64_ref(amp_idx)),
                GpuBufferRef::F64(bufs.f64_ref(lac_idx)),
                GpuBufferRef::F64(bufs.f64_ref(org_idx)),
                GpuBufferRef::F64(bufs.f64_ref(res_idx)),
            ],
        );
        if ok {
            bufs.download_f64(&self.device, res_idx, results)?;
        } else {
            cpu_shift_a_batch(sampler, xz_positions, results);
        }

        bufs.free_all(&self.device)?;
        Ok(())
    }

    /// 使用 JIT 特化 kernel 进行 ShiftA 批量采样。
    #[cfg(feature = "pumpkin-util")]
    pub fn sample_shift_a_jit(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        xz_positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(xz_positions.len(), n * 2);
        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_shift_a_batch(sampler, xz_positions, results);
            return Ok(());
        }
        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let config = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);

        let max_unroll = crate::jit::get_jit_max_unroll();
        if let Some(jit_kernel) = crate::jit::specialize_shift("shift_a", &config, max_unroll) {
            let m = config.num_octaves();
            let mut d_pos = self.device.alloc_f64(n * 2)?;
            let d_res = self.device.alloc_f64(n)?;
            let mut d_perm = self.device.alloc_u8(m * 256)?;
            self.device.copy_to_device(&mut d_pos, xz_positions)?;
            self.device
                .copy_to_device(&mut d_perm, &config.packed_permutations())?;

            if !self
                .device
                .kernel_launcher()
                .is_some_and(|l| l.has_kernel(&jit_kernel.name))
            {
                let _ = self.device.compile_jit_kernel(&jit_kernel);
            }
            let ok = self.try_launch(
                &jit_kernel.name,
                n,
                vec![
                    KernelArg::BufferRef(0),
                    KernelArg::BufferRef(1),
                    KernelArg::BufferRef(2),
                    KernelArg::I32(n as i32),
                ],
                vec![
                    GpuBufferRef::F64(&d_pos),
                    GpuBufferRef::U8(&d_perm),
                    GpuBufferRef::F64(&d_res),
                ],
            );
            if ok {
                self.device.copy_from_device(&d_res, results)?;
                self.device.free(d_pos)?;
                self.device.free(d_res)?;
                self.device.free(d_perm)?;
                return Ok(());
            }
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_perm)?;
        }
        self.sample_shift_a_batch(sampler, xz_positions, results)
    }

    #[cfg(feature = "pumpkin-util")]
    pub fn sample_shift_b_batch(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        zx_positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(zx_positions.len(), n * 2);
        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_shift_b_batch(sampler, zx_positions, results);
            return Ok(());
        }
        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let config = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);
        let m = config.num_octaves();

        let mut bufs = GpuBufferSet::new();
        let pos_idx = bufs.alloc_f64(&self.device, n * 2)?;
        let res_idx = bufs.alloc_f64(&self.device, n)?;
        let (perm_idx, amp_idx, lac_idx, org_idx) =
            bufs.load_octave_config(&self.device, &config)?;
        bufs.upload_f64(&self.device, pos_idx, zx_positions)?;

        let ok = self.try_launch(
            "shift_b_sample_f64",
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::BufferRef(3),
                KernelArg::BufferRef(4),
                KernelArg::BufferRef(5),
                KernelArg::I32(n as i32),
                KernelArg::I32(m as i32),
            ],
            vec![
                GpuBufferRef::F64(bufs.f64_ref(pos_idx)),
                GpuBufferRef::U8(bufs.u8_ref(perm_idx)),
                GpuBufferRef::F64(bufs.f64_ref(amp_idx)),
                GpuBufferRef::F64(bufs.f64_ref(lac_idx)),
                GpuBufferRef::F64(bufs.f64_ref(org_idx)),
                GpuBufferRef::F64(bufs.f64_ref(res_idx)),
            ],
        );
        if ok {
            bufs.download_f64(&self.device, res_idx, results)?;
        } else {
            cpu_shift_b_batch(sampler, zx_positions, results);
        }

        bufs.free_all(&self.device)?;
        Ok(())
    }

    /// 使用 JIT 特化 kernel 进行 ShiftB 批量采样。
    #[cfg(feature = "pumpkin-util")]
    pub fn sample_shift_b_jit(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        zx_positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(zx_positions.len(), n * 2);
        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_shift_b_batch(sampler, zx_positions, results);
            return Ok(());
        }
        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let config = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);

        let max_unroll = crate::jit::get_jit_max_unroll();
        if let Some(jit_kernel) = crate::jit::specialize_shift("shift_b", &config, max_unroll) {
            let m = config.num_octaves();
            let mut d_pos = self.device.alloc_f64(n * 2)?;
            let d_res = self.device.alloc_f64(n)?;
            let mut d_perm = self.device.alloc_u8(m * 256)?;
            self.device.copy_to_device(&mut d_pos, zx_positions)?;
            self.device
                .copy_to_device(&mut d_perm, &config.packed_permutations())?;

            if !self
                .device
                .kernel_launcher()
                .is_some_and(|l| l.has_kernel(&jit_kernel.name))
            {
                let _ = self.device.compile_jit_kernel(&jit_kernel);
            }
            let ok = self.try_launch(
                &jit_kernel.name,
                n,
                vec![
                    KernelArg::BufferRef(0),
                    KernelArg::BufferRef(1),
                    KernelArg::BufferRef(2),
                    KernelArg::I32(n as i32),
                ],
                vec![
                    GpuBufferRef::F64(&d_pos),
                    GpuBufferRef::U8(&d_perm),
                    GpuBufferRef::F64(&d_res),
                ],
            );
            if ok {
                self.device.copy_from_device(&d_res, results)?;
                self.device.free(d_pos)?;
                self.device.free(d_res)?;
                self.device.free(d_perm)?;
                return Ok(());
            }
            self.device.free(d_pos)?;
            self.device.free(d_res)?;
            self.device.free(d_perm)?;
        }
        self.sample_shift_b_batch(sampler, zx_positions, results)
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

    pub fn batch_trilinear(
        &mut self,
        corners: &[f64],
        deltas: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        assert_eq!(corners.len(), n * 8);
        assert_eq!(deltas.len(), n * 3);
        if n == 0 {
            return Ok(());
        }

        if self.device.device_type() == crate::DeviceType::Cpu {
            cpu_trilinear(corners, deltas, results);
            return Ok(());
        }

        let mut d_c = self.device.alloc_f64(n * 8)?;
        let mut d_d = self.device.alloc_f64(n * 3)?;
        let d_r = self.device.alloc_f64(n)?;
        self.device.copy_to_device(&mut d_c, corners)?;
        self.device.copy_to_device(&mut d_d, deltas)?;
        let ok = self.try_launch(
            "trilinear_interpolate_f64",
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::I32(n as i32),
            ],
            vec![
                GpuBufferRef::F64(&d_c),
                GpuBufferRef::F64(&d_d),
                GpuBufferRef::F64(&d_r),
            ],
        );
        if ok {
            self.device.copy_from_device(&d_r, results)?;
        } else {
            cpu_trilinear(corners, deltas, results);
        }
        self.device.free(d_c)?;
        self.device.free(d_d)?;
        self.device.free(d_r)?;
        Ok(())
    }

    // ==== FlatCache 2D Precomputation ====

    #[cfg(feature = "pumpkin-util")]
    pub fn precompute_flatcache(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        xz: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        assert_eq!(xz.len(), n * 2);
        if n == 0 {
            return Ok(());
        }

        if self.device.device_type() == crate::DeviceType::Cpu {
            for i in 0..n {
                results[i] = sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
            }
            return Ok(());
        }

        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let c = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);
        let m = c.num_octaves();
        let mut d_pos = self.device.alloc_f64(n * 2)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_perm = self.device.alloc_u8(m * 256)?;
        let mut d_amp = self.device.alloc_f64(m)?;
        let mut d_lac = self.device.alloc_f64(m)?;
        let mut d_org = self.device.alloc_f64(m * 3)?;
        self.device.copy_to_device(&mut d_pos, xz)?;
        self.device
            .copy_to_device(&mut d_perm, &c.packed_permutations())?;
        self.device
            .copy_to_device(&mut d_amp, &c.packed_amplitudes())?;
        self.device
            .copy_to_device(&mut d_lac, &c.packed_lacunarities())?;
        self.device
            .copy_to_device(&mut d_org, &c.packed_origins())?;
        let ok = self.try_launch(
            "flatcache_precompute_f64",
            n,
            vec![
                KernelArg::BufferRef(0),
                KernelArg::BufferRef(1),
                KernelArg::BufferRef(2),
                KernelArg::BufferRef(3),
                KernelArg::BufferRef(4),
                KernelArg::BufferRef(5),
                KernelArg::I32(n as i32),
                KernelArg::I32(m as i32),
            ],
            vec![
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::U8(&d_perm),
                GpuBufferRef::F64(&d_amp),
                GpuBufferRef::F64(&d_lac),
                GpuBufferRef::F64(&d_org),
                GpuBufferRef::F64(&d_res),
            ],
        );
        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            for i in 0..n {
                results[i] = sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
            }
        }
        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_perm)?;
        self.device.free(d_amp)?;
        self.device.free(d_lac)?;
        self.device.free(d_org)?;
        Ok(())
    }
}

// ========== CPU fallbacks ==========

#[cfg(feature = "pumpkin-util")]
fn cpu_octave_batch(sampler: &OctavePerlinNoiseSampler, pos: &[f64], res: &mut [f64]) {
    for i in 0..res.len() {
        res[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
}

#[cfg(feature = "pumpkin-util")]
fn cpu_double_perlin_batch(
    first: &OctavePerlinNoiseSampler,
    second: &OctavePerlinNoiseSampler,
    amplitude: f64,
    pos: &[f64],
    res: &mut [f64],
) {
    let c = 1.0181268882175227f64;
    for i in 0..res.len() {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        res[i] = (first.sample(x, y, z) + second.sample(x * c, y * c, z * c)) * amplitude;
    }
}

#[cfg(feature = "pumpkin-util")]
fn cpu_shift_a_batch(sampler: &OctavePerlinNoiseSampler, xz: &[f64], res: &mut [f64]) {
    for i in 0..res.len() {
        res[i] = sampler.sample(xz[i * 2] * 0.25, 0.0, xz[i * 2 + 1] * 0.25) * 4.0;
    }
}

#[cfg(feature = "pumpkin-util")]
fn cpu_shift_b_batch(sampler: &OctavePerlinNoiseSampler, zx: &[f64], res: &mut [f64]) {
    for i in 0..res.len() {
        res[i] = sampler.sample(zx[i * 2 + 1] * 0.25, 0.0, zx[i * 2] * 0.25) * 4.0;
    }
}

fn cpu_trilinear(corners: &[f64], deltas: &[f64], results: &mut [f64]) {
    for i in 0..results.len() {
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

#[cfg(all(test, feature = "pumpkin-util"))]
mod tests {
    use super::*;
    use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

    const SEED: u64 = 138_782_381_985_206;

    fn mk_sampler(octaves: &[i32]) -> OctavePerlinNoiseSampler {
        let r = Xoroshiro::from_seed(SEED);
        let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
        let mut g = RandomGenerator::Xoroshiro(r);
        OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
    }

    fn mk_pos(n: usize) -> Vec<f64> {
        let mut p = Vec::with_capacity(n * 3);
        let mut s = SEED;
        for _ in 0..n {
            p.push((s.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8);
            s = s.wrapping_mul(1442695040888963407);
            p.push((s as f64) * 1e-8);
            s = s.wrapping_mul(1442695040888963407);
            p.push((s as f64) * 1e-8);
        }
        p
    }

    fn fnv1a(d: &[f64]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &v in d {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    #[test]
    fn octave_consistency() {
        let sampler = mk_sampler(&[0, 1, 2]);
        let pos = mk_pos(1024);
        let n = 1024;
        let mut cpu = vec![0f64; n];
        let mut gpu = vec![0f64; n];
        cpu_octave_batch(&sampler, &pos, &mut cpu);
        let mut s = GpuNoiseSampler::new(GpuDevice::init());
        s.sample_octave_batch(&sampler, &pos, &mut gpu).unwrap();
        assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "octave hash mismatch");
    }

    #[test]
    fn double_perlin_consistency() {
        let a = mk_sampler(&[0, 1, 2]);
        let b = mk_sampler(&[1, 2, 3]);
        let pos = mk_pos(1024);
        let n = 1024;
        let mut cpu = vec![0f64; n];
        let mut gpu = vec![0f64; n];
        cpu_double_perlin_batch(&a, &b, 0.5, &pos, &mut cpu);
        let mut s = GpuNoiseSampler::new(GpuDevice::init());
        s.sample_double_perlin_batch(&a, &b, 0.5, &pos, &mut gpu)
            .unwrap();
        assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "double_perlin hash mismatch");
    }

    #[test]
    fn shift_a_consistency() {
        let sampler = mk_sampler(&[0, 1]);
        let xz: Vec<f64> = mk_pos(512)
            .iter()
            .step_by(3)
            .copied()
            .zip(mk_pos(512).iter().skip(2).step_by(3).copied())
            .flat_map(|(x, z)| [x, z])
            .collect();
        let n = xz.len() / 2;
        let mut cpu = vec![0f64; n];
        let mut gpu = vec![0f64; n];
        cpu_shift_a_batch(&sampler, &xz, &mut cpu);
        let mut s = GpuNoiseSampler::new(GpuDevice::init());
        s.sample_shift_a_batch(&sampler, &xz, &mut gpu).unwrap();
        assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "shift_a hash mismatch");
    }

    #[test]
    fn shift_b_consistency() {
        let sampler = mk_sampler(&[0, 1]);
        let zx: Vec<f64> = mk_pos(512)
            .iter()
            .skip(2)
            .step_by(3)
            .copied()
            .zip(mk_pos(512).iter().step_by(3).copied())
            .flat_map(|(z, x)| [z, x])
            .collect();
        let n = zx.len() / 2;
        let mut cpu = vec![0f64; n];
        let mut gpu = vec![0f64; n];
        cpu_shift_b_batch(&sampler, &zx, &mut cpu);
        let mut s = GpuNoiseSampler::new(GpuDevice::init());
        s.sample_shift_b_batch(&sampler, &zx, &mut gpu).unwrap();
        assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "shift_b hash mismatch");
    }
}
