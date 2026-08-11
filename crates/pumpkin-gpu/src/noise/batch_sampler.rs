//! GPU 噪声批量采样器 — 全类型支持。
//!
//! 支持：Octave, `DoublePerlin`, ShiftA/B, `ShiftedNoise`, `InterpolatedNoise`,
//! `VeinNoise`, `AquiferDensity`。
#![allow(
    clippy::separated_literal_suffix,
    clippy::as_ptr_cast_mut,
    clippy::ptr_as_ptr
)]

use crate::GpuDevice;
use crate::common::DeviceError;
use crate::common::kernel::GpuBufferRef;
use crate::common::kernel::KernelArg;
use crate::noise::cache::{NoiseCache, SerializedOctaveConfig};

#[cfg(feature = "pumpkin-util")]
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

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
        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_perm = self.device.alloc_u8(m * 256)?;
        let mut d_amp = self.device.alloc_f64(m)?;
        let mut d_lac = self.device.alloc_f64(m)?;
        let mut d_org = self.device.alloc_f64(m * 3)?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device
            .copy_to_device(&mut d_perm, &config.packed_permutations())?;
        self.device
            .copy_to_device(&mut d_amp, &config.packed_amplitudes())?;
        self.device
            .copy_to_device(&mut d_lac, &config.packed_lacunarities())?;
        self.device
            .copy_to_device(&mut d_org, &config.packed_origins())?;

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
            cpu_octave_batch(sampler, positions, results);
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_perm)?;
        self.device.free(d_amp)?;
        self.device.free(d_lac)?;
        self.device.free(d_org)?;
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
        if let Some(jit_kernel) = crate::jit::specialize_octave_perlin(&config) {
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
            let ok = self.try_launch(&jit_kernel.name, n, vec![KernelArg::I32(n as i32)], vec![]);
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

        let mut d_pos = self.device.alloc_f64(n * 3)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_p1 = self.device.alloc_u8(m1 * 256)?;
        let mut d_a1 = self.device.alloc_f64(m1)?;
        let mut d_l1 = self.device.alloc_f64(m1)?;
        let mut d_o1 = self.device.alloc_f64(m1 * 3)?;
        let mut d_p2 = self.device.alloc_u8(m2 * 256)?;
        let mut d_a2 = self.device.alloc_f64(m2)?;
        let mut d_l2 = self.device.alloc_f64(m2)?;
        let mut d_o2 = self.device.alloc_f64(m2 * 3)?;

        self.device.copy_to_device(&mut d_pos, positions)?;
        self.device
            .copy_to_device(&mut d_p1, &c1.packed_permutations())?;
        self.device
            .copy_to_device(&mut d_a1, &c1.packed_amplitudes())?;
        self.device
            .copy_to_device(&mut d_l1, &c1.packed_lacunarities())?;
        self.device
            .copy_to_device(&mut d_o1, &c1.packed_origins())?;
        self.device
            .copy_to_device(&mut d_p2, &c2.packed_permutations())?;
        self.device
            .copy_to_device(&mut d_a2, &c2.packed_amplitudes())?;
        self.device
            .copy_to_device(&mut d_l2, &c2.packed_lacunarities())?;
        self.device
            .copy_to_device(&mut d_o2, &c2.packed_origins())?;

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
                GpuBufferRef::F64(&d_pos),
                GpuBufferRef::U8(&d_p1),
                GpuBufferRef::F64(&d_a1),
                GpuBufferRef::F64(&d_l1),
                GpuBufferRef::F64(&d_o1),
                GpuBufferRef::U8(&d_p2),
                GpuBufferRef::F64(&d_a2),
                GpuBufferRef::F64(&d_l2),
                GpuBufferRef::F64(&d_o2),
                GpuBufferRef::F64(&d_res),
            ],
        );
        if ok {
            self.device.copy_from_device(&d_res, results)?;
        } else {
            cpu_double_perlin_batch(first, second, amplitude, positions, results);
        }

        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_p1)?;
        self.device.free(d_a1)?;
        self.device.free(d_l1)?;
        self.device.free(d_o1)?;
        self.device.free(d_p2)?;
        self.device.free(d_a2)?;
        self.device.free(d_l2)?;
        self.device.free(d_o2)?;
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
        let mut d_pos = self.device.alloc_f64(n * 2)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_perm = self.device.alloc_u8(m * 256)?;
        let mut d_amp = self.device.alloc_f64(m)?;
        let mut d_lac = self.device.alloc_f64(m)?;
        let mut d_org = self.device.alloc_f64(m * 3)?;
        self.device.copy_to_device(&mut d_pos, xz_positions)?;
        self.device
            .copy_to_device(&mut d_perm, &config.packed_permutations())?;
        self.device
            .copy_to_device(&mut d_amp, &config.packed_amplitudes())?;
        self.device
            .copy_to_device(&mut d_lac, &config.packed_lacunarities())?;
        self.device
            .copy_to_device(&mut d_org, &config.packed_origins())?;
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
            cpu_shift_a_batch(sampler, xz_positions, results);
        }
        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_perm)?;
        self.device.free(d_amp)?;
        self.device.free(d_lac)?;
        self.device.free(d_org)?;
        Ok(())
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
        let mut d_pos = self.device.alloc_f64(n * 2)?;
        let d_res = self.device.alloc_f64(n)?;
        let mut d_perm = self.device.alloc_u8(m * 256)?;
        let mut d_amp = self.device.alloc_f64(m)?;
        let mut d_lac = self.device.alloc_f64(m)?;
        let mut d_org = self.device.alloc_f64(m * 3)?;
        self.device.copy_to_device(&mut d_pos, zx_positions)?;
        self.device
            .copy_to_device(&mut d_perm, &config.packed_permutations())?;
        self.device
            .copy_to_device(&mut d_amp, &config.packed_amplitudes())?;
        self.device
            .copy_to_device(&mut d_lac, &config.packed_lacunarities())?;
        self.device
            .copy_to_device(&mut d_org, &config.packed_origins())?;
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
            cpu_shift_b_batch(sampler, zx_positions, results);
        }
        self.device.free(d_pos)?;
        self.device.free(d_res)?;
        self.device.free(d_perm)?;
        self.device.free(d_amp)?;
        self.device.free(d_lac)?;
        self.device.free(d_org)?;
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

    // ==== Trilinear Interpolation ====

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
