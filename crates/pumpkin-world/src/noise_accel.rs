//! 噪声阶段的 GPU 加速接口。
#![allow(clippy::doc_markdown)]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

pub struct NoiseAccelerator {
    #[cfg(feature = "gpu")]
    inner: Option<pumpkin_gpu::noise::GpuNoiseSampler>,
    #[cfg(not(feature = "gpu"))]
    _config: GpuConfig,
}

impl NoiseAccelerator {
    #[must_use]
    pub fn new(config: &GpuConfig) -> Self {
        #[cfg(feature = "gpu")]
        {
            if config.enabled && config.noise_acceleration {
                let device = pumpkin_gpu::GpuDevice::from_config(config);
                if device.device_type() != pumpkin_gpu::DeviceType::Cpu {
                    tracing::info!("噪声加速已启用: {}", device.device_name());
                    return Self {
                        inner: Some(pumpkin_gpu::noise::GpuNoiseSampler::new(device)),
                    };
                }
            }
            Self { inner: None }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Self {
                _config: config.clone(),
            }
        }
    }

    #[must_use]
    pub const fn is_active(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.inner.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    pub fn sample_octave(&mut self, s: &OctavePerlinNoiseSampler, pos: &[f64], res: &mut [f64]) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut i) = self.inner {
            // 尝试 JIT 路径（如果配置启用且八度数 ≤ 16）
            if i.sample_octave_jit(s, pos, res).is_ok() {
                return;
            }
            // 回退标准 GPU 路径
            if i.sample_octave_batch(s, pos, res).is_ok() {
                return;
            }
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU octave sample failed".into(),
                ),
                "noise_accel::sample_octave",
            );
        }
        // CPU 路径
        for i in 0..res.len() {
            res[i] = s.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        }
    }
    pub fn sample_double_perlin(
        &mut self,
        a: &OctavePerlinNoiseSampler,
        b: &OctavePerlinNoiseSampler,
        amp: f64,
        pos: &[f64],
        res: &mut [f64],
    ) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut i) = self.inner {
            // 尝试 JIT 路径
            if i.sample_double_perlin_jit(a, b, amp, pos, res).is_ok() {
                return;
            }
            // 回退标准 GPU 路径
            if i.sample_double_perlin_batch(a, b, amp, pos, res).is_ok() {
                return;
            }
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU double perlin sample failed".into(),
                ),
                "noise_accel::sample_double_perlin",
            );
        }
        let c = 1.0181268882175227f64;
        for i in 0..res.len() {
            let x = pos[i * 3];
            let y = pos[i * 3 + 1];
            let z = pos[i * 3 + 2];
            res[i] = (a.sample(x, y, z) + b.sample(x * c, y * c, z * c)) * amp;
        }
    }
    pub fn sample_shift_a(&mut self, s: &OctavePerlinNoiseSampler, xz: &[f64], res: &mut [f64]) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut i) = self.inner {
            if i.sample_shift_a_jit(s, xz, res).is_ok() {
                return;
            }
            if i.sample_shift_a_batch(s, xz, res).is_ok() {
                return;
            }
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU shift A sample failed".into(),
                ),
                "noise_accel::sample_shift_a",
            );
        }
        for i in 0..res.len() {
            res[i] = s.sample(xz[i * 2] * 0.25, 0.0, xz[i * 2 + 1] * 0.25) * 4.0;
        }
    }
    pub fn sample_shift_b(&mut self, s: &OctavePerlinNoiseSampler, zx: &[f64], res: &mut [f64]) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut i) = self.inner {
            if i.sample_shift_b_jit(s, zx, res).is_ok() {
                return;
            }
            if i.sample_shift_b_batch(s, zx, res).is_ok() {
                return;
            }
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU shift B sample failed".into(),
                ),
                "noise_accel::sample_shift_b",
            );
        }
        for i in 0..res.len() {
            res[i] = s.sample(zx[i * 2 + 1] * 0.25, 0.0, zx[i * 2] * 0.25) * 4.0;
        }
    }
    pub fn batch_trilinear(&mut self, corners: &[f64], deltas: &[f64], results: &mut [f64]) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut i) = self.inner {
            if i.batch_trilinear(corners, deltas, results).is_ok() {
                return;
            }
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU batch trilinear failed".into(),
                ),
                "noise_accel::batch_trilinear",
            );
        }
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
    pub fn precompute_flatcache(
        &mut self,
        s: &OctavePerlinNoiseSampler,
        xz: &[f64],
        results: &mut [f64],
    ) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut i) = self.inner
            && i.precompute_flatcache(s, xz, results).is_ok()
        {
            return;
        }
        #[cfg(feature = "gpu")]
        if self.inner.is_some() {
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU flatcache precompute failed".into(),
                ),
                "noise_accel::precompute_flatcache",
            );
        }
        for i in 0..results.len() {
            results[i] = s.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
        }
    }

    /// 预计算 Surface 阶段噪声缓存（256 列批量 `DoublePerlin` 采样）。
    #[allow(clippy::too_many_arguments)]
    pub fn precompute_surface(
        &mut self,
        surface_a: &OctavePerlinNoiseSampler,
        surface_b: &OctavePerlinNoiseSampler,
        surface_amp: f64,
        secondary_a: &OctavePerlinNoiseSampler,
        secondary_b: &OctavePerlinNoiseSampler,
        secondary_amp: f64,
        start_x: i32,
        start_z: i32,
    ) -> crate::generation::surface::CachedSurfaceNoise {
        let n = 256usize;
        let mut xz_flat = Vec::with_capacity(n * 2);
        for lx in 0i32..16 {
            for lz in 0i32..16 {
                xz_flat.push((start_x + lx) as f64);
                xz_flat.push((start_z + lz) as f64);
            }
        }
        let pos_3d: Vec<f64> = (0..n)
            .flat_map(|i| {
                let x = xz_flat[i * 2];
                let z = xz_flat[i * 2 + 1];
                [x, 0.0f64, z]
            })
            .collect();

        let mut surf = vec![0.0f64; n];
        let mut sec = vec![0.0f64; n];

        // GPU 路径：JIT → batch → CPU 级联回退。
        #[cfg(feature = "gpu")]
        let gpu_ok = self.inner.as_mut().is_some_and(|inner| {
            // 优先 JIT 特化路径（常量内联 + 循环展开，1.5-2.5× 加速）
            let surf_jit = inner.sample_double_perlin_jit(
                surface_a,
                surface_b,
                surface_amp,
                &pos_3d,
                &mut surf,
            );
            let sec_jit = inner.sample_double_perlin_jit(
                secondary_a,
                secondary_b,
                secondary_amp,
                &pos_3d,
                &mut sec,
            );
            if surf_jit.is_ok() && sec_jit.is_ok() {
                return true;
            }
            // JIT 失败 → 回退标准 batch 路径
            inner
                .sample_double_perlin_batch(surface_a, surface_b, surface_amp, &pos_3d, &mut surf)
                .is_ok()
                && inner
                    .sample_double_perlin_batch(
                        secondary_a,
                        secondary_b,
                        secondary_amp,
                        &pos_3d,
                        &mut sec,
                    )
                    .is_ok()
        });
        #[cfg(not(feature = "gpu"))]
        let gpu_ok = false;

        // CPU 回退（GPU 不可用或失败时）
        if !gpu_ok {
            #[cfg(feature = "gpu")]
            if self.inner.is_some() {
                pumpkin_gpu::logging::log_fallback(
                    &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                        "GPU surface precompute failed".into(),
                    ),
                    "noise_accel::precompute_surface",
                );
            }
            let c = 1.0181268882175227f64;
            for i in 0..n {
                let x = pos_3d[i * 3];
                let z = pos_3d[i * 3 + 2];
                surf[i] = (surface_a.sample(x, 0.0, z) + surface_b.sample(x * c, 0.0, z * c))
                    * surface_amp;
                sec[i] = (secondary_a.sample(x, 0.0, z) + secondary_b.sample(x * c, 0.0, z * c))
                    * secondary_amp;
            }
        }

        let mut cache = crate::generation::surface::CachedSurfaceNoise::new_empty();
        for i in 0..n {
            cache.surface[i] = surf[i];
            cache.secondary[i] = sec[i];
        }
        cache
    }
}
