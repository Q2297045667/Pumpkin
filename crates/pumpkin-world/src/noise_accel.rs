//! 噪声阶段的 GPU 加速接口。

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
            if i.sample_octave_batch(s, pos, res).is_ok() {
                return;
            }
        }
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
            if i.sample_double_perlin_batch(a, b, amp, pos, res).is_ok() {
                return;
            }
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
            if i.sample_shift_a_batch(s, xz, res).is_ok() {
                return;
            }
        }
        for i in 0..res.len() {
            res[i] = s.sample(xz[i * 2] * 0.25, 0.0, xz[i * 2 + 1] * 0.25) * 4.0;
        }
    }
    pub fn sample_shift_b(&mut self, s: &OctavePerlinNoiseSampler, zx: &[f64], res: &mut [f64]) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut i) = self.inner {
            if i.sample_shift_b_batch(s, zx, res).is_ok() {
                return;
            }
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
        if let Some(ref mut i) = self.inner {
            if i.precompute_flatcache(s, xz, results).is_ok() {
                return;
            }
        }
        for i in 0..results.len() {
            results[i] = s.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
        }
    }
}
