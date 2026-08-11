//! 光照阶段的 GPU 加速接口。

use pumpkin_config::gpu::GpuConfig;

/// 光照 GPU 加速器。
pub struct LightAccelerator {
    #[cfg(feature = "gpu")]
    inner: Option<pumpkin_gpu::light::GpuLightSampler>,
    #[cfg(not(feature = "gpu"))]
    _config: GpuConfig,
}

impl LightAccelerator {
    #[must_use]
    pub fn new(config: &GpuConfig) -> Self {
        #[cfg(feature = "gpu")]
        {
            if config.enabled && config.light_acceleration {
                let device = pumpkin_gpu::GpuDevice::from_config(config);
                if device.device_type() != pumpkin_gpu::DeviceType::Cpu {
                    tracing::info!("光照加速已启用: {}", device.device_name());
                    return Self {
                        inner: Some(pumpkin_gpu::light::GpuLightSampler::new(device)),
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

    /// GPU 批量天空光填充 (CPU fallback included).
    pub fn batch_sky_fill(&mut self, hm: &[i32], op: &[u8], sl: &mut [u8], n: usize, h: usize) {
        #[cfg(feature = "gpu")]
        if let Some(ref mut s) = self.inner {
            if s.batch_sky_fill(hm, op, sl, n, h).is_ok() {
                return;
            }
        }
        for col in 0..n {
            let top = hm[col] as i32;
            for y in (top + 1)..h as i32 {
                sl[col * h + y as usize] = 15;
            }
            let mut lt: u8 = 15;
            for y in (0..=top).rev() {
                let i = col * h + y as usize;
                lt = lt.saturating_sub(op[i]);
                sl[i] = lt;
            }
        }
    }

    /// GPU 批量方块光扫描 (returns source indices).
    pub fn batch_block_scan(&mut self, lum: &[u8], bl: &mut [u8], n: usize) -> Vec<i32> {
        #[cfg(feature = "gpu")]
        if let Some(ref mut s) = self.inner {
            if let Ok(src) = s.batch_block_scan(lum, bl, n) {
                return src;
            }
        }
        let mut src = Vec::new();
        for i in 0..n {
            bl[i] = lum[i];
            if lum[i] > 0 {
                src.push(i as i32);
            }
        }
        src
    }

    /// GPU 迭代距离场传播 (returns iteration count).
    pub fn iterative_propagate(
        &mut self,
        light: &mut [u8],
        opacity: &[u8],
        neighbors: &[i32],
        n: usize,
        max_iters: usize,
    ) -> usize {
        #[cfg(feature = "gpu")]
        if let Some(ref mut s) = self.inner {
            if let Ok(it) = s.iterative_propagate(light, opacity, neighbors, n, max_iters) {
                return it;
            }
        }
        let mut it = 0;
        for _ in 0..max_iters {
            let mut ch = false;
            for i in 0..n {
                let cur = light[i];
                let _op = opacity[i];
                let mut best = cur;
                for d in 0..6 {
                    let ni = neighbors[i * 6 + d] as usize;
                    if ni < n {
                        let nl = light[ni];
                        let no = opacity[ni];
                        let p = if nl > 1 + no { nl - 1 - no } else { 0 };
                        if p > best {
                            best = p;
                        }
                    }
                }
                if best > cur {
                    light[i] = best;
                    ch = true;
                }
            }
            it += 1;
            if !ch {
                break;
            }
        }
        it
    }
}
