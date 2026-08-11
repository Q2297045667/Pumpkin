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
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU batch sky fill failed".into(),
                ),
                "light_accel::batch_sky_fill",
            );
        }
        for col in 0..n {
            let top = hm[col];
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
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU batch block scan failed".into(),
                ),
                "light_accel::batch_block_scan",
            );
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
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU iterative propagate failed".into(),
                ),
                "light_accel::iterative_propagate",
            );
        }
        let mut it = 0;
        for _ in 0..max_iters {
            let mut ch = false;
            for i in 0..n {
                let cur = light[i];
                let _ = &opacity[i];
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

    /// GPU 天空光水平传播 + 向下级联（CPU fallback included）。
    ///
    /// 在垂直填充后调用，迭代执行水平传播（4 方向衰减 1）
    /// 和向下级联（15 透过空气保持 15），直至收敛。
    ///
    /// 返回实际执行迭代次数。
    #[allow(clippy::too_many_lines)]
    pub fn sky_horizontal_propagate(
        &mut self,
        sky_light: &mut [u8],
        opacity: &[u8],
        width: usize,
        depth: usize,
        height: usize,
        max_iters: usize,
    ) -> usize {
        #[cfg(feature = "gpu")]
        if let Some(ref mut s) = self.inner {
            if let Ok(it) =
                s.sky_horizontal_propagate(sky_light, opacity, width, depth, height, max_iters)
            {
                return it;
            }
            pumpkin_gpu::logging::log_fallback(
                &pumpkin_gpu::logging::FallbackReason::UnsupportedOperation(
                    "GPU sky horizontal propagate failed".into(),
                ),
                "light_accel::sky_horizontal_propagate",
            );
        }
        // CPU fallback: iterative 2D horizontal BFS + downward cascade
        let stride_x = height;
        let stride_z = width * height;
        let mut iterations = 0;
        for _ in 0..max_iters {
            let mut changed = false;
            for z in 0..depth {
                for x in 0..width {
                    for y in (0..height).rev() {
                        let idx = z * stride_z + x * stride_x + y;
                        let cur = sky_light[idx];
                        let mut best = cur;

                        // Horizontal: 4 neighbors
                        if x > 0 {
                            let nl = sky_light[idx - stride_x];
                            if nl > 1 && nl - 1 > best {
                                best = nl - 1;
                            }
                        }
                        if x < width - 1 {
                            let nl = sky_light[idx + stride_x];
                            if nl > 1 && nl - 1 > best {
                                best = nl - 1;
                            }
                        }
                        if z > 0 {
                            let nl = sky_light[idx - stride_z];
                            if nl > 1 && nl - 1 > best {
                                best = nl - 1;
                            }
                        }
                        if z < depth - 1 {
                            let nl = sky_light[idx + stride_z];
                            if nl > 1 && nl - 1 > best {
                                best = nl - 1;
                            }
                        }

                        // Downward cascade: light 15 through air
                        if y < height - 1 {
                            let above = sky_light[idx + 1];
                            if above == 15 && opacity[idx] == 0 && 15 > best {
                                best = 15;
                            }
                        }

                        if best > cur {
                            sky_light[idx] = best;
                            changed = true;
                        }
                    }
                }
            }
            iterations += 1;
            if !changed {
                break;
            }
        }
        iterations
    }
}
