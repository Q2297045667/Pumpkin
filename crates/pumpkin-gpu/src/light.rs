//! GPU 光照加速采样器。
//!
//! 提供天空光垂直填充、方块光扫描和迭代距离场传播的 GPU 批量加速。

use crate::GpuDevice;
use crate::common::DeviceError;
use crate::common::kernel::{GpuBufferRef, KernelArg};

/// GPU 光照加速器。
pub struct GpuLightSampler {
    device: GpuDevice,
}

impl GpuLightSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self { device }
    }

    /// GPU 批量天空光垂直填充。
    ///
    /// 对 N 个列为独立线程，每列从上往下累积透明度。
    pub fn batch_sky_fill(
        &mut self,
        heightmap: &[i32],
        opacity: &[u8],
        sky_light: &mut [u8],
        n: usize,
        max_height: usize,
    ) -> Result<(), DeviceError> {
        if n == 0 {
            return Ok(());
        }

        if let Some(l) = self.device.kernel_launcher() {
            if l.has_kernel("sky_light_fill_u8") {
                let mut d_hm = self.device.alloc_i32(n)?;
                let mut d_op = self.device.alloc_u8(n * max_height)?;
                let d_sl = self.device.alloc_u8(n * max_height)?;
                self.device.copy_to_device(&mut d_hm, heightmap)?;
                self.device.copy_to_device(&mut d_op, opacity)?;
                l.launch(crate::common::kernel::KernelLaunch {
                    name: "sky_light_fill_u8",
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args: vec![
                        KernelArg::BufferRef(0),
                        KernelArg::BufferRef(1),
                        KernelArg::BufferRef(2),
                        KernelArg::I32(n as i32),
                        KernelArg::I32(max_height as i32),
                    ],
                    gpu_buffers: vec![
                        GpuBufferRef::I32(&d_hm),
                        GpuBufferRef::U8(&d_op),
                        GpuBufferRef::U8(&d_sl),
                    ],
                })?;
                l.synchronize()?;
                self.device.copy_from_device(&d_sl, sky_light)?;
                self.device.free(d_hm)?;
                self.device.free(d_op)?;
                self.device.free(d_sl)?;
                return Ok(());
            }
        }

        // CPU fallback
        for col in 0..n {
            let top = heightmap[col];
            for y in (top + 1)..max_height as i32 {
                sky_light[col * max_height + y as usize] = 15;
            }
            let mut light: u8 = 15;
            for y in (0..=top).rev() {
                let idx = col * max_height + y as usize;
                let op = opacity[idx];
                light = light.saturating_sub(op);
                sky_light[idx] = light;
            }
        }
        Ok(())
    }

    /// GPU 批量方块光扫描。返回光源索引列表。
    pub fn batch_block_scan(
        &mut self,
        luminances: &[u8],
        block_light: &mut [u8],
        n: usize,
    ) -> Result<Vec<i32>, DeviceError> {
        let mut sources = Vec::new();

        if let Some(l) = self.device.kernel_launcher() {
            if l.has_kernel("block_light_scan_u8") {
                let mut d_lum = self.device.alloc_u8(n)?;
                let d_bl = self.device.alloc_u8(n)?;
                let d_src = self.device.alloc_i32(n)?;
                let mut d_cnt = self.device.alloc_i32(1)?;
                self.device.copy_to_device(&mut d_lum, luminances)?;
                self.device.copy_to_device(&mut d_cnt, &[0i32])?;
                l.launch(crate::common::kernel::KernelLaunch {
                    name: "block_light_scan_u8",
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args: vec![
                        KernelArg::BufferRef(0),
                        KernelArg::BufferRef(1),
                        KernelArg::BufferRef(2),
                        KernelArg::BufferRef(3),
                        KernelArg::I32(n as i32),
                    ],
                    gpu_buffers: vec![
                        GpuBufferRef::U8(&d_lum),
                        GpuBufferRef::U8(&d_bl),
                        GpuBufferRef::I32(&d_src),
                        GpuBufferRef::I32(&d_cnt),
                    ],
                })?;
                l.synchronize()?;
                self.device.copy_from_device(&d_bl, block_light)?;
                let mut count = [0i32];
                self.device.copy_from_device(&d_cnt, &mut count)?;
                let k = count[0] as usize;
                if k > 0 {
                    let mut src = vec![0i32; k];
                    self.device.copy_from_device(&d_src, &mut src)?;
                    sources = src;
                }
                self.device.free(d_lum)?;
                self.device.free(d_bl)?;
                self.device.free(d_src)?;
                self.device.free(d_cnt)?;
                return Ok(sources);
            }
        }

        for i in 0..n {
            let lum = luminances[i];
            block_light[i] = lum;
            if lum > 0 {
                sources.push(i as i32);
            }
        }
        Ok(sources)
    }

    /// GPU 迭代距离场光照传播。
    /// 重复调用 `light_propagate_u8` kernel 直到收敛。
    /// 如果 persistent kernel 可用，使用单次 cooperative launch。
    #[allow(clippy::too_many_lines)]
    pub fn iterative_propagate(
        &mut self,
        light: &mut [u8],
        opacity: &[u8],
        neighbors: &[i32],
        n: usize,
        max_iters: usize,
    ) -> Result<usize, DeviceError> {
        if n == 0 {
            return Ok(0);
        }

        if let Some(l) = self.device.kernel_launcher() {
            // 优先使用 persistent kernel（单次 cooperative launch）
            if l.has_kernel("light_propagate_u8_persistent") {
                let mut d_light = self.device.alloc_u8(n)?;
                let mut d_opacity = self.device.alloc_u8(n)?;
                let mut d_neighbors = self.device.alloc_i32(n * 6)?;
                let mut d_converged = self.device.alloc_i32(1)?;
                let mut d_sync_counter = self.device.alloc_i32(1)?;

                self.device.copy_to_device(&mut d_light, light)?;
                self.device.copy_to_device(&mut d_opacity, opacity)?;
                self.device.copy_to_device(&mut d_neighbors, neighbors)?;
                self.device.copy_to_device(&mut d_converged, &[0i32])?;
                self.device.copy_to_device(&mut d_sync_counter, &[0i32])?;

                l.launch(crate::common::kernel::KernelLaunch {
                    name: "light_propagate_u8_persistent",
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args: vec![
                        KernelArg::BufferRef(0), // light
                        KernelArg::BufferRef(1), // opacity
                        KernelArg::BufferRef(2), // neighbors
                        KernelArg::BufferRef(3), // converged flag
                        KernelArg::BufferRef(4), // sync counter
                        KernelArg::I32(n as i32),
                        KernelArg::I32(max_iters as i32),
                    ],
                    gpu_buffers: vec![
                        GpuBufferRef::U8(&d_light),
                        GpuBufferRef::U8(&d_opacity),
                        GpuBufferRef::I32(&d_neighbors),
                        GpuBufferRef::I32(&d_converged),
                        GpuBufferRef::I32(&d_sync_counter),
                    ],
                })?;
                l.synchronize()?;
                self.device.copy_from_device(&d_light, light)?;
                self.device.free(d_light)?;
                self.device.free(d_opacity)?;
                self.device.free(d_neighbors)?;
                self.device.free(d_converged)?;
                self.device.free(d_sync_counter)?;
                return Ok(1); // 单次迭代
            }

            if l.has_kernel("light_propagate_u8") {
                let mut d_light = self.device.alloc_u8(n)?;
                let mut d_opacity = self.device.alloc_u8(n)?;
                let mut d_neighbors = self.device.alloc_i32(n * 6)?;
                let mut d_changed = self.device.alloc_i32(1)?;
                self.device.copy_to_device(&mut d_light, light)?;
                self.device.copy_to_device(&mut d_opacity, opacity)?;
                self.device.copy_to_device(&mut d_neighbors, neighbors)?;

                let mut iterations = 0;
                for _ in 0..max_iters {
                    self.device.copy_to_device(&mut d_changed, &[0i32])?;
                    l.launch(crate::common::kernel::KernelLaunch {
                        name: "light_propagate_u8",
                        global_work_size: [n, 1, 1],
                        local_work_size: Some([256, 1, 1]),
                        args: vec![
                            KernelArg::BufferRef(0),
                            KernelArg::BufferRef(1),
                            KernelArg::BufferRef(2),
                            KernelArg::BufferRef(3),
                            KernelArg::I32(n as i32),
                        ],
                        gpu_buffers: vec![
                            GpuBufferRef::U8(&d_light),
                            GpuBufferRef::U8(&d_opacity),
                            GpuBufferRef::I32(&d_neighbors),
                            GpuBufferRef::I32(&d_changed),
                        ],
                    })?;
                    l.synchronize()?;
                    iterations += 1;
                    let mut c = [0i32];
                    self.device.copy_from_device(&d_changed, &mut c)?;
                    if c[0] == 0 {
                        break;
                    }
                }
                self.device.copy_from_device(&d_light, light)?;
                self.device.free(d_light)?;
                self.device.free(d_opacity)?;
                self.device.free(d_neighbors)?;
                self.device.free(d_changed)?;
                return Ok(iterations);
            }
        }

        // CPU fallback: iterative distance field
        let mut iterations = 0;
        for _ in 0..max_iters {
            let mut changed = false;
            for i in 0..n {
                let cur = light[i];
                let mut best = cur;
                for d in 0..6 {
                    let n_idx = neighbors[i * 6 + d] as usize;
                    if n_idx < n {
                        let nl = light[n_idx];
                        let n_op = opacity[n_idx];
                        let prop = if nl > 1 + n_op {
                            nl.saturating_sub(1 + n_op)
                        } else {
                            0
                        };
                        if prop > best {
                            best = prop;
                        }
                    }
                }
                if best > cur {
                    light[i] = best;
                    changed = true;
                }
            }
            iterations += 1;
            if !changed {
                break;
            }
        }
        Ok(iterations)
    }
}
