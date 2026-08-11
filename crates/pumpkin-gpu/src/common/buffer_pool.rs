//! GPU 缓冲区池 — 持久化复用，减少分配/释放开销。
//!
//! 池按 buffer 长度分组，支持同一长度多个 buffer 共存。
//! 适用于跨方法调用复用的场景（如 `GpuCellBatchSampler`、`GpuNoiseSampler`）。

use crate::GpuBuffer;
use crate::GpuDevice;
use crate::common::DeviceError;
use std::collections::HashMap;

/// GPU 缓冲区池，按长度分组回收。
pub struct GpuBufferPool {
    f64: HashMap<usize, Vec<GpuBuffer<f64>>>,
    u8: HashMap<usize, Vec<GpuBuffer<u8>>>,
    i32: HashMap<usize, Vec<GpuBuffer<i32>>>,
}

impl GpuBufferPool {
    #[must_use]
    pub fn new() -> Self {
        Self {
            f64: HashMap::default(),
            u8: HashMap::default(),
            i32: HashMap::default(),
        }
    }

    /// 从池中取一个 f64 buffer（复用或新建）。
    pub fn take_f64(
        &mut self,
        device: &GpuDevice,
        len: usize,
    ) -> Result<GpuBuffer<f64>, DeviceError> {
        self.f64
            .get_mut(&len)
            .and_then(Vec::pop)
            .map_or_else(|| device.alloc_f64(len), Ok)
    }

    /// 将 f64 buffer 归还池中。
    pub fn put_f64(&mut self, buf: GpuBuffer<f64>) {
        let len = buf.len();
        self.f64.entry(len).or_default().push(buf);
    }

    /// 从池中取一个 u8 buffer（复用或新建）。
    pub fn take_u8(
        &mut self,
        device: &GpuDevice,
        len: usize,
    ) -> Result<GpuBuffer<u8>, DeviceError> {
        self.u8
            .get_mut(&len)
            .and_then(Vec::pop)
            .map_or_else(|| device.alloc_u8(len), Ok)
    }

    /// 将 u8 buffer 归还池中。
    pub fn put_u8(&mut self, buf: GpuBuffer<u8>) {
        let len = buf.len();
        self.u8.entry(len).or_default().push(buf);
    }

    /// 从池中取一个 i32 buffer（复用或新建）。
    pub fn take_i32(
        &mut self,
        device: &GpuDevice,
        len: usize,
    ) -> Result<GpuBuffer<i32>, DeviceError> {
        self.i32
            .get_mut(&len)
            .and_then(Vec::pop)
            .map_or_else(|| device.alloc_i32(len), Ok)
    }

    /// 将 i32 buffer 归还池中。
    pub fn put_i32(&mut self, buf: GpuBuffer<i32>) {
        let len = buf.len();
        self.i32.entry(len).or_default().push(buf);
    }

    /// 上传数据到 buffer（使用 device 的 copy_to_device）。
    pub fn upload_f64(
        device: &GpuDevice,
        buf: &mut GpuBuffer<f64>,
        data: &[f64],
    ) -> Result<(), DeviceError> {
        device.copy_to_device(buf, data)
    }

    /// 从 buffer 下载数据（使用 device 的 copy_from_device）。
    pub fn download_f64(
        device: &GpuDevice,
        buf: &GpuBuffer<f64>,
        data: &mut [f64],
    ) -> Result<(), DeviceError> {
        device.copy_from_device(buf, data)
    }

    /// 一次性释放池中所有 buffer。
    pub fn free_all(self, device: &GpuDevice) -> Result<(), DeviceError> {
        for (_, bufs) in self.f64 {
            for buf in bufs {
                device.free(buf)?;
            }
        }
        for (_, bufs) in self.u8 {
            for buf in bufs {
                device.free(buf)?;
            }
        }
        for (_, bufs) in self.i32 {
            for buf in bufs {
                device.free(buf)?;
            }
        }
        Ok(())
    }
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}
