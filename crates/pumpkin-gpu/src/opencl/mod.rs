//! OpenCL 后端（基于 [`opencl3`]）。

mod buffer;
pub mod context;
pub(crate) mod kernel;

pub use context::is_opencl_available;

use crate::common::{DeviceError, GpuBuffer, KernelLauncher};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;

/// OpenCL 后端实现。
pub struct OpenClBackend {
    pub(crate) ctx: Context,
    #[allow(dead_code)]
    pub(crate) device: opencl3::device::Device,
    pub(crate) name: String,
    pub(crate) launcher: kernel::OpenClKernelLauncher,
}

// SAFETY: OpenClBackend 的所有内部字段（Context, CommandQueue, Device）
// 都实现了 Send，且 OpenCL API 本身是线程安全的（命令队列序列化访问）。
unsafe impl Send for OpenClBackend {}

impl OpenClBackend {
    /// 尝试初始化 OpenCL 后端。
    pub fn try_init(
        device_index: Option<usize>,
        device_name_filter: Option<&str>,
        prefer_integrated: bool,
        flags: Option<&[String]>,
        pipeline_queues: usize,
    ) -> Result<Self, DeviceError> {
        let (ctx, queues, device, name) = context::init_opencl(
            device_index,
            device_name_filter,
            prefer_integrated,
            pipeline_queues,
        )
        .map_err(|e| DeviceError::InitFailed(format!("OpenCL: {e}")))?;

        tracing::info!("OpenCL 设备: {name}");
        let mut launcher = kernel::OpenClKernelLauncher::new();
        launcher.init(&ctx, &device, queues, flags);
        Ok(Self {
            ctx,
            device,
            name,
            launcher,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    fn queue(&self) -> &CommandQueue {
        self.launcher.queue()
    }

    pub fn alloc_f64(&self, len: usize) -> Result<GpuBuffer<f64>, DeviceError> {
        buffer::alloc_f64(&self.ctx, self.queue(), len)
    }

    pub fn alloc_i32(&self, len: usize) -> Result<GpuBuffer<i32>, DeviceError> {
        buffer::alloc_i32(&self.ctx, self.queue(), len)
    }

    pub fn alloc_u8(&self, len: usize) -> Result<GpuBuffer<u8>, DeviceError> {
        buffer::alloc_u8(&self.ctx, self.queue(), len)
    }

    pub fn copy_to_device<T: bytemuck::Pod>(
        &self,
        buffer: &mut GpuBuffer<T>,
        data: &[T],
    ) -> Result<(), DeviceError> {
        buffer::copy_to_device::<T>(&self.ctx, self.queue(), buffer, data)
    }

    pub fn copy_from_device<T: bytemuck::Pod>(
        &self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), DeviceError> {
        buffer::copy_from_device::<T>(&self.ctx, self.queue(), buffer, data)
    }

    pub fn free<T: bytemuck::Pod>(&self, buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        buffer::free::<T>(&self.ctx, buffer)
    }

    pub fn kernel_launcher(&self) -> Option<&dyn KernelLauncher> {
        Some(&self.launcher)
    }

    /// 编译一个 JIT 特化 kernel。
    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &mut self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        self.launcher
            .compile_jit_kernel(&self.ctx, &self.device, jit_kernel)
    }
}
