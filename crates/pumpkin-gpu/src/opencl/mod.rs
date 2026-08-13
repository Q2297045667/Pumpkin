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
    ///
    /// 编译标志策略：
    /// - 用户显式提供标志时以用户为准（允许覆盖精度选项）；
    /// - 未提供时使用标准标志 `-cl-fp32-correctly-rounded-divide-sqrt`。
    ///
    /// f64 的 FMA 收缩（`a*b+c` → `fma`）不是通过编译标志控制的，
    /// 而是在 `opencl_compile::compile_one` 中向所有 kernel 源码注入
    /// 标准 pragma `#pragma OPENCL FP_CONTRACT OFF` 来禁用——NVIDIA 的
    /// OpenCL 编译器拒绝 CUDA 风格的 `-fmad=false` 标志，但遵守该 pragma。
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
        let flags = flags.map_or_else(
            || vec!["-cl-fp32-correctly-rounded-divide-sqrt".to_string()],
            <[String]>::to_vec,
        );
        let mut launcher = kernel::OpenClKernelLauncher::new();
        launcher.init(&ctx, &device, queues, Some(&flags));
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
        &self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        self.launcher
            .compile_jit_kernel(&self.ctx, &self.device, jit_kernel)
    }

    /// 按需编译单个预注册 kernel（延迟加载）。
    ///
    /// 从全局 OpenCL 源码注册表查找源码并编译；失败仅记录日志，
    /// 上层 `try_launch_kernel` 会回退到 CPU 路径。
    pub fn compile_kernel_by_name(&self, name: &str) {
        self.launcher
            .compile_kernel_by_name(&self.ctx, &self.device, name);
    }
}
