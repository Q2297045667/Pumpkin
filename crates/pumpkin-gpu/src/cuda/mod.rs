//! CUDA 后端（基于 [`cudarc`]）。
//!
//! # 实现状态
//!
//! | 功能 | 状态 | 说明 |
//! |------|------|------|
//! | 驱动初始化 | ✅ | cudarc driver API |
//! | NVRTC kernel 编译 | ✅ | CUDA .cu kernel 源码 |
//! | GPU 内存分配 (标准) | ✅ | `CudaStream::alloc` → `CudaSlice` |
//! | GPU 内存分配 (零拷贝) | ⚠️ | PinnedHostSlice 框架就绪 |
//! | HtoD/DtoH 拷贝 | ✅ | `memcpy_htod` / `memcpy_dtoh` |
//! | Kernel 启动 | ⚠️ | `LaunchArgs` builder 框架就绪 |
//! | 设备选择 ByIndex | ✅ | |
//! | CPU 回退 | ✅ | 内存分配/传输失败自动回退 |

mod context;
pub(crate) mod kernel;
mod memory;

pub(crate) use context::cuda_driver_available;

use crate::common::{DeviceError, GpuBuffer, KernelLauncher};
use std::sync::Arc;

/// CUDA 后端实现。
pub struct CudaBackend {
    #[allow(dead_code)]
    pub(crate) ctx: Arc<cudarc::driver::CudaContext>,
    pub(crate) stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) name: String,
    pub(crate) launcher: kernel::CudaKernelLauncher,
    /// 零拷贝阈值（字节）
    zero_copy_threshold_bytes: usize,
}

// SAFETY: CudaBackend's internal state is Send by cudarc specification.
unsafe impl Send for CudaBackend {}

impl CudaBackend {
    pub fn try_init(
        device_index: Option<usize>,
        flags: Option<&[String]>,
        zero_copy_threshold_kb: usize,
        persistent_enabled: bool,
    ) -> Result<Self, DeviceError> {
        let idx = device_index.unwrap_or(0);

        // 初始化 CUDA 驱动
        let ctx =
            context::init_cuda(idx).map_err(|e| DeviceError::InitFailed(format!("CUDA: {e}")))?;

        // 获取设备名称和默认流（可能在无 GPU 时失败）
        let name = ctx
            .name()
            .unwrap_or_else(|_| String::from("Unknown CUDA Device"));
        let stream = ctx.default_stream();

        tracing::info!("CUDA 设备: {name}");
        if zero_copy_threshold_kb > 0 {
            tracing::debug!("CUDA 零拷贝阈值: {} KB", zero_copy_threshold_kb);
        }

        let mut launcher = kernel::CudaKernelLauncher::new();
        launcher.init(ctx.clone(), stream.clone(), flags, persistent_enabled);
        Ok(Self {
            ctx,
            stream,
            name,
            launcher,
            zero_copy_threshold_bytes: zero_copy_threshold_kb.saturating_mul(1024),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    pub fn alloc_f64(&self, len: usize) -> Result<GpuBuffer<f64>, DeviceError> {
        memory::CudaMemory::alloc_f64(&self.stream, len, self.zero_copy_threshold_bytes)
    }

    pub fn alloc_i32(&self, len: usize) -> Result<GpuBuffer<i32>, DeviceError> {
        memory::CudaMemory::alloc_i32(&self.stream, len, self.zero_copy_threshold_bytes)
    }

    pub fn alloc_u8(&self, len: usize) -> Result<GpuBuffer<u8>, DeviceError> {
        memory::CudaMemory::alloc_u8(&self.stream, len, self.zero_copy_threshold_bytes)
    }

    pub fn copy_to_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        &self,
        buffer: &mut GpuBuffer<T>,
        data: &[T],
    ) -> Result<(), DeviceError> {
        memory::CudaMemory::copy_to_device::<T>(&self.stream, buffer, data)
    }

    pub fn copy_from_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        &self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), DeviceError> {
        memory::CudaMemory::copy_from_device::<T>(&self.stream, buffer, data)
    }

    pub fn free<T: bytemuck::Pod>(&self, buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        memory::CudaMemory::free(buffer)
    }

    pub fn kernel_launcher(&self) -> Option<&dyn KernelLauncher> {
        Some(&self.launcher)
    }

    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &mut self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        self.launcher.compile_jit_kernel(jit_kernel)
    }
}
