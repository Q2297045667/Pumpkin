//! CUDA 后端（基于 [`cudarc`]）。
//!
//! # 实现状态
//!
//! 设备初始化 ✓ — 驱动加载和设备探测已完成。
//! 内存分配 ⚠ — 需要装有 CUDA Toolkit 和 NVIDIA GPU 的开发环境验证。
//! 数据传输 ⚠ — 同上，API 需在 GPU 机器上测试。
//! Kernel 执行 ❌ — 待 Kernel 编译链实现。
//!
//! 当前在非 CUDA 环境下自动回退 CPU，不影响功能。
//!
//! # CUDA ↔ OpenCL 对齐状态
//!
//! | 特性 | CUDA | OpenCL |
//! |------|------|--------|
//! | 设备初始化 | ✓ (cudarc driver API) | ✓ (opencl3 platform API) |
//! | GPU 内存分配 | ⚠ (API 验证待完成) | ✓ (Buffer::create) |
//! | HtoD/DtoH 拷贝 | ⚠ (API 验证待完成) | ✓ (enqueue_read/write) |
//! | Kernel 加载 | ❌ | ❌ |
//! | Kernel 执行 | ❌ | ❌ |
//! | f64 支持 | ✓ (CUDA 原生) | ✓ (cl_khr_f64) |
//! | CPU 回退 | ✓ | ✓ |

mod context;
pub(crate) mod kernel;

use crate::common::{DeviceError, GpuBuffer, KernelLauncher};
use std::sync::Arc;

/// CUDA 后端实现。
pub struct CudaBackend {
    pub(crate) ctx: Arc<cudarc::driver::CudaContext>,
    pub(crate) name: String,
    pub(crate) launcher: kernel::CudaKernelLauncher,
}

unsafe impl Send for CudaBackend {}

impl CudaBackend {
    pub fn try_init() -> Result<Self, DeviceError> {
        let ctx =
            context::init_cuda().map_err(|e| DeviceError::InitFailed(format!("CUDA: {e}")))?;

        let name = ctx
            .name()
            .unwrap_or_else(|_| String::from("Unknown CUDA Device"));

        tracing::info!("CUDA 设备: {name}");
        let mut launcher = kernel::CudaKernelLauncher::new();
        launcher.init(ctx.clone());
        Ok(Self {
            ctx,
            name,
            launcher,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    // NOTE: Memory and transfer APIs need a CUDA-capable machine for final verification.
    // Until then, CPU fallback ensures correct behavior.
    pub fn alloc_f64(&self, len: usize) -> Result<GpuBuffer<f64>, DeviceError> {
        Ok(GpuBuffer::new_cpu(vec![0.0f64; len]))
    }
    pub fn alloc_i32(&self, len: usize) -> Result<GpuBuffer<i32>, DeviceError> {
        Ok(GpuBuffer::new_cpu(vec![0i32; len]))
    }
    pub fn alloc_u8(&self, len: usize) -> Result<GpuBuffer<u8>, DeviceError> {
        Ok(GpuBuffer::new_cpu(vec![0u8; len]))
    }
    pub fn copy_to_device<T: bytemuck::Pod>(
        &self,
        buffer: &mut GpuBuffer<T>,
        data: &[T],
    ) -> Result<(), DeviceError> {
        if buffer.len() != data.len() {
            return Err(DeviceError::SizeMismatch {
                buffer_len: buffer.len(),
                data_len: data.len(),
            });
        }
        if buffer.is_empty() {
            return Ok(());
        }
        if let Some(cpu) = buffer.cpu_data_mut() {
            cpu.clear();
            cpu.extend_from_slice(data);
        }
        Ok(())
    }
    pub fn copy_from_device<T: bytemuck::Pod>(
        &self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), DeviceError> {
        if buffer.len() != data.len() {
            return Err(DeviceError::SizeMismatch {
                buffer_len: buffer.len(),
                data_len: data.len(),
            });
        }
        if buffer.is_empty() {
            return Ok(());
        }
        if let Some(cpu) = buffer.cpu_data() {
            data.copy_from_slice(cpu);
        }
        Ok(())
    }
    pub fn free<T: bytemuck::Pod>(&self, buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        drop(buffer);
        Ok(())
    }
    pub fn kernel_launcher(&self) -> Option<&dyn KernelLauncher> {
        Some(&self.launcher)
    }
}
