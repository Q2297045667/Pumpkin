//! CUDA 后端（基于 [`cudarc`]）。
//!
//! # 实现状态
//!
//! | 功能 | 状态 | 阻塞条件 |
//! |------|------|---------|
//! | 驱动初始化 | ✅ 已完成 | |
//! | NVRTC kernel 编译 | ⚠️ 需要 CUDA kernel 源码 | 当前 kernel 为 OpenCL C 语法，NVRTC 无法编译 |
//! | GPU 内存分配 | ❌ 使用 CPU fallback | 需实现 `CudaSlice` 分配 |
//! | Kernel 启动 | ❌ 硬编码 Unsupported | 需 GPU 硬件 + CUDA kernel 源码 |
//! | 设备选择 (ByIndex) | ✅ 已完成 | |
//! | 设备选择 (ByName) | ⚠️ 部分完成 | 需 CUDA 设备枚举 API |
//! | CPU 回退 | ✅ 已完成 | |
//!
//! # CUDA ↔ OpenCL 对齐状态
//!
//! | 特性 | CUDA | OpenCL |
//! |------|------|--------|
//! | 设备初始化 | ✅ (cudarc driver API) | ✅ (opencl3 platform API) |
//! | GPU 内存分配 | ❌ (CPU fallback) | ✅ (Buffer::create) |
//! | HtoD/DtoH 拷贝 | ❌ (CPU fallback) | ✅ (enqueue_read/write) |
//! | Kernel 编译 | ⚠️ (NVRTC 编译 OpenCL 语法失败) | ✅ (create_from_source + build) |
//! | Kernel 启动 | ❌ (硬编码 Unsupported) | ⚠️ (标量参数已绑定，buffer 参数需接线) |
//! | f64 支持 | ✅ (CUDA 原生) | ✅ (cl_khr_f64) |
//! | 设备选择 | ✅ (ByIndex) | ✅ (ByIndex/ByName/Integrated) |
//! | CPU 回退 | ✅ | ✅ |

mod context;
pub(crate) mod kernel;

use crate::common::{DeviceError, GpuBuffer, KernelLauncher};
use std::sync::Arc;

/// CUDA 后端实现。
pub struct CudaBackend {
    #[allow(dead_code)]
    pub(crate) ctx: Arc<cudarc::driver::CudaContext>,
    pub(crate) name: String,
    pub(crate) launcher: kernel::CudaKernelLauncher,
}

// SAFETY: CudaBackend's internal state (Arc<CudaContext>) is Send by cudarc specification.
unsafe impl Send for CudaBackend {}

impl CudaBackend {
    pub fn try_init(
        device_index: Option<usize>,
        flags: Option<&[String]>,
    ) -> Result<Self, DeviceError> {
        let idx = device_index.unwrap_or(0);
        let ctx =
            context::init_cuda(idx).map_err(|e| DeviceError::InitFailed(format!("CUDA: {e}")))?;

        let name = ctx
            .name()
            .unwrap_or_else(|_| String::from("Unknown CUDA Device"));

        tracing::info!("CUDA 设备: {name}");
        let mut launcher = kernel::CudaKernelLauncher::new();
        launcher.init(ctx.clone(), flags);
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

    /// 编译一个 JIT 特化 kernel。
    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &mut self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        self.launcher.compile_jit_kernel(jit_kernel)
    }
}
