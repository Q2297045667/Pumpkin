//! CPU 回退后端。

use crate::common::{DeviceError, GpuBuffer, KernelLauncher, buffer::RawBuffer};
#[cfg(feature = "gpu")]
use std::sync::OnceLock;

/// 缓存 CPU 品牌名——`sysinfo::System::new_all()` 成本高（~100ms），
/// 且 `CpuBackend` 与 `logging::log_gpu_startup` 都会频繁获取 CPU 名。
#[cfg(feature = "gpu")]
static CPU_NAME: OnceLock<String> = OnceLock::new();

/// 获取 CPU 品牌名（进程内缓存）。
///
/// `sysinfo::System::new_all()` 需要枚举全部硬件（~100ms），
/// 因此首次调用后缓存结果，后续调用零成本。
#[cfg(feature = "gpu")]
pub(crate) fn cpu_name() -> String {
    use sysinfo::System;

    CPU_NAME
        .get_or_init(|| {
            let sys = System::new_all();
            sys.cpus().first().map_or_else(
                || String::from("CPU Fallback"),
                |cpu| format!("CPU: {} ({})", cpu.brand(), cpu.name()),
            )
        })
        .clone()
}

/// CPU 回退后端。
pub struct CpuBackend {
    name: String,
    launcher: CpuKernelLauncher,
}

impl CpuBackend {
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "gpu")]
            name: cpu_name(),
            #[cfg(not(feature = "gpu"))]
            name: String::from("CPU Fallback"),
            launcher: CpuKernelLauncher,
        }
    }

    fn check_size<T: bytemuck::Pod>(
        buffer: &GpuBuffer<T>,
        data_len: usize,
    ) -> Result<(), DeviceError> {
        if buffer.len() != data_len {
            return Err(DeviceError::SizeMismatch {
                buffer_len: buffer.len(),
                data_len,
            });
        }
        Ok(())
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub fn device_name(&self) -> &str {
        &self.name
    }

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
        Self::check_size(buffer, data.len())?;
        match &mut buffer.raw {
            RawBuffer::Cpu(vec) => {
                vec.clear();
                vec.extend_from_slice(data);
            }
            #[cfg(feature = "cuda")]
            RawBuffer::Cuda(_) | RawBuffer::CudaMapped(_) => {
                return Err(DeviceError::Unsupported(
                    "CPU 后端不能操作 CUDA 缓冲区".into(),
                ));
            }
            #[cfg(feature = "opencl")]
            RawBuffer::OpenCl(_) => {
                return Err(DeviceError::Unsupported(
                    "CPU 后端不能操作 OpenCL 缓冲区".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn copy_from_device<T: bytemuck::Pod>(
        &self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), DeviceError> {
        Self::check_size(buffer, data.len())?;
        match &buffer.raw {
            RawBuffer::Cpu(vec) => {
                data.copy_from_slice(vec);
            }
            #[cfg(feature = "cuda")]
            RawBuffer::Cuda(_) | RawBuffer::CudaMapped(_) => {
                return Err(DeviceError::Unsupported(
                    "CPU 后端不能操作 CUDA 缓冲区".into(),
                ));
            }
            #[cfg(feature = "opencl")]
            RawBuffer::OpenCl(_) => {
                return Err(DeviceError::Unsupported(
                    "CPU 后端不能操作 OpenCL 缓冲区".into(),
                ));
            }
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

struct CpuKernelLauncher;

impl KernelLauncher for CpuKernelLauncher {
    fn launch(&self, _launch: crate::common::kernel::KernelLaunch<'_>) -> Result<(), DeviceError> {
        // CPU 回退不通过 KernelLaunch 路径。
        // 各采样器模块在检测到 DeviceType::Cpu 后直接调用内置 CPU fallback 函数。
        Err(DeviceError::Unsupported(
            "CPU backend does not use KernelLaunch".into(),
        ))
    }

    fn has_kernel(&self, _name: &str) -> bool {
        false // CPU 路径不通过 kernel launcher
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        Ok(())
    }
}
