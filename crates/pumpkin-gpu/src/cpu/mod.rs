//! CPU 回退后端。

mod fallback;

use crate::common::{DeviceError, GpuBuffer, KernelLauncher, buffer::RawBuffer};

/// CPU 回退后端。
pub struct CpuBackend {
    name: String,
    launcher: CpuKernelLauncher,
}

impl CpuBackend {
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: get_cpu_name(),
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
            RawBuffer::Cuda(_) => {
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
            RawBuffer::Cuda(_) => {
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

#[cfg(feature = "gpu")]
fn get_cpu_name() -> String {
    use sysinfo::System;

    let sys = System::new_all();
    sys.cpus().first().map_or_else(
        || String::from("CPU Fallback"),
        |cpu| format!("CPU: {} ({})", cpu.brand(), cpu.name()),
    )
}

#[cfg(not(feature = "gpu"))]
fn get_cpu_name() -> String {
    String::from("CPU Fallback")
}

struct CpuKernelLauncher;

impl KernelLauncher for CpuKernelLauncher {
    fn launch(&self, launch: crate::common::kernel::KernelLaunch<'_>) -> Result<(), DeviceError> {
        fallback::dispatch(&launch)
    }

    fn has_kernel(&self, name: &str) -> bool {
        fallback::has_kernel(name)
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        Ok(())
    }
}
