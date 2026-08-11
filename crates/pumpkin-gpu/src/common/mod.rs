//! 平台无关的共享抽象层。

pub mod buffer;
pub mod error;
pub mod kernel;
pub mod layout;

pub use buffer::GpuBuffer;
pub use error::DeviceError;
pub use kernel::KernelLauncher;

/// 后端类型标记。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cuda,
    OpenCl,
    Cpu,
}

/// 后端实现枚举。
/// 通过枚举派发（enum dispatch）替代 trait object，避免泛型方法的 dyn 兼容问题。
#[allow(variant_size_differences)]
pub(crate) enum BackendImpl {
    Cpu(crate::cpu::CpuBackend),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaBackend),
    #[cfg(feature = "opencl")]
    OpenCl(crate::opencl::OpenClBackend),
}

impl BackendImpl {
    #[allow(dead_code)]
    pub(crate) fn backend_type(&self) -> BackendType {
        match self {
            Self::Cpu(_) => BackendType::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendType::Cuda,
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => BackendType::OpenCl,
        }
    }

    pub(crate) fn device_name(&self) -> &str {
        match self {
            Self::Cpu(b) => b.device_name(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.device_name(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.device_name(),
        }
    }

    pub(crate) fn alloc_f64(&self, len: usize) -> Result<GpuBuffer<f64>, DeviceError> {
        match self {
            Self::Cpu(b) => b.alloc_f64(len),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.alloc_f64(len),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.alloc_f64(len),
        }
    }

    pub(crate) fn alloc_i32(&self, len: usize) -> Result<GpuBuffer<i32>, DeviceError> {
        match self {
            Self::Cpu(b) => b.alloc_i32(len),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.alloc_i32(len),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.alloc_i32(len),
        }
    }

    pub(crate) fn alloc_u8(&self, len: usize) -> Result<GpuBuffer<u8>, DeviceError> {
        match self {
            Self::Cpu(b) => b.alloc_u8(len),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.alloc_u8(len),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.alloc_u8(len),
        }
    }

    pub(crate) fn copy_to_device<T: bytemuck::Pod>(
        &self,
        buffer: &mut GpuBuffer<T>,
        data: &[T],
    ) -> Result<(), DeviceError> {
        match self {
            Self::Cpu(b) => b.copy_to_device(buffer, data),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.copy_to_device(buffer, data),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.copy_to_device(buffer, data),
        }
    }

    pub(crate) fn copy_from_device<T: bytemuck::Pod>(
        &self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), DeviceError> {
        match self {
            Self::Cpu(b) => b.copy_from_device(buffer, data),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.copy_from_device(buffer, data),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.copy_from_device(buffer, data),
        }
    }

    pub(crate) fn free<T: bytemuck::Pod>(&self, buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        match self {
            Self::Cpu(b) => b.free(buffer),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.free(buffer),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.free(buffer),
        }
    }

    pub(crate) fn kernel_launcher(&self) -> Option<&dyn KernelLauncher> {
        match self {
            Self::Cpu(b) => b.kernel_launcher(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.kernel_launcher(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.kernel_launcher(),
        }
    }

    /// 编译一个 JIT 特化 kernel。
    ///
    /// 仅在 GPU 后端（CUDA / OpenCL）下有效，CPU 后端返回错误。
    #[cfg(feature = "pumpkin-util")]
    pub(crate) fn compile_jit_kernel(
        &mut self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        match self {
            Self::Cpu(_) => Err(DeviceError::Unsupported(
                "JIT compilation not supported on CPU backend".into(),
            )),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.compile_jit_kernel(jit_kernel),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.compile_jit_kernel(jit_kernel),
        }
    }
}

// SAFETY: All backend variants (CpuBackend, CudaBackend, OpenClBackend) implement Send.
unsafe impl Send for BackendImpl {}
