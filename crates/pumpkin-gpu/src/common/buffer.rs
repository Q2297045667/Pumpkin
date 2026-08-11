//! 设备端缓冲区抽象。

use super::BackendType;
#[cfg(feature = "opencl")]
use opencl3::memory::ClMem;

/// CUDA 设备内存包装。
///
/// 注意：在 GPU 硬件验证完成前，`CudaSliceHolder` 字段暂时未使用。
/// 待 CUDA 后端完成后，此类型将被激活。
#[allow(dead_code)]
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(crate) struct CudaSliceHolder<T: bytemuck::Pod> {
    pub slice: cudarc::driver::CudaSlice<T>,
}

// SAFETY: CudaSlice is Send by specification. The contained pointer is device-side memory that cudarc manages.
#[cfg(feature = "cuda")]
unsafe impl<T: bytemuck::Pod + Send> Send for CudaSliceHolder<T> {}

/// `OpenCL` 设备内存包装。使用 `UnsafeCell` 以允许对 `Buffer` 的合法互斥可变访问。
#[cfg(feature = "opencl")]
#[derive(Debug)]
pub(crate) struct OpenClBufferHolder {
    pub buffer: std::cell::UnsafeCell<opencl3::memory::Buffer<u8>>,
}

// SAFETY: OpenCL Buffer is Send. UnsafeCell provides interior mutability needed for enqueue operations.
#[cfg(feature = "opencl")]
unsafe impl Send for OpenClBufferHolder {}

/// 后端特定的缓冲区内部表示。
#[allow(dead_code)]
pub(crate) enum RawBuffer<T: bytemuck::Pod> {
    Cpu(Vec<T>),
    #[cfg(feature = "cuda")]
    Cuda(Box<CudaSliceHolder<T>>),
    #[cfg(feature = "opencl")]
    OpenCl(Box<OpenClBufferHolder>),
}

// SAFETY: All variant types (Vec<T>, CudaSliceHolder<T>, OpenClBufferHolder) are Send.
unsafe impl<T: bytemuck::Pod + Send> Send for RawBuffer<T> {}

/// 统一的 GPU 缓冲区句柄。
pub struct GpuBuffer<T: bytemuck::Pod> {
    len: usize,
    backend_type: BackendType,
    pub(crate) raw: RawBuffer<T>,
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    #[must_use]
    pub(crate) fn new_cpu(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            len,
            backend_type: BackendType::Cpu,
            raw: RawBuffer::Cpu(data),
        }
    }

    #[allow(dead_code)]
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn new_cuda(len: usize, slice: cudarc::driver::CudaSlice<T>) -> Self {
        Self {
            len,
            backend_type: BackendType::Cuda,
            raw: RawBuffer::Cuda(Box::new(CudaSliceHolder { slice })),
        }
    }

    #[cfg(feature = "opencl")]
    #[must_use]
    pub(crate) fn new_opencl(len: usize, buffer: opencl3::memory::Buffer<u8>) -> Self {
        Self {
            len,
            backend_type: BackendType::OpenCl,
            raw: RawBuffer::OpenCl(Box::new(OpenClBufferHolder {
                buffer: std::cell::UnsafeCell::new(buffer),
            })),
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(dead_code)]
    #[must_use]
    pub(crate) const fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    #[must_use]
    pub(crate) fn cpu_data(&self) -> Option<&[T]> {
        match &self.raw {
            RawBuffer::Cpu(v) => Some(v),
            #[cfg(feature = "cuda")]
            RawBuffer::Cuda(_) => None,
            #[cfg(feature = "opencl")]
            RawBuffer::OpenCl(_) => None,
        }
    }

    pub(crate) fn cpu_data_mut(&mut self) -> Option<&mut Vec<T>> {
        match &mut self.raw {
            RawBuffer::Cpu(v) => Some(v),
            #[cfg(feature = "cuda")]
            RawBuffer::Cuda(_) => None,
            #[cfg(feature = "opencl")]
            RawBuffer::OpenCl(_) => None,
        }
    }

    #[allow(dead_code)]
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn cuda_slice(&self) -> Result<&cudarc::driver::CudaSlice<T>, &'static str> {
        match &self.raw {
            RawBuffer::Cuda(holder) => Ok(&holder.slice),
            _ => Err("不是 CUDA 缓冲区"),
        }
    }

    #[allow(dead_code)]
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn cuda_slice_mut(
        &mut self,
    ) -> Result<&mut cudarc::driver::CudaSlice<T>, &'static str> {
        match &mut self.raw {
            RawBuffer::Cuda(holder) => Ok(&mut holder.slice),
            _ => Err("不是 CUDA 缓冲区"),
        }
    }

    /// 获取字节大小。
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// 获取 OpenCL cl_mem 句柄（仅 OpenCL 后端有效）。
    #[cfg(feature = "opencl")]
    #[must_use]
    pub fn opencl_handle(&self) -> Option<opencl3::types::cl_mem> {
        match &self.raw {
            RawBuffer::OpenCl(holder) => {
                // SAFETY: UnsafeCell provides interior mutability.
                // We only need read access to the handle for set_arg.
                let buf = unsafe { &*holder.buffer.get() };
                Some(buf.get())
            }
            _ => None,
        }
    }
}

impl<T: bytemuck::Pod> std::fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.len)
            .field("backend", &self.backend_type)
            .field("elem_type", &std::any::type_name::<T>())
            .finish()
    }
}
