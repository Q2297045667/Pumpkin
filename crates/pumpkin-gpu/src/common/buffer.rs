//! 设备端缓冲区抽象。

use super::BackendType;
#[cfg(feature = "opencl")]
use opencl3::memory::ClMem;

/// CUDA 设备内存包装（原始驱动 API 持有，支持零拷贝 kernel 参数）。
///
/// 使用 `cuMemAlloc` 分配的裸设备指针：cudarc 的 `CudaSlice`/`LaunchArgs`
/// 公共 API 无法传递任意设备指针（字段均私有），而零拷贝映射内存需要
/// 在 kernel 参数中直接传递 `CUdeviceptr` 值，因此 CUDA 后端的内存与启动
/// 层基于原始驱动 API 实现。
#[cfg(feature = "cuda")]
pub(crate) struct CudaRawHolder<T: bytemuck::Pod> {
    pub ptr: cudarc::driver::sys::CUdeviceptr,
    /// 上下文句柄：`cuMemFree` 等操作要求当前上下文正确绑定。
    pub(crate) ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    _marker: std::marker::PhantomData<T>,
}

// SAFETY: ptr 指向由 cuMemAlloc 分配的设备内存；应用层保证互斥访问。
#[cfg(feature = "cuda")]
unsafe impl<T: bytemuck::Pod + Send> Send for CudaRawHolder<T> {}

#[cfg(feature = "cuda")]
impl<T: bytemuck::Pod> Drop for CudaRawHolder<T> {
    fn drop(&mut self) {
        // 绑定上下文后释放（cuMemFree 作用于当前上下文）。
        let _ = self.ctx.bind_to_thread();
        // SAFETY: ptr 由 cuMemAlloc 分配且未被释放过。
        unsafe {
            let _ = cudarc::driver::result::free_sync(self.ptr);
        }
    }
}

/// CUDA 零拷贝（映射主机内存）包装。
///
/// `cuMemHostAlloc(CU_MEMHOSTALLOC_DEVICEMAP)` 分配的主机内存同时映射到设备
/// 地址空间：主机通过 `host_ptr` 直接读写，kernel 通过 `device_ptr` 访问，
/// 无需显式 `memcpy`。仅用于小于零拷贝阈值的小缓冲区。
#[cfg(feature = "cuda")]
pub(crate) struct CudaMappedHolder<T: bytemuck::Pod> {
    pub host_ptr: std::ptr::NonNull<T>,
    pub device_ptr: cudarc::driver::sys::CUdeviceptr,
    /// 上下文句柄：`cuMemFreeHost` 要求当前上下文正确绑定。
    pub(crate) ctx: std::sync::Arc<cudarc::driver::CudaContext>,
}

// SAFETY: host_ptr 指向的映射内存由 cuMemHostAlloc 分配，跨线程访问安全；
// 应用层保证对同一缓冲区的互斥访问。
#[cfg(feature = "cuda")]
unsafe impl<T: bytemuck::Pod + Send> Send for CudaMappedHolder<T> {}

#[cfg(feature = "cuda")]
impl<T: bytemuck::Pod> Drop for CudaMappedHolder<T> {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: host_ptr 由 cuMemHostAlloc 分配且未被释放过。
        unsafe {
            let _ = cudarc::driver::result::free_host(self.host_ptr.as_ptr().cast());
        }
    }
}

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
    Cuda(Box<CudaRawHolder<T>>),
    #[cfg(feature = "cuda")]
    CudaMapped(Box<CudaMappedHolder<T>>),
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
    pub(crate) fn new_cuda(
        len: usize,
        ptr: cudarc::driver::sys::CUdeviceptr,
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    ) -> Self {
        Self {
            len,
            backend_type: BackendType::Cuda,
            raw: RawBuffer::Cuda(Box::new(CudaRawHolder {
                ptr,
                ctx,
                _marker: std::marker::PhantomData,
            })),
        }
    }

    /// 创建 CUDA 零拷贝（映射内存）缓冲区。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn new_cuda_mapped(len: usize, holder: CudaMappedHolder<T>) -> Self {
        Self {
            len,
            backend_type: BackendType::Cuda,
            raw: RawBuffer::CudaMapped(Box::new(holder)),
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

    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn cpu_data(&self) -> Option<&[T]> {
        match &self.raw {
            RawBuffer::Cpu(v) => Some(v),
            #[cfg(feature = "cuda")]
            RawBuffer::Cuda(_) | RawBuffer::CudaMapped(_) => None,
            #[cfg(feature = "opencl")]
            RawBuffer::OpenCl(_) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cpu_data_mut(&mut self) -> Option<&mut Vec<T>> {
        match &mut self.raw {
            RawBuffer::Cpu(v) => Some(v),
            #[cfg(feature = "cuda")]
            RawBuffer::Cuda(_) | RawBuffer::CudaMapped(_) => None,
            #[cfg(feature = "opencl")]
            RawBuffer::OpenCl(_) => None,
        }
    }

    /// 获取 CUDA 缓冲区的设备指针（标准或零拷贝映射内存均返回 `Some`）。
    /// 供原始 kernel 启动路径传递参数。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn cuda_device_ptr(&self) -> Option<cudarc::driver::sys::CUdeviceptr> {
        match &self.raw {
            RawBuffer::Cuda(holder) => Some(holder.ptr),
            RawBuffer::CudaMapped(holder) => Some(holder.device_ptr),
            _ => None,
        }
    }

    /// 获取设备指针字段的地址（供 `cuLaunchKernel` 参数数组使用：
    /// 指针参数需要「指向设备指针值的指针」，驱动会解引用取得指针值）。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn cuda_device_ptr_addr(&self) -> Option<*const cudarc::driver::sys::CUdeviceptr> {
        match &self.raw {
            RawBuffer::Cuda(holder) => Some(std::ptr::addr_of!(holder.ptr)),
            RawBuffer::CudaMapped(holder) => Some(std::ptr::addr_of!(holder.device_ptr)),
            _ => None,
        }
    }

    /// 获取零拷贝映射缓冲区的引用（主机指针 + 设备指针 + 长度）。
    /// 仅 CUDA 零拷贝缓冲区返回 `Some`。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn cuda_mapped(&self) -> Option<&CudaMappedHolder<T>> {
        match &self.raw {
            RawBuffer::CudaMapped(holder) => Some(holder),
            _ => None,
        }
    }

    /// 获取 CUDA 零拷贝映射缓冲区的主机指针（仅零拷贝缓冲区返回 `Some`）。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) fn cuda_mapped_host_ptr(&self) -> Option<*mut T> {
        self.cuda_mapped().map(|h| h.host_ptr.as_ptr())
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
