//! CUDA 内存分配与传输。
//!
//! 基于原始驱动 API：`cuMemAlloc` / `cuMemcpy*` / `cuMemHostAlloc(DEVICEMAP)`。
//! cudarc 的 `CudaSlice` 字段为私有，公共 API 无法传递任意设备指针
//! （零拷贝映射内存的 kernel 参数需要 `CUdeviceptr` 值），因此内存与
//! 启动层直接使用 cudarc 重新导出的原始驱动函数。
//! 失败时自动回退 CPU 缓冲区。

use crate::common::buffer::CudaMappedHolder;
use crate::common::{DeviceError, GpuBuffer};
use std::sync::Arc;

pub struct CudaMemory;

impl CudaMemory {
    // ── f64 ──────────────────────────────────────────────

    pub fn alloc_f64(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
        zero_copy_threshold_bytes: usize,
    ) -> Result<GpuBuffer<f64>, DeviceError> {
        if len == 0 {
            return Ok(GpuBuffer::new_cpu(Vec::new()));
        }
        if zero_copy_threshold_bytes > 0 && len * size_of::<f64>() <= zero_copy_threshold_bytes {
            return Self::try_alloc_mapped::<f64>(stream, len);
        }
        Self::try_alloc_device::<f64>(stream, len)
    }

    // ── i32 ──────────────────────────────────────────────

    pub fn alloc_i32(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
        zero_copy_threshold_bytes: usize,
    ) -> Result<GpuBuffer<i32>, DeviceError> {
        if len == 0 {
            return Ok(GpuBuffer::new_cpu(Vec::new()));
        }
        if zero_copy_threshold_bytes > 0 && len * size_of::<i32>() <= zero_copy_threshold_bytes {
            return Self::try_alloc_mapped::<i32>(stream, len);
        }
        Self::try_alloc_device::<i32>(stream, len)
    }

    // ── u8 ───────────────────────────────────────────────

    pub fn alloc_u8(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
        zero_copy_threshold_bytes: usize,
    ) -> Result<GpuBuffer<u8>, DeviceError> {
        if len == 0 {
            return Ok(GpuBuffer::new_cpu(Vec::new()));
        }
        if zero_copy_threshold_bytes > 0 && len <= zero_copy_threshold_bytes {
            return Self::try_alloc_mapped::<u8>(stream, len);
        }
        Self::try_alloc_device::<u8>(stream, len)
    }

    /// 分配标准设备内存（原始 `cuMemAlloc`）。失败时回退 CPU 缓冲区。
    fn try_alloc_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<GpuBuffer<T>, DeviceError> {
        let num_bytes = len * size_of::<T>();
        // cuMemAlloc 作用于当前上下文——绑定后分配。
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| DeviceError::OutOfMemory {
                requested: num_bytes,
                detail: format!("CUDA 绑定上下文失败: {e:?}"),
            })?;
        // SAFETY: 分配未初始化设备内存，调用后按需写入。
        match unsafe { cudarc::driver::result::malloc_sync(num_bytes) } {
            Ok(ptr) => Ok(GpuBuffer::new_cuda(len, ptr, stream.context().clone())),
            Err(e) => {
                tracing::trace!("CUDA alloc failed ({e:?}), CPU fallback");
                // SAFETY: Pod 要求 Zeroable，全零表示对任意 Pod 类型有效。
                Ok(GpuBuffer::new_cpu(vec![T::zeroed(); len]))
            }
        }
    }

    /// 分配零拷贝（映射主机内存）缓冲区：
    /// `cuMemHostAlloc(CU_MEMHOSTALLOC_DEVICEMAP)` 分配的主机内存同时映射到
    /// 设备地址空间，主机与 kernel 直接读写同一物理内存，无需 memcpy。
    ///
    /// 分配失败时回退为标准设备内存（保证功能可用）。
    fn try_alloc_mapped<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<GpuBuffer<T>, DeviceError> {
        let num_bytes = len * size_of::<T>();
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| DeviceError::OutOfMemory {
                requested: num_bytes,
                detail: format!("CUDA 绑定上下文失败: {e:?}"),
            })?;
        // SAFETY: cuMemHostAlloc 分配未初始化内存；调用后立即按需写入。
        let host_ptr = unsafe {
            cudarc::driver::result::malloc_host(
                num_bytes,
                cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        }
        .map_err(|e| DeviceError::OutOfMemory {
            requested: num_bytes,
            detail: format!("CUDA 映射内存分配失败: {e:?}"),
        })?;
        let mut device_ptr: cudarc::driver::sys::CUdeviceptr = 0;
        // SAFETY: host_ptr 由 cuMemHostAlloc 分配，有效且非空。
        let get_ptr_result = unsafe {
            cudarc::driver::sys::cuMemHostGetDevicePointer_v2(
                std::ptr::addr_of_mut!(device_ptr),
                host_ptr,
                0,
            )
            .result()
        };
        if let Err(e) = get_ptr_result {
            // SAFETY: host_ptr 由 malloc_host 分配。
            unsafe {
                let _ = cudarc::driver::result::free_host(host_ptr);
            }
            return Err(DeviceError::OutOfMemory {
                requested: num_bytes,
                detail: format!("CUDA 获取映射设备指针失败: {e:?}"),
            });
        }
        let holder = CudaMappedHolder {
            // SAFETY: host_ptr 非空（cuMemHostAlloc 成功后必非空）。
            host_ptr: unsafe { std::ptr::NonNull::new_unchecked(host_ptr.cast::<T>()) },
            device_ptr,
            ctx: stream.context().clone(),
        };
        Ok(GpuBuffer::new_cuda_mapped(len, holder))
    }

    // ── HtoD ─────────────────────────────────────────────

    pub fn copy_to_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
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

        // 零拷贝路径：直接写入映射的主机内存（无 memcpy）。
        // 后续 kernel 启动建立主机写 → 设备读的可见性顺序。
        if let Some(host_ptr) = buffer.cuda_mapped_host_ptr() {
            // SAFETY: host_ptr 指向长度为 len 的映射内存；data 为等长有效切片。
            let _: () = unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), host_ptr, data.len());
            };
            return Ok(());
        }

        if let Some(ptr) = buffer.cuda_device_ptr() {
            // SAFETY: ptr 为有效设备内存；data 为等长有效主机切片。
            // 同步 memcpy 阻塞至完成，且与先前流内工作保持顺序。
            unsafe { cudarc::driver::result::memcpy_htod_sync(ptr, data) }
                .map_err(|e| DeviceError::TransferFailed(format!("CUDA HtoD: {e:?}")))?;
        } else if let Some(cpu) = buffer.cpu_data_mut() {
            cpu.clear();
            cpu.extend_from_slice(data);
        } else {
            let _ = stream;
            return Err(DeviceError::Unsupported("未知的 CUDA 缓冲区类型".into()));
        }
        Ok(())
    }

    // ── DtoH ─────────────────────────────────────────────

    pub fn copy_from_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        stream: &Arc<cudarc::driver::CudaStream>,
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

        // 零拷贝路径：kernel 写映射内存后，主机必须先同步（确保完成与可见性），
        // 然后直接从映射内存读取（无 memcpy）。
        if let Some(host_ptr) = buffer.cuda_mapped_host_ptr() {
            stream
                .synchronize()
                .map_err(|e| DeviceError::TransferFailed(format!("CUDA 同步失败: {e:?}")))?;
            // SAFETY: host_ptr 指向长度为 len 的映射内存；data 为等长有效切片。
            let _: () = unsafe {
                std::ptr::copy_nonoverlapping(host_ptr, data.as_mut_ptr(), data.len());
            };
            return Ok(());
        }

        if let Some(ptr) = buffer.cuda_device_ptr() {
            // SAFETY: ptr 为有效设备内存；data 为等长有效主机切片。
            // 同步 memcpy 等待先前设备工作完成后拷贝。
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(data, ptr) }
                .map_err(|e| DeviceError::TransferFailed(format!("CUDA DtoH: {e:?}")))?;
        } else if let Some(cpu) = buffer.cpu_data() {
            data.copy_from_slice(cpu);
        } else {
            return Err(DeviceError::Unsupported("未知的 CUDA 缓冲区类型".into()));
        }
        Ok(())
    }

    // ── Free ─────────────────────────────────────────────

    pub fn free<T: bytemuck::Pod>(buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        drop(buffer);
        Ok(())
    }
}
