//! CUDA 内存分配与传输。
//!
//! 使用 cudarc 0.19 API：`CudaStream::alloc` + `memcpy_htod` / `memcpy_dtoh`。
//! 失败时自动回退 CPU 缓冲区。
//!
//! 零拷贝（pinned memory）：`CudaStream::alloc_pinned` 框架就绪，
//! 需 GPU 硬件验证后激活。

use crate::common::{DeviceError, GpuBuffer};
use std::sync::Arc;

pub struct CudaMemory;

impl CudaMemory {
    // ── f64 ──────────────────────────────────────────────

    pub fn alloc_f64(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
        _zero_copy_threshold_bytes: usize,
    ) -> Result<GpuBuffer<f64>, DeviceError> {
        if len == 0 {
            return Ok(GpuBuffer::new_cpu(Vec::new()));
        }
        Self::try_alloc_device_f64(stream, len)
    }

    fn try_alloc_device_f64(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<GpuBuffer<f64>, DeviceError> {
        // SAFETY: CudaStream is valid, f64 implements DeviceRepr
        match unsafe { stream.alloc::<f64>(len) } {
            Ok(slice) => Ok(GpuBuffer::new_cuda(len, slice)),
            Err(e) => {
                tracing::trace!("CUDA alloc f64 failed ({e:?}), CPU fallback");
                Ok(GpuBuffer::new_cpu(vec![0.0f64; len]))
            }
        }
    }

    // ── i32 ──────────────────────────────────────────────

    pub fn alloc_i32(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
        _zero_copy_threshold_bytes: usize,
    ) -> Result<GpuBuffer<i32>, DeviceError> {
        if len == 0 {
            return Ok(GpuBuffer::new_cpu(Vec::new()));
        }
        match unsafe { stream.alloc::<i32>(len) } {
            Ok(slice) => Ok(GpuBuffer::new_cuda(len, slice)),
            Err(e) => {
                tracing::trace!("CUDA alloc i32 failed ({e:?}), CPU fallback");
                Ok(GpuBuffer::new_cpu(vec![0i32; len]))
            }
        }
    }

    // ── u8 ───────────────────────────────────────────────

    pub fn alloc_u8(
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
        _zero_copy_threshold_bytes: usize,
    ) -> Result<GpuBuffer<u8>, DeviceError> {
        if len == 0 {
            return Ok(GpuBuffer::new_cpu(Vec::new()));
        }
        match unsafe { stream.alloc::<u8>(len) } {
            Ok(slice) => Ok(GpuBuffer::new_cuda(len, slice)),
            Err(e) => {
                tracing::trace!("CUDA alloc u8 failed ({e:?}), CPU fallback");
                Ok(GpuBuffer::new_cpu(vec![0u8; len]))
            }
        }
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

        match buffer.cuda_slice_mut() {
            Ok(slice) => {
                // SAFETY: slice is valid, data lives long enough
                stream
                    .memcpy_htod(data, slice)
                    .map_err(|e| DeviceError::TransferFailed(format!("CUDA HtoD: {e:?}")))?;
                Ok(())
            }
            Err(_) => {
                if let Some(cpu) = buffer.cpu_data_mut() {
                    cpu.clear();
                    cpu.extend_from_slice(data);
                }
                Ok(())
            }
        }
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

        match buffer.cuda_slice() {
            Ok(slice) => {
                // SAFETY: slice is valid, data is valid mutable host buffer
                stream
                    .memcpy_dtoh(slice, data)
                    .map_err(|e| DeviceError::TransferFailed(format!("CUDA DtoH: {e:?}")))?;
                Ok(())
            }
            Err(_) => {
                if let Some(cpu) = buffer.cpu_data() {
                    data.copy_from_slice(cpu);
                }
                Ok(())
            }
        }
    }

    // ── Free ─────────────────────────────────────────────

    pub fn free<T: bytemuck::Pod>(buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        drop(buffer);
        Ok(())
    }
}
