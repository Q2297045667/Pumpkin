//! OpenCL 缓冲区管理。

use crate::common::DeviceError;
use crate::common::GpuBuffer;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::memory::Buffer;

fn alloc_raw(ctx: &Context, num_bytes: usize) -> Result<Buffer<u8>, DeviceError> {
    let size = if num_bytes == 0 { 1 } else { num_bytes };

    unsafe {
        Buffer::<u8>::create(
            ctx,
            opencl3::memory::CL_MEM_READ_WRITE,
            size,
            std::ptr::null_mut::<std::ffi::c_void>(),
        )
    }
    .map_err(|e| DeviceError::OutOfMemory {
        requested: num_bytes,
        detail: format!("OpenCL 分配失败: {e}"),
    })
}

/// 在 OpenCL 设备上分配 f64 缓冲区。
pub fn alloc_f64(
    ctx: &Context,
    _queue: &CommandQueue,
    len: usize,
) -> Result<GpuBuffer<f64>, DeviceError> {
    let num_bytes = len.checked_mul(size_of::<f64>()).unwrap_or(0);
    let buf = alloc_raw(ctx, num_bytes)?;
    Ok(GpuBuffer::new_opencl(len, buf))
}

/// 在 OpenCL 设备上分配 i32 缓冲区。
pub fn alloc_i32(
    ctx: &Context,
    _queue: &CommandQueue,
    len: usize,
) -> Result<GpuBuffer<i32>, DeviceError> {
    let num_bytes = len.checked_mul(size_of::<i32>()).unwrap_or(0);
    let buf = alloc_raw(ctx, num_bytes)?;
    Ok(GpuBuffer::new_opencl(len, buf))
}

/// 在 OpenCL 设备上分配 u8 缓冲区。
pub fn alloc_u8(
    ctx: &Context,
    _queue: &CommandQueue,
    len: usize,
) -> Result<GpuBuffer<u8>, DeviceError> {
    let buf = alloc_raw(ctx, len)?;
    Ok(GpuBuffer::new_opencl(len, buf))
}

/// 获取 OpenCL buffer 的可变引用（通过 wrapper 的 interior mutability）。
fn get_mut_buffer(buf: &GpuBuffer<impl bytemuck::Pod>) -> Result<&mut Buffer<u8>, &'static str> {
    let holder = match &buf.raw {
        crate::common::buffer::RawBuffer::OpenCl(h) => h,
        _ => return Err("不是 OpenCL 缓冲区"),
    };
    // SAFETY: We ensure exclusive access at the application level
    Ok(unsafe { &mut *holder.buffer.get() })
}

/// HtoD: 主机 → 设备。
pub fn copy_to_device<T: bytemuck::Pod>(
    _ctx: &Context,
    queue: &CommandQueue,
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

    let cl_buf = get_mut_buffer(buffer).map_err(|e| DeviceError::Unsupported(e.into()))?;

    let host_bytes: &[u8] = bytemuck::cast_slice(data);

    unsafe {
        queue.enqueue_write_buffer(
            cl_buf,
            opencl3::command_queue::CL_BLOCKING,
            0,
            host_bytes,
            &[],
        )
    }
    .map_err(|e| DeviceError::TransferFailed(format!("OpenCL HtoD 拷贝失败: {e}")))?;

    Ok(())
}

/// DtoH: 设备 → 主机。
pub fn copy_from_device<T: bytemuck::Pod>(
    _ctx: &Context,
    queue: &CommandQueue,
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

    let cl_buf = get_mut_buffer(buffer).map_err(|e| DeviceError::Unsupported(e.into()))?;

    let host_bytes: &mut [u8] = bytemuck::cast_slice_mut(data);

    unsafe {
        queue.enqueue_read_buffer(
            cl_buf,
            opencl3::command_queue::CL_BLOCKING,
            0,
            host_bytes,
            &[],
        )
    }
    .map_err(|e| DeviceError::TransferFailed(format!("OpenCL DtoH 拷贝失败: {e}")))?;

    Ok(())
}

/// 释放 OpenCL 缓冲区。
pub fn free<T: bytemuck::Pod>(_ctx: &Context, buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
    drop(buffer);
    Ok(())
}
