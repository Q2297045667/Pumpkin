//! Kernel 启动的抽象接口。
//!
//! GPU 后端需要实现此 trait 以支持 Kernel 的执行。

use super::error::DeviceError;

/// GPU 缓冲区引用（跨后端抽象）。
///
/// 用于在 [`KernelLaunch`] 中传递 GPU buffer 的引用，
/// 后端实现可根据此引用提取原始句柄（cl_mem / CUdeviceptr）。
#[derive(Debug, Clone, Copy)]
pub enum GpuBufferRef<'a> {
    /// `f64` GPU 缓冲区引用
    F64(&'a super::buffer::GpuBuffer<f64>),
    /// `i32` GPU 缓冲区引用
    I32(&'a super::buffer::GpuBuffer<i32>),
    /// `u8` GPU 缓冲区引用
    U8(&'a super::buffer::GpuBuffer<u8>),
}

/// Kernel 参数类型。
///
/// 表示一个 Kernel 参数，可以是标量或缓冲区引用。
#[derive(Debug)]
pub enum KernelArg<'a> {
    /// `f64` 标量值
    F64(f64),
    /// `i32` 标量值
    I32(i32),
    /// `u32` 标量值
    U32(u32),
    /// `usize` 标量值
    USize(usize),
    /// 指向设备缓冲区的引用（通过索引标识）
    BufferRef(usize),
    /// `f64` 切片引用（仅 CPU 后端使用）
    F64Slice(&'a [f64]),
    /// `f64` 可变切片引用（仅 CPU 后端使用）
    F64SliceMut(&'a mut [f64]),
    /// `i32` 切片引用（仅 CPU 后端使用）
    I32Slice(&'a [i32]),
    /// `i32` 可变切片引用（仅 CPU 后端使用）
    I32SliceMut(&'a mut [i32]),
    /// `u8` 切片引用（仅 CPU 后端使用）
    U8Slice(&'a [u8]),
    /// `u8` 可变切片引用（仅 CPU 后端使用）
    U8SliceMut(&'a mut [u8]),
}

/// Kernel 启动描述。
pub struct KernelLaunch<'a> {
    /// Kernel 名称
    pub name: &'a str,
    /// 全局工作大小（总线程数）
    pub global_work_size: [usize; 3],
    /// 本地工作大小（每个 work-group 的线程数）
    pub local_work_size: Option<[usize; 3]>,
    /// Kernel 参数列表
    pub args: Vec<KernelArg<'a>>,
    /// GPU 缓冲区引用列表（通过 `BufferRef(usize)` 索引引用）
    pub gpu_buffers: Vec<GpuBufferRef<'a>>,
    /// 尾部 local / shared 内存参数的大小（字节），按 kernel 签名顺序。
    /// CUDA 后端汇总为 `LaunchConfig.shared_mem_bytes`；
    /// OpenCL 后端通过 `set_arg_local_buffer` 依次设置。
    pub local_mem_bytes: Vec<usize>,
}

/// Kernel 启动器 trait。
///
/// 每个 GPU 后端需要实现此 trait 来提供 Kernel 执行能力。
pub trait KernelLauncher {
    /// 启动一个 Kernel。
    ///
    /// # Errors
    /// Kernel 不存在、参数错误或设备执行失败时返回错误。
    fn launch(&self, launch: KernelLaunch<'_>) -> Result<(), DeviceError>;

    /// 检查指定名称的 Kernel 是否存在。
    #[must_use]
    fn has_kernel(&self, name: &str) -> bool;

    /// 等待所有已提交的 Kernel 执行完成。
    ///
    /// # Errors
    /// 同步失败时返回错误。
    fn synchronize(&self) -> Result<(), DeviceError>;
}
