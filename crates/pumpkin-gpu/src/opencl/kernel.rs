//! OpenCL Kernel 启动器。
//!
//! 集成 OpenCL kernel 编译、参数设置和命令入队。
//! 支持多 CommandQueue 流水线：`pipeline_queues > 1` 时使用轮转分配。

use crate::common::DeviceError;
use crate::common::kernel::{GpuBufferRef, KernelArg, KernelLaunch, KernelLauncher};
use crate::compile::opencl_compile::OpenClKernelCompiler;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct OpenClKernelLauncher {
    compiler: Option<OpenClKernelCompiler>,
    queues: Vec<CommandQueue>,
    /// 轮转计数器，用于多队列流水线
    next_queue: AtomicUsize,
}

impl OpenClKernelLauncher {
    #[must_use]
    pub fn new() -> Self {
        Self {
            compiler: None,
            queues: Vec::new(),
            next_queue: AtomicUsize::new(0),
        }
    }

    /// 初始化启动器：编译所有 kernel 并保存命令队列。
    ///
    /// `queues` 的所有权被移入启动器。
    /// 当 `queues.len() > 1` 时启用轮转流水线模式。
    pub fn init(
        &mut self,
        ctx: &Context,
        device: &Device,
        queues: Vec<CommandQueue>,
        flags: Option<&[String]>,
    ) {
        let mut compiler = OpenClKernelCompiler::new();
        let flags = flags.map_or_else(
            || vec!["-cl-fp32-correctly-rounded-divide-sqrt".to_string()],
            <[String]>::to_vec,
        );
        if let Err(e) = compiler.compile_all(ctx, device.id(), &flags) {
            tracing::warn!("OpenCL kernel compilation failed: {e}. CPU fallback will be used.");
        }
        if queues.len() > 1 {
            tracing::debug!("OpenCL 流水线: {} 个命令队列（轮转模式）", queues.len());
        }
        self.compiler = Some(compiler);
        self.queues = queues;
    }

    /// 编译一个 JIT 特化 kernel。
    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &mut self,
        ctx: &Context,
        device: &Device,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), crate::common::DeviceError> {
        let compiler = self.compiler.as_mut().ok_or_else(|| {
            crate::common::DeviceError::Internal("OpenCL compiler not initialized".into())
        })?;
        compiler.compile_jit_kernel(ctx, device.id(), jit_kernel)
    }

    /// 获取命令队列引用（供 `OpenClBackend` 的 buffer 操作使用）。
    /// 返回第一个队列（buffer 操作不需要流水线）。
    ///
    /// # Panics
    ///
    /// 如果在 `init()` 之前调用会 panic。
    #[allow(clippy::expect_used)]
    pub fn queue(&self) -> &CommandQueue {
        self.queues
            .first()
            .expect("OpenClKernelLauncher::queue() called before init()")
    }

    /// 获取当前轮转索引的命令队列。
    /// 在流水线模式下每次调用返回不同的队列（round-robin）。
    fn next_queue(&self) -> &CommandQueue {
        let idx = self.next_queue.fetch_add(1, Ordering::Relaxed) % self.queues.len();
        &self.queues[idx]
    }
}

impl Default for OpenClKernelLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelLauncher for OpenClKernelLauncher {
    fn launch(&self, launch: KernelLaunch<'_>) -> Result<(), DeviceError> {
        let compiler = self.compiler.as_ref().ok_or_else(|| {
            DeviceError::Unsupported("OpenCL kernel compiler not initialized".into())
        })?;
        let queue = self.next_queue();

        let kernel = compiler
            .get_kernel(launch.name)
            .ok_or_else(|| DeviceError::KernelError(format!("'{}' not compiled", launch.name)))?;

        // 设置 kernel 参数
        for (arg_index, arg) in (0u32..).zip(launch.args.iter()) {
            match arg {
                KernelArg::I32(v) => {
                    // SAFETY: kernel is valid and compiled; arg_index is within signature bounds
                    unsafe { kernel.set_arg(arg_index, v) }.map_err(|e| {
                        DeviceError::LaunchFailed(format!("set_arg {arg_index}: {e}"))
                    })?;
                }
                KernelArg::F64(v) => {
                    // SAFETY: kernel is valid and compiled; arg_index is within signature bounds
                    unsafe { kernel.set_arg(arg_index, v) }.map_err(|e| {
                        DeviceError::LaunchFailed(format!("set_arg {arg_index}: {e}"))
                    })?;
                }
                KernelArg::BufferRef(idx) => {
                    let buf_ref = launch.gpu_buffers.get(*idx).ok_or_else(|| {
                        DeviceError::LaunchFailed(format!(
                            "BufferRef({idx}) out of bounds ({} buffers)",
                            launch.gpu_buffers.len()
                        ))
                    })?;
                    let handle = match buf_ref {
                        GpuBufferRef::F64(b) => b.opencl_handle(),
                        GpuBufferRef::I32(b) => b.opencl_handle(),
                        GpuBufferRef::U8(b) => b.opencl_handle(),
                    }
                    .ok_or_else(|| {
                        DeviceError::Unsupported("Buffer is not an OpenCL buffer".into())
                    })?;
                    // SAFETY: kernel is valid and compiled; cl_mem handle is valid for duration
                    unsafe { kernel.set_arg(arg_index, &handle) }.map_err(|e| {
                        DeviceError::LaunchFailed(format!("set_arg {arg_index} (buffer): {e}"))
                    })?;
                }
                KernelArg::F64Slice(_)
                | KernelArg::F64SliceMut(_)
                | KernelArg::U8Slice(_)
                | KernelArg::U8SliceMut(_)
                | KernelArg::I32Slice(_)
                | KernelArg::I32SliceMut(_)
                | KernelArg::USize(_)
                | KernelArg::U32(_) => {
                    let msg = format!("Arg type not supported on OpenCL GPU path: {arg:?}");
                    return Err(DeviceError::Unsupported(msg));
                }
            }
        }

        // 执行 kernel
        let gs = launch.global_work_size[0];
        let ls = launch.local_work_size.map_or(256.min(gs), |l| l[0].min(gs));
        let gws: [usize; 1] = [gs];
        let lws: [usize; 1] = [ls];

        // SAFETY: all args set, kernel valid, work sizes valid
        let result = unsafe {
            queue.enqueue_nd_range_kernel(
                kernel.get(),
                1,
                std::ptr::null::<usize>(),
                gws.as_ptr(),
                lws.as_ptr(),
                &[],
            )
        };
        result.map_err(|e| DeviceError::LaunchFailed(format!("'{}': {e}", launch.name)))?;

        Ok(())
    }

    fn has_kernel(&self, name: &str) -> bool {
        self.compiler.as_ref().is_some_and(|c| c.has(name))
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        for q in &self.queues {
            q.finish()
                .map_err(|e| DeviceError::TransferFailed(format!("finish: {e}")))?;
        }
        Ok(())
    }
}
