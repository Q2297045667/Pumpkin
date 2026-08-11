//! CUDA Kernel 启动器 — 集成 NVRTC 编译和 LaunchArgs 执行。

use crate::common::DeviceError;
use crate::common::kernel::{GpuBufferRef, KernelArg, KernelLaunch, KernelLauncher};
use crate::compile::cuda_compile::CudaKernelCompiler;
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

/// CUDA Kernel 启动器。
pub struct CudaKernelLauncher {
    compiler: Option<CudaKernelCompiler>,
    stream: Option<Arc<cudarc::driver::CudaStream>>,
    /// 是否启用 persistent kernel 模式（光照传播等迭代算法）
    persistent_enabled: bool,
}

impl CudaKernelLauncher {
    #[must_use]
    pub fn new() -> Self {
        Self {
            compiler: None,
            stream: None,
            persistent_enabled: false,
        }
    }

    /// 初始化编译器并编译所有 Kernel。
    pub fn init(
        &mut self,
        ctx: &Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
        flags: Option<&[String]>,
        persistent_enabled: bool,
        compile_ptx: Option<&str>,
    ) {
        let mut compiler = CudaKernelCompiler::new(compile_ptx);
        let default_flags: &[String] = &[];
        let flags = flags.unwrap_or(default_flags);
        if let Err(e) = compiler.compile_all(ctx, flags) {
            tracing::warn!("CUDA NVRTC kernel compilation failed: {e}. CPU fallback will be used.");
        }
        self.compiler = Some(compiler);
        self.stream = Some(stream);
        self.persistent_enabled = persistent_enabled;
        if persistent_enabled {
            tracing::debug!("CUDA persistent kernel 模式已启用");
        }
    }

    /// 编译一个 JIT 特化 kernel。
    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &mut self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        let ctx = self
            .stream
            .as_ref()
            .map(|s| s.context().clone())
            .ok_or_else(|| DeviceError::Internal("CUDA stream not initialized".into()))?;
        self.compiler.as_mut().map_or_else(
            || {
                Err(DeviceError::Unsupported(
                    "CUDA compiler not initialized".into(),
                ))
            },
            |compiler| compiler.compile_jit_kernel(&ctx, jit_kernel),
        )
    }
}

impl Default for CudaKernelLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelLauncher for CudaKernelLauncher {
    fn launch(&self, launch: KernelLaunch<'_>) -> Result<(), DeviceError> {
        let compiler = self.compiler.as_ref().ok_or_else(|| {
            DeviceError::Unsupported("CUDA kernel compiler not initialized".into())
        })?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| DeviceError::Internal("CUDA stream not initialized".into()))?;

        let func = compiler
            .get_function(launch.name)
            .ok_or_else(|| DeviceError::KernelError(format!("'{}' not compiled", launch.name)))?;

        // 构建 LaunchArgs builder
        let mut builder = stream.launch_builder(func);

        for arg in &launch.args {
            match arg {
                KernelArg::I32(v) => {
                    builder.arg(v);
                }
                KernelArg::F64(v) => {
                    builder.arg(v);
                }
                KernelArg::U32(v) => {
                    builder.arg(v);
                }
                KernelArg::USize(v) => {
                    builder.arg(v);
                }
                KernelArg::BufferRef(idx) => {
                    let buf_ref = launch.gpu_buffers.get(*idx).ok_or_else(|| {
                        DeviceError::LaunchFailed(format!(
                            "BufferRef({idx}) out of bounds ({} buffers)",
                            launch.gpu_buffers.len()
                        ))
                    })?;
                    match buf_ref {
                        GpuBufferRef::F64(b) => {
                            let slice = b.cuda_slice().map_err(|e| {
                                DeviceError::LaunchFailed(format!("F64 buffer: {e}"))
                            })?;
                            builder.arg(slice);
                        }
                        GpuBufferRef::I32(b) => {
                            let slice = b.cuda_slice().map_err(|e| {
                                DeviceError::LaunchFailed(format!("I32 buffer: {e}"))
                            })?;
                            builder.arg(slice);
                        }
                        GpuBufferRef::U8(b) => {
                            let slice = b.cuda_slice().map_err(|e| {
                                DeviceError::LaunchFailed(format!("U8 buffer: {e}"))
                            })?;
                            builder.arg(slice);
                        }
                    }
                }
                // CPU-only arg types — CUDA GPU path 不支持
                KernelArg::F64Slice(_)
                | KernelArg::F64SliceMut(_)
                | KernelArg::I32Slice(_)
                | KernelArg::I32SliceMut(_)
                | KernelArg::U8Slice(_)
                | KernelArg::U8SliceMut(_) => {
                    return Err(DeviceError::Unsupported(
                        "Slice args not supported on CUDA GPU path".into(),
                    ));
                }
            }
        }

        // 执行 kernel
        let n = launch.global_work_size[0] as u32;
        let block_dim = launch
            .local_work_size
            .map_or(256u32, |l| l[0] as u32)
            .min(n);
        let grid_dim = n.div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        // 检测是否为 persistent kernel 变体（名称含 "_persistent"）
        let is_persistent = launch.name.contains("_persistent") && self.persistent_enabled;

        // SAFETY: all kernel args have been pushed in correct order matching __global__ signature.
        // GPU buffers are valid for the duration of the kernel execution.
        let result = if is_persistent {
            tracing::debug!("CUDA: launching persistent kernel '{}'", launch.name);
            // SAFETY: kernel args match signature; cooperative launch requires SM 6.0+
            // Config (grid/block dimensions, shared memory) is valid for this kernel.
            unsafe { builder.launch_cooperative(cfg) }
        } else {
            // SAFETY: kernel args match signature; config (grid/block dimensions) is valid.
            unsafe { builder.launch(cfg) }
        };

        result.map_err(|e| DeviceError::LaunchFailed(format!("'{}': {e:?}", launch.name)))?;

        Ok(())
    }

    fn has_kernel(&self, name: &str) -> bool {
        self.compiler.as_ref().is_some_and(|c| c.has(name))
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        // CUDA default stream is implicitly synchronized on DtoH copy
        Ok(())
    }
}
