//! CUDA Kernel 启动器 — 集成 NVRTC 编译和 LaunchArgs 执行。

use crate::common::DeviceError;
use crate::common::kernel::{GpuBufferRef, KernelArg, KernelLaunch, KernelLauncher};
use crate::compile::cuda_compile::CudaKernelCompiler;
use parking_lot::Mutex;
use std::sync::Arc;

/// CUDA Kernel 启动器。
pub struct CudaKernelLauncher {
    /// 编译器由 `Mutex` 包裹以支持延迟编译（`KernelLauncher` trait 方法为 `&self`）。
    compiler: Mutex<Option<CudaKernelCompiler>>,
    stream: Option<Arc<cudarc::driver::CudaStream>>,
    /// 是否启用 persistent kernel 模式（光照传播等迭代算法）
    persistent_enabled: bool,
}

impl CudaKernelLauncher {
    #[must_use]
    pub fn new() -> Self {
        Self {
            compiler: Mutex::new(None),
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
        let default_flags: &[String] = &[];
        let flags = flags.unwrap_or(default_flags);
        let mut compiler = CudaKernelCompiler::new(compile_ptx, flags);
        if let Err(e) = compiler.compile_all(ctx) {
            tracing::warn!("CUDA NVRTC kernel compilation failed: {e}. CPU fallback will be used.");
        }
        *self.compiler.lock() = Some(compiler);
        self.stream = Some(stream);
        self.persistent_enabled = persistent_enabled;
        if persistent_enabled {
            tracing::debug!("CUDA persistent kernel 模式已启用");
        }
    }

    /// 编译一个 JIT 特化 kernel。
    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        let ctx = self
            .stream
            .as_ref()
            .map(|s| s.context().clone())
            .ok_or_else(|| DeviceError::Internal("CUDA stream not initialized".into()))?;
        self.compiler.lock().as_mut().map_or_else(
            || {
                Err(DeviceError::Unsupported(
                    "CUDA compiler not initialized".into(),
                ))
            },
            |compiler| compiler.compile_jit_kernel(&ctx, jit_kernel),
        )
    }

    /// 按需编译单个预注册 kernel（延迟加载）。
    ///
    /// 从全局 CUDA 源码注册表查找源码，使用与 `compile_all` 相同的编译选项。
    /// 编译失败仅记录日志——上层 `try_launch_kernel` 会看到 kernel 仍不存在并回退 CPU。
    pub fn compile_kernel_by_name(&self, name: &str) {
        let Some(source) = crate::compile::lookup_cuda_kernel_source(name) else {
            tracing::debug!("CUDA lazy: '{name}' not in registry");
            return;
        };
        let Some(ctx) = self.stream.as_ref().map(|s| s.context().clone()) else {
            tracing::debug!("CUDA lazy: stream not initialized for '{name}'");
            return;
        };
        let mut guard = self.compiler.lock();
        let Some(compiler) = guard.as_mut() else {
            return;
        };
        if let Err(e) = compiler.compile_by_name(&ctx, name, source) {
            tracing::warn!("CUDA lazy: compile '{name}' failed: {e}");
            crate::logging::log_fallback(
                &crate::logging::FallbackReason::KernelCompileFailed(format!(
                    "CUDA lazy '{name}': {e}"
                )),
                "cuda_kernel::compile_kernel_by_name",
            );
        }
    }
}

impl Default for CudaKernelLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelLauncher for CudaKernelLauncher {
    fn launch(&self, launch: KernelLaunch<'_>) -> Result<(), DeviceError> {
        let compiler_guard = self.compiler.lock();
        let compiler = compiler_guard.as_ref().ok_or_else(|| {
            DeviceError::Unsupported("CUDA kernel compiler not initialized".into())
        })?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| DeviceError::Internal("CUDA stream not initialized".into()))?;

        let kernel = compiler
            .get_function(launch.name)
            .ok_or_else(|| DeviceError::KernelError(format!("'{}' not compiled", launch.name)))?;

        // 构建原始 kernel 参数数组：
        // - 标量参数：指向 `launch.args` 中值的指针（cuLaunchKernel 调用期间有效）；
        // - 缓冲区参数：直接以 `CUdeviceptr` 值作为指针传入（含零拷贝映射内存）。
        let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(launch.args.len());
        for arg in &launch.args {
            match arg {
                KernelArg::I32(v) => {
                    params.push(std::ptr::from_ref(v).cast_mut().cast());
                }
                KernelArg::F64(v) => {
                    params.push(std::ptr::from_ref(v).cast_mut().cast());
                }
                KernelArg::U32(v) => {
                    params.push(std::ptr::from_ref(v).cast_mut().cast());
                }
                KernelArg::USize(v) => {
                    params.push(std::ptr::from_ref(v).cast_mut().cast());
                }
                KernelArg::BufferRef(idx) => {
                    let buf_ref = launch.gpu_buffers.get(*idx).ok_or_else(|| {
                        DeviceError::LaunchFailed(format!(
                            "BufferRef({idx}) out of bounds ({} buffers)",
                            launch.gpu_buffers.len()
                        ))
                    })?;
                    // 指针参数：params 数组元素为「指向设备指针值的指针」
                    // （驱动解引用取得指针值）；地址指向 GpuBuffer 内的稳定字段，
                    // 在启动调用期间保持有效。
                    let ptr_addr = match buf_ref {
                        GpuBufferRef::F64(b) => b.cuda_device_ptr_addr().ok_or_else(|| {
                            DeviceError::Unsupported("F64 buffer is not a CUDA buffer".into())
                        })?,
                        GpuBufferRef::I32(b) => b.cuda_device_ptr_addr().ok_or_else(|| {
                            DeviceError::Unsupported("I32 buffer is not a CUDA buffer".into())
                        })?,
                        GpuBufferRef::U8(b) => b.cuda_device_ptr_addr().ok_or_else(|| {
                            DeviceError::Unsupported("U8 buffer is not a CUDA buffer".into())
                        })?,
                    };
                    params.push(ptr_addr.cast_mut().cast());
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
        // 动态共享内存（extern __shared__ kernel 使用）：汇总各参数大小
        let shared_mem_bytes = launch.local_mem_bytes.iter().sum::<usize>() as u32;

        // 检测是否为 persistent kernel 变体（名称含 "_persistent"）
        let is_persistent = launch.name.contains("_persistent") && self.persistent_enabled;

        // cuLaunchKernel 要求调用线程的上下文正确绑定（与 cudarc 的 LaunchArgs::launch 一致）。
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| DeviceError::LaunchFailed(format!("CUDA 绑定上下文: {e:?}")))?;

        // SAFETY: 参数数组按 kernel 签名顺序构建；缓冲区设备指针有效；
        // 标量指针指向 launch.args 中的值，在启动调用期间保持有效。
        let result = if is_persistent {
            tracing::debug!("CUDA: launching persistent kernel '{}'", launch.name);
            // SAFETY: kernel args match signature; cooperative launch requires SM 6.0+
            unsafe {
                cudarc::driver::result::launch_cooperative_kernel(
                    kernel.function,
                    (grid_dim, 1, 1),
                    (block_dim, 1, 1),
                    shared_mem_bytes,
                    stream.cu_stream(),
                    &mut params,
                )
            }
        } else {
            // SAFETY: kernel args match signature; config (grid/block dimensions) is valid.
            unsafe {
                cudarc::driver::result::launch_kernel(
                    kernel.function,
                    (grid_dim, 1, 1),
                    (block_dim, 1, 1),
                    shared_mem_bytes,
                    stream.cu_stream(),
                    &mut params,
                )
            }
        };

        result.map_err(|e| DeviceError::LaunchFailed(format!("'{}': {e:?}", launch.name)))?;

        Ok(())
    }

    fn has_kernel(&self, name: &str) -> bool {
        self.compiler.lock().as_ref().is_some_and(|c| c.has(name))
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        if let Some(stream) = self.stream.as_ref() {
            stream
                .synchronize()
                .map_err(|e| DeviceError::TransferFailed(format!("CUDA synchronize: {e:?}")))?;
        }
        Ok(())
    }
}
