//! CUDA Kernel 启动器 — 集成 NVRTC 编译。

use crate::common::DeviceError;
use crate::common::kernel::{KernelLaunch, KernelLauncher};
use crate::compile::cuda_compile::CudaKernelCompiler;
use crate::noise::kernels;

/// CUDA Kernel 启动器。
pub struct CudaKernelLauncher {
    compiler: Option<CudaKernelCompiler>,
    device_ctx: Option<std::sync::Arc<cudarc::driver::CudaContext>>,
}

impl CudaKernelLauncher {
    #[must_use]
    pub fn new() -> Self {
        Self {
            compiler: None,
            device_ctx: None,
        }
    }

    /// 初始化编译器并编译所有 Kernel。
    pub fn init(&mut self, ctx: std::sync::Arc<cudarc::driver::CudaContext>) {
        let mut compiler = CudaKernelCompiler::new();
        let flags = vec![
            "--fmad=false".to_string(),
            "--ftz=false".to_string(),
            "--prec-div=true".to_string(),
            "--prec-sqrt=true".to_string(),
        ];
        if let Err(e) = compiler.compile_all(&ctx, &flags) {
            tracing::warn!("CUDA NVRTC kernel compilation failed: {e}. CPU fallback will be used.");
        }
        self.compiler = Some(compiler);
        self.device_ctx = Some(ctx);
    }
}

impl Default for CudaKernelLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelLauncher for CudaKernelLauncher {
    fn launch(&self, _launch: KernelLaunch<'_>) -> Result<(), DeviceError> {
        if let Some(ref compiler) = self.compiler {
            let n = _launch.global_work_size[0];
            return compiler.launch(_launch.name, n);
        }
        Err(DeviceError::Unsupported(
            "CUDA kernel compiler not initialized".into(),
        ))
    }

    fn has_kernel(&self, name: &str) -> bool {
        self.compiler.as_ref().is_some_and(|c| c.has(name))
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        // CUDA streams are implicitly synchronized on dtoh copy
        Ok(())
    }
}
