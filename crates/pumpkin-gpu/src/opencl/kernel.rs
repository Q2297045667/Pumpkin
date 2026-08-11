//! OpenCL Kernel 启动器。

use crate::common::DeviceError;
use crate::common::kernel::{KernelLaunch, KernelLauncher};
use crate::compile::opencl_compile::OpenClKernelCompiler;
use opencl3::context::Context;
use opencl3::device::Device;

pub struct OpenClKernelLauncher {
    compiler: Option<OpenClKernelCompiler>,
}

impl OpenClKernelLauncher {
    #[must_use]
    pub fn new() -> Self {
        Self { compiler: None }
    }

    pub fn init(&mut self, ctx: &Context, device: &Device) {
        let mut compiler = OpenClKernelCompiler::new();
        let flags = vec!["-cl-fp32-correctly-rounded-divide-sqrt".to_string()];
        if let Err(e) = compiler.compile_all(ctx, device.id(), &flags) {
            tracing::warn!("OpenCL kernel compilation failed: {e}. CPU fallback will be used.");
        }
        self.compiler = Some(compiler);
    }
}

impl Default for OpenClKernelLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelLauncher for OpenClKernelLauncher {
    fn launch(&self, _launch: KernelLaunch<'_>) -> Result<(), DeviceError> {
        if let Some(ref c) = self.compiler {
            return c.launch(_launch.name, _launch.global_work_size[0]);
        }
        Err(DeviceError::Unsupported(
            "OpenCL kernel compiler not initialized".into(),
        ))
    }
    fn has_kernel(&self, name: &str) -> bool {
        self.compiler.as_ref().is_some_and(|c| c.has(name))
    }
    fn synchronize(&self) -> Result<(), DeviceError> {
        Ok(())
    }
}
