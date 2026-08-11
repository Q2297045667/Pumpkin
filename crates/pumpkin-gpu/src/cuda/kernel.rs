//! CUDA Kernel 启动器。
//!
//! 管理 CUDA Kernel 的加载和启动。

use crate::common::DeviceError;
use crate::common::kernel::{KernelLaunch, KernelLauncher};

/// CUDA Kernel 启动器（存根实现）。
///
/// 当前阶段仅提供框架代码，不加载实际的 PTX。
/// 后续迭代将在此处集成 NVRTC 编译或预编译 PTX 加载。
pub struct CudaKernelLauncher;

impl CudaKernelLauncher {
    /// 创建新的 CUDA Kernel 启动器。
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for CudaKernelLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelLauncher for CudaKernelLauncher {
    fn launch(&self, launch: KernelLaunch<'_>) -> Result<(), DeviceError> {
        // TODO: 加载 PTX 模块并启动 Kernel
        // 当前返回 "未实现" 错误，等待后续版本补充
        let _ = launch;
        Err(DeviceError::Unsupported(format!(
            "CUDA Kernel '{}' 尚未实现",
            launch.name
        )))
    }

    fn has_kernel(&self, name: &str) -> bool {
        // 当前未加载任何 Kernel
        let _ = name;
        false
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        // TODO: 调用 cuCtxSynchronize
        Ok(())
    }
}
