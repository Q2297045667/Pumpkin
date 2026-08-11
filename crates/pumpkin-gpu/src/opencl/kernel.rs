//! OpenCL Kernel 启动器。
//!
//! 管理 OpenCL Program 和 Kernel 的加载与启动。

use crate::common::DeviceError;
use crate::common::kernel::{KernelLaunch, KernelLauncher};

/// OpenCL Kernel 启动器（存根实现）。
///
/// 当前阶段仅提供框架代码，不加载实际的 OpenCL 程序。
/// 后续迭代将在此处集成 SPIR-V 或 OpenCL C 源码的编译。
pub struct OpenClKernelLauncher;

impl OpenClKernelLauncher {
    /// 创建新的 OpenCL Kernel 启动器。
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenClKernelLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelLauncher for OpenClKernelLauncher {
    fn launch(&self, launch: KernelLaunch<'_>) -> Result<(), DeviceError> {
        // TODO: 编译 OpenCL 程序并启动 Kernel
        let _ = launch;
        Err(DeviceError::Unsupported(format!(
            "OpenCL Kernel '{}' 尚未实现",
            launch.name
        )))
    }

    fn has_kernel(&self, name: &str) -> bool {
        let _ = name;
        false
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        // TODO: 调用 clFinish
        Ok(())
    }
}
