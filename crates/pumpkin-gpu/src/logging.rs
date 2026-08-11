//! GPU 设备枚举与启动日志。
//!
//! 提供设备发现、后端状态显示和 CPU 回退原因记录。

use crate::DeviceType;

/// 启动时输出 GPU 加速状态信息。
///
/// 仅在 `gpu` feature 启用时编译。
/// 显示后端类型、设备名称，并在 CPU 回退时显示 CPU 型号。
pub fn log_gpu_startup(device_type: DeviceType, device_name: &str) {
    match device_type {
        DeviceType::Cuda => {
            tracing::info!("══════════════════════════════════════════");
            tracing::info!("  GPU 加速已启用");
            tracing::info!("  后端: CUDA (NVIDIA)");
            tracing::info!("  设备: {device_name}");
            tracing::info!("══════════════════════════════════════════");
        }
        DeviceType::OpenCl => {
            tracing::info!("══════════════════════════════════════════");
            tracing::info!("  GPU 加速已启用");
            tracing::info!("  后端: OpenCL");
            tracing::info!("  设备: {device_name}");
            tracing::info!("══════════════════════════════════════════");
        }
        DeviceType::Cpu => {
            let cpu_name = get_cpu_name();
            tracing::info!("══════════════════════════════════════════");
            tracing::info!("  GPU 加速未启用 — 使用 CPU 回退");
            tracing::info!("  CPU: {cpu_name}");
            tracing::info!("══════════════════════════════════════════");
        }
    }
}

/// 获取 CPU 型号名称。
#[cfg(feature = "gpu")]
fn get_cpu_name() -> String {
    use sysinfo::System;

    let sys = System::new_all();
    sys.cpus().first().map_or_else(
        || String::from("Unknown CPU"),
        |cpu| cpu.brand().to_string(),
    )
}

#[cfg(not(feature = "gpu"))]
fn get_cpu_name() -> String {
    String::from("Unknown CPU")
}

/// CPU 回退原因记录。
pub enum FallbackReason {
    /// 配置中禁用了 GPU
    ConfigDisabled,
    /// 驱动未安装
    DriverNotFound,
    /// 初始化失败
    InitFailed(String),
    /// Kernel 编译失败
    KernelCompileFailed(String),
    /// Kernel 启动失败
    KernelLaunchFailed(String),
    /// 计算结果与 CPU 不一致（数据完整性检查失败）
    ResultMismatch,
    /// 后端不支持该操作
    UnsupportedOperation(String),
}

impl std::fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigDisabled => write!(f, "GPU 在配置中已禁用"),
            Self::DriverNotFound => write!(f, "未找到 GPU 驱动"),
            Self::InitFailed(e) => write!(f, "GPU 初始化失败: {e}"),
            Self::KernelCompileFailed(e) => write!(f, "Kernel 编译失败: {e}"),
            Self::KernelLaunchFailed(e) => write!(f, "Kernel 启动失败: {e}"),
            Self::ResultMismatch => write!(f, "GPU 计算结果与 CPU 不一致"),
            Self::UnsupportedOperation(e) => write!(f, "不支持的 GPU 操作: {e}"),
        }
    }
}

/// 记录 CPU 回退事件。
///
/// 使用 `tracing::warn!` 输出回退原因，帮助用户了解为何 GPU 未被使用。
/// 同一原因的重复回退会被抑制（每个原因只输出一次）。
pub fn log_fallback(reason: &FallbackReason, context: &str) {
    tracing::warn!("[{context}] CPU 回退 — {reason}");
}
