//! GPU 设备枚举与启动日志。
//!
//! 提供设备发现、后端状态显示和 CPU 回退原因记录。

use crate::DeviceType;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

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
            #[cfg(feature = "gpu")]
            let cpu_name = crate::cpu::cpu_name();
            #[cfg(not(feature = "gpu"))]
            let cpu_name = String::from("Unknown CPU");
            tracing::info!("══════════════════════════════════════════");
            tracing::info!("  GPU 加速未启用 — 使用 CPU 回退");
            tracing::info!("  CPU: {cpu_name}");
            tracing::info!("══════════════════════════════════════════");
        }
    }
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
/// 同一类原因的重复回退会被抑制，避免启动时每个后端或 kernel 重复刷屏。
pub fn log_fallback(reason: &FallbackReason, context: &str) {
    static LOGGED_KINDS: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let logged_kinds = LOGGED_KINDS.get_or_init(|| Mutex::new(HashSet::new()));
    let kind = reason.kind();
    let Ok(mut logged_kinds) = logged_kinds.lock() else {
        tracing::debug!("[{context}] CPU 回退 — {reason}");
        return;
    };
    if !logged_kinds.insert(kind) {
        tracing::debug!("[{context}] CPU 回退 — {reason}");
        return;
    }
    tracing::warn!("GPU 不可用，已切换到 CPU 路径 — {reason}");
}

impl FallbackReason {
    fn kind(&self) -> &'static str {
        match self {
            Self::ConfigDisabled => "config_disabled",
            Self::DriverNotFound => "driver_not_found",
            Self::InitFailed(_) => "init_failed",
            Self::KernelCompileFailed(_) => "kernel_compile_failed",
            Self::KernelLaunchFailed(_) => "kernel_launch_failed",
            Self::ResultMismatch => "result_mismatch",
            Self::UnsupportedOperation(_) => "unsupported_operation",
        }
    }
}
