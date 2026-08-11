//! GPU 加速模块接口。
//!
//! 当 `gpu` feature 启用时，提供 GPU 加速的噪声采样、密度计算、
//! 三线性插值和光照传播等能力。
//!
//! 当 GPU 不可用或 feature 未启用时，所有操作自动回退到 CPU 路径。

use std::sync::OnceLock;

use pumpkin_config::gpu::GpuConfig;

/// 全局 GPU 配置（由 Server 启动时设置）。
static GPU_CONFIG: OnceLock<GpuConfig> = OnceLock::new();

/// 在启动阶段注入 GPU 配置，使其对后续所有子系统可用。
///
/// # Panics
///
/// 如果配置已被设置则 panic（保证只设置一次）。
pub fn init_gpu_config(config: GpuConfig) {
    GPU_CONFIG
        .set(config)
        .expect("GPU config has already been set");
}

/// 获取全局 GPU 配置（如果已被初始化）。
#[must_use]
pub fn get_gpu_config() -> Option<&'static GpuConfig> {
    GPU_CONFIG.get()
}

/// GPU 计算状态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuStatus {
    /// GPU 已激活，附带设备名称
    Active(String),
    /// 使用 CPU 回退
    Fallback,
}

/// GPU 计算句柄。
///
/// 封装后端选择和回退逻辑。
pub struct GpuCompute {
    status: GpuStatus,
    /// 配置副本（用于运行时查询各个子系统的启用状态）
    pub config: GpuConfig,
}

impl GpuCompute {
    /// 使用指定配置创建 GPU 计算句柄。
    #[must_use]
    pub fn new(config: GpuConfig) -> Self {
        #[cfg(feature = "gpu")]
        {
            let device = pumpkin_gpu::GpuDevice::from_config(&config);
            #[allow(clippy::single_match_else)]
            match device.device_type() {
                pumpkin_gpu::DeviceType::Cpu => Self {
                    status: GpuStatus::Fallback,
                    config,
                },
                _ => {
                    let name = device.device_name().to_string();
                    Self {
                        status: GpuStatus::Active(name),
                        config,
                    }
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            Self {
                status: GpuStatus::Fallback,
                config,
            }
        }
    }

    /// 使用默认配置创建（默认不启用 GPU）。
    #[must_use]
    pub fn default_disabled() -> Self {
        Self::new(GpuConfig::default())
    }

    /// 返回当前 GPU 计算状态。
    #[must_use]
    pub const fn status(&self) -> &GpuStatus {
        &self.status
    }

    /// 是否使用了 GPU 加速。
    #[must_use]
    pub const fn is_gpu_active(&self) -> bool {
        matches!(self.status, GpuStatus::Active(_))
    }

    /// 噪声加速是否启用（需同时满足全局启用和子系统启用）。
    #[must_use]
    pub const fn noise_enabled(&self) -> bool {
        self.is_gpu_active() && self.config.noise_acceleration
    }

    /// 光照加速是否启用。
    #[must_use]
    pub const fn light_enabled(&self) -> bool {
        self.is_gpu_active() && self.config.light_acceleration
    }

    /// 地表加速是否启用。
    #[must_use]
    pub const fn surface_enabled(&self) -> bool {
        self.is_gpu_active() && self.config.surface_acceleration
    }

    /// JIT 编译是否启用。
    #[must_use]
    pub const fn jit_enabled(&self) -> bool {
        self.is_gpu_active() && self.config.jit_enabled
    }
}

impl Default for GpuCompute {
    fn default() -> Self {
        Self::default_disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        let gpu = GpuCompute::default();
        assert!(!gpu.is_gpu_active());
        assert!(!gpu.noise_enabled());
        assert!(!gpu.light_enabled());
        assert!(!gpu.surface_enabled());
    }

    #[test]
    fn with_disabled_config() {
        let config = GpuConfig {
            enabled: false,
            noise_acceleration: true,
            ..Default::default()
        };
        let gpu = GpuCompute::new(config);
        assert!(!gpu.is_gpu_active());
        // 即使噪声开关打开，全局禁用时也不启用
        assert!(!gpu.noise_enabled());
    }

    #[test]
    fn gpu_compute_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<GpuCompute>();
    }
}
