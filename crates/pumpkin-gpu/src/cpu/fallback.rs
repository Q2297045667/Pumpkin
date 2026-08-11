//! CPU Kernel 分派器。
//!
//! 将 Kernel 名称映射到对应的 CPU 实现函数。
//! 目前仅包含基础算子的存根实现，完整实现将在后续迭代中补充。

use crate::common::DeviceError;
use crate::common::kernel::KernelLaunch;

/// 已注册的 CPU Kernel 名称集合。
const REGISTERED_KERNELS: &[&str] = &[
    "perlin_sample_f64",
    "trilinear_interpolate_f64",
    "light_propagate_u8",
];

/// 分派 Kernel 调用到对应的 CPU 实现。
///
/// # Errors
/// 如果 Kernel 名称未注册或参数不匹配，返回错误。
pub fn dispatch(launch: &KernelLaunch<'_>) -> Result<(), DeviceError> {
    match launch.name {
        "perlin_sample_f64" => {
            // TODO: 实现 Perlin 噪声批采样
            Err(DeviceError::Unsupported(
                "perlin_sample_f64 CPU 实现尚未完成".into(),
            ))
        }
        "trilinear_interpolate_f64" => {
            // TODO: 实现三线性插值批处理
            Err(DeviceError::Unsupported(
                "trilinear_interpolate_f64 CPU 实现尚未完成".into(),
            ))
        }
        "light_propagate_u8" => {
            // TODO: 实现光照传播
            Err(DeviceError::Unsupported(
                "light_propagate_u8 CPU 实现尚未完成".into(),
            ))
        }
        unknown => Err(DeviceError::KernelError(format!("未知 Kernel: {unknown}"))),
    }
}

/// 检查指定 Kernel 名称是否已注册。
#[must_use]
pub fn has_kernel(name: &str) -> bool {
    REGISTERED_KERNELS.contains(&name)
}
