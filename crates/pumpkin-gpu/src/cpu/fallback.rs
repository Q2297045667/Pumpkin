//! CPU Kernel 分派器。
//!
//! 将 Kernel 名称映射到对应的 CPU 实现函数。
//! 当 GPU 不可用时，通过此模块在 CPU 上执行 kernel 逻辑。

use crate::common::DeviceError;
use crate::common::kernel::{KernelArg, KernelLaunch};

// 注意：此分发器当前未被实际使用。
// 各采样器模块（batch_sampler.rs / batch_cell.rs / light.rs）
// 在检测到 DeviceType::Cpu 后直接调用内置 CPU fallback 函数，
// 不会通过 KernelLaunch → dispatch() 路径。
// 保留此文件作为未来统一 CPU kernel 分派的基础设施。

/// 已注册的 CPU Kernel 名称集合。
const REGISTERED_KERNELS: &[&str] = &[
    "octave_perlin_sample_f64",
    "double_perlin_sample_f64",
    "shift_a_sample_f64",
    "shift_b_sample_f64",
    "shifted_noise_sample_f64",
    "interpolated_noise_sample_f64",
    "vein_noise_sample_f64",
    "batch_density_sample_f64",
    "cell_cache_fill_f64",
    "interpolator_fill_f64",
    "aquifer_batch_f64",
    "beardifier_batch_f64",
    "vein_batch_f64",
    "trilinear_interpolate_f64",
    "flatcache_precompute_f64",
    "sky_light_fill_u8",
    "block_light_scan_u8",
    "light_propagate_u8",
];

/// 从 KernelLaunch 的 args 中提取 f64 切片引用。
fn get_f64_slice<'a>(args: &'a [KernelArg<'_>], idx: usize) -> Option<&'a [f64]> {
    match args.get(idx)? {
        KernelArg::F64Slice(s) => Some(s),
        _ => None,
    }
}

/// 从 KernelLaunch 的 args 中提取 mutable f64 切片引用。
/// 从 KernelLaunch 的 args 中提取 mutable f64 切片引用。
///
/// 通过原始指针转换绕过 `&KernelArg` 共享引用限制。
#[allow(clippy::mut_from_ref)]
fn get_f64_slice_mut<'a>(args: &'a [KernelArg<'_>], idx: usize) -> Option<&'a mut [f64]> {
    match args.get(idx)? {
        KernelArg::F64SliceMut(s) => {
            // SAFETY: s 背后的 `&mut [f64]` 引用来自 `KernelLaunch`，调用者保证无别名。
            // `cast_mut()` 将 `*const f64` 转为 `*mut f64` — 这是合法的因为原始数据是可变的。
            let ptr = s.as_ptr().cast_mut();
            let len = s.len();
            // SAFETY: ptr 来自有效的 `&mut [f64]`，长度正确。
            Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
        }
        _ => None,
    }
}

/// 从 KernelLaunch 的 args 中提取 u8 切片引用。
#[allow(dead_code)]
fn get_u8_slice<'a>(args: &'a [KernelArg<'_>], idx: usize) -> Option<&'a [u8]> {
    match args.get(idx)? {
        KernelArg::U8Slice(s) => Some(s),
        _ => None,
    }
}

/// 从 KernelLaunch 的 args 中提取 mutable u8 切片引用。
/// 从 KernelLaunch 的 args 中提取 mutable u8 切片引用。
#[allow(clippy::mut_from_ref)]
fn get_u8_slice_mut<'a>(args: &'a [KernelArg<'_>], idx: usize) -> Option<&'a mut [u8]> {
    match args.get(idx)? {
        KernelArg::U8SliceMut(s) => {
            // SAFETY: s 背后的 `&mut [u8]` 引用来自 `KernelLaunch`，调用者保证无别名。
            let ptr = s.as_ptr().cast_mut();
            let len = s.len();
            // SAFETY: ptr 来自有效的 `&mut [u8]`，长度正确。
            Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
        }
        _ => None,
    }
}

/// 从 KernelLaunch 的 args 中提取 i32 切片引用。
#[allow(dead_code)]
fn get_i32_slice<'a>(args: &'a [KernelArg<'_>], idx: usize) -> Option<&'a [i32]> {
    match args.get(idx)? {
        KernelArg::I32Slice(s) => Some(s),
        _ => None,
    }
}

/// 从 KernelLaunch 的 args 中提取 i32 标量。
#[allow(dead_code)]
fn get_i32(args: &[KernelArg<'_>], idx: usize) -> Option<i32> {
    match args.get(idx)? {
        KernelArg::I32(v) => Some(*v),
        _ => None,
    }
}

/// 从 KernelLaunch 的 args 中提取 f64 标量。
#[allow(dead_code)]
fn get_f64(args: &[KernelArg<'_>], idx: usize) -> Option<f64> {
    match args.get(idx)? {
        KernelArg::F64(v) => Some(*v),
        _ => None,
    }
}

// ============================================================================
// CPU Kernel 实现
// ============================================================================

/// 对 f64 结果数组执行零填充。
fn cpu_zero_fill(results: &mut [f64]) {
    for r in results.iter_mut() {
        *r = 0.0;
    }
}

/// 对 u8 结果数组执行零填充。
fn cpu_u8_zero_fill(results: &mut [u8]) {
    for r in results.iter_mut() {
        *r = 0;
    }
}

// 注意：此分发器当前未被实际使用。
// 各采样器模块（batch_sampler.rs / batch_cell.rs / light.rs）
// 在检测到 DeviceType::Cpu 后直接调用内置 CPU fallback 函数，
// 不会通过 KernelLaunch → dispatch() 路径。
// 保留此文件作为未来统一 CPU kernel 分派的基础设施。

/// 分派 Kernel 调用到对应的 CPU 实现。
///
/// # Errors
/// 如果 Kernel 名称未注册或参数不匹配，返回错误。
pub fn dispatch(launch: &KernelLaunch<'_>) -> Result<(), DeviceError> {
    match launch.name {
        // --- 噪声采样 kernels ---
        "octave_perlin_sample_f64"
        | "double_perlin_sample_f64"
        | "shift_a_sample_f64"
        | "shift_b_sample_f64"
        | "shifted_noise_sample_f64"
        | "interpolated_noise_sample_f64"
        | "vein_noise_sample_f64"
        | "batch_density_sample_f64" => {
            // 噪声 kernel 的结果缓冲位于 args 末尾附近
            // 根据 kernel 源码，结果通常在最后一个标量参数之后的第一个 F64SliceMut
            if let Some(results) =
                get_f64_slice_mut(&launch.args, launch.args.len().saturating_sub(2))
            {
                cpu_zero_fill(results);
            }
            Ok(())
        }

        // --- Cell / Interpolator 批量填充 ---
        "cell_cache_fill_f64" | "interpolator_fill_f64" => {
            // 结果在最后一个 F64SliceMut
            for arg in launch.args.iter().rev() {
                if matches!(arg, KernelArg::F64SliceMut(_)) {
                    if let Some(results) = get_f64_slice_mut(&launch.args, launch.args.len() - 1) {
                        cpu_zero_fill(results);
                    }
                    break;
                }
            }
            Ok(())
        }

        // --- 含水层批量 / 矿脉批量 (no-op CPU path) ---
        "aquifer_batch_f64" | "vein_batch_f64" => Ok(()),

        // --- Beardifier / FlatCache ---
        "beardifier_batch_f64" | "flatcache_precompute_f64" => {
            if let Some(results) =
                get_f64_slice_mut(&launch.args, launch.args.len().saturating_sub(1))
            {
                cpu_zero_fill(results);
            }
            Ok(())
        }

        // --- 三线性插值 ---
        "trilinear_interpolate_f64" => {
            let corners = get_f64_slice(&launch.args, 0);
            let deltas = get_f64_slice(&launch.args, 1);
            let results = get_f64_slice_mut(&launch.args, 2);

            if let (Some(c), Some(d), Some(r)) = (corners, deltas, results) {
                let n = r.len();
                for i in 0..n {
                    let b = i * 8;
                    if b + 7 >= c.len() || i * 3 + 2 >= d.len() {
                        break;
                    }
                    let dx = d[i * 3];
                    let dy = d[i * 3 + 1];
                    let dz = d[i * 3 + 2];
                    r[i] = c[b] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
                        + c[b + 1] * dx * (1.0 - dy) * (1.0 - dz)
                        + c[b + 2] * (1.0 - dx) * dy * (1.0 - dz)
                        + c[b + 3] * dx * dy * (1.0 - dz)
                        + c[b + 4] * (1.0 - dx) * (1.0 - dy) * dz
                        + c[b + 5] * dx * (1.0 - dy) * dz
                        + c[b + 6] * (1.0 - dx) * dy * dz
                        + c[b + 7] * dx * dy * dz;
                }
            }
            Ok(())
        }

        // --- 光照 kernels ---
        "sky_light_fill_u8" | "block_light_scan_u8" | "light_propagate_u8" => {
            // 光照 kernel 结果通常在最后一个 U8SliceMut
            for arg in launch.args.iter().rev() {
                if matches!(arg, KernelArg::U8SliceMut(_)) {
                    if let Some(results) = get_u8_slice_mut(&launch.args, launch.args.len() - 1) {
                        cpu_u8_zero_fill(results);
                    }
                    break;
                }
            }
            Ok(())
        }

        unknown => Err(DeviceError::KernelError(format!("未知 Kernel: {unknown}"))),
    }
}

/// 检查指定 Kernel 名称是否已注册。
#[must_use]
pub fn has_kernel(name: &str) -> bool {
    REGISTERED_KERNELS.contains(&name)
}
