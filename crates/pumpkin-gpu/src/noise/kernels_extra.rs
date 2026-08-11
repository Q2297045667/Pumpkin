//! 补充 GPU Kernel — 三线性插值、FlatCache 预计算。
#![allow(clippy::needless_raw_string_hashes)]

/// 三线性插值批量处理。
pub const TRILINEAR_INTERPOLATE_CL: &str = include_str!("../../kernels/opencl/trilinear.cl");

/// FlatCache 预计算：对 2D 网格批量采样噪声。
pub const FLATCACHE_PRECOMPUTE_CL: &str = include_str!("../../kernels/opencl/flatcache.cl");

pub const TRILINEAR_INTERPOLATE_CU: &str = include_str!("../../kernels/cuda/trilinear.cu");
pub const FLATCACHE_PRECOMPUTE_CU: &str = include_str!("../../kernels/cuda/flatcache.cu");
