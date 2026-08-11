//! Lighting GPU Kernel 实现。
#![allow(clippy::needless_raw_string_hashes)]

/// 天空光垂直填充：对每个 (x,z) 列计算 light = 15 - cumulative_opacity。
pub const SKY_LIGHT_FILL_CL: &str = include_str!("../../kernels/opencl/light_sky.cl");

/// 方块光扫描：找出所有发光方块并设置初始光等级。
pub const BLOCK_LIGHT_SCAN_CL: &str = include_str!("../../kernels/opencl/light_block.cl");

/// 光照传播单步迭代（迭代式距离场替代 BFS）。
pub const LIGHT_PROPAGATE_CL: &str = include_str!("../../kernels/opencl/light_propagate.cl");
