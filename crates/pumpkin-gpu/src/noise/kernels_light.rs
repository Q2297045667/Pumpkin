//! Lighting GPU Kernel 实现。
#![allow(clippy::needless_raw_string_hashes)]

/// 天空光垂直填充：对每个 (x,z) 列计算 light = 15 - cumulative_opacity。
pub const SKY_LIGHT_FILL_CL: &str = include_str!("../../kernels/opencl/light_sky.cl");

/// 方块光扫描：找出所有发光方块并设置初始光等级。
pub const BLOCK_LIGHT_SCAN_CL: &str = include_str!("../../kernels/opencl/light_block.cl");

/// 光照传播单步迭代（迭代式距离场替代 BFS）。
pub const LIGHT_PROPAGATE_CL: &str = include_str!("../../kernels/opencl/light_propagate.cl");

pub const SKY_LIGHT_FILL_CU: &str = include_str!("../../kernels/cuda/light_sky.cu");
pub const BLOCK_LIGHT_SCAN_CU: &str = include_str!("../../kernels/cuda/light_block.cu");
pub const LIGHT_PROPAGATE_CU: &str = include_str!("../../kernels/cuda/light_propagate.cu");

/// 光照传播 Persistent Kernel（CUDA 专用，内部迭代直至收敛）。
pub const LIGHT_PROPAGATE_PERSISTENT_CU: &str =
    include_str!("../../kernels/cuda/light_propagate_persistent.cu");
