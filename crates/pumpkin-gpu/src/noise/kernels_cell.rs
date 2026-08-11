//! GPU Kernel 实现 — Cell Cache、插值器、含水层、Beardifier、矿脉。
//!
//! 所有 Kernel 保证与 pumpkin-world CPU 路径逐位一致。
#![allow(clippy::needless_raw_string_hashes, clippy::too_many_lines)]

/// 批量 Cell Cache 填充：对每个 3D 位置计算密度，
/// 使用扁平化的 `component_stack` 参数和 `cell_indices` 映射。
///
/// 每个 work-item 处理一个位置，
/// 通过类似 `fill_from_stack` 的简化路径计算密度。
pub const CELL_CACHE_FILL_CL: &str = include_str!("../../kernels/opencl/cell_cache.cl");

/// 批量插值器缓冲区填充：对 YZ 切片位置数组计算密度。
///
/// DAG 参数驱动插值器噪声配置；
/// 每个 work-item 处理一个 YZ 切片位置。
pub const INTERPOLATOR_FILL_CL: &str = include_str!("../../kernels/opencl/interpolator_fill.cl");

/// 批量含水层判定：4-NN 搜索 + 屏障密度 + 流体判定。
///
/// 每个 work-item 处理一个块，
/// 返回 `block_state_id` (i32) 和 `should_schedule_fluid_update` (u8)。
pub const AQUIFER_BATCH_CL: &str = include_str!("../../kernels/opencl/aquifer_batch.cl");

/// 批量 Beardifier：对每个位置遍历结构和连接点，
/// 使用预计算的 24³ 核表累加 beard 贡献。
pub const BEARDIFIER_BATCH_CL: &str = include_str!("../../kernels/opencl/beardifier_batch.cl");

/// 批量矿脉判定（独立于含水层）：
/// 对每个位置计算矿脉类型。
///
/// 返回值：0 = 无矿脉，1 = 矿石，2 = 粗矿，3 = 围岩。
pub const VEIN_BATCH_CL: &str = include_str!("../../kernels/opencl/vein_batch.cl");
