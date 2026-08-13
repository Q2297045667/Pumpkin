//! GPU Kernel 实现 — 含水层、Beardifier。
//!
//! 所有 Kernel 保证与 pumpkin-world CPU 路径逐位一致。
#![allow(clippy::needless_raw_string_hashes, clippy::too_many_lines)]

/// 批量含水层判定：4-NN 搜索 + 屏障密度 + 流体判定。
///
/// 每个 work-item 处理一个块，
/// 返回 `block_state_id` (i32) 和 `should_schedule_fluid_update` (u8)。
pub const AQUIFER_BATCH_CL: &str = include_str!("../../kernels/opencl/aquifer_batch.cl");

/// 批量含水层判定 tiled 变体：使用 local memory 协作加载 packed 数据。
///
/// 当 M <= 2048 时使用此 kernel 以利用 local memory 带宽优势。
pub const AQUIFER_BATCH_TILED_CL: &str =
    include_str!("../../kernels/opencl/aquifer_batch_tiled.cl");

/// 批量 Beardifier：对每个位置遍历结构和连接点，
/// 使用预计算的 24³ 核表累加 beard 贡献。
pub const BEARDIFIER_BATCH_CL: &str = include_str!("../../kernels/opencl/beardifier_batch.cl");

pub const AQUIFER_BATCH_CU: &str = include_str!("../../kernels/cuda/aquifer_batch.cu");
pub const AQUIFER_BATCH_TILED_CU: &str = include_str!("../../kernels/cuda/aquifer_batch_tiled.cu");
pub const BEARDIFIER_BATCH_CU: &str = include_str!("../../kernels/cuda/beardifier_batch.cu");
