//! GPU Kernel 实现 — 完整噪声类型。
//!
//! 所有 Kernel 保证与 pumpkin-world CPU 路径逐位一致。
#![allow(clippy::needless_raw_string_hashes, clippy::too_many_lines)]

/// 基础 Perlin 噪声采样器 (sample_no_fade)。
/// 由 octave_perlin 和 double_perlin 共享使用。
pub const PERLIN_CORE_CL: &str = include_str!("../../kernels/opencl/perlin_core.cl");

/// 八度 Perlin 噪声批量采样。
pub const OCTAVE_PERLIN_SAMPLE_CL: &str = include_str!("../../kernels/opencl/noise_octave.cl");

/// 八度 Perlin 噪声批量采样 — SoA 变体（独立 X/Y/Z 数组）。
pub const OCTAVE_PERLIN_SAMPLE_SOA_CL: &str =
    include_str!("../../kernels/opencl/noise_octave_soa.cl");

/// 双 Perlin 噪声批量采样。
pub const DOUBLE_PERLIN_SAMPLE_CL: &str = include_str!("../../kernels/opencl/noise_double.cl");

/// 偏移噪声批量采样 (ShiftA / ShiftB)。
pub const SHIFT_A_SAMPLE_CL: &str = include_str!("../../kernels/opencl/noise_shift_a.cl");

/// ShiftB 专用。
pub const SHIFT_B_SAMPLE_CL: &str = include_str!("../../kernels/opencl/noise_shift_b.cl");

/// CUDA 版本 — 基础 Perlin 噪声采样器
pub const PERLIN_CORE_CU: &str = include_str!("../../kernels/cuda/perlin_core.cu");
/// CUDA 版本 — 八度 Perlin 噪声
pub const OCTAVE_PERLIN_SAMPLE_CU: &str = include_str!("../../kernels/cuda/noise_octave.cu");
/// CUDA 版本 — 八度 Perlin 噪声 SoA 变体
pub const OCTAVE_PERLIN_SAMPLE_SOA_CU: &str =
    include_str!("../../kernels/cuda/noise_octave_soa.cu");
/// CUDA 版本 — 双 Perlin 噪声
pub const DOUBLE_PERLIN_SAMPLE_CU: &str = include_str!("../../kernels/cuda/noise_double.cu");
pub const SHIFT_A_SAMPLE_CU: &str = include_str!("../../kernels/cuda/noise_shift_a.cu");
pub const SHIFT_B_SAMPLE_CU: &str = include_str!("../../kernels/cuda/noise_shift_b.cu");
