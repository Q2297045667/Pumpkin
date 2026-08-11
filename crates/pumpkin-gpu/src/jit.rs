//! JIT (Just-In-Time) Kernel 参数特化。
//!
//! 在运行时根据采样器参数生成特化版本的 kernel 源码，
//! 将八度数、振幅表等常量烘焙进源码中，使 GPU 编译器能够：
//! - 完全展开循环
//! - 消除间接数组访问
//! - 将小数组放入常量内存

use crate::noise::cache::SerializedOctaveConfig;
use std::fmt::Write;
use std::sync::OnceLock;

/// JIT 循环展开上限的全局配置。
/// 由 [`crate::GpuDevice::from_config`] 在初始化时设置。
static JIT_MAX_UNROLL: OnceLock<usize> = OnceLock::new();

/// 设置 JIT 循环展开上限（从配置读取）。
pub(crate) fn set_jit_max_unroll(max: usize) {
    let _ = JIT_MAX_UNROLL.set(max);
}

/// 获取 JIT 循环展开上限。
/// 如果尚未设置，返回默认值 `16`。
#[must_use]
pub(crate) fn get_jit_max_unroll() -> usize {
    JIT_MAX_UNROLL.get().copied().unwrap_or(16)
}

/// JIT 特化后的 kernel 元数据。
pub struct JitSpecializedKernel {
    /// 特化后的 kernel 名称（如 "octave_perlin_sample_f64_jit_M4"）
    pub name: String,
    /// 特化后的完整 OpenCL C 源码
    pub source: String,
}

/// 生成八度 Perlin 噪声的 JIT 特化 kernel。
///
/// 当八度数 ≤ `max_unroll` 时收益显著（消除循环 + 间接访存）。
/// 当八度数 > `max_unroll` 时返回 `None`（展开的指令缓存压力可能抵消收益）。
#[must_use]
pub fn specialize_octave_perlin(
    config: &SerializedOctaveConfig,
    max_unroll: usize,
) -> Option<JitSpecializedKernel> {
    let m = config.num_octaves();
    if m > max_unroll {
        return None;
    }

    let amps = config.packed_amplitudes();
    let lacs = config.packed_lacunarities();
    let orgs = config.packed_origins();

    // 生成特化 kernel 名称
    let name = format!("octave_perlin_sample_f64_jit_m{m}");

    // 生成特化源码
    let mut src = String::new();

    let _ = writeln!(src, "// JIT specialized: octave_perlin_sample_f64");
    let _ = writeln!(src, "// num_octaves = {m}");
    let _ = writeln!(src, "#pragma OPENCL EXTENSION cl_khr_f64 : enable");
    let _ = writeln!(src);

    // 包含辅助函数（sample_no_fade_core 等）— 由 compile.rs 的 compile_one 自动添加

    let _ = writeln!(src, "__kernel void {name}(");
    let _ = writeln!(src, "    __global const double* pos,");
    let _ = writeln!(src, "    __global const uchar* perms,  // {m}*256 bytes");
    let _ = writeln!(src, "    __global double* res,");
    let _ = writeln!(src, "    int N");
    let _ = writeln!(src, ") {{");
    let _ = writeln!(src, "    int i = get_global_id(0); if (i >= N) return;");
    let _ = writeln!(
        src,
        "    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];"
    );
    let _ = writeln!(src, "    double sum = 0.0;");

    // 内联所有八度的计算
    for o in 0..m {
        let _ = writeln!(
            src,
            "    sum += {amp} * sample_no_fade_core(perms + {o}*256,",
            amp = amps[o],
            o = o
        );
        let _ = writeln!(
            src,
            "        {ox}, {oy}, {oz},",
            ox = orgs[o * 3],
            oy = orgs[o * 3 + 1],
            oz = orgs[o * 3 + 2]
        );
        let _ = writeln!(
            src,
            "        maintain_precision(x*{lac}), maintain_precision(y*{lac}), maintain_precision(z*{lac}));",
            lac = lacs[o]
        );
    }

    let _ = writeln!(src, "    res[i] = sum;");
    let _ = writeln!(src, "}}");

    Some(JitSpecializedKernel { name, source: src })
}

/// 检查 JIT 是否应该被使用。
#[must_use]
pub fn should_jit_specialize(num_octaves: usize, max_unroll: usize) -> bool {
    num_octaves > 0 && num_octaves <= max_unroll
}
