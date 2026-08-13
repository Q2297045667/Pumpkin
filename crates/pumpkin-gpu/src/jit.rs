//! JIT (Just-In-Time) Kernel 参数特化。
//!
//! 在运行时根据采样器参数生成特化版本的 kernel 源码，
//! 将八度数、振幅表等常量烘焙进源码中，使 GPU 编译器能够：
//! - 完全展开循环
//! - 消除间接数组访问
//! - 将小数组放入常量内存
//!
//! 每个特化函数同时生成 **OpenCL C** 与 **CUDA C++** 两种方言的源码，
//! 分别供 OpenCL 后端与 CUDA 后端（NVRTC）的 JIT 编译路径使用。
//! 两种方言的 kernel 语义必须与 CPU 路径逐位一致。

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
    /// 特化后的 kernel 名称（如 "octave_perlin_sample_f64_jit_m4"）
    pub name: String,
    /// 特化后的 OpenCL C 源码（供 OpenCL 后端）
    pub source: String,
    /// 特化后的 CUDA C++ 源码（供 CUDA 后端 NVRTC）
    pub cuda_source: String,
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
    let pers = config.packed_persistences();
    let lacs = config.packed_lacunarities();
    let orgs = config.packed_origins();

    // 生成特化 kernel 名称。必须包含配置内容指纹：
    // 八度数相同的不同采样器（不同种子/原点/置换表）烘焙的常量不同，
    // 若共用 kernel 名会导致 JIT kernel 被错误复用、输出错误数值。
    let name = format!(
        "octave_perlin_sample_f64_jit_m{m}_h{:016x}",
        config.fingerprint()
    );

    // 两种方言共用的 kernel 体（除参数表与线程索引行外完全一致）
    let mut body = String::new();
    let _ = writeln!(
        body,
        "    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];"
    );
    let _ = writeln!(body, "    double sum = 0.0;");
    for o in 0..m {
        let _ = writeln!(
            body,
            "    sum += ({amp} * sample_no_fade_core(perms + {o}*256,",
            amp = amps[o],
            o = o
        );
        let _ = writeln!(
            body,
            "        {ox}, {oy}, {oz},",
            ox = orgs[o * 3],
            oy = orgs[o * 3 + 1],
            oz = orgs[o * 3 + 2]
        );
        let _ = writeln!(
            body,
            "        maintain_precision(x*{lac}), maintain_precision(y*{lac}), maintain_precision(z*{lac}))) * {pers};",
            lac = lacs[o],
            pers = pers[o]
        );
    }
    let _ = writeln!(body, "    res[i] = sum;");

    let source = format!(
        "// JIT specialized: octave_perlin_sample_f64\n\
         // num_octaves = {m}\n\
         __kernel void {name}(\n\
         \x20   __global const double* pos,\n\
         \x20   __global const uchar* perms,  // {m}*256 bytes\n\
         \x20   __global double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = get_global_id(0); if (i >= N) return;\n\
         {body}}}"
    );
    let cuda_source = format!(
        "// JIT specialized (CUDA): octave_perlin_sample_f64\n\
         // num_octaves = {m}\n\
         extern \"C\" __global__ void {name}(\n\
         \x20   const double* pos,\n\
         \x20   const unsigned char* perms,  // {m}*256 bytes\n\
         \x20   double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;\n\
         {body}}}"
    );

    Some(JitSpecializedKernel {
        name,
        source,
        cuda_source,
    })
}

/// 检查 JIT 是否应该被使用。
#[must_use]
pub fn should_jit_specialize(num_octaves: usize, max_unroll: usize) -> bool {
    num_octaves > 0 && num_octaves <= max_unroll
}

/// 生成双 Perlin 噪声的 JIT 特化 kernel。
///
/// 将两组八度参数（amp1/lac1/org1 + amp2/lac2/org2）全部硬编码为 GPU 常量，
/// 展开两个循环。预期加速 1.5×–2.5×（当前 kernel 参数最多，内存读取最频繁）。
#[must_use]
pub fn specialize_double_perlin(
    config1: &SerializedOctaveConfig,
    config2: &SerializedOctaveConfig,
    amplitude: f64,
    max_unroll: usize,
) -> Option<JitSpecializedKernel> {
    let m1 = config1.num_octaves();
    let m2 = config2.num_octaves();
    if m1 > max_unroll || m2 > max_unroll {
        return None;
    }

    let amps1 = config1.packed_amplitudes();
    let pers1 = config1.packed_persistences();
    let lacs1 = config1.packed_lacunarities();
    let orgs1 = config1.packed_origins();
    let amps2 = config2.packed_amplitudes();
    let pers2 = config2.packed_persistences();
    let lacs2 = config2.packed_lacunarities();
    let orgs2 = config2.packed_origins();

    let name = format!(
        "double_perlin_sample_f64_jit_m{m1}_{m2}_h{:016x}",
        config1.fingerprint().wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ config2.fingerprint()
            ^ amplitude.to_bits()
    );
    let c = 1.0181268882175227f64;

    let mut body = String::new();
    let _ = writeln!(
        body,
        "    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];"
    );

    // 第一组八度
    let _ = writeln!(body, "    double sum1 = 0.0;");
    for o in 0..m1 {
        let _ = writeln!(
            body,
            "    sum1 += ({amp} * sample_no_fade_core(perms1 + {o}*256,",
            amp = amps1[o],
            o = o
        );
        let _ = writeln!(
            body,
            "        {ox}, {oy}, {oz},",
            ox = orgs1[o * 3],
            oy = orgs1[o * 3 + 1],
            oz = orgs1[o * 3 + 2]
        );
        let _ = writeln!(
            body,
            "        maintain_precision(x*{lac}), maintain_precision(y*{lac}), maintain_precision(z*{lac}))) * {pers};",
            lac = lacs1[o],
            pers = pers1[o]
        );
    }

    // 第二组八度 (带常量 c 缩放)。
    // 注意：与 CPU `sample(x*c, ...)` 及 batch kernel 一致，
    // `maintain_precision` 只在 `(x*c)*lac` 上应用一次——
    // 不得对 `x*c` 提前应用（双重应用会破坏逐位一致）。
    let _ = writeln!(body, "    double x2 = x*{c}, y2 = y*{c}, z2 = z*{c};");
    let _ = writeln!(body, "    double sum2 = 0.0;");
    for o in 0..m2 {
        let _ = writeln!(
            body,
            "    sum2 += ({amp} * sample_no_fade_core(perms2 + {o}*256,",
            amp = amps2[o],
            o = o
        );
        let _ = writeln!(
            body,
            "        {ox}, {oy}, {oz},",
            ox = orgs2[o * 3],
            oy = orgs2[o * 3 + 1],
            oz = orgs2[o * 3 + 2]
        );
        let _ = writeln!(
            body,
            "        maintain_precision(x2*{lac}), maintain_precision(y2*{lac}), maintain_precision(z2*{lac}))) * {pers};",
            lac = lacs2[o],
            pers = pers2[o]
        );
    }

    let _ = writeln!(body, "    res[i] = (sum1 + sum2) * {amplitude};");

    let source = format!(
        "// JIT specialized: double_perlin_sample_f64\n\
         // num_octaves = {m1}, {m2}, amp = {amplitude}, c = {c}\n\
         __kernel void {name}(\n\
         \x20   __global const double* pos,\n\
         \x20   __global const uchar* perms1,  // {m1}*256 bytes\n\
         \x20   __global const uchar* perms2,  // {m2}*256 bytes\n\
         \x20   __global double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = get_global_id(0); if (i >= N) return;\n\
         {body}}}"
    );
    let cuda_source = format!(
        "// JIT specialized (CUDA): double_perlin_sample_f64\n\
         // num_octaves = {m1}, {m2}, amp = {amplitude}, c = {c}\n\
         extern \"C\" __global__ void {name}(\n\
         \x20   const double* pos,\n\
         \x20   const unsigned char* perms1,  // {m1}*256 bytes\n\
         \x20   const unsigned char* perms2,  // {m2}*256 bytes\n\
         \x20   double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;\n\
         {body}}}"
    );

    Some(JitSpecializedKernel {
        name,
        source,
        cuda_source,
    })
}

/// 生成 ShiftA/ShiftB 的 JIT 特化 kernel。
///
/// 输入为 2D (xz 或 zx)，八度参数硬编码为常量，y 固定为 0。
/// ShiftA: `sample(x * 0.25, 0.0, z * 0.25) * 4.0`
/// ShiftB: `sample(z * 0.25, 0.0, x * 0.25) * 4.0`
#[must_use]
pub fn specialize_shift(
    shift_type: &str,
    config: &SerializedOctaveConfig,
    max_unroll: usize,
) -> Option<JitSpecializedKernel> {
    let m = config.num_octaves();
    if m > max_unroll {
        return None;
    }

    let amps = config.packed_amplitudes();
    let pers = config.packed_persistences();
    let lacs = config.packed_lacunarities();
    let orgs = config.packed_origins();

    let name = format!(
        "{shift_type}_sample_f64_jit_m{m}_h{:016x}",
        config.fingerprint()
    );

    let mut body = String::new();
    if shift_type == "shift_a" {
        let _ = writeln!(
            body,
            "    double x = pos[i*2] * 0.25, z = pos[i*2+1] * 0.25;"
        );
    } else {
        let _ = writeln!(
            body,
            "    double z = pos[i*2] * 0.25, x = pos[i*2+1] * 0.25;"
        );
    }
    let _ = writeln!(body, "    double sum = 0.0;");

    for o in 0..m {
        let _ = writeln!(
            body,
            "    sum += ({amp} * sample_no_fade_core(perms + {o}*256,",
            amp = amps[o],
            o = o
        );
        let _ = writeln!(
            body,
            "        {ox}, {oy}, {oz},",
            ox = orgs[o * 3],
            oy = orgs[o * 3 + 1],
            oz = orgs[o * 3 + 2]
        );
        let _ = writeln!(
            body,
            "        maintain_precision(x*{lac}), 0.0, maintain_precision(z*{lac}))) * {pers};",
            lac = lacs[o],
            pers = pers[o]
        );
    }

    let _ = writeln!(body, "    res[i] = sum * 4.0;");

    let source = format!(
        "// JIT specialized: {shift_type}_sample_f64\n\
         // num_octaves = {m}\n\
         __kernel void {name}(\n\
         \x20   __global const double* pos,\n\
         \x20   __global const uchar* perms,\n\
         \x20   __global double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = get_global_id(0); if (i >= N) return;\n\
         {body}}}"
    );
    let cuda_source = format!(
        "// JIT specialized (CUDA): {shift_type}_sample_f64\n\
         // num_octaves = {m}\n\
         extern \"C\" __global__ void {name}(\n\
         \x20   const double* pos,\n\
         \x20   const unsigned char* perms,\n\
         \x20   double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;\n\
         {body}}}"
    );

    Some(JitSpecializedKernel {
        name,
        source,
        cuda_source,
    })
}

/// 生成 FlatCache 的 JIT 特化 kernel。
///
/// 输入为 2D (xz)，八度参数硬编码为常量，y 固定为 0。
/// 语义与 `flatcache_precompute_f64` 及 CPU `sample(x, 0.0, z)` 逐位一致。
#[must_use]
pub fn specialize_flatcache(
    config: &SerializedOctaveConfig,
    max_unroll: usize,
) -> Option<JitSpecializedKernel> {
    let m = config.num_octaves();
    if m > max_unroll {
        return None;
    }

    let amps = config.packed_amplitudes();
    let pers = config.packed_persistences();
    let lacs = config.packed_lacunarities();
    let orgs = config.packed_origins();

    let name = format!(
        "flatcache_precompute_f64_jit_m{m}_h{:016x}",
        config.fingerprint()
    );

    let mut body = String::new();
    let _ = writeln!(body, "    double x = pos[i*2], z = pos[i*2+1];");
    let _ = writeln!(body, "    double sum = 0.0;");

    for o in 0..m {
        let _ = writeln!(
            body,
            "    sum += ({amp} * sample_no_fade_core(perms + {o}*256,",
            amp = amps[o],
            o = o
        );
        let _ = writeln!(
            body,
            "        {ox}, {oy}, {oz},",
            ox = orgs[o * 3],
            oy = orgs[o * 3 + 1],
            oz = orgs[o * 3 + 2]
        );
        let _ = writeln!(
            body,
            "        maintain_precision(x*{lac}), 0.0, maintain_precision(z*{lac}))) * {pers};",
            lac = lacs[o],
            pers = pers[o]
        );
    }

    let _ = writeln!(body, "    res[i] = sum;");

    let source = format!(
        "// JIT specialized: flatcache_precompute_f64\n\
         // num_octaves = {m}\n\
         __kernel void {name}(\n\
         \x20   __global const double* pos,\n\
         \x20   __global const uchar* perms,\n\
         \x20   __global double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = get_global_id(0); if (i >= N) return;\n\
         {body}}}"
    );
    let cuda_source = format!(
        "// JIT specialized (CUDA): flatcache_precompute_f64\n\
         // num_octaves = {m}\n\
         extern \"C\" __global__ void {name}(\n\
         \x20   const double* pos,\n\
         \x20   const unsigned char* perms,\n\
         \x20   double* res,\n\
         \x20   int N\n\
         ) {{\n\
         \x20   int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;\n\
         {body}}}"
    );

    Some(JitSpecializedKernel {
        name,
        source,
        cuda_source,
    })
}
