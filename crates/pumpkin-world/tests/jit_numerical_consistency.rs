//! JIT 数值正确性测试 — 验证 JIT kernel 输出与 CPU 基准一致。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::float_cmp
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::{GpuBackend, GpuConfig};
use pumpkin_gpu::GpuDevice;
use pumpkin_gpu::noise::GpuNoiseSampler;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

const SEED: u64 = 138_782_381_985_206;

fn fnv1a_f64(data: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn mk_sampler(seed: u64, octaves: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
    let mut g = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
}

fn mk_pos3d(n: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(n * 3);
    let mut s = SEED;
    for _ in 0..n {
        p.push((s.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
    }
    p
}

fn mk_pos2d(n: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(n * 2);
    let mut s = SEED;
    for _ in 0..n {
        p.push((s.wrapping_mul(6364136223846793005) as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
    }
    p
}

// ============================================================================
// JIT Octave Perlin 数值正确性
// ============================================================================

#[test]
fn jit_octave_vs_batch_3oct() {
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let n = 512;
    let pos = mk_pos3d(n);

    let mut batch_res = vec![0.0; n];
    let mut jit_res = vec![0.0; n];

    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));

    // Batch path
    gpu.sample_octave_batch(&sampler, &pos, &mut batch_res)
        .expect("batch should succeed");

    // JIT path
    gpu.sample_octave_jit(&sampler, &pos, &mut jit_res)
        .expect("JIT should succeed");

    assert_eq!(
        fnv1a_f64(&batch_res),
        fnv1a_f64(&jit_res),
        "JIT octave 3oct output must match batch output"
    );
}

#[test]
fn jit_octave_vs_batch_5oct() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3, 4]);
    let n = 256;
    let pos = mk_pos3d(n);

    let mut batch_res = vec![0.0; n];
    let mut jit_res = vec![0.0; n];

    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));

    gpu.sample_octave_batch(&sampler, &pos, &mut batch_res)
        .expect("batch should succeed");
    gpu.sample_octave_jit(&sampler, &pos, &mut jit_res)
        .expect("JIT should succeed");

    assert_eq!(
        fnv1a_f64(&batch_res),
        fnv1a_f64(&jit_res),
        "JIT octave 5oct output must match batch output"
    );
}

#[test]
fn jit_octave_vs_cpu_direct() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 512;
    let pos = mk_pos3d(n);

    let mut cpu_res = vec![0.0; n];
    let mut jit_res = vec![0.0; n];

    // CPU direct
    for i in 0..n {
        cpu_res[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }

    // GPU JIT
    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));
    gpu.sample_octave_jit(&sampler, &pos, &mut jit_res)
        .expect("JIT should succeed");

    assert_eq!(
        fnv1a_f64(&cpu_res),
        fnv1a_f64(&jit_res),
        "JIT octave output must match CPU direct output"
    );
}

// ============================================================================
// JIT DoublePerlin 数值正确性
// ============================================================================

#[test]
fn jit_double_perlin_vs_batch() {
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[1, 2, 3]);
    let amp = 0.5;
    let n = 256;
    let pos = mk_pos3d(n);

    let mut batch_res = vec![0.0; n];
    let mut jit_res = vec![0.0; n];

    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));

    gpu.sample_double_perlin_batch(&a, &b, amp, &pos, &mut batch_res)
        .expect("batch should succeed");
    gpu.sample_double_perlin_jit(&a, &b, amp, &pos, &mut jit_res)
        .expect("JIT should succeed");

    assert_eq!(
        fnv1a_f64(&batch_res),
        fnv1a_f64(&jit_res),
        "JIT double_perlin output must match batch output"
    );
}

// ============================================================================
// JIT ShiftA / ShiftB 数值正确性
// ============================================================================

#[test]
fn jit_shift_a_vs_batch() {
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let n = 256;
    let xz = mk_pos2d(n);

    let mut batch_res = vec![0.0; n];
    let mut jit_res = vec![0.0; n];

    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));

    gpu.sample_shift_a_batch(&sampler, &xz, &mut batch_res)
        .expect("batch should succeed");
    gpu.sample_shift_a_jit(&sampler, &xz, &mut jit_res)
        .expect("JIT should succeed");

    assert_eq!(
        fnv1a_f64(&batch_res),
        fnv1a_f64(&jit_res),
        "JIT shift_a output must match batch output"
    );
}

#[test]
fn jit_shift_b_vs_batch() {
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let n = 256;
    let zx = mk_pos2d(n);

    let mut batch_res = vec![0.0; n];
    let mut jit_res = vec![0.0; n];

    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));

    gpu.sample_shift_b_batch(&sampler, &zx, &mut batch_res)
        .expect("batch should succeed");
    gpu.sample_shift_b_jit(&sampler, &zx, &mut jit_res)
        .expect("JIT should succeed");

    assert_eq!(
        fnv1a_f64(&batch_res),
        fnv1a_f64(&jit_res),
        "JIT shift_b output must match batch output"
    );
}

// ============================================================================
// JIT FlatCache 数值正确性
// ============================================================================

#[test]
fn jit_flatcache_vs_batch() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 512;
    let xz = mk_pos2d(n);

    let mut batch_res = vec![0.0; n];
    let mut jit_res = vec![0.0; n];

    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));

    gpu.precompute_flatcache(&sampler, &xz, &mut batch_res)
        .expect("flatcache should succeed");

    // 直接调用 JIT 专用 kernel 生成路径：禁用 JIT 时 flatcache 走标准路径，
    // 这里通过 jit_enabled=false 的对比验证 JIT 入口一致（CPU 设备上两种路径同源）。
    let mut gpu_nojit = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: false,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));
    gpu_nojit
        .precompute_flatcache(&sampler, &xz, &mut jit_res)
        .expect("flatcache should succeed");

    assert_eq!(
        fnv1a_f64(&batch_res),
        fnv1a_f64(&jit_res),
        "flatcache output must be independent of JIT flag on CPU device"
    );

    // 与 CPU 直接采样逐位一致
    let mut cpu_res = vec![0.0; n];
    for i in 0..n {
        cpu_res[i] = sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
    }
    assert_eq!(
        fnv1a_f64(&batch_res),
        fnv1a_f64(&cpu_res),
        "flatcache output must match CPU direct sampling"
    );
}

// ============================================================================
// JIT 跳过大量八度 -> 回退 batch
// ============================================================================

#[test]
fn jit_skip_large_octaves_falls_back_to_batch() {
    let sampler = mk_sampler(
        SEED,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    );
    let n = 64;
    let pos = mk_pos3d(n);

    let mut res = vec![0.0; n];

    let mut gpu = GpuNoiseSampler::new(GpuDevice::from_config(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Cpu,
        ..Default::default()
    }));

    // 18 octaves > max_unroll(16) → JIT 应回退到 batch
    gpu.sample_octave_jit(&sampler, &pos, &mut res)
        .expect("JIT fallback to batch should succeed");

    assert!(
        res.iter().all(|&v| v.is_finite()),
        "all outputs must be finite"
    );
}

// ============================================================================
// 真实 GPU 后端（CUDA / OpenCL）JIT 全族一致性
// ============================================================================

/// 在真实 GPU 后端上验证五种 JIT 特化 kernel 与 CPU 参考逐位一致，
/// 并断言 JIT kernel 确实被编译（而非静默回退 batch）。
///
/// 无可用 GPU 时跳过（CPU 后端路径已由上方 Cpu 后端测试覆盖）。
#[test]
fn jit_gpu_backend_all_families_bitwise_cpu() {
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Auto,
        ..Default::default()
    };
    let device = GpuDevice::from_config(&config);
    if device.device_type() == pumpkin_gpu::DeviceType::Cpu {
        println!("SKIP: 无可用 GPU 设备，跳过真实 GPU JIT 一致性测试");
        return;
    }
    let mut gpu = GpuNoiseSampler::new(device);
    let n = 1024;

    // octave
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let pos3 = mk_pos3d(n);
    let mut cpu = vec![0.0; n];
    let mut out = vec![0.0; n];
    for i in 0..n {
        cpu[i] = sampler.sample(pos3[i * 3], pos3[i * 3 + 1], pos3[i * 3 + 2]);
    }
    gpu.sample_octave_jit(&sampler, &pos3, &mut out)
        .expect("octave JIT");
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&out), "octave JIT vs CPU");
    let oct_jit_name = pumpkin_gpu::jit::specialize_octave_perlin(
        &pumpkin_gpu::noise::cache::SerializedOctaveConfig::from_sampler(&sampler),
        16,
    )
    .expect("specialize")
    .name;
    assert!(
        gpu.device
            .kernel_launcher()
            .is_some_and(|l| l.has_kernel(&oct_jit_name)),
        "octave JIT kernel 必须真实编译"
    );

    // double perlin
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[-1, 0, 1]);
    let c = 1.0181268882175227f64;
    for i in 0..n {
        let x = pos3[i * 3];
        let y = pos3[i * 3 + 1];
        let z = pos3[i * 3 + 2];
        cpu[i] = (a.sample(x, y, z) + b.sample(x * c, y * c, z * c)) * 0.5;
    }
    gpu.sample_double_perlin_jit(&a, &b, 0.5, &pos3, &mut out)
        .expect("double perlin JIT");
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&out), "double perlin JIT vs CPU");
    let dp_jit_name = pumpkin_gpu::jit::specialize_double_perlin(
        &pumpkin_gpu::noise::cache::SerializedOctaveConfig::from_sampler(&a),
        &pumpkin_gpu::noise::cache::SerializedOctaveConfig::from_sampler(&b),
        0.5,
        16,
    )
    .expect("specialize")
    .name;
    assert!(
        gpu.device
            .kernel_launcher()
            .is_some_and(|l| l.has_kernel(&dp_jit_name)),
        "double perlin JIT kernel 必须真实编译"
    );

    // shift a / b（2D 输入）
    let pos2 = mk_pos2d(n);
    let cfg = pumpkin_gpu::noise::cache::SerializedOctaveConfig::from_sampler(&sampler);
    for (is_a, shift_type) in [(true, "shift_a"), (false, "shift_b")] {
        for i in 0..n {
            cpu[i] = if is_a {
                sampler.sample(pos2[i * 2] * 0.25, 0.0, pos2[i * 2 + 1] * 0.25) * 4.0
            } else {
                sampler.sample(pos2[i * 2 + 1] * 0.25, 0.0, pos2[i * 2] * 0.25) * 4.0
            };
        }
        if is_a {
            gpu.sample_shift_a_jit(&sampler, &pos2, &mut out)
                .expect("shift a JIT");
        } else {
            gpu.sample_shift_b_jit(&sampler, &pos2, &mut out)
                .expect("shift b JIT");
        }
        assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&out), "{shift_type} vs CPU");
        let shift_jit_name = pumpkin_gpu::jit::specialize_shift(shift_type, &cfg, 16)
            .expect("specialize")
            .name;
        assert!(
            gpu.device
                .kernel_launcher()
                .is_some_and(|l| l.has_kernel(&shift_jit_name)),
            "{shift_jit_name} 必须真实编译"
        );
    }

    // flatcache
    for i in 0..n {
        cpu[i] = sampler.sample(pos2[i * 2], 0.0, pos2[i * 2 + 1]);
    }
    gpu.precompute_flatcache(&sampler, &pos2, &mut out)
        .expect("flatcache JIT");
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&out), "flatcache vs CPU");
}

/// 回归：两个**八度数相同但内容不同**的采样器（不同种子）必须各自得到
/// 正确的 JIT 结果。
///
/// 历史 bug：JIT kernel 名只含八度数（如 `..._jit_m3`），常量（原点/振幅等）
/// 烘焙在源码中——第二个采样器会错误复用第一个采样器的 kernel，输出错误数值。
/// 修复：kernel 名追加配置内容指纹（`SerializedOctaveConfig::fingerprint`）。
#[test]
fn jit_same_octave_count_different_seeds_no_collision() {
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        backend: GpuBackend::Auto,
        ..Default::default()
    };
    let device = GpuDevice::from_config(&config);
    if device.device_type() == pumpkin_gpu::DeviceType::Cpu {
        println!("SKIP: 无可用 GPU 设备");
        return;
    }
    let mut gpu = GpuNoiseSampler::new(device);

    let n = 512usize;
    let pos = mk_pos3d(n);
    let sa = mk_sampler(1, &[0, 1, 2]);
    let sb = mk_sampler(2, &[0, 1, 2]);

    let mut out_a = vec![0.0; n];
    let mut out_b = vec![0.0; n];
    gpu.sample_octave_jit(&sa, &pos, &mut out_a)
        .expect("sample a");
    gpu.sample_octave_jit(&sb, &pos, &mut out_b)
        .expect("sample b");

    let mut cpu_a = vec![0.0; n];
    let mut cpu_b = vec![0.0; n];
    for i in 0..n {
        cpu_a[i] = sa.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        cpu_b[i] = sb.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    assert_eq!(
        fnv1a_f64(&cpu_a),
        fnv1a_f64(&out_a),
        "sampler a 的 JIT 结果必须与 CPU 一致"
    );
    assert_eq!(
        fnv1a_f64(&cpu_b),
        fnv1a_f64(&out_b),
        "sampler b（同八度数、不同种子）的 JIT 结果必须与 CPU 一致"
    );
}
