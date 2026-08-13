//! 真 GPU 后端对齐测试 — CUDA / `OpenCL` 在真硬件上的 JIT 与 batch 路径逐位一致。
//!
//! 无 GPU 时自动跳过（`GpuDevice::init()` 回退 CPU）。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::print_stderr
)]
#![cfg(feature = "gpu")]

use pumpkin_gpu::DeviceType;
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

fn mk_pos3d(n: usize, seed: u64) -> Vec<f64> {
    let mut p = Vec::with_capacity(n * 3);
    let mut s = seed;
    for _ in 0..n {
        p.push((s.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
    }
    p
}

fn mk_pos2d(n: usize, seed: u64) -> Vec<f64> {
    let mut p = Vec::with_capacity(n * 2);
    let mut s = seed;
    for _ in 0..n {
        p.push((s.wrapping_mul(6364136223846793005) as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
    }
    p
}

/// 有真 GPU 时返回设备，否则返回 None（测试跳过）。
fn gpu_or_none() -> Option<GpuDevice> {
    let device = GpuDevice::init();
    if device.device_type() == DeviceType::Cpu {
        eprintln!("SKIP: 无可用 GPU 设备");
        None
    } else {
        Some(device)
    }
}

/// JIT 八度采样与标准 batch kernel 在真 GPU 上逐位一致。
#[test]
fn gpu_jit_octave_parity() {
    let Some(device) = gpu_or_none() else {
        return;
    };
    let mut gpu = GpuNoiseSampler::new(device);
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 1024;
    let pos = mk_pos3d(n, SEED);

    let mut batch = vec![0.0f64; n];
    let mut jit = vec![0.0f64; n];
    gpu.sample_octave_batch(&sampler, &pos, &mut batch)
        .expect("batch launch");
    gpu.sample_octave_jit(&sampler, &pos, &mut jit)
        .expect("jit launch");

    assert_eq!(
        fnv1a_f64(&batch),
        fnv1a_f64(&jit),
        "真 GPU 上 JIT 八度采样与 batch 不一致"
    );
}

/// JIT 双 Perlin 采样与标准 batch kernel 在真 GPU 上逐位一致。
#[test]
fn gpu_jit_double_perlin_parity() {
    let Some(device) = gpu_or_none() else {
        return;
    };
    let mut gpu = GpuNoiseSampler::new(device);
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[0, 1, 2]);
    let n = 512;
    let pos = mk_pos3d(n, SEED.wrapping_add(1));

    let mut batch = vec![0.0f64; n];
    let mut jit = vec![0.0f64; n];
    gpu.sample_double_perlin_batch(&a, &b, 0.5, &pos, &mut batch)
        .expect("batch launch");
    gpu.sample_double_perlin_jit(&a, &b, 0.5, &pos, &mut jit)
        .expect("jit launch");

    assert_eq!(
        fnv1a_f64(&batch),
        fnv1a_f64(&jit),
        "真 GPU 上 JIT 双 Perlin 采样与 batch 不一致"
    );
}

/// JIT `FlatCache` 与标准 kernel 在真 GPU 上逐位一致。
#[test]
fn gpu_jit_flatcache_parity() {
    let Some(device) = gpu_or_none() else {
        return;
    };
    let mut gpu = GpuNoiseSampler::new(device);
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 512;
    let xz = mk_pos2d(n, SEED.wrapping_add(2));

    let mut reference = vec![0.0f64; n];
    let mut jit = vec![0.0f64; n];
    for i in 0..n {
        reference[i] = sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
    }
    // `precompute_flatcache` 在 JIT 开启（max_unroll 默认 16）时优先走 JIT kernel。
    gpu.precompute_flatcache(&sampler, &xz, &mut jit)
        .expect("flatcache launch");

    assert_eq!(
        fnv1a_f64(&reference),
        fnv1a_f64(&jit),
        "真 GPU 上 JIT FlatCache 与 CPU 参考不一致"
    );
}

/// 真 GPU 上内核启动器可用，且注册了核心 kernel。
#[test]
fn gpu_kernel_launcher_registered() {
    let Some(device) = gpu_or_none() else {
        return;
    };
    let launcher = device.kernel_launcher().expect("GPU launcher should exist");
    for name in [
        "octave_perlin_sample_f64",
        "double_perlin_sample_f64",
        "flatcache_precompute_f64",
        "trilinear_interpolate_f64",
    ] {
        assert!(
            launcher.has_kernel(name),
            "核心 kernel '{name}' 未注册（编译失败将回退 CPU）"
        );
    }
}
