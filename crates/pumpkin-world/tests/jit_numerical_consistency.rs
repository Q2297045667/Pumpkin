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
