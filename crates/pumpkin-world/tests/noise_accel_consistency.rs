//! `NoiseAccelerator` 完整一致性测试 — 覆盖全部噪声类型。
//!
//! 固定种子 138782381985206，对比 CPU 直接采样与 GPU 路径的 FNV-1a 指纹。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::noise_accel::NoiseAccelerator;

const SEED: u64 = 138_782_381_985_206;

// ============================================================================
// 辅助函数
// ============================================================================

fn mk_sampler(seed: u64, octaves: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
    let mut g = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
}

fn mk_positions(n: usize) -> Vec<f64> {
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

/// 生成 2D (x, z) 位置用于 ShiftA/ShiftB/FlatCache
fn mk_positions_2d(n: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(n * 2);
    let mut s = SEED;
    for _ in 0..n {
        p.push((s.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
    }
    p
}

fn fnv1a_f64(d: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in d {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn mk_accel() -> NoiseAccelerator {
    NoiseAccelerator::new(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    })
}

// ============================================================================
// Octave Perlin — 基础类型
// ============================================================================

#[test]
fn octave_single() {
    let s = mk_sampler(SEED, &[0]);
    let n = 256;
    let p = mk_positions(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    let mut accel = mk_accel();
    accel.sample_octave(&s, &p, &mut gpu);
    assert_eq!(
        fnv1a_f64(&cpu),
        fnv1a_f64(&gpu),
        "octave_single fingerprint mismatch"
    );
}

#[test]
fn octave_multi_3() {
    let s = mk_sampler(SEED, &[-2, 0, 2]);
    let n = 512;
    let p = mk_positions(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    let mut accel = mk_accel();
    accel.sample_octave(&s, &p, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "octave_multi_3");
}

#[test]
fn octave_multi_5() {
    let s = mk_sampler(SEED, &[-3, -1, 0, 1, 3]);
    let n = 1024;
    let p = mk_positions(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    let mut accel = mk_accel();
    accel.sample_octave(&s, &p, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "octave_multi_5");
}

#[test]
fn octave_zero_positions() {
    // 零位置（原点）应可正常采样
    let s = mk_sampler(SEED, &[0, 1]);
    let n = 64;
    let p = vec![0.0f64; n * 3];
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(0.0, 0.0, 0.0);
    }
    let mut accel = mk_accel();
    accel.sample_octave(&s, &p, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "octave_zero_pos");
}

// ============================================================================
// Double Perlin
// ============================================================================

#[test]
fn double_perlin_consistency() {
    let a = mk_sampler(SEED, &[0, 1]);
    let b = mk_sampler(SEED.wrapping_add(1), &[-1, 0, 1]);
    let amp = 0.5;
    let n = 512;
    let p = mk_positions(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    let c = 1.0181268882175227f64;
    for i in 0..n {
        cpu[i] = (a.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2])
            + b.sample(p[i * 3] * c, p[i * 3 + 1] * c, p[i * 3 + 2] * c))
            * amp;
    }
    let mut accel = mk_accel();
    accel.sample_double_perlin(&a, &b, amp, &p, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "double_perlin");
}

#[test]
fn double_perlin_small() {
    let a = mk_sampler(SEED, &[0]);
    let b = mk_sampler(SEED.wrapping_add(2), &[0]);
    let n = 32;
    let p = mk_positions(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    let c = 1.0181268882175227f64;
    for i in 0..n {
        cpu[i] = (a.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2])
            + b.sample(p[i * 3] * c, p[i * 3 + 1] * c, p[i * 3 + 2] * c))
            * 1.0;
    }
    let mut accel = mk_accel();
    accel.sample_double_perlin(&a, &b, 1.0, &p, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "double_perlin_small");
}

// ============================================================================
// ShiftA / ShiftB
// ============================================================================

#[test]
fn shift_a_consistency() {
    let s = mk_sampler(SEED, &[0, 1, 2]);
    let n = 512;
    let xz = mk_positions_2d(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(xz[i * 2] * 0.25, 0.0, xz[i * 2 + 1] * 0.25) * 4.0;
    }
    let mut accel = mk_accel();
    accel.sample_shift_a(&s, &xz, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "shift_a");
}

#[test]
fn shift_b_consistency() {
    let s = mk_sampler(SEED, &[0, 1, 2]);
    let n = 512;
    let zx = mk_positions_2d(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(zx[i * 2 + 1] * 0.25, 0.0, zx[i * 2] * 0.25) * 4.0;
    }
    let mut accel = mk_accel();
    accel.sample_shift_b(&s, &zx, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "shift_b");
}

// ============================================================================
// 三线性插值
// ============================================================================

#[test]
fn trilinear_consistency() {
    let n = 256;
    // 生成 8 个角落值和 3 个 delta 值
    let mut s = SEED;
    let mut corners = Vec::with_capacity(n * 8);
    let mut deltas = Vec::with_capacity(n * 3);
    for _ in 0..n {
        for _ in 0..8 {
            corners.push((s.wrapping_mul(6364136223846793005) as f64) * 1e-12);
            s = s.wrapping_mul(1442695040888963407);
        }
        deltas.push((s as f64 % 1000.0) / 1000.0);
        s = s.wrapping_mul(1442695040888963407);
        deltas.push((s as f64 % 1000.0) / 1000.0);
        s = s.wrapping_mul(1442695040888963407);
        deltas.push((s as f64 % 1000.0) / 1000.0);
    }

    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        let b = i * 8;
        let dx = deltas[i * 3];
        let dy = deltas[i * 3 + 1];
        let dz = deltas[i * 3 + 2];
        cpu[i] = corners[b] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + corners[b + 1] * dx * (1.0 - dy) * (1.0 - dz)
            + corners[b + 2] * (1.0 - dx) * dy * (1.0 - dz)
            + corners[b + 3] * dx * dy * (1.0 - dz)
            + corners[b + 4] * (1.0 - dx) * (1.0 - dy) * dz
            + corners[b + 5] * dx * (1.0 - dy) * dz
            + corners[b + 6] * (1.0 - dx) * dy * dz
            + corners[b + 7] * dx * dy * dz;
    }

    let mut accel = mk_accel();
    accel.batch_trilinear(&corners, &deltas, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "trilinear");
}

#[test]
fn trilinear_identity() {
    // 当所有 delta=0 时，结果应等于 corners[0]
    let n = 64;
    let mut corners = vec![0.0f64; n * 8];
    for i in 0..n {
        corners[i * 8] = i as f64 + 1.0;
    }
    let deltas = vec![0.0f64; n * 3];
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = corners[i * 8]; // delta=0 → 仅第一项
    }
    let mut accel = mk_accel();
    accel.batch_trilinear(&corners, &deltas, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "trilinear_identity");
}

// ============================================================================
// FlatCache 预计算
// ============================================================================

#[test]
fn flatcache_consistency() {
    let s = mk_sampler(SEED, &[0, 1]);
    let n = 256;
    let xz = mk_positions_2d(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
    }
    let mut accel = mk_accel();
    accel.precompute_flatcache(&s, &xz, &mut gpu);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "flatcache");
}

// ============================================================================
// Surface 噪声预计算
// ============================================================================

#[test]
fn surface_noise_consistency() {
    let surface_a = mk_sampler(SEED, &[0]);
    let surface_b = mk_sampler(SEED.wrapping_add(3), &[0]);
    let secondary_a = mk_sampler(SEED.wrapping_add(5), &[0]);
    let secondary_b = mk_sampler(SEED.wrapping_add(7), &[0]);

    let (start_x, start_z) = (0, 0);
    let n = 256;
    let mut xz = Vec::with_capacity(n * 2);
    for lx in 0i32..16 {
        for lz in 0i32..16 {
            xz.push((start_x + lx) as f64);
            xz.push((start_z + lz) as f64);
        }
    }

    // CPU 参考
    let c = 1.0181268882175227f64;
    let mut cpu_surf = vec![0.0f64; n];
    let mut cpu_sec = vec![0.0f64; n];
    for i in 0..n {
        let x = xz[i * 2];
        let z = xz[i * 2 + 1];
        cpu_surf[i] = (surface_a.sample(x, 0.0, z) + surface_b.sample(x * c, 0.0, z * c)) * 1.0;
        cpu_sec[i] = (secondary_a.sample(x, 0.0, z) + secondary_b.sample(x * c, 0.0, z * c)) * 1.0;
    }

    let mut accel = mk_accel();
    let gpu = accel.precompute_surface(
        &surface_a,
        &surface_b,
        1.0,
        &secondary_a,
        &secondary_b,
        1.0,
        start_x,
        start_z,
    );

    assert_eq!(
        fnv1a_f64(&cpu_surf),
        fnv1a_f64(&*gpu.surface),
        "surface_surface"
    );
    assert_eq!(
        fnv1a_f64(&cpu_sec),
        fnv1a_f64(&*gpu.secondary),
        "surface_secondary"
    );
}

// ============================================================================
// 空输入边界测试
// ============================================================================

#[test]
fn noise_empty_input() {
    let s = mk_sampler(SEED, &[0]);
    let mut accel = mk_accel();
    let mut res: Vec<f64> = vec![];
    // 不应 panic
    accel.sample_octave(&s, &[], &mut res);
    accel.sample_double_perlin(&s, &s, 1.0, &[], &mut res);
    accel.sample_shift_a(&s, &[], &mut res);
    accel.sample_shift_b(&s, &[], &mut res);
    accel.batch_trilinear(&[], &[], &mut res);
    accel.precompute_flatcache(&s, &[], &mut res);
}

// ============================================================================
// 多次调用一致性（验证缓存稳定性）
// ============================================================================

#[test]
fn octave_cache_stability() {
    let s = mk_sampler(SEED, &[0, 1, 2]);
    let n = 256;
    let p = mk_positions(n);
    let mut accel = mk_accel();
    let mut r1 = vec![0.0f64; n];
    let mut r2 = vec![0.0f64; n];
    let mut r3 = vec![0.0f64; n];

    // 三次调用应产生相同结果（验证缓存不会破坏数据）
    accel.sample_octave(&s, &p, &mut r1);
    accel.sample_octave(&s, &p, &mut r2);
    accel.sample_octave(&s, &p, &mut r3);

    let f1 = fnv1a_f64(&r1);
    assert_eq!(f1, fnv1a_f64(&r2), "cache_stability_2nd");
    assert_eq!(f1, fnv1a_f64(&r3), "cache_stability_3rd");
}

// ============================================================================
// 基准测试（含计时）
// ============================================================================

#[test]
fn bench_octave_large() {
    let s = mk_sampler(SEED, &[-2, -1, 0, 1, 2, 3]);
    let n = 65536;
    let p = mk_positions(n);
    let mut cpu = vec![0.0f64; n];
    let mut gpu = vec![0.0f64; n];

    let t0 = std::time::Instant::now();
    for i in 0..n {
        cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut accel = mk_accel();
    let t1 = std::time::Instant::now();
    accel.sample_octave(&s, &p, &mut gpu);
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "bench_large");

    let speedup = cpu_ms / gpu_ms.max(1e-9);
    println!("octave_bench(n={n}): cpu={cpu_ms:.2}ms, gpu={gpu_ms:.2}ms, speedup={speedup:.2}x");
}
