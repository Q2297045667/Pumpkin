//! 世界生成性能基准和压力测试。
//!
//! 测试 GPU 和 CPU 路径在各种规模下的性能和稳定性。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::many_single_char_names
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_gpu::noise::batch_cell::BeardifierStructureData;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::batch_accel::BatchAccelerator;
use pumpkin_world::noise_accel::NoiseAccelerator;
use std::hint::black_box;

const SEED: u64 = 138_782_381_985_206;

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

fn batch() -> BatchAccelerator {
    BatchAccelerator::new(&GpuConfig::default())
}

// ============================================================================
// 性能基准
// ============================================================================

macro_rules! bench {
    ($name:ident, $n:expr, $iters:expr, $body:block) => {
        #[test]
        fn $name() {
            let start = std::time::Instant::now();
            for _ in 0..$iters {
                black_box({ $body });
            }
            let elapsed = start.elapsed();
            println!(
                "{} (n={}, x{}): {:?}",
                stringify!($name),
                $n,
                $iters,
                elapsed
            );
        }
    };
}

bench!(bench_trilinear_1k, 1024, 20, {
    let n = 1024;
    let mut c = vec![0.0; n * 8];
    let mut d = vec![0.0; n * 3];
    let mut s = SEED;
    for i in 0..n * 8 {
        c[i] = (s.wrapping_mul(6364136223846793005) as f64) * 1e-12;
        s = s.wrapping_mul(1442695040888963407);
    }
    for i in 0..n * 3 {
        d[i] = ((s >> 32) as f64) / (u32::MAX as f64);
        s = s.wrapping_mul(1442695040888963407);
    }
    let mut r = vec![0.0; n];
    batch().batch_trilinear(&c, &d, &mut r);
});

bench!(bench_octave_1k, 1024, 10, {
    let mut accel = NoiseAccelerator::new(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    });
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3, 4]);
    let pos = mk_pos3d(1024);
    let mut res = vec![0.0; 1024];
    accel.sample_octave(&sampler, &pos, &mut res);
    assert!(res.iter().all(|v| v.is_finite()));
});

// ============================================================================
// 压力测试 — 大输入不崩溃
// ============================================================================

#[test]
fn stress_trilinear_131k() {
    let n = 131072;
    let mut c = vec![0.0; n * 8];
    let mut d = vec![0.0; n * 3];
    let mut s = SEED;
    for i in 0..n * 8 {
        c[i] = (s.wrapping_mul(6364136223846793005) as f64) * 1e-12;
        s = s.wrapping_mul(1442695040888963407);
    }
    for i in 0..n * 3 {
        d[i] = ((s >> 32) as f64) / (u32::MAX as f64);
        s = s.wrapping_mul(1442695040888963407);
    }
    let mut r = vec![0.0; n];
    batch().batch_trilinear(&c, &d, &mut r);
    assert!(r.iter().all(|v| v.is_finite()));
}

#[test]
fn stress_all_ops_chained() {
    // 链式执行所有 batch 操作，验证无内存损坏
    let pos3 = mk_pos3d(1024);
    let mut f64buf = vec![0.0; 1024];

    // Trilinear
    let mut c = vec![0.0; 256 * 8];
    let mut d = vec![0.0; 256 * 3];
    for i in 0..256 * 8 {
        c[i] = (i as f64).sin();
    }
    for i in 0..256 * 3 {
        d[i] = (i as f64 * 0.7).fract();
    }
    batch().batch_trilinear(&c, &d, &mut f64buf[..256]);
    assert!(f64buf[..256].iter().all(|v| v.is_finite()));

    // Beardifier
    let structures = [BeardifierStructureData {
        box_min_x: -5,
        box_min_y: 60,
        box_min_z: -5,
        box_max_x: 5,
        box_max_y: 70,
        box_max_z: 5,
        adaptation: 1, // BeardThin
        ground_delta: 5,
    }];
    batch().batch_beardifier(
        &pos3[..9],
        &structures,
        &[],
        [-10, 55, -10, 10, 75, 10],
        &mut f64buf[..3],
    );
    assert!(f64buf[..3].iter().all(|v| v.is_finite()));

    // Aquifer
    let mut pg = Vec::new();
    for i in 0..4 {
        pg.push(((i as f64) * 20.0).to_bits() as i64);
        pg.push((64.0f64).to_bits() as i64);
        pg.push(((i as f64) * 20.0).to_bits() as i64);
        pg.push(0.3f64.to_bits() as i64);
    }
    let dens: Vec<f64> = (0..4).map(|i| -(i as f64) * 0.01).collect();
    batch().batch_aquifer_apply(&pos3[..12], &dens, &pg, -10000.0, 0.3);
}

#[test]
fn noise_accel_cpu_fallback() {
    let mut accel = NoiseAccelerator::new(&GpuConfig::default());
    assert!(!accel.is_active(), "default config should be inactive");
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let pos = mk_pos3d(256);
    let mut res = vec![0.0; 256];
    // Should NOT panic even with GPU disabled
    accel.sample_octave(&sampler, &pos, &mut res);
    assert!(res.iter().all(|v| v.is_finite()));
}
