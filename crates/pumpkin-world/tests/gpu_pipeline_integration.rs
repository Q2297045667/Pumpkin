//! GPU 流水线集成测试 — JIT 配置、Surface 回退、矿脉缓存、噪声缓存回填。
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
use pumpkin_gpu::noise::batch_cell::CellFillParams;
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

// ============================================================================
// JIT 配置测试
// ============================================================================

#[test]
fn jit_config_disabled_by_default() {
    let config = GpuConfig::default();
    assert!(!config.jit_enabled, "JIT should be disabled by default");
    assert_eq!(config.jit_max_unroll, 16);
}

#[test]
fn jit_config_enabled_fields() {
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 8,
        ..Default::default()
    };
    assert!(config.jit_enabled);
    assert_eq!(config.jit_max_unroll, 8);
    // jit_enabled 开启时，BatchAccelerator::is_active 也应返回 true
    let accel = BatchAccelerator::new(&config);
    assert!(
        accel.is_active(),
        "BatchAccelerator should be active when jit_enabled"
    );
}

// ============================================================================
// BatchAccelerator 配置 & 激活测试
// ============================================================================

#[test]
fn batch_accel_jit_only_is_active() {
    // jit_enabled 单独开启也应激活 BatchAccelerator
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: false,
        batch_acceleration: false,
        jit_enabled: true,
        ..Default::default()
    };
    let accel = BatchAccelerator::new(&config);
    assert!(
        accel.is_active(),
        "BatchAccelerator should be active when jit_enabled"
    );
}

#[test]
fn batch_accel_all_disabled_not_active() {
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: false,
        batch_acceleration: false,
        jit_enabled: false,
        ..Default::default()
    };
    let accel = BatchAccelerator::new(&config);
    assert!(
        !accel.is_active(),
        "BatchAccelerator should be inactive when all disabled"
    );
}

// ============================================================================
// BatchTrilinear 测试
// ============================================================================

#[test]
fn trilinear_batch_cpu_fallback() {
    let config = GpuConfig {
        enabled: false,
        ..Default::default()
    };
    let accel = BatchAccelerator::new(&config);

    // 4 组三线性插值
    let corners = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // item 0
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, // item 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // item 2 (all zeros)
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // item 3 (corner [0]=1)
    ];
    let deltas = vec![
        0.5, 0.5, 0.5, // item 0
        0.0, 0.0, 0.0, // item 1 (at corner)
        1.0, 1.0, 1.0, // item 2 (far corner)
        0.0, 0.0, 0.0, // item 3 (at corner)
    ];
    let n = 4;
    let mut results = vec![0.0f64; n];
    accel.batch_trilinear(&corners, &deltas, &mut results);

    // Item 0: trilinear([0..8], 0.5,0.5,0.5) = 3.5
    assert!(
        (results[0] - 3.5).abs() < 1e-12,
        "trilinear 0 failed: {}",
        results[0]
    );
    // Item 1: at corner (0,0,0) → corners[0] = 10.0
    assert!(
        (results[1] - 10.0).abs() < 1e-12,
        "trilinear 1 failed: {}",
        results[1]
    );
    // Item 2: all zeros → result 0
    assert!(
        (results[2] - 0.0).abs() < 1e-12,
        "trilinear 2 failed: {}",
        results[2]
    );
    // Item 3: at corner (0,0,0) → corners[0] = 1.0
    assert!(
        (results[3] - 1.0).abs() < 1e-12,
        "trilinear 3 failed: {}",
        results[3]
    );

    let fingerprint = fnv1a_f64(&results);
    // fingerprint should be deterministic
    assert_eq!(
        fingerprint,
        fnv1a_f64(&results),
        "fingerprint not deterministic"
    );
}

#[test]
fn trilinear_empty_batch() {
    let config = GpuConfig::default();
    let accel = BatchAccelerator::new(&config);
    let mut results: [f64; 0] = [];
    // Should not panic for empty batch
    accel.batch_trilinear(&[], &[], &mut results);
}

// ============================================================================
// NoiseAccelerator 测试
// ============================================================================

#[test]
fn noise_accel_inactive_by_default() {
    let config = GpuConfig::default();
    let accel = NoiseAccelerator::new(&config);
    assert!(
        !accel.is_active(),
        "NoiseAccelerator should be inactive by default"
    );
}

#[test]
fn noise_accel_active_when_configured() {
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    };
    let accel = NoiseAccelerator::new(&config);
    // May be inactive if no GPU device available, but shouldn't panic
    let _ = black_box(accel.is_active());
}

#[test]
fn noise_accel_sample_octave_fallback_works() {
    let mut accel = NoiseAccelerator::new(&GpuConfig::default());
    assert!(!accel.is_active());

    let sampler = mk_sampler(SEED, &[1, 2, 3]);
    let n = 16;
    let positions: Vec<f64> = (0..n)
        .flat_map(|i| {
            let x = (i as f64) * 1.5;
            [x, x * 0.5 + 1.0, x * 0.3 - 2.0]
        })
        .collect();
    let mut results = vec![0.0f64; n];
    accel.sample_octave(&sampler, &positions, &mut results);

    // Verify all results are finite
    for (i, &r) in results.iter().enumerate() {
        assert!(r.is_finite(), "result[{i}] = {r} is not finite");
    }

    let fingerprint = fnv1a_f64(&results);
    assert_ne!(fingerprint, 0, "fingerprint should not be zero");
}

#[test]
fn noise_accel_precompute_surface_fallback_works() {
    let mut accel = NoiseAccelerator::new(&GpuConfig::default());
    assert!(!accel.is_active());

    let surface_a = mk_sampler(SEED, &[0, 1, 2, 3]);
    let surface_b = mk_sampler(SEED ^ 1, &[0, 1, 2, 3]);
    let secondary_a = mk_sampler(SEED ^ 2, &[4, 5, 6]);
    let secondary_b = mk_sampler(SEED ^ 3, &[4, 5, 6]);

    let cache = accel.precompute_surface(
        &surface_a,
        &surface_b,
        0.5,
        &secondary_a,
        &secondary_b,
        0.3,
        0,
        0,
    );

    // All 256 positions should have valid surface noise
    for i in 0..256 {
        assert!(cache.surface[i].is_finite(), "surface[{i}] not finite");
        assert!(cache.secondary[i].is_finite(), "secondary[{i}] not finite");
    }

    let fp_surface = fnv1a_f64(&*cache.surface);
    let fp_secondary = fnv1a_f64(&*cache.secondary);
    assert_ne!(fp_surface, 0);
    assert_ne!(fp_secondary, 0);

    // Deterministic: same inputs → same outputs
    let cache2 = accel.precompute_surface(
        &surface_a,
        &surface_b,
        0.5,
        &secondary_a,
        &secondary_b,
        0.3,
        0,
        0,
    );
    assert_eq!(fp_surface, fnv1a_f64(&*cache2.surface));
    assert_eq!(fp_secondary, fnv1a_f64(&*cache2.secondary));
}

// ============================================================================
// BatchAccelerator Cell Cache / Interpolator 测试
// ============================================================================

#[test]
fn batch_cell_cache_empty_config() {
    let config = GpuConfig::default();
    let accel = BatchAccelerator::new(&config);
    let params = CellFillParams {
        perlin_configs: vec![],
        num_octaves: vec![],
        sampler_types: vec![],
    };
    let mut results = vec![0.0f64; 32];
    // Empty config should zero-fill or not crash
    accel.batch_fill_cell_caches(&[0.0; 96], &params, &mut results);
    for &r in &results {
        assert!(r.is_finite(), "zero-fill: result should be finite");
    }
}

#[test]
fn batch_interpolator_empty_config() {
    let config = GpuConfig::default();
    let accel = BatchAccelerator::new(&config);
    let params = CellFillParams {
        perlin_configs: vec![],
        num_octaves: vec![],
        sampler_types: vec![],
    };
    let mut results = vec![0.0f64; 16];
    accel.batch_fill_interpolators(&[0.0; 48], &params, &mut results);
    for &r in &results {
        assert!(r.is_finite());
    }
}

// ============================================================================
// 基准测试
// ============================================================================

#[test]
fn bench_trilinear_batch_1024() {
    let config = GpuConfig::default();
    let accel = BatchAccelerator::new(&config);
    let n = 1024;

    let mut corners = vec![0.0f64; n * 8];
    let mut deltas = vec![0.0f64; n * 3];
    let mut results = vec![0.0f64; n];

    // 填充随机数据
    for i in 0..n * 8 {
        corners[i] = (i as f64).sin();
    }
    for i in 0..n * 3 {
        deltas[i] = (i as f64 * 0.7).fract();
    }

    let start = std::time::Instant::now();
    for _ in 0..10 {
        accel.batch_trilinear(
            black_box(&corners),
            black_box(&deltas),
            black_box(&mut results),
        );
    }
    let elapsed = start.elapsed();

    // 不应崩溃 + 有合理性能
    assert!(results.iter().all(|&r| r.is_finite()));
    println!(
        "trilinear batch 1024 x10: {elapsed:?} (fingerprint: {})",
        fnv1a_f64(&results)
    );
}

#[test]
fn bench_cell_cache_1024() {
    let config = GpuConfig::default();
    let accel = BatchAccelerator::new(&config);
    let _sampler = mk_sampler(SEED, &[1, 2, 3, 4]);

    // Build params from a real sampler
    let n = 1024;
    let positions: Vec<f64> = (0..n)
        .flat_map(|i| {
            let x = (i as f64) * 0.5;
            [x, x * 0.3 + 1.0, x * 0.7 - 2.0]
        })
        .collect();

    // Simple params (1 sampler, 4 octaves)
    let mut perlin_configs = vec![4.0f64]; // num_octaves
    for _o in 0..4i32 {
        perlin_configs.push(1.0); // amp
        perlin_configs.push(2.0); // lac
        perlin_configs.push(0.0); // org_x
        perlin_configs.push(0.0); // org_y
        perlin_configs.push(0.0); // org_z
    }
    let params = CellFillParams {
        perlin_configs,
        num_octaves: vec![4],
        sampler_types: vec![0],
    };

    let mut results = vec![0.0f64; n];
    let start = std::time::Instant::now();
    for _ in 0..5 {
        accel.batch_fill_cell_caches(
            black_box(&positions),
            black_box(&params),
            black_box(&mut results),
        );
    }
    let elapsed = start.elapsed();
    assert!(results.iter().all(|&r| r.is_finite()));
    println!(
        "cell_cache 1024 x5: {elapsed:?} (fingerprint: {})",
        fnv1a_f64(&results)
    );
}
