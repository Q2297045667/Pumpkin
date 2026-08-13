//! 世界生成指纹测试 — 覆盖 Aquifer/Beardifier/Trilinear/噪声采样全路径。
//!
//! 使用固定种子生成确定性噪声配置，验证 GPU 批量路径输出的确定性与有限性。
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

const SEED: u64 = 138_782_381_985_206;

// ============================================================================
// Helpers
// ============================================================================

fn f64_hash(data: &[f64]) -> u64 {
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

fn accel() -> BatchAccelerator {
    BatchAccelerator::new(&GpuConfig::default())
}

fn noise_accel() -> NoiseAccelerator {
    NoiseAccelerator::new(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    })
}

// ============================================================================
// Aquifer 指纹
// ============================================================================

#[test]
fn aquifer_grid4() {
    // 4-grid: minimal test
    let n = 16;
    let positions: Vec<f64> = (0..n)
        .flat_map(|i| {
            let x = (i as f64) * 10.0;
            [x, 64.0, x]
        })
        .collect();
    let densities: Vec<f64> = (0..n).map(|i| -(i as f64) * 0.05).collect();
    let mut packed_grid = Vec::new();
    for i in 0..4 {
        packed_grid.push(((i as f64) * 20.0).to_bits() as i64);
        packed_grid.push((64.0f64).to_bits() as i64);
        packed_grid.push(((i as f64) * 20.0).to_bits() as i64);
        packed_grid.push(0.3f64.to_bits() as i64);
    }
    let result = accel().batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
    assert_eq!(result.block_ids.len(), n);
    assert_eq!(result.fluid_updates.len(), n);
}

#[test]
fn aquifer_empty_grid() {
    let result = accel().batch_aquifer_apply(&[], &[], &[], -10000.0, 0.3);
    assert!(result.block_ids.is_empty());
}

// ============================================================================
// Beardifier 指纹
// ============================================================================

#[test]
fn beardier_1struct() {
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
    let junctions = [];
    let positions = [0.0f64, 64.0, 0.0, 3.0, 64.0, 3.0, -3.0, 64.0, -3.0];
    let n = 3;
    let mut res = vec![0.0; n];
    accel().batch_beardifier(
        &positions,
        &structures,
        &junctions,
        [-10, 55, -10, 10, 75, 10],
        &mut res,
    );
    assert!(res.iter().all(|&v| v.is_finite()));
    // Center position should have positive contribution
    assert!(
        res[0] >= 0.0,
        "center should have non-negative beard: {}",
        res[0]
    );
}

// ============================================================================
// NoiseAccelerator 全噪声类型指纹
// ============================================================================

#[test]
fn noise_octave_fingerprint() {
    let mut accel = noise_accel();
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 1024;
    let pos = mk_pos3d(n);
    let mut res = vec![0.0; n];
    accel.sample_octave(&sampler, &pos, &mut res);
    assert!(res.iter().all(|&v| v.is_finite()));
    let hash = f64_hash(&res);
    assert_ne!(hash, 0);
    // Deterministic
    let mut res2 = vec![0.0; n];
    accel.sample_octave(&sampler, &pos, &mut res2);
    assert_eq!(
        f64_hash(&res2),
        hash,
        "octave fingerprint must be deterministic"
    );
}

#[test]
fn noise_double_perlin_fingerprint() {
    let mut accel = noise_accel();
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[0, 1, 2]);
    let n = 512;
    let pos = mk_pos3d(n);
    let mut res = vec![0.0; n];
    accel.sample_double_perlin(&a, &b, 0.5, &pos, &mut res);
    assert!(res.iter().all(|&v| v.is_finite()));
    assert_ne!(f64_hash(&res), 0);
}

#[test]
fn noise_shift_a_fingerprint() {
    let mut accel = noise_accel();
    let s = mk_sampler(SEED, &[0, 1, 2]);
    let n = 256;
    let xz = mk_pos2d(n);
    let mut res = vec![0.0; n];
    accel.sample_shift_a(&s, &xz, &mut res);
    assert!(res.iter().all(|&v| v.is_finite()));
    assert_ne!(f64_hash(&res), 0);
}

#[test]
fn noise_shift_b_fingerprint() {
    let mut accel = noise_accel();
    let s = mk_sampler(SEED, &[0, 1, 2]);
    let n = 256;
    let zx = mk_pos2d(n);
    let mut res = vec![0.0; n];
    accel.sample_shift_b(&s, &zx, &mut res);
    assert!(res.iter().all(|&v| v.is_finite()));
    assert_ne!(f64_hash(&res), 0);
}

// ============================================================================
// Trilinear 指纹
// ============================================================================

#[test]
fn trilinear_fingerprint() {
    let n = 64;
    let mut corners = vec![0.0; n * 8];
    let mut deltas = vec![0.0; n * 3];
    let mut s = SEED;
    for i in 0..n * 8 {
        corners[i] = (s.wrapping_mul(6364136223846793005) as f64) * 1e-12;
        s = s.wrapping_mul(1442695040888963407);
    }
    for i in 0..n * 3 {
        deltas[i] = ((s >> 32) as f64) / (u32::MAX as f64);
        s = s.wrapping_mul(1442695040888963407);
    }
    let mut res1 = vec![0.0; n];
    let mut res2 = vec![0.0; n];
    accel().batch_trilinear(&corners, &deltas, &mut res1);
    accel().batch_trilinear(&corners, &deltas, &mut res2);
    assert_eq!(f64_hash(&res1), f64_hash(&res2), "trilinear deterministic");
}
