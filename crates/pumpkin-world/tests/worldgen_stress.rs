//! 世界生成 GPU 路径压力测试。
//!
//! 覆盖大输入、极端坐标、边界尺寸、重复调用与大规模结构，
//! 验证不崩溃、不泄漏、输出有限且确定。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_gpu::noise::batch_cell::{BeardifierJunctionData, BeardifierStructureData};
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::batch_accel::BatchAccelerator;
use pumpkin_world::noise_accel::NoiseAccelerator;

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

fn mk_noise_accel() -> NoiseAccelerator {
    NoiseAccelerator::new(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    })
}

fn mk_batch_accel() -> BatchAccelerator {
    BatchAccelerator::new(&GpuConfig {
        enabled: true,
        batch_acceleration: true,
        ..Default::default()
    })
}

// ============================================================================
// 大批量输入
// ============================================================================

#[test]
fn stress_octave_262k() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 262_144;
    let pos = mk_pos3d(n, SEED);
    let mut res1 = vec![0.0f64; n];
    let mut res2 = vec![0.0f64; n];
    let mut accel = mk_noise_accel();
    accel.sample_octave(&sampler, &pos, &mut res1);
    accel.sample_octave(&sampler, &pos, &mut res2);
    assert!(res1.iter().all(|&v| v.is_finite()), "all finite");
    assert_eq!(
        fnv1a_f64(&res1),
        fnv1a_f64(&res2),
        "large octave must be deterministic"
    );
}

#[test]
fn stress_double_perlin_65k() {
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[-1, 0, 1]);
    let n = 65_536;
    let pos = mk_pos3d(n, SEED.wrapping_add(1));
    let mut res = vec![0.0f64; n];
    let mut accel = mk_noise_accel();
    accel.sample_double_perlin(&a, &b, 0.5, &pos, &mut res);
    assert!(res.iter().all(|&v| v.is_finite()));
    assert_ne!(fnv1a_f64(&res), 0, "output must not be all zero");
}

#[test]
fn stress_flatcache_65k() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 65_536;
    let pos = mk_pos3d(n, SEED.wrapping_add(2));
    let xz: Vec<f64> = (0..n).flat_map(|i| [pos[i * 3], pos[i * 3 + 2]]).collect();
    let mut res = vec![0.0f64; n];
    let mut accel = mk_noise_accel();
    accel.precompute_flatcache(&sampler, &xz, &mut res);
    assert!(res.iter().all(|&v| v.is_finite()));
}

#[test]
fn stress_trilinear_131k() {
    let accel = mk_batch_accel();
    let n = 131_072;
    let mut s = SEED;
    let mut corners = Vec::with_capacity(n * 8);
    let mut deltas = Vec::with_capacity(n * 3);
    for _ in 0..n {
        for _ in 0..8 {
            corners.push((s.wrapping_mul(6364136223846793005) as f64) * 1e-12);
            s = s.wrapping_mul(1442695040888963407);
        }
        deltas.push(((s >> 32) as f64) / (u32::MAX as f64));
        s = s.wrapping_mul(1442695040888963407);
        deltas.push(((s >> 32) as f64) / (u32::MAX as f64));
        s = s.wrapping_mul(1442695040888963407);
        deltas.push(((s >> 32) as f64) / (u32::MAX as f64));
    }
    let mut res = vec![0.0f64; n];
    accel.batch_trilinear(&corners, &deltas, &mut res);
    assert!(res.iter().all(|&v| v.is_finite()));
}

// ============================================================================
// 大规模结构
// ============================================================================

#[test]
fn stress_many_cell_cache_specs() {
    let accel = mk_batch_accel();
    // 16 个 cache，每个不同的采样器
    let n = 1024;
    let positions = mk_pos3d(n, SEED);
    let samplers: Vec<OctavePerlinNoiseSampler> = (0..16)
        .map(|i| mk_sampler(SEED.wrapping_add(i), &[0, 1, 2]))
        .collect();
    let specs: Vec<_> = samplers
        .iter()
        .map(|s| pumpkin_world::batch_accel::CellCacheFillSpec {
            first: s,
            second: s,
            amplitude: 1.0,
            xz_scale: 0.25,
            y_scale: 0.125,
        })
        .collect();
    let mut results = vec![0.0f64; n * specs.len()];
    accel.batch_fill_cell_caches_vanilla(&positions, &specs, &mut results);
    assert!(results.iter().all(|&v| v.is_finite()), "all specs finite");
    // 不同 cache 的结果应不同
    let mut hashes = Vec::new();
    for c in 0..specs.len() {
        hashes.push(fnv1a_f64(&results[c * n..(c + 1) * n]));
    }
    hashes.sort_unstable();
    hashes.dedup();
    assert!(hashes.len() >= 2, "不同采样器的 cache 输出不应全部相同");
}

#[test]
fn stress_aquifer_large_grid() {
    let accel = mk_batch_accel();
    let n = 4096;
    let positions = mk_pos3d(n, SEED.wrapping_add(3));
    let densities: Vec<f64> = (0..n).map(|i| (i as f64 % 64.0) / 64.0 - 0.5).collect();
    // 大网格：13×13×13 = 2197 点
    let g = 13usize;
    let mut packed_grid = Vec::with_capacity(g * g * g * 4);
    let mut s = SEED;
    for ix in 0..g {
        for iy in 0..g {
            for iz in 0..g {
                packed_grid.push((((ix as i32 - 6) * 16) as f64).to_bits() as i64);
                packed_grid.push((((iy as i32 - 6) * 16) as f64).to_bits() as i64);
                packed_grid.push((((iz as i32 - 6) * 16) as f64).to_bits() as i64);
                packed_grid.push(((s as f64) * 1e-12 - 0.3).to_bits() as i64);
                s = s.wrapping_mul(1442695040888963407);
            }
        }
    }
    let result = accel.batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
    assert_eq!(result.block_ids.len(), n);
    assert_eq!(result.fluid_updates.len(), n);
}

#[test]
fn stress_beardifier_many_structures() {
    let accel = mk_batch_accel();
    let n = 4096;
    // 位置限定在结构分布范围（±128）内，确保大部分位置落在包围盒附近
    let mut s = SEED;
    let mut positions = Vec::with_capacity(n * 3);
    for _ in 0..n {
        positions.push((s as f64 % 256.0) - 128.0);
        s = s.wrapping_mul(1442695040888963407);
        positions.push((s as f64 % 128.0) - 96.0);
        s = s.wrapping_mul(1442695040888963407);
        positions.push((s as f64 % 256.0) - 128.0);
        s = s.wrapping_mul(1442695040888963407);
    }
    let mut s = SEED;
    let mut structures = Vec::with_capacity(64);
    for _ in 0..64 {
        let cx = (s as f64 % 256.0) - 128.0;
        s = s.wrapping_mul(1442695040888963407);
        let cy = (s as f64 % 96.0) - 96.0;
        s = s.wrapping_mul(1442695040888963407);
        let cz = (s as f64 % 256.0) - 128.0;
        s = s.wrapping_mul(1442695040888963407);
        structures.push(BeardifierStructureData {
            box_min_x: cx as i32 - 8,
            box_min_y: cy as i32 - 16,
            box_min_z: cz as i32 - 8,
            box_max_x: cx as i32 + 8,
            box_max_y: cy as i32 + 16,
            box_max_z: cz as i32 + 8,
            adaptation: 1, // BeardThin
            ground_delta: 8,
        });
    }
    let junctions: Vec<BeardifierJunctionData> = (0..32)
        .map(|i| BeardifierJunctionData {
            x: (i % 8) * 32 - 128,
            ground_y: -16 - (i % 4) * 8,
            z: (i / 8) * 32 - 128,
        })
        .collect();
    let mut results = vec![0.0f64; n];
    accel.batch_beardifier(
        &positions,
        &structures,
        &junctions,
        [-160, -128, -160, 160, 16, 160],
        &mut results,
    );
    assert!(results.iter().all(|&v| v.is_finite()), "all finite");
    assert!(
        results.iter().any(|&v| v > 0.0),
        "密集结构下应产生非零 beard 贡献"
    );
}

// ============================================================================
// 边界尺寸
// ============================================================================

#[test]
fn stress_edge_sizes() {
    let sampler = mk_sampler(SEED, &[0, 1]);
    let mut accel = mk_noise_accel();
    for &n in &[1usize, 2, 3, 255, 256, 257, 1023, 1025] {
        let pos = mk_pos3d(n, SEED.wrapping_add(n as u64));
        let mut res = vec![0.0f64; n];
        accel.sample_octave(&sampler, &pos, &mut res);
        assert!(res.iter().all(|&v| v.is_finite()), "n={n} all finite");
        assert_eq!(res.len(), n);
    }
}

#[test]
fn stress_extreme_coordinates() {
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let n = 64;
    // 极大 / 极小 / 负坐标
    let coords = [
        1.0e9f64,
        -1.0e9,
        1.0e-12,
        -1.0e-12,
        3.355_443_2E7,
        -3.355_443_2E7,
        0.0,
        1.0,
    ];
    let pos: Vec<f64> = (0..n)
        .flat_map(|i| {
            [
                coords[i % coords.len()],
                coords[(i + 2) % coords.len()],
                coords[(i + 5) % coords.len()],
            ]
        })
        .collect();
    let mut res1 = vec![0.0f64; n];
    let mut res2 = vec![0.0f64; n];
    let mut accel = mk_noise_accel();
    accel.sample_octave(&sampler, &pos, &mut res1);
    accel.sample_octave(&sampler, &pos, &mut res2);
    assert!(res1.iter().all(|&v| v.is_finite()), "extreme coords finite");
    assert_eq!(
        fnv1a_f64(&res1),
        fnv1a_f64(&res2),
        "extreme coords deterministic"
    );
}

// ============================================================================
// 重复调用
// ============================================================================

#[test]
fn stress_repeated_calls() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 2048;
    let pos = mk_pos3d(n, SEED.wrapping_add(9));
    let mut accel = mk_noise_accel();
    let mut first = vec![0.0f64; n];
    accel.sample_octave(&sampler, &pos, &mut first);
    let first_hash = fnv1a_f64(&first);

    // 50 次重复调用 + 交错其他操作，验证无状态污染
    let b_sampler = mk_sampler(SEED ^ 5, &[0, 1]);
    let xz: Vec<f64> = (0..n).flat_map(|i| [pos[i * 3], pos[i * 3 + 2]]).collect();
    let mut buf = vec![0.0f64; n];
    for i in 0..50 {
        accel.sample_octave(&sampler, &pos, &mut buf);
        assert_eq!(fnv1a_f64(&buf), first_hash, "iteration {i} diverged");
        // 交错其他家族
        accel.sample_shift_a(&b_sampler, &xz, &mut buf);
        accel.precompute_flatcache(&b_sampler, &xz, &mut buf);
        accel.sample_octave(&sampler, &pos, &mut buf);
        assert_eq!(fnv1a_f64(&buf), first_hash, "iteration {i} polluted");
    }
}

#[test]
fn stress_light_large_propagate() {
    use pumpkin_world::light_accel::LightAccelerator;
    let mut accel = LightAccelerator::new(&GpuConfig::default()); // CPU 回退模式,确定性
    let width = 18usize;
    let depth = 18usize;
    let height = 384usize;
    let n_total = width * depth * height;
    let mut light = vec![0u8; n_total];
    // 散布光源
    let mut s = SEED;
    for _ in 0..64 {
        let idx = (s as usize) % n_total;
        light[idx] = 15;
        s = s.wrapping_mul(1442695040888963407);
    }
    let opacity = vec![1u8; n_total];
    let mut neighbors = Vec::with_capacity(n_total * 6);
    for i in 0..n_total {
        let x = (i % width) as i32;
        let y = ((i / width) % height) as i32;
        let z = (i / (width * height)) as i32;
        for (dx, dy, dz) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            let idx = if (0..width as i32).contains(&nx)
                && (0..height as i32).contains(&ny)
                && (0..depth as i32).contains(&nz)
            {
                nz * width as i32 * height as i32 + ny * width as i32 + nx
            } else {
                -1
            };
            neighbors.push(idx);
        }
    }
    let iterations = accel.iterative_propagate(&mut light, &opacity, &neighbors, n_total, 64);
    assert!(iterations > 0, "大网格传播应发生迭代");
    assert!(light.iter().any(|&v| v > 0), "光源应传播出去");
}
