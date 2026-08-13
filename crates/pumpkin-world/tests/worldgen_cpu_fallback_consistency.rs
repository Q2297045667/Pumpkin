//! 强制 CPU 回退路径一致性测试。
//!
//! 显式使用 `enabled: false` 配置，验证每个加速器的 CPU 回退实现与
//! 独立参考计算逐位一致。这些测试在任何环境（有/无 GPU）下都确定性地
//! 覆盖 CPU 回退分支，与 GPU 路径测试互补。
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
use pumpkin_data::chunk::DoublePerlinNoiseParameters;
use pumpkin_gpu::noise::batch_cell::{BeardifierJunctionData, BeardifierStructureData};
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::batch_accel::{BatchAccelerator, CellCacheFillSpec};
use pumpkin_world::generation::noise::perlin::DoublePerlinNoiseSampler;
use pumpkin_world::light_accel::LightAccelerator;
use pumpkin_world::noise_accel::NoiseAccelerator;

const SEED: u64 = 138_782_381_985_206;

fn fnv1a_u8(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
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

fn mk_test_double_perlin(seed: u64) -> DoublePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let mut g = RandomGenerator::Xoroshiro(r);
    let params = DoublePerlinNoiseParameters::new(
        0,
        0,
        &[1.0f64],
        0,
        0,
        DoublePerlinNoiseSampler::get_amplitude(&[1.0f64]),
    );
    DoublePerlinNoiseSampler::from_params(&mut g, &params, false)
}

// ============================================================================
// NoiseAccelerator — CPU 回退
// ============================================================================

#[test]
fn noise_accel_cpu_fallback_all_families() {
    let config = GpuConfig::default(); // enabled=false → inner=None
    let mut accel = NoiseAccelerator::new(&config);
    assert!(
        !accel.is_active(),
        "disabled config must produce inactive accel"
    );

    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let n = 256;
    let pos = mk_pos3d(n);
    let xz: Vec<f64> = (0..n).flat_map(|i| [pos[i * 3], pos[i * 3 + 2]]).collect();

    // octave
    let mut out = vec![0.0f64; n];
    accel.sample_octave(&sampler, &pos, &mut out);
    for i in 0..n {
        assert_eq!(
            out[i],
            sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]),
            "octave cpu fallback[{i}]"
        );
    }

    // double perlin
    let b = mk_sampler(SEED ^ 1, &[0, 1]);
    let c = 1.0181268882175227f64;
    accel.sample_double_perlin(&sampler, &b, 0.5, &pos, &mut out);
    for i in 0..n {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        assert_eq!(
            out[i],
            (sampler.sample(x, y, z) + b.sample(x * c, y * c, z * c)) * 0.5,
            "double perlin cpu fallback[{i}]"
        );
    }

    // shift a / shift b
    accel.sample_shift_a(&sampler, &xz, &mut out);
    for i in 0..n {
        assert_eq!(
            out[i],
            sampler.sample(xz[i * 2] * 0.25, 0.0, xz[i * 2 + 1] * 0.25) * 4.0,
            "shift_a cpu fallback[{i}]"
        );
    }
    accel.sample_shift_b(&sampler, &xz, &mut out);
    for i in 0..n {
        assert_eq!(
            out[i],
            sampler.sample(xz[i * 2 + 1] * 0.25, 0.0, xz[i * 2] * 0.25) * 4.0,
            "shift_b cpu fallback[{i}]"
        );
    }

    // flatcache
    accel.precompute_flatcache(&sampler, &xz, &mut out);
    for i in 0..n {
        assert_eq!(
            out[i],
            sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]),
            "flatcache cpu fallback[{i}]"
        );
    }

    // trilinear
    let corners: Vec<f64> = (0..n * 8).map(|i| (i as f64) * 1e-3).collect();
    let deltas: Vec<f64> = (0..n * 3).map(|i| (i as f64 % 7.0) / 7.0).collect();
    accel.batch_trilinear(&corners, &deltas, &mut out);
    for i in 0..n {
        let bb = i * 8;
        let dx = deltas[i * 3];
        let dy = deltas[i * 3 + 1];
        let dz = deltas[i * 3 + 2];
        let expected = corners[bb] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + corners[bb + 1] * dx * (1.0 - dy) * (1.0 - dz)
            + corners[bb + 2] * (1.0 - dx) * dy * (1.0 - dz)
            + corners[bb + 3] * dx * dy * (1.0 - dz)
            + corners[bb + 4] * (1.0 - dx) * (1.0 - dy) * dz
            + corners[bb + 5] * dx * (1.0 - dy) * dz
            + corners[bb + 6] * (1.0 - dx) * dy * dz
            + corners[bb + 7] * dx * dy * dz;
        assert_eq!(out[i], expected, "trilinear cpu fallback[{i}]");
    }
}

// ============================================================================
// BatchAccelerator — CPU 回退
// ============================================================================

#[test]
fn batch_accel_cpu_fallback_cell_cache_vanilla() {
    let accel = BatchAccelerator::new(&GpuConfig::default());
    let n = 128;
    let positions = mk_pos3d(n);
    let dbl = mk_test_double_perlin(SEED);
    let specs = vec![CellCacheFillSpec {
        first: dbl.first_sampler(),
        second: dbl.second_sampler(),
        amplitude: dbl.amplitude(),
        xz_scale: 0.25,
        y_scale: 0.125,
    }];
    let mut results = vec![0.0f64; n];
    accel.batch_fill_cell_caches_vanilla(&positions, &specs, &mut results);
    for i in 0..n {
        let x = positions[i * 3] * 0.25;
        let y = positions[i * 3 + 1] * 0.125;
        let z = positions[i * 3 + 2] * 0.25;
        assert_eq!(results[i], dbl.sample(x, y, z), "cell cache cpu[{i}]");
    }
}

#[test]
fn batch_accel_cpu_fallback_aquifer_and_beardifier() {
    let accel = BatchAccelerator::new(&GpuConfig::default());
    let n = 64;
    let positions = mk_pos3d(n);
    let densities: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3 - 0.5).collect();

    // 最小 4 点网格
    let mut packed_grid = Vec::new();
    for i in 0..4 {
        packed_grid.push(((i as f64) * 20.0).to_bits() as i64);
        packed_grid.push((64.0f64).to_bits() as i64);
        packed_grid.push(((i as f64) * 20.0).to_bits() as i64);
        packed_grid.push(0.3f64.to_bits() as i64);
    }
    let result = accel.batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
    assert_eq!(result.block_ids.len(), n);
    assert_eq!(result.fluid_updates.len(), n);

    let structures = [BeardifierStructureData {
        box_min_x: -8,
        box_min_y: -16,
        box_min_z: -8,
        box_max_x: 8,
        box_max_y: 0,
        box_max_z: 8,
        adaptation: 1, // BeardThin
        ground_delta: 8,
    }];
    let junctions = [BeardifierJunctionData {
        x: 0,
        ground_y: 0,
        z: 0,
    }];
    // 位置位于结构包围盒附近，确保产生非零贡献
    let near_positions = vec![
        0.0, 0.0, 0.0, // 中心
        3.0, -8.0, 3.0, // 盒内
        -3.0, -12.0, -3.0, // 盒内
        20.0, 0.0, 20.0, // 盒外
    ];
    let mut beard = vec![0.0f64; 4];
    accel.batch_beardifier(
        &near_positions,
        &structures,
        &junctions,
        [-16, -24, -16, 16, 8, 16],
        &mut beard,
    );
    assert!(beard.iter().all(|v| v.is_finite()));
    assert!(
        beard.iter().take(3).any(|&v| v > 0.0),
        "结构包围盒内的位置应产生非零贡献"
    );
}

// ============================================================================
// LightAccelerator — CPU 回退
// ============================================================================

#[test]
fn light_accel_cpu_fallback_sky_fill_and_scan() {
    let mut accel = LightAccelerator::new(&GpuConfig::default());
    assert!(!accel.is_active());

    let n = 16usize;
    let h = 64usize;
    let mut s = SEED;
    let hm: Vec<i32> = (0..n)
        .map(|_| {
            let v = ((s % 40) + 48) as i32;
            s = s.wrapping_mul(1442695040888963407);
            v
        })
        .collect();
    let op: Vec<u8> = (0..n * h)
        .map(|_| {
            let v = (s % 16) as u8;
            s = s.wrapping_mul(1442695040888963407);
            v
        })
        .collect();

    // sky fill
    let mut gpu_sky = vec![0u8; n * h];
    accel.batch_sky_fill(&hm, &op, &mut gpu_sky, n, h);
    let mut cpu_sky = vec![0u8; n * h];
    for col in 0..n {
        let top = hm[col];
        for y in (top + 1)..h as i32 {
            cpu_sky[col * h + y as usize] = 15;
        }
        let mut lt: u8 = 15;
        for y in (0..=top).rev() {
            let i = col * h + y as usize;
            lt = lt.saturating_sub(op[i]);
            cpu_sky[i] = lt;
        }
    }
    assert_eq!(
        fnv1a_u8(&cpu_sky),
        fnv1a_u8(&gpu_sky),
        "sky fill cpu fallback"
    );

    // block scan
    let lum: Vec<u8> = (0..n * h).map(|i| (i % 17) as u8).collect();
    let mut gpu_bl = vec![0u8; n * h];
    let sources = accel.batch_block_scan(&lum, &mut gpu_bl, n * h);
    let mut cpu_sources = Vec::new();
    for (i, &v) in lum.iter().enumerate() {
        if v > 0 {
            cpu_sources.push(i as i32);
        }
    }
    assert_eq!(sources, cpu_sources, "block scan sources");
    assert_eq!(gpu_bl, lum, "block scan values");
}

#[test]
fn light_accel_cpu_fallback_propagate() {
    let mut accel = LightAccelerator::new(&GpuConfig::default());
    let n = 27usize; // 3×3×3
    let mut light = vec![0u8; n];
    light[13] = 15; // 中心光源
    let opacity = vec![1u8; n];
    // 邻居表：为 3×3×3 网格构建 6 邻居
    let mut neighbors = Vec::with_capacity(n * 6);
    for i in 0..n {
        let x = (i % 3) as i32;
        let y = ((i / 3) % 3) as i32;
        let z = (i / 9) as i32;
        for (dx, dy, dz) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            let idx = if (0..3).contains(&nx) && (0..3).contains(&ny) && (0..3).contains(&nz) {
                nz * 9 + ny * 3 + nx
            } else {
                -1
            };
            neighbors.push(idx);
        }
    }

    let mut gpu_light = light.clone();
    let iterations = accel.iterative_propagate(&mut gpu_light, &opacity, &neighbors, n, 32);
    assert!(iterations > 0, "propagation should iterate at least once");

    // CPU 参考
    let mut cpu_light = light.clone();
    for _ in 0..32 {
        let mut changed = false;
        for i in 0..n {
            let cur = cpu_light[i];
            let mut best = cur;
            for d in 0..6 {
                let ni = neighbors[i * 6 + d] as usize;
                if ni < n {
                    let nl = cpu_light[ni];
                    let no = opacity[ni];
                    let prop = if nl > 1 + no { nl - 1 - no } else { 0 };
                    if prop > best {
                        best = prop;
                    }
                }
            }
            if best > cur {
                cpu_light[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    assert_eq!(
        fnv1a_u8(&cpu_light),
        fnv1a_u8(&gpu_light),
        "iterative propagate cpu fallback"
    );
}
