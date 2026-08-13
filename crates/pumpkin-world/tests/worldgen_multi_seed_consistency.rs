//! 世界生成 GPU 路径多种子 / 多配置一致性扫描。
//!
//! 对每个噪声家族、批量操作以多组种子与八度配置对比加速器输出与 CPU 参考指纹。
//! 有 GPU 时走 GPU kernel（真硬件验证），无 GPU 时走 CPU 回退——两种情况下
//! 输出都必须与 CPU 参考逐位一致。
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
use pumpkin_world::noise_accel::NoiseAccelerator;

/// 多组种子覆盖确定性与哈希质量。
const SEEDS: [u64; 5] = [0, 1, 42, 138_782_381_985_206, 1_779_920_288_596_261_407];

/// 多组八度配置覆盖不同长度与正负八度。
const OCTAVE_SETS: [&[i32]; 4] = [&[0], &[-2, 0, 2], &[0, 1, 2, 3, 4], &[-4, -2, 0, 2, 4]];

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

fn fnv1a_i32(data: &[i32]) -> u64 {
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

fn mk_positions_3d(n: usize, seed: u64) -> Vec<f64> {
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

fn mk_positions_2d(n: usize, seed: u64) -> Vec<f64> {
    let mut p = Vec::with_capacity(n * 2);
    let mut s = seed;
    for _ in 0..n {
        p.push((s.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8);
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
// 噪声家族多种子扫描
// ============================================================================

#[test]
fn octave_multi_seed_multi_config() {
    let mut accel = mk_noise_accel();
    for &seed in &SEEDS {
        for octaves in OCTAVE_SETS {
            let sampler = mk_sampler(seed, octaves);
            let n = 384;
            let pos = mk_positions_3d(n, seed);
            let mut cpu = vec![0.0f64; n];
            let mut gpu = vec![0.0f64; n];
            for i in 0..n {
                cpu[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
            }
            accel.sample_octave(&sampler, &pos, &mut gpu);
            assert_eq!(
                fnv1a_f64(&cpu),
                fnv1a_f64(&gpu),
                "octave mismatch: seed={seed}, octaves={octaves:?}"
            );
        }
    }
}

#[test]
fn double_perlin_multi_seed() {
    let mut accel = mk_noise_accel();
    let c = 1.0181268882175227f64;
    for &seed in &SEEDS {
        let a = mk_sampler(seed, &[0, 1, 2]);
        let b = mk_sampler(seed ^ 0x9E37_79B9, &[-1, 0, 1]);
        let amp = 0.5;
        let n = 512;
        let pos = mk_positions_3d(n, seed);
        let mut cpu = vec![0.0f64; n];
        let mut gpu = vec![0.0f64; n];
        for i in 0..n {
            cpu[i] = (a.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2])
                + b.sample(pos[i * 3] * c, pos[i * 3 + 1] * c, pos[i * 3 + 2] * c))
                * amp;
        }
        accel.sample_double_perlin(&a, &b, amp, &pos, &mut gpu);
        assert_eq!(
            fnv1a_f64(&cpu),
            fnv1a_f64(&gpu),
            "double perlin mismatch: seed={seed}"
        );
    }
}

#[test]
fn shift_a_multi_seed() {
    let mut accel = mk_noise_accel();
    for &seed in &SEEDS {
        let sampler = mk_sampler(seed, &[0, 1, 2]);
        let n = 512;
        let xz = mk_positions_2d(n, seed);
        let mut cpu = vec![0.0f64; n];
        let mut gpu = vec![0.0f64; n];
        for i in 0..n {
            cpu[i] = sampler.sample(xz[i * 2] * 0.25, 0.0, xz[i * 2 + 1] * 0.25) * 4.0;
        }
        accel.sample_shift_a(&sampler, &xz, &mut gpu);
        assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "shift_a: seed={seed}");
    }
}

#[test]
fn shift_b_multi_seed() {
    let mut accel = mk_noise_accel();
    for &seed in &SEEDS {
        let sampler = mk_sampler(seed, &[0, 1, 2]);
        let n = 512;
        let zx = mk_positions_2d(n, seed);
        let mut cpu = vec![0.0f64; n];
        let mut gpu = vec![0.0f64; n];
        for i in 0..n {
            cpu[i] = sampler.sample(zx[i * 2 + 1] * 0.25, 0.0, zx[i * 2] * 0.25) * 4.0;
        }
        accel.sample_shift_b(&sampler, &zx, &mut gpu);
        assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "shift_b: seed={seed}");
    }
}

#[test]
fn flatcache_multi_seed() {
    let mut accel = mk_noise_accel();
    for &seed in &SEEDS {
        let sampler = mk_sampler(seed, &[0, 1, 2, 3]);
        let n = 512;
        let xz = mk_positions_2d(n, seed);
        let mut cpu = vec![0.0f64; n];
        let mut gpu = vec![0.0f64; n];
        for i in 0..n {
            cpu[i] = sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
        }
        accel.precompute_flatcache(&sampler, &xz, &mut gpu);
        assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "flatcache: seed={seed}");
    }
}

#[test]
fn surface_multi_seed() {
    let mut accel = mk_noise_accel();
    for &seed in &SEEDS {
        let surface_a = mk_sampler(seed, &[0, 1]);
        let surface_b = mk_sampler(seed.wrapping_add(3), &[0, 1]);
        let secondary_a = mk_sampler(seed.wrapping_add(5), &[0]);
        let secondary_b = mk_sampler(seed.wrapping_add(7), &[0]);

        let (start_x, start_z) = (-8, -8);
        let n = 256;
        let mut xz = Vec::with_capacity(n * 2);
        for lx in 0i32..16 {
            for lz in 0i32..16 {
                xz.push((start_x + lx) as f64);
                xz.push((start_z + lz) as f64);
            }
        }
        let c = 1.0181268882175227f64;
        let mut cpu_surf = vec![0.0f64; n];
        let mut cpu_sec = vec![0.0f64; n];
        for i in 0..n {
            let x = xz[i * 2];
            let z = xz[i * 2 + 1];
            cpu_surf[i] = (surface_a.sample(x, 0.0, z) + surface_b.sample(x * c, 0.0, z * c)) * 0.7;
            cpu_sec[i] =
                (secondary_a.sample(x, 0.0, z) + secondary_b.sample(x * c, 0.0, z * c)) * 0.3;
        }
        let gpu = accel.precompute_surface(
            &surface_a,
            &surface_b,
            0.7,
            &secondary_a,
            &secondary_b,
            0.3,
            start_x,
            start_z,
        );
        assert_eq!(
            fnv1a_f64(&cpu_surf),
            fnv1a_f64(&*gpu.surface),
            "surface: seed={seed}"
        );
        assert_eq!(
            fnv1a_f64(&cpu_sec),
            fnv1a_f64(&*gpu.secondary),
            "secondary: seed={seed}"
        );
    }
}

// ============================================================================
// 批量操作多种子扫描
// ============================================================================

#[test]
fn trilinear_multi_seed() {
    let accel = mk_batch_accel();
    for &seed in &SEEDS {
        let n = 256;
        let mut s = seed;
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
        accel.batch_trilinear(&corners, &deltas, &mut gpu);
        assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "trilinear: seed={seed}");
    }
}

#[test]
fn cell_cache_vanilla_multi_seed() {
    let accel = mk_batch_accel();
    for &seed in &SEEDS {
        let n = 256;
        let positions = mk_positions_3d(n, seed);
        let dbl1 = mk_test_double_perlin(seed);
        let dbl2 = mk_test_double_perlin(seed.wrapping_add(7));
        let specs = vec![
            CellCacheFillSpec {
                first: dbl1.first_sampler(),
                second: dbl1.second_sampler(),
                amplitude: dbl1.amplitude(),
                xz_scale: 0.25,
                y_scale: 0.125,
            },
            CellCacheFillSpec {
                first: dbl2.first_sampler(),
                second: dbl2.second_sampler(),
                amplitude: dbl2.amplitude(),
                xz_scale: 1.0,
                y_scale: 1.0,
            },
        ];
        let mut results = vec![0.0f64; n * 2];
        accel.batch_fill_cell_caches_vanilla(&positions, &specs, &mut results);
        let mut reference = vec![0.0f64; n * 2];
        for (c, spec) in specs.iter().enumerate() {
            let out = &mut reference[c * n..(c + 1) * n];
            for (i, res) in out.iter_mut().enumerate() {
                let x = positions[i * 3] * spec.xz_scale;
                let y = positions[i * 3 + 1] * spec.y_scale;
                let z = positions[i * 3 + 2] * spec.xz_scale;
                *res = if c == 0 {
                    dbl1.sample(x, y, z)
                } else {
                    dbl2.sample(x, y, z)
                };
            }
        }
        assert_eq!(
            fnv1a_f64(&results),
            fnv1a_f64(&reference),
            "cell cache vanilla: seed={seed}"
        );
    }
}

#[test]
fn aquifer_multi_seed() {
    let accel = mk_batch_accel();
    for &seed in &SEEDS {
        let n = 128;
        let positions = mk_positions_3d(n, seed);
        let mut densities = vec![0.0f64; n];
        let mut s = seed;
        for d in &mut densities {
            *d = (s as f64) * 1e-12 - 0.5;
            s = s.wrapping_mul(1442695040888963407);
        }
        // 4×4×4 网格
        let mut packed_grid = Vec::with_capacity(64 * 4);
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    packed_grid.push((((ix - 1) * 24) as f64).to_bits() as i64);
                    packed_grid.push((((iy - 2) * 24) as f64).to_bits() as i64);
                    packed_grid.push((((iz - 1) * 24) as f64).to_bits() as i64);
                    packed_grid.push(((s as f64) * 1e-12 - 0.3).to_bits() as i64);
                    s = s.wrapping_mul(1442695040888963407);
                }
            }
        }

        // CPU 参考：4-NN 搜索（与 cpu_aquifer_apply 逻辑一致）
        let m = packed_grid.len() / 4;
        let grid_positions: Vec<[f64; 3]> = (0..m)
            .map(|i| {
                [
                    f64::from_bits(packed_grid[i * 4] as u64),
                    f64::from_bits(packed_grid[i * 4 + 1] as u64),
                    f64::from_bits(packed_grid[i * 4 + 2] as u64),
                ]
            })
            .collect();
        let grid_densities: Vec<f64> = (0..m)
            .map(|i| f64::from_bits(packed_grid[i * 4 + 3] as u64))
            .collect();
        let mut cpu_block_ids = vec![0i32; n];
        let mut cpu_fluid = vec![0u8; n];
        for i in 0..n {
            let qx = positions[i * 3];
            let qy = positions[i * 3 + 1];
            let qz = positions[i * 3 + 2];
            let mut best_dist = [f64::INFINITY; 4];
            let mut best_idx = [0usize; 4];
            for j in 0..m {
                let dx = qx - grid_positions[j][0];
                let dy = qy - grid_positions[j][1];
                let dz = qz - grid_positions[j][2];
                let dist = dx * dx + dy * dy + dz * dz;
                for k in 0..4 {
                    if dist < best_dist[k] {
                        for kk in (k + 1..4).rev() {
                            best_idx[kk] = best_idx[kk - 1];
                            best_dist[kk] = best_dist[kk - 1];
                        }
                        best_idx[k] = j;
                        best_dist[k] = dist;
                        break;
                    }
                }
            }
            let barrier: f64 = best_idx.iter().map(|&j| grid_densities[j]).sum::<f64>() / 4.0;
            let eff = densities[i] + barrier * 0.3;
            if eff > 0.0 {
                cpu_block_ids[i] = 1;
            } else if qy < -10000.0 {
                cpu_block_ids[i] = 2;
                cpu_fluid[i] = 1;
            }
        }

        let gpu = accel.batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
        assert_eq!(
            fnv1a_i32(&cpu_block_ids),
            fnv1a_i32(&gpu.block_ids),
            "aquifer block_ids: seed={seed}"
        );
        assert_eq!(cpu_fluid, gpu.fluid_updates, "aquifer fluid: seed={seed}");
    }
}

#[test]
fn beardifier_multi_seed() {
    let accel = mk_batch_accel();
    for &seed in &SEEDS {
        let n = 256;
        // 位置限定在结构分布范围（±64）内
        let mut ps = seed;
        let mut positions = Vec::with_capacity(n * 3);
        for _ in 0..n {
            positions.push((ps as f64 % 128.0) - 64.0);
            ps = ps.wrapping_mul(1442695040888963407);
            positions.push((ps as f64 % 64.0) - 48.0);
            ps = ps.wrapping_mul(1442695040888963407);
            positions.push((ps as f64 % 128.0) - 64.0);
            ps = ps.wrapping_mul(1442695040888963407);
        }
        let mut s = seed;
        let mut structures = Vec::new();
        for _ in 0..3 {
            let cx = (s as f64 % 64.0) - 32.0;
            s = s.wrapping_mul(1442695040888963407);
            let cy = (s as f64 % 32.0) - 48.0;
            s = s.wrapping_mul(1442695040888963407);
            let cz = (s as f64 % 64.0) - 32.0;
            s = s.wrapping_mul(1442695040888963407);
            structures.push(BeardifierStructureData {
                box_min_x: cx as i32 - 16,
                box_min_y: cy as i32 - 24,
                box_min_z: cz as i32 - 16,
                box_max_x: cx as i32 + 16,
                box_max_y: cy as i32 + 24,
                box_max_z: cz as i32 + 16,
                adaptation: 1 + (s % 2) as i32, // BeardThin / BeardBox
                ground_delta: 8,
            });
        }
        let junctions = vec![
            BeardifierJunctionData {
                x: 0,
                ground_y: 0,
                z: 0,
            },
            BeardifierJunctionData {
                x: 48,
                ground_y: -16,
                z: 48,
            },
        ];
        let affected_box = [-80, -80, -80, 80, 16, 80];

        // CPU 参考（vanilla `Beardifier::sample` 逐位等价）
        let mut cpu = vec![0.0f64; n];
        for i in 0..n {
            let x = positions[i * 3] as i32;
            let y = positions[i * 3 + 1] as i32;
            let z = positions[i * 3 + 2] as i32;
            if !(-80..=80).contains(&x) || !(-80..=16).contains(&y) || !(-80..=80).contains(&z) {
                cpu[i] = 0.0;
                continue;
            }
            let mut weight = 0.0;
            for st in &structures {
                let dx = 0.max((st.box_min_x - x).max(x - st.box_max_x));
                let dz = 0.max((st.box_min_z - z).max(z - st.box_max_z));
                let ground_y = st.box_min_y + st.ground_delta;
                let dy_to_ground = y - ground_y;
                let dy = match st.adaptation {
                    0 => 0,
                    1 | 3 => dy_to_ground,
                    2 => 0.max((ground_y - y).max(y - st.box_max_y)),
                    _ => 0.max((st.box_min_y - y).max(y - st.box_max_y)),
                };
                let contrib = match st.adaptation {
                    0 => 0.0,
                    3 => ref_bury(f64::from(dx), f64::from(dy) / 2.0, f64::from(dz)),
                    1 | 2 => ref_beard(dx, dy, dz, dy_to_ground) * 0.8,
                    _ => {
                        ref_bury(
                            f64::from(dx) / 2.0,
                            f64::from(dy) / 2.0,
                            f64::from(dz) / 2.0,
                        ) * 0.8
                    }
                };
                weight += contrib;
            }
            for j in &junctions {
                let jdx = x - j.x;
                let jdy = y - j.ground_y;
                let jdz = z - j.z;
                weight += ref_beard(jdx, jdy, jdz, jdy) * 0.4;
            }
            cpu[i] = weight;
        }

        let mut gpu = vec![0.0f64; n];
        accel.batch_beardifier(&positions, &structures, &junctions, affected_box, &mut gpu);
        assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&gpu), "beardifier: seed={seed}");
    }
}

/// vanilla `get_beard_contribution` 逐位等价参考。
fn ref_beard(dx: i32, dy: i32, dz: i32, y_to_ground: i32) -> f64 {
    let xi = dx + 12;
    let yi = dy + 12;
    let zi = dz + 12;
    if (0..24).contains(&xi) && (0..24).contains(&yi) && (0..24).contains(&zi) {
        let dy_off = f64::from(y_to_ground) + 0.5;
        let dsq = f64::from(dx).powi(2) + dy_off.powi(2) + f64::from(dz).powi(2);
        let value = -dy_off * (dsq / 2.0).sqrt().recip() / 2.0;
        let kdsq = f64::from(dx).powi(2) + (f64::from(dy) + 0.5).powi(2) + f64::from(dz).powi(2);
        value * std::f64::consts::E.powf(-kdsq / 16.0)
    } else {
        0.0
    }
}

/// vanilla `get_bury_contribution` 逐位等价参考。
fn ref_bury(dx: f64, dy: f64, dz: f64) -> f64 {
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    if distance < 0.0 {
        1.0
    } else if distance > 6.0 {
        0.0
    } else {
        1.0 - distance / 6.0
    }
}
