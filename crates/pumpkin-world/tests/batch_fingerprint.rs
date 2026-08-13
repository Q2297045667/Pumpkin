//! 批量 GPU 加速指纹测试 — Cell Cache（vanilla 规格）、Aquifer、Beardifier。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::semicolon_inside_block,
    clippy::semicolon_outside_block,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_data::chunk::DoublePerlinNoiseParameters;
use pumpkin_gpu::noise::batch_cell::{
    AquiferBatchResult, BeardifierJunctionData, BeardifierStructureData,
};
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::batch_accel::{BatchAccelerator, CellCacheFillSpec};
use pumpkin_world::generation::noise::perlin::DoublePerlinNoiseSampler;

const SEED: u64 = 138_782_381_985_206;

fn make_jit_accel() -> BatchAccelerator {
    BatchAccelerator::new(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        batch_acceleration: true,
        jit_enabled: true,
        jit_max_unroll: 16,
        ..Default::default()
    })
}

// ============================================================================
// FNV-1a 哈希辅助函数
// ============================================================================

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

fn fnv1a_i32(d: &[i32]) -> u64 {
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
// 辅助函数
// ============================================================================

fn make_accel() -> BatchAccelerator {
    BatchAccelerator::new(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    })
}

fn make_positions_3d(n: usize, seed: u64) -> Vec<f64> {
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

// ============================================================================
// CPU 参考实现（用于与 GPU 输出比对）
// ============================================================================

/// CPU 参考：含水层 4-NN 搜索。
/// 与 `batch_accel::cpu_aquifer_apply` 保持逻辑一致。
fn cpu_aquifer_ref(
    positions: &[f64],
    densities: &[f64],
    packed_grid: &[i64],
    fluid_level: f64,
    barrier_scale: f64,
) -> AquiferBatchResult {
    let n = densities.len();
    let mut block_ids = vec![0i32; n];
    let mut fluid_updates = vec![0u8; n];

    let m = packed_grid.len() / 4;
    if m < 4 {
        return AquiferBatchResult {
            block_ids,
            fluid_updates,
        };
    }

    // 预提取网格位置和密度
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

    for i in 0..n {
        let qx = positions[i * 3];
        let qy = positions[i * 3 + 1];
        let qz = positions[i * 3 + 2];
        let q_density = densities[i];

        // 4-NN 线性搜索
        let mut best_idx = [0usize; 4];
        let mut best_dist = [f64::INFINITY; 4];

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

        let barrier_sum: f64 = best_idx.iter().map(|&idx| grid_densities[idx]).sum();
        let barrier_density = barrier_sum / 4.0;
        let effective = q_density + barrier_density * barrier_scale;

        if effective > 0.0 {
            block_ids[i] = 1; // 石头
            fluid_updates[i] = 0;
        } else if qy < fluid_level {
            block_ids[i] = 2; // 水
            fluid_updates[i] = 1;
        } else {
            block_ids[i] = 0; // 空气
            fluid_updates[i] = 0;
        }
    }

    AquiferBatchResult {
        block_ids,
        fluid_updates,
    }
}

/// 与 vanilla `Beardifier::get_beard_contribution` 逐位一致的参考实现。
fn ref_beard_contrib(dx: i32, dy: i32, dz: i32, y_to_ground: i32) -> f64 {
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

/// 与 vanilla `Beardifier::get_bury_contribution` 逐位一致。
fn ref_bury_contrib(dx: f64, dy: f64, dz: f64) -> f64 {
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    if distance < 0.0 {
        1.0
    } else if distance > 6.0 {
        0.0
    } else {
        1.0 - distance / 6.0
    }
}

/// CPU 参考：vanilla `Beardifier::sample` 的逐位等价实现。
fn cpu_beardifier_ref(
    positions: &[f64],
    structures: &[BeardifierStructureData],
    junctions: &[BeardifierJunctionData],
    affected_box: [i32; 6],
    results: &mut [f64],
) {
    let [aminx, aminy, aminz, amaxx, amaxy, amaxz] = affected_box;
    for i in 0..results.len() {
        let x = positions[i * 3] as i32;
        let y = positions[i * 3 + 1] as i32;
        let z = positions[i * 3 + 2] as i32;

        if x < aminx || x > amaxx || y < aminy || y > amaxy || z < aminz || z > amaxz {
            results[i] = 0.0;
            continue;
        }

        let mut weight = 0.0;
        for s in structures {
            let dx = 0.max((s.box_min_x - x).max(x - s.box_max_x));
            let dz = 0.max((s.box_min_z - z).max(z - s.box_max_z));
            let ground_y = s.box_min_y + s.ground_delta;
            let dy_to_ground = y - ground_y;
            let dy = match s.adaptation {
                0 => 0,
                1 | 3 => dy_to_ground,
                2 => 0.max((ground_y - y).max(y - s.box_max_y)),
                _ => 0.max((s.box_min_y - y).max(y - s.box_max_y)),
            };
            let contrib = match s.adaptation {
                0 => 0.0,
                3 => ref_bury_contrib(f64::from(dx), f64::from(dy) / 2.0, f64::from(dz)),
                1 | 2 => ref_beard_contrib(dx, dy, dz, dy_to_ground) * 0.8,
                _ => {
                    ref_bury_contrib(
                        f64::from(dx) / 2.0,
                        f64::from(dy) / 2.0,
                        f64::from(dz) / 2.0,
                    ) * 0.8
                }
            };
            weight += contrib;
        }
        for j in junctions {
            let jdx = x - j.x;
            let jdy = y - j.ground_y;
            let jdz = z - j.z;
            weight += ref_beard_contrib(jdx, jdy, jdz, jdy) * 0.4;
        }

        results[i] = weight;
    }
}

// ============================================================================
// 测试用例
// ============================================================================

/// 构造测试用 `DoublePerlinNoiseSampler`。
fn make_test_double_perlin(seed: u64) -> DoublePerlinNoiseSampler {
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

/// 多 cache 批量填充（vanilla `DoublePerlin` 语义）与 `DoublePerlinNoiseSampler::sample` 逐位一致。
/// 验证：每 cache 独立采样器、NoiseData 缩放、GPU/CPU 路径一致性。
#[test]
fn cell_cache_fill_vanilla_double_perlin_parity() {
    let n = 256;
    let positions = make_positions_3d(n, SEED);

    let dbl1 = make_test_double_perlin(SEED);
    let dbl2 = make_test_double_perlin(SEED.wrapping_add(7));

    // 两个 cache 使用不同的采样器与缩放（与 vanilla NoiseData 一致）
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
    make_accel().batch_fill_cell_caches_vanilla(&positions, &specs, &mut results);

    // 参考：DoublePerlinNoiseSampler::sample（vanilla Noise.compute 语义）
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
        "多 cache vanilla 双 Perlin 批量填充与 vanilla 参考不一致"
    );
}

/// vanilla spec 路径在 JIT 开启时与 vanilla 参考逐位一致。
///
/// 验证 `batch_fill_cell_caches_vanilla` 的 JIT → batch → CPU 级联：
/// 有 GPU 时走 JIT 专用 kernel，无 GPU 时走 CPU 回退，两者都必须与
/// `DoublePerlinNoiseSampler::sample` 逐位一致。
#[test]
fn cell_cache_fill_vanilla_jit_parity() {
    let n = 256;
    let positions = make_positions_3d(n, SEED);

    let dbl1 = make_test_double_perlin(SEED);
    let dbl2 = make_test_double_perlin(SEED.wrapping_add(7));

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
    make_jit_accel().batch_fill_cell_caches_vanilla(&positions, &specs, &mut results);

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
        "JIT 路径批量填充与 vanilla 参考不一致"
    );
}

/// 含水层判定一致性
#[test]
fn aquifer_apply_consistency() {
    let n = 256;
    let positions = make_positions_3d(n, SEED.wrapping_add(1));

    // 确定性生成 densities
    let mut densities = vec![0.0f64; n];
    let mut s = SEED.wrapping_mul(6364136223846793005);
    for i in 0..n {
        densities[i] = (s as f64) * 1e-12 - 0.5;
        s = s.wrapping_mul(1442695040888963407);
    }

    // 确定性生成 packed_grid（5×7×5 网格 = 175 个点，每点 4 个 i64 = 700 个值）
    let gx = 5usize;
    let gy = 7usize;
    let gz = 5usize;
    let grid_size = gx * gy * gz; // 175
    let mut packed_grid = Vec::with_capacity(grid_size * 4);
    let mut sg = SEED.wrapping_add(42);
    for ix in 0..gx {
        for yi in 0..gy {
            for zi in 0..gz {
                let px = ((ix as i32 - 2) * 16) as f64;
                let py = ((yi as i32 - 3) * 16) as f64;
                let pz = ((zi as i32 - 2) * 16) as f64;
                packed_grid.push(px.to_bits() as i64);
                packed_grid.push(py.to_bits() as i64);
                packed_grid.push(pz.to_bits() as i64);
                let den = ((sg as f64) * 1e-12 - 0.3) as f64;
                packed_grid.push(den.to_bits() as i64);
                sg = sg.wrapping_mul(1442695040888963407);
            }
        }
    }
    assert_eq!(packed_grid.len(), grid_size * 4);

    let fluid_level: f64 = -10000.0;
    let barrier_scale: f64 = 0.3;

    // CPU 参考路径
    let cpu_result = cpu_aquifer_ref(
        &positions,
        &densities,
        &packed_grid,
        fluid_level,
        barrier_scale,
    );

    // GPU 路径（通过 BatchAccelerator）
    let gpu_result = make_accel().batch_aquifer_apply(
        &positions,
        &densities,
        &packed_grid,
        fluid_level,
        barrier_scale,
    );

    // 比对 block_ids
    let cpu_block_hash = fnv1a_i32(&cpu_result.block_ids);
    let gpu_block_hash = fnv1a_i32(&gpu_result.block_ids);
    assert_eq!(
        cpu_block_hash, gpu_block_hash,
        "aquifer block_ids: CPU={cpu_block_hash:#x} GPU={gpu_block_hash:#x}"
    );

    // 逐元素比对 fluid_updates
    assert_eq!(
        cpu_result.fluid_updates, gpu_result.fluid_updates,
        "aquifer fluid_updates mismatch"
    );

    // 额外检查：block_ids 也逐元素比对
    assert_eq!(
        cpu_result.block_ids, gpu_result.block_ids,
        "aquifer block_ids mismatch"
    );
}

/// Beardifier 一致性（GPU kernel 与 vanilla `Beardifier::sample` 逐位一致）。
#[test]
fn beardifier_consistency() {
    let n = 512;
    // 位置限定在受影响盒与结构包围盒附近（vanilla 语义下盒外输出 0，
    // 若用随机大坐标会退化成全零的平凡比较）
    let mut ps = SEED.wrapping_add(2);
    let mut positions = Vec::with_capacity(n * 3);
    for _ in 0..n {
        positions.push((ps as f64 % 160.0) - 80.0);
        ps = ps.wrapping_mul(1442695040888963407);
        positions.push((ps as f64 % 96.0) - 48.0);
        ps = ps.wrapping_mul(1442695040888963407);
        positions.push((ps as f64 % 160.0) - 80.0);
        ps = ps.wrapping_mul(1442695040888963407);
    }

    // 创建 3 个结构数据（不同的 bbox + terrain_adaptation）
    let structures = vec![
        BeardifierStructureData {
            box_min_x: -32,
            box_min_y: -64,
            box_min_z: -32,
            box_max_x: 32,
            box_max_y: 0,
            box_max_z: 32,
            adaptation: 1, // BeardThin
            ground_delta: 8,
        },
        BeardifierStructureData {
            box_min_x: 48,
            box_min_y: -48,
            box_min_z: 48,
            box_max_x: 80,
            box_max_y: 16,
            box_max_z: 80,
            adaptation: 3, // Bury
            ground_delta: 0,
        },
        BeardifierStructureData {
            box_min_x: -80,
            box_min_y: -32,
            box_min_z: 64,
            box_max_x: -48,
            box_max_y: 32,
            box_max_z: 96,
            adaptation: 2, // BeardBox
            ground_delta: 16,
        },
    ];

    // 创建 2 个连接点
    let junctions = vec![
        BeardifierJunctionData {
            x: 0,
            ground_y: 0,
            z: 0,
        },
        BeardifierJunctionData {
            x: 64,
            ground_y: -16,
            z: 64,
        },
    ];

    let affected_box = [-96, -80, -96, 96, 48, 112];

    // CPU 参考路径
    let mut cpu_results = vec![0.0f64; n];
    cpu_beardifier_ref(
        &positions,
        &structures,
        &junctions,
        affected_box,
        &mut cpu_results,
    );

    // GPU 路径
    let mut gpu_results = vec![0.0f64; n];
    make_accel().batch_beardifier(
        &positions,
        &structures,
        &junctions,
        affected_box,
        &mut gpu_results,
    );

    let cpu_hash = fnv1a_f64(&cpu_results);
    let gpu_hash = fnv1a_f64(&gpu_results);
    assert_eq!(
        cpu_hash, gpu_hash,
        "beardifier_consistency: CPU={cpu_hash:#x} GPU={gpu_hash:#x}"
    );
    // 非平凡：结构包围盒内应产生非零贡献
    assert!(
        cpu_results.iter().any(|&v| v != 0.0),
        "beardifier 参考输出不应全零"
    );
}

/// Aquifer + Beardifier 批量综合测试
#[test]
fn aquifer_beardifier_combined() {
    let accel = make_accel();

    // --- Aquifer (128 位置) ---
    let n_aq = 128;
    let pos_aq = make_positions_3d(n_aq, SEED.wrapping_add(12));
    let mut densities = vec![0.0f64; n_aq];
    let mut sa = SEED.wrapping_add(555);
    for i in 0..n_aq {
        densities[i] = (sa as f64) * 1e-12 - 0.5;
        sa = sa.wrapping_mul(1442695040888963407);
    }
    let gx = 3usize;
    let gy = 5usize;
    let gz = 3usize;
    let grid_size = gx * gy * gz; // 45
    let mut packed_grid = Vec::with_capacity(grid_size * 4);
    let mut sg = SEED.wrapping_add(84);
    for ix in 0..gx {
        for yi in 0..gy {
            for zi in 0..gz {
                let px = ((ix as i32 - 1) * 20) as f64;
                let py = ((yi as i32 - 2) * 20) as f64;
                let pz = ((zi as i32 - 1) * 20) as f64;
                packed_grid.push(px.to_bits() as i64);
                packed_grid.push(py.to_bits() as i64);
                packed_grid.push(pz.to_bits() as i64);
                let den = ((sg as f64) * 1e-12 - 0.3) as f64;
                packed_grid.push(den.to_bits() as i64);
                sg = sg.wrapping_mul(1442695040888963407);
            }
        }
    }
    let fluid_level: f64 = -10000.0;
    let barrier_scale: f64 = 0.3;
    let cpu_aq = cpu_aquifer_ref(
        &pos_aq,
        &densities,
        &packed_grid,
        fluid_level,
        barrier_scale,
    );
    let gpu_aq = accel.batch_aquifer_apply(
        &pos_aq,
        &densities,
        &packed_grid,
        fluid_level,
        barrier_scale,
    );
    assert_eq!(
        fnv1a_i32(&cpu_aq.block_ids),
        fnv1a_i32(&gpu_aq.block_ids),
        "aquifer block_ids"
    );
    assert_eq!(
        cpu_aq.fluid_updates, gpu_aq.fluid_updates,
        "aquifer fluid_updates"
    );

    // --- Beardifier (256 位置) ---
    let n_beard = 256;
    let mut pb = SEED.wrapping_add(13);
    let mut pos_beard = Vec::with_capacity(n_beard * 3);
    for _ in 0..n_beard {
        pos_beard.push((pb as f64 % 96.0) - 48.0);
        pb = pb.wrapping_mul(1442695040888963407);
        pos_beard.push((pb as f64 % 64.0) - 32.0);
        pb = pb.wrapping_mul(1442695040888963407);
        pos_beard.push((pb as f64 % 96.0) - 48.0);
        pb = pb.wrapping_mul(1442695040888963407);
    }
    let structures = vec![
        BeardifierStructureData {
            box_min_x: -16,
            box_min_y: -32,
            box_min_z: -16,
            box_max_x: 16,
            box_max_y: 0,
            box_max_z: 16,
            adaptation: 1, // BeardThin
            ground_delta: 8,
        },
        BeardifierStructureData {
            box_min_x: 32,
            box_min_y: -16,
            box_min_z: 32,
            box_max_x: 64,
            box_max_y: 16,
            box_max_z: 64,
            adaptation: 2, // BeardBox
            ground_delta: 0,
        },
    ];
    let junctions = vec![BeardifierJunctionData {
        x: 48,
        ground_y: -8,
        z: 48,
    }];
    let affected_box = [-64, -48, -64, 96, 32, 96];
    let mut cpu_beard = vec![0.0f64; n_beard];
    let mut gpu_beard = vec![0.0f64; n_beard];
    cpu_beardifier_ref(
        &pos_beard,
        &structures,
        &junctions,
        affected_box,
        &mut cpu_beard,
    );
    accel.batch_beardifier(
        &pos_beard,
        &structures,
        &junctions,
        affected_box,
        &mut gpu_beard,
    );
    assert_eq!(fnv1a_f64(&cpu_beard), fnv1a_f64(&gpu_beard), "beardifier");
}

/// 空输入测试
#[test]
fn empty_batch() {
    let accel = make_accel();

    // Aquifer
    {
        let result = accel.batch_aquifer_apply(&[], &[], &[], -10000.0, 0.3);
        assert!(result.block_ids.is_empty());
        assert!(result.fluid_updates.is_empty());
    }

    // Beardifier
    {
        let mut results = vec![];
        accel.batch_beardifier(&[], &[], &[], [0; 6], &mut results);
        assert!(results.is_empty());
    }
}
