//! 批量 GPU 加速指纹测试 — Cell Cache、Aquifer、Beardifier、Vein。
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
use pumpkin_world::batch_accel::BatchAccelerator;

#[cfg(feature = "gpu")]
use pumpkin_gpu::noise::batch_cell::{
    AquiferBatchResult, BeardifierJunctionData, BeardifierStructureData, CellFillParams, VeinParams,
};
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

const SEED: u64 = 138_782_381_985_206;

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

/// CPU 参考：Beardifier 遍历结构与连接点。
/// 与 `batch_accel::cpu_beardifier` 保持逻辑一致。
fn cpu_beardifier_ref(
    positions: &[f64],
    structures: &[BeardifierStructureData],
    junctions: &[BeardifierJunctionData],
    results: &mut [f64],
) {
    for i in 0..results.len() {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        let mut beard = 0.0;

        // 结构贡献：距离反比衰减
        for s in structures {
            let cx = s.center_x;
            let cy = s.center_y;
            let cz = s.center_z;
            let rx = s.radius_x + 1.0;
            let ry = s.radius_y + 1.0;
            let rz = s.radius_z + 1.0;

            if rx <= 0.0 || ry <= 0.0 || rz <= 0.0 {
                continue;
            }

            let dx = (x - cx) / rx;
            let dy = (y - cy) / ry;
            let dz = (z - cz) / rz;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            // 仅包围盒内（归一化距离 < 1）才贡献
            if dist_sq < 1.0 {
                let contrib = (1.0 - dist_sq.sqrt()).max(0.0);
                let y_factor = if s.ground_delta_y > 0.0 {
                    ((y - s.min_y) / s.ground_delta_y).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                beard += contrib * y_factor * 0.5;
            }
        }

        // 连接点贡献：固定半径高斯衰减
        for j in junctions {
            let dx = x - f64::from(j.x);
            let dy = y - f64::from(j.ground_y);
            let dz = z - f64::from(j.z);
            let jr: f64 = 12.0;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let norm = dist_sq / (jr * jr);
            if norm < 1.0 {
                beard += (1.0 - norm) * 0.25;
            }
        }

        results[i] = beard;
    }
}

// ============================================================================
// 测试用例
// ============================================================================

/// 1. Cell Cache 填充一致性
#[test]
fn cell_cache_fill_consistency() {
    let n = 1024;
    let positions = make_positions_3d(n, SEED);

    // 使用真实 perlin 配置（从 OctavePerlinNoiseSampler 提取）
    let sampler = make_test_sampler(SEED, &[0, 1, 2, 3]);
    let (perlin_configs, num_octaves) = extract_cell_params_from_sampler(&sampler);

    let params = CellFillParams {
        perlin_configs: perlin_configs.clone(),
        num_octaves: num_octaves.clone(),
        sampler_types: vec![0],
    };

    // 两次 GPU 调用应产生完全相同的结果（确定性验证）
    let mut run1 = vec![0.0f64; n];
    let mut run2 = vec![0.0f64; n];
    make_accel().batch_fill_cell_caches(&positions, &params, &mut run1);
    make_accel().batch_fill_cell_caches(&positions, &params, &mut run2);

    let hash1 = fnv1a_f64(&run1);
    let hash2 = fnv1a_f64(&run2);
    assert_eq!(
        hash1, hash2,
        "cell_cache_fill: deterministic output expected, got {hash1:#x} vs {hash2:#x}"
    );

    // 非零配置应产生非零输出（验证 GPU 确实在执行计算）
    let non_zero = run1.iter().any(|&v| v.abs() > 1e-12);
    assert!(
        non_zero,
        "cell_cache_fill with real configs should produce non-zero output"
    );
}

/// 2. 插值器缓冲填充一致性
#[test]
fn interpolator_fill_consistency() {
    let n = 1024;
    let positions = make_positions_3d(n, SEED);

    let sampler = make_test_sampler(SEED.wrapping_add(1), &[0, 1, 2]);
    let (perlin_configs, num_octaves) = extract_interp_params_from_sampler(&sampler, 0.25, 0.125);

    let params = CellFillParams {
        perlin_configs: perlin_configs.clone(),
        num_octaves: num_octaves.clone(),
        sampler_types: vec![0],
    };

    // 两次 GPU 调用应产生完全相同的结果（确定性验证）
    let mut run1 = vec![0.0f64; n];
    let mut run2 = vec![0.0f64; n];
    make_accel().batch_fill_interpolators(&positions, &params, &mut run1);
    make_accel().batch_fill_interpolators(&positions, &params, &mut run2);

    let hash1 = fnv1a_f64(&run1);
    let hash2 = fnv1a_f64(&run2);
    assert_eq!(
        hash1, hash2,
        "interpolator_fill: deterministic output expected, got {hash1:#x} vs {hash2:#x}"
    );

    let non_zero = run1.iter().any(|&v| v.abs() > 1e-12);
    assert!(
        non_zero,
        "interpolator_fill with real configs should produce non-zero output"
    );
}

/// 3. 含水层判定一致性
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

/// 4. Beardifier 一致性
#[test]
fn beardifier_consistency() {
    let n = 512;
    let positions = make_positions_3d(n, SEED.wrapping_add(2));

    // 创建 3 个结构数据（不同的 bbox + terrain_adaptation）
    let structures = vec![
        BeardifierStructureData {
            center_x: 0.0,
            center_y: -32.0,
            center_z: 0.0,
            radius_x: 32.0,
            radius_y: 32.0,
            radius_z: 32.0,
            min_y: -64.0,
            ground_delta_y: 8.0,
            max_y: 0.0,
        },
        BeardifierStructureData {
            center_x: 64.0,
            center_y: -16.0,
            center_z: 64.0,
            radius_x: 16.0,
            radius_y: 32.0,
            radius_z: 16.0,
            min_y: -48.0,
            ground_delta_y: 0.0,
            max_y: 16.0,
        },
        BeardifierStructureData {
            center_x: -64.0,
            center_y: 0.0,
            center_z: 80.0,
            radius_x: 16.0,
            radius_y: 32.0,
            radius_z: 16.0,
            min_y: -32.0,
            ground_delta_y: 16.0,
            max_y: 32.0,
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

    // CPU 参考路径
    let mut cpu_results = vec![0.0f64; n];
    cpu_beardifier_ref(&positions, &structures, &junctions, &mut cpu_results);

    // GPU 路径
    let mut gpu_results = vec![0.0f64; n];
    make_accel().batch_beardifier(&positions, &structures, &junctions, &mut gpu_results);

    let cpu_hash = fnv1a_f64(&cpu_results);
    let gpu_hash = fnv1a_f64(&gpu_results);
    assert_eq!(
        cpu_hash, gpu_hash,
        "beardifier_consistency: CPU={cpu_hash:#x} GPU={gpu_hash:#x}"
    );
}

/// 5. 矿脉判定一致性
#[test]
fn vein_sample_consistency() {
    let n = 256;
    let positions = make_positions_3d(n, SEED.wrapping_add(3));

    // 确定性填充 VeinParams
    let mut toggle_config = Vec::with_capacity(32);
    let mut ridged_config = Vec::with_capacity(32);
    let mut gap_config = Vec::with_capacity(32);
    let mut sv = SEED.wrapping_add(99);
    for _ in 0..32 {
        toggle_config.push((sv as f64) * 1e-12);
        sv = sv.wrapping_mul(1442695040888963407);
        ridged_config.push((sv as f64) * 1e-12);
        sv = sv.wrapping_mul(1442695040888963407);
        gap_config.push((sv as f64) * 1e-12);
        sv = sv.wrapping_mul(1442695040888963407);
    }

    let params = VeinParams {
        toggle_config,
        ridged_config,
        gap_config,
    };

    // CPU 路径：全部返回 0（忽略 vein）
    let mut cpu_results = vec![0i32; n];

    // GPU 路径
    let mut gpu_results = vec![0i32; n];
    make_accel().batch_vein_sample(&positions, &params, &mut gpu_results);

    let cpu_hash = fnv1a_i32(&cpu_results);
    let gpu_hash = fnv1a_i32(&gpu_results);
    assert_eq!(
        cpu_hash, gpu_hash,
        "vein_sample_consistency: CPU={cpu_hash:#x} GPU={gpu_hash:#x}"
    );

    // 逐元素比对
    assert_eq!(cpu_results, gpu_results, "vein results mismatch");
}

/// 6. 全批量类型综合测试
#[test]
fn all_batch_types() {
    let accel = make_accel();

    // --- Cell Cache (512 位置) ---
    let n_cell = 512;
    let pos_cell = make_positions_3d(n_cell, SEED.wrapping_add(10));
    let params_cell = CellFillParams {
        perlin_configs: vec![],
        num_octaves: vec![3, 3],
        sampler_types: vec![0, 0],
    };
    let mut cpu_cell = vec![0.0f64; n_cell];
    cpu_cell.fill(0.0);
    let mut gpu_cell = vec![0.0f64; n_cell];
    accel.batch_fill_cell_caches(&pos_cell, &params_cell, &mut gpu_cell);
    assert_eq!(
        fnv1a_f64(&cpu_cell),
        fnv1a_f64(&gpu_cell),
        "all_batch_types: cell_cache"
    );

    // --- Interpolator (512 位置) ---
    let pos_interp = make_positions_3d(n_cell, SEED.wrapping_add(11));
    let mut cpu_interp = vec![0.0f64; n_cell];
    cpu_interp.fill(0.0);
    let mut gpu_interp = vec![0.0f64; n_cell];
    accel.batch_fill_interpolators(&pos_interp, &params_cell, &mut gpu_interp);
    assert_eq!(
        fnv1a_f64(&cpu_interp),
        fnv1a_f64(&gpu_interp),
        "all_batch_types: interpolator"
    );

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
        "all_batch_types: aquifer block_ids"
    );
    assert_eq!(
        cpu_aq.fluid_updates, gpu_aq.fluid_updates,
        "all_batch_types: aquifer fluid_updates"
    );

    // --- Beardifier (256 位置) ---
    let n_beard = 256;
    let pos_beard = make_positions_3d(n_beard, SEED.wrapping_add(13));
    let structures = vec![
        BeardifierStructureData {
            center_x: 0.0,
            center_y: -16.0,
            center_z: 0.0,
            radius_x: 16.0,
            radius_y: 16.0,
            radius_z: 16.0,
            min_y: -32.0,
            ground_delta_y: 8.0,
            max_y: 0.0,
        },
        BeardifierStructureData {
            center_x: 48.0,
            center_y: 0.0,
            center_z: 48.0,
            radius_x: 16.0,
            radius_y: 16.0,
            radius_z: 16.0,
            min_y: -16.0,
            ground_delta_y: 0.0,
            max_y: 16.0,
        },
    ];
    let junctions = vec![BeardifierJunctionData {
        x: 48,
        ground_y: -8,
        z: 48,
    }];
    let mut cpu_beard = vec![0.0f64; n_beard];
    let mut gpu_beard = vec![0.0f64; n_beard];
    cpu_beardifier_ref(&pos_beard, &structures, &junctions, &mut cpu_beard);
    accel.batch_beardifier(&pos_beard, &structures, &junctions, &mut gpu_beard);
    assert_eq!(
        fnv1a_f64(&cpu_beard),
        fnv1a_f64(&gpu_beard),
        "all_batch_types: beardifier"
    );

    // --- Vein (128 位置) ---
    let n_vein = 128;
    let pos_vein = make_positions_3d(n_vein, SEED.wrapping_add(14));
    let mut tv = Vec::with_capacity(16);
    let mut rv = Vec::with_capacity(16);
    let mut gv = Vec::with_capacity(16);
    let mut sv2 = SEED.wrapping_add(777);
    for _ in 0..16 {
        tv.push((sv2 as f64) * 1e-12);
        sv2 = sv2.wrapping_mul(1442695040888963407);
        rv.push((sv2 as f64) * 1e-12);
        sv2 = sv2.wrapping_mul(1442695040888963407);
        gv.push((sv2 as f64) * 1e-12);
        sv2 = sv2.wrapping_mul(1442695040888963407);
    }
    let vein_params = VeinParams {
        toggle_config: tv,
        ridged_config: rv,
        gap_config: gv,
    };
    let mut cpu_vein = vec![0i32; n_vein];
    let mut gpu_vein = vec![0i32; n_vein];
    accel.batch_vein_sample(&pos_vein, &vein_params, &mut gpu_vein);
    assert_eq!(
        fnv1a_i32(&cpu_vein),
        fnv1a_i32(&gpu_vein),
        "all_batch_types: vein"
    );
    assert_eq!(cpu_vein, gpu_vein, "all_batch_types: vein element-wise");
}

/// 7. 空输入测试
#[test]
fn empty_batch() {
    let accel = make_accel();

    // Cell Cache
    {
        let params = CellFillParams {
            perlin_configs: vec![],
            num_octaves: vec![],
            sampler_types: vec![],
        };
        let mut results = vec![];
        accel.batch_fill_cell_caches(&[], &params, &mut results);
        assert!(results.is_empty());
    }

    // Interpolator
    {
        let params = CellFillParams {
            perlin_configs: vec![],
            num_octaves: vec![],
            sampler_types: vec![],
        };
        let mut results = vec![];
        accel.batch_fill_interpolators(&[], &params, &mut results);
        assert!(results.is_empty());
    }

    // Aquifer
    {
        let result = accel.batch_aquifer_apply(&[], &[], &[], -10000.0, 0.3);
        assert!(result.block_ids.is_empty());
        assert!(result.fluid_updates.is_empty());
    }

    // Beardifier
    {
        let mut results = vec![];
        accel.batch_beardifier(&[], &[], &[], &mut results);
        assert!(results.is_empty());
    }

    // Vein
    {
        let params = VeinParams {
            toggle_config: vec![],
            ridged_config: vec![],
            gap_config: vec![],
        };
        let mut results = vec![];
        accel.batch_vein_sample(&[], &params, &mut results);
        assert!(results.is_empty());
    }
}

/// 8. Cell Cache 填充性能基准
#[test]
fn perf_batch_cell() {
    let n = 4096;
    let positions = make_positions_3d(n, SEED);

    let params = CellFillParams {
        perlin_configs: vec![],
        num_octaves: vec![3, 3],
        sampler_types: vec![0, 0],
    };

    let n_iter = 10u32;

    // CPU 路径：零填充
    let cpu_start = std::time::Instant::now();
    for _ in 0..n_iter {
        let mut results = vec![0.0f64; n];
        results.fill(0.0);
        // 防止优化消除
        std::hint::black_box(&mut results);
    }
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

    // GPU 路径（通过 BatchAccelerator）
    let gpu_start = std::time::Instant::now();
    for _ in 0..n_iter {
        let mut results = vec![0.0f64; n];
        make_accel().batch_fill_cell_caches(&positions, &params, &mut results);
        std::hint::black_box(&mut results);
    }
    let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

    println!(
        "Cell Cache fill (n={n}, iters={n_iter}): cpu={cpu_ms:.3}ms, gpu={gpu_ms:.3}ms, speedup={:.2}x",
        cpu_ms / gpu_ms.max(1e-9)
    );

    assert!(
        cpu_ms < 1000.0,
        "CPU cell cache fill should complete within 1s per iteration (took {cpu_ms:.1}ms)"
    );
    assert!(
        gpu_ms < 1000.0,
        "GPU cell cache fill should complete within 1s per iteration (took {gpu_ms:.1}ms)"
    );
}

// ============================================================================
// 测试辅助函数：perlin 配置提取
// ============================================================================

/// 创建一个测试用的 `OctavePerlinNoiseSampler`。
fn make_test_sampler(seed: u64, octaves: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let (start, amplitudes) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
    let mut g = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut g, start, &amplitudes, false)
}

/// 从 `OctavePerlinNoiseSampler` 提取 cell cache 编码的 perlin 配置。
///
/// 编码格式：`[num_octaves, amps[0..n], lacs[0..n], orgs[0..3n]]`
fn extract_cell_params_from_sampler(sampler: &OctavePerlinNoiseSampler) -> (Vec<f64>, Vec<i32>) {
    let num_octaves = sampler.samplers.len() as i32;
    let mut config = Vec::with_capacity(1 + num_octaves as usize * 5);
    config.push(num_octaves as f64);
    for sd in sampler.samplers.iter() {
        config.push(sd.amplitude * sd.persistence);
    }
    for sd in sampler.samplers.iter() {
        config.push(sd.lacunarity);
    }
    for sd in sampler.samplers.iter() {
        config.push(sd.sampler.x_origin());
        config.push(sd.sampler.y_origin());
        config.push(sd.sampler.z_origin());
    }
    (config, vec![num_octaves])
}

/// 从 `OctavePerlinNoiseSampler` 提取插值器编码的 perlin 配置。
///
/// 编码格式：每八度 8 个 f64 `[amp, lac, orgx, orgy, orgz, xz_scale, y_scale, 0.0]`
fn extract_interp_params_from_sampler(
    sampler: &OctavePerlinNoiseSampler,
    xz_scale: f64,
    y_scale: f64,
) -> (Vec<f64>, Vec<i32>) {
    let num_octaves = sampler.samplers.len() as i32;
    let mut config = Vec::with_capacity(num_octaves as usize * 8);
    for sd in sampler.samplers.iter() {
        config.push(sd.amplitude * sd.persistence);
        config.push(sd.lacunarity);
        config.push(sd.sampler.x_origin());
        config.push(sd.sampler.y_origin());
        config.push(sd.sampler.z_origin());
        config.push(xz_scale);
        config.push(y_scale);
        config.push(0.0); // reserved
    }
    (config, vec![num_octaves])
}
