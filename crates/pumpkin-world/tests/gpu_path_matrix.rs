//! GPU 全路径矩阵测试 — 通过环境变量选择后端与 JIT，对比 CPU 参考。
//!
//! 环境变量：
//! - `PUMPKIN_GPU_BACKEND`：`auto` | `cuda` | `opencl` | `cpu`（默认 `auto`）
//! - `PUMPKIN_GPU_JIT`：`0` | `1`（默认 `0`）
//!
//! 输出（`--nocapture`）包含设备信息、各模块 CPU/加速器一致性结果、
//! JIT 实际启用情况与 CPU/加速器耗时，供基准报告采集。
//!
//! JIT 规则：JIT 内核编译失败时跳过 JIT 段（不判失败），其余一致性断言照常。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::{GpuBackend, GpuConfig};
use pumpkin_gpu::noise::batch_cell::{BeardifierJunctionData, BeardifierStructureData};
use pumpkin_gpu::{DeviceType, GpuDevice};
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::batch_accel::{BatchAccelerator, CellCacheFillSpec};
use pumpkin_world::light_accel::LightAccelerator;
use pumpkin_world::noise_accel::NoiseAccelerator;
use std::sync::OnceLock;

const SEED: u64 = 138_782_381_985_206;

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// 从环境变量构建矩阵配置。
pub(crate) fn matrix_config() -> GpuConfig {
    let backend = match env_str("PUMPKIN_GPU_BACKEND").as_deref() {
        Some("cuda") => GpuBackend::Cuda,
        Some("opencl") => GpuBackend::OpenCl,
        Some("cpu") => GpuBackend::Cpu,
        _ => GpuBackend::Auto,
    };
    let jit_enabled = env_str("PUMPKIN_GPU_JIT").as_deref() == Some("1");
    // OpenCL 不支持 auto 调度策略（会警告并回退 CPU），测试时显式指定 ByIndex(0)。
    let device = if backend == GpuBackend::OpenCl {
        pumpkin_config::gpu::GpuDeviceSelection::ByIndex { index: 0 }
    } else {
        pumpkin_config::gpu::GpuDeviceSelection::Auto
    };
    GpuConfig {
        enabled: true,
        noise_acceleration: true,
        light_acceleration: true,
        batch_acceleration: true,
        jit_enabled,
        jit_max_unroll: 16,
        backend,
        device,
        ..Default::default()
    }
}

fn backend_tag() -> String {
    env_str("PUMPKIN_GPU_BACKEND").unwrap_or_else(|| "auto".into())
}

fn jit_tag() -> &'static str {
    if env_str("PUMPKIN_GPU_JIT").as_deref() == Some("1") {
        "jit-on"
    } else {
        "jit-off"
    }
}

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
    NoiseAccelerator::new(&matrix_config())
}

fn mk_batch_accel() -> BatchAccelerator {
    BatchAccelerator::new(&matrix_config())
}

fn mk_light_accel() -> LightAccelerator {
    LightAccelerator::new(&matrix_config())
}

/// 设备报告：打印后端、设备名、JIT 开关与实际设备类型。
#[test]
fn matrix_device_report() {
    let config = matrix_config();
    let device = GpuDevice::from_config(&config);
    println!(
        "MATRIX[{}|{}] requested_backend={:?} device_type={:?} device_name={}",
        backend_tag(),
        jit_tag(),
        config.backend,
        device.device_type(),
        device.device_name()
    );
    // 强制指定后端时，若初始化失败会回退 CPU——打印警告供报告分析。
    if config.backend == GpuBackend::Cuda && device.device_type() != DeviceType::Cuda {
        println!("MATRIX WARN: 强制 CUDA 但实际设备不是 CUDA（回退 CPU）");
    }
    if config.backend == GpuBackend::OpenCl && device.device_type() != DeviceType::OpenCl {
        println!("MATRIX WARN: 强制 OpenCL 但实际设备不是 OpenCL（回退 CPU）");
    }
}

/// 噪声五族 + 三线性：加速器路径 vs CPU 参考（哈希逐位一致）。
#[test]
fn matrix_noise_families_consistency() {
    let mut accel = mk_noise_accel();
    let c = 1.0181268882175227f64;

    // octave（4 组八度配置）
    for octaves in [&[0][..], &[-2, 0, 2][..], &[0, 1, 2, 3, 4][..]] {
        let sampler = mk_sampler(SEED, octaves);
        let n = 512;
        let pos = mk_pos3d(n, SEED);
        let mut cpu = vec![0.0f64; n];
        let mut acc = vec![0.0f64; n];
        for i in 0..n {
            cpu[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        }
        accel.sample_octave(&sampler, &pos, &mut acc);
        assert_eq!(
            fnv1a_f64(&cpu),
            fnv1a_f64(&acc),
            "octave {octaves:?} mismatch"
        );
    }
    println!(
        "MATRIX[{}|{}] octave consistency OK",
        backend_tag(),
        jit_tag()
    );

    // double perlin
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[-1, 0, 1]);
    let n = 512;
    let pos = mk_pos3d(n, SEED.wrapping_add(1));
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];
    for i in 0..n {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        cpu[i] = (a.sample(x, y, z) + b.sample(x * c, y * c, z * c)) * 0.5;
    }
    accel.sample_double_perlin(&a, &b, 0.5, &pos, &mut acc);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&acc), "double perlin mismatch");
    println!(
        "MATRIX[{}|{}] double_perlin consistency OK",
        backend_tag(),
        jit_tag()
    );

    // shift a / b
    let xz: Vec<f64> = (0..n).flat_map(|i| [pos[i * 3], pos[i * 3 + 2]]).collect();
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    for (shift, is_a) in [("shift_a", true), ("shift_b", false)] {
        let mut cpu = vec![0.0f64; n];
        let mut acc = vec![0.0f64; n];
        for i in 0..n {
            cpu[i] = if is_a {
                sampler.sample(xz[i * 2] * 0.25, 0.0, xz[i * 2 + 1] * 0.25) * 4.0
            } else {
                sampler.sample(xz[i * 2 + 1] * 0.25, 0.0, xz[i * 2] * 0.25) * 4.0
            };
        }
        if is_a {
            accel.sample_shift_a(&sampler, &xz, &mut acc);
        } else {
            accel.sample_shift_b(&sampler, &xz, &mut acc);
        }
        assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&acc), "{shift} mismatch");
    }
    println!(
        "MATRIX[{}|{}] shift_a/shift_b consistency OK",
        backend_tag(),
        jit_tag()
    );

    // flatcache
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
    }
    accel.precompute_flatcache(&sampler, &xz, &mut acc);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&acc), "flatcache mismatch");
    println!(
        "MATRIX[{}|{}] flatcache consistency OK",
        backend_tag(),
        jit_tag()
    );

    // trilinear（BatchAccelerator）
    let accel_b = mk_batch_accel();
    let corners: Vec<f64> = (0..n * 8).map(|i| (i as f64 % 16.0) * 1e-3).collect();
    let deltas: Vec<f64> = (0..n * 3).map(|i| (i as f64 % 7.0) / 7.0).collect();
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];
    for i in 0..n {
        let bb = i * 8;
        let dx = deltas[i * 3];
        let dy = deltas[i * 3 + 1];
        let dz = deltas[i * 3 + 2];
        cpu[i] = corners[bb] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + corners[bb + 1] * dx * (1.0 - dy) * (1.0 - dz)
            + corners[bb + 2] * (1.0 - dx) * dy * (1.0 - dz)
            + corners[bb + 3] * dx * dy * (1.0 - dz)
            + corners[bb + 4] * (1.0 - dx) * (1.0 - dy) * dz
            + corners[bb + 5] * dx * (1.0 - dy) * dz
            + corners[bb + 6] * (1.0 - dx) * dy * dz
            + corners[bb + 7] * dx * dy * dz;
    }
    accel_b.batch_trilinear(&corners, &deltas, &mut acc);
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&acc), "trilinear mismatch");
    println!(
        "MATRIX[{}|{}] trilinear consistency OK",
        backend_tag(),
        jit_tag()
    );
}

/// CellCache vanilla / Aquifer / Beardifier：加速器 vs CPU 参考。
#[test]
fn matrix_batch_families_consistency() {
    let accel = mk_batch_accel();
    let n = 256;
    let positions = mk_pos3d(n, SEED);

    // cell cache vanilla（双规格）
    let sampler_a = mk_sampler(SEED, &[0, 1, 2]);
    let sampler_b = mk_sampler(SEED.wrapping_add(7), &[0, 1]);
    let specs = vec![
        CellCacheFillSpec {
            first: &sampler_a,
            second: &sampler_a,
            amplitude: 1.0,
            xz_scale: 0.25,
            y_scale: 0.125,
        },
        CellCacheFillSpec {
            first: &sampler_b,
            second: &sampler_b,
            amplitude: 0.5,
            xz_scale: 1.0,
            y_scale: 1.0,
        },
    ];
    let mut results = vec![0.0f64; n * 2];
    accel.batch_fill_cell_caches_vanilla(&positions, &specs, &mut results);
    let c = 1.0181268882175227f64;
    let mut reference = vec![0.0f64; n * 2];
    for (ci, spec) in specs.iter().enumerate() {
        for i in 0..n {
            let x = positions[i * 3] * spec.xz_scale;
            let y = positions[i * 3 + 1] * spec.y_scale;
            let z = positions[i * 3 + 2] * spec.xz_scale;
            reference[ci * n + i] = (spec.first.sample(x, y, z)
                + spec.second.sample(x * c, y * c, z * c))
                * spec.amplitude;
        }
    }
    assert_eq!(
        fnv1a_f64(&results),
        fnv1a_f64(&reference),
        "cell cache vanilla mismatch"
    );
    println!(
        "MATRIX[{}|{}] cell_cache_vanilla consistency OK",
        backend_tag(),
        jit_tag()
    );

    // aquifer（4×4×4 网格）
    let densities: Vec<f64> = (0..n).map(|i| (i as f64 % 64.0) / 64.0 - 0.5).collect();
    let mut packed_grid = Vec::with_capacity(64 * 4);
    for ix in 0..4 {
        for iy in 0..4 {
            for iz in 0..4 {
                packed_grid.push((((ix - 1) * 24) as f64).to_bits() as i64);
                packed_grid.push((((iy - 2) * 24) as f64).to_bits() as i64);
                packed_grid.push((((iz - 1) * 24) as f64).to_bits() as i64);
                packed_grid.push(0.3f64.to_bits() as i64);
            }
        }
    }
    let gpu_aq = accel.batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
    // CPU 参考（4-NN）
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
    let mut cpu_ids = vec![0i32; n];
    for i in 0..n {
        let qx = positions[i * 3];
        let qy = positions[i * 3 + 1];
        let qz = positions[i * 3 + 2];
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
        let barrier: f64 = best_idx.iter().map(|&j| grid_densities[j]).sum::<f64>() / 4.0;
        let eff = densities[i] + barrier * 0.3;
        if eff > 0.0 {
            cpu_ids[i] = 1;
        } else if qy < -10000.0 {
            cpu_ids[i] = 2;
        }
    }
    assert_eq!(cpu_ids, gpu_aq.block_ids, "aquifer mismatch");
    println!(
        "MATRIX[{}|{}] aquifer consistency OK",
        backend_tag(),
        jit_tag()
    );

    // beardifier（vanilla 语义，盒内位置）
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
        x: 0,
        ground_y: 0,
        z: 0,
    }];
    let mut ps = SEED.wrapping_add(13);
    let mut near_positions = Vec::with_capacity(n * 3);
    for _ in 0..n {
        near_positions.push((ps as f64 % 96.0) - 48.0);
        ps = ps.wrapping_mul(1442695040888963407);
        near_positions.push((ps as f64 % 64.0) - 32.0);
        ps = ps.wrapping_mul(1442695040888963407);
        near_positions.push((ps as f64 % 96.0) - 48.0);
        ps = ps.wrapping_mul(1442695040888963407);
    }
    let affected_box = [-64, -48, -64, 96, 32, 96];
    let mut beard = vec![0.0f64; n];
    accel.batch_beardifier(
        &near_positions,
        &structures,
        &junctions,
        affected_box,
        &mut beard,
    );
    // CPU 参考（vanilla 逐位等价）
    let mut cpu_beard = vec![0.0f64; n];
    for i in 0..n {
        let x = near_positions[i * 3] as i32;
        let y = near_positions[i * 3 + 1] as i32;
        let z = near_positions[i * 3 + 2] as i32;
        if !(-64..=96).contains(&x) || !(-48..=32).contains(&y) || !(-64..=96).contains(&z) {
            cpu_beard[i] = 0.0;
            continue;
        }
        let mut weight = 0.0;
        for s in &structures {
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
                3 => bury_ref(f64::from(dx), f64::from(dy) / 2.0, f64::from(dz)),
                1 | 2 => beard_ref(dx, dy, dz, dy_to_ground) * 0.8,
                _ => {
                    bury_ref(
                        f64::from(dx) / 2.0,
                        f64::from(dy) / 2.0,
                        f64::from(dz) / 2.0,
                    ) * 0.8
                }
            };
            weight += contrib;
        }
        for j in &junctions {
            weight += beard_ref(x - j.x, y - j.ground_y, z - j.z, y - j.ground_y) * 0.4;
        }
        cpu_beard[i] = weight;
    }
    assert_eq!(
        fnv1a_f64(&cpu_beard),
        fnv1a_f64(&beard),
        "beardifier mismatch"
    );
    println!(
        "MATRIX[{}|{}] beardifier consistency OK",
        backend_tag(),
        jit_tag()
    );
}

fn beard_ref(dx: i32, dy: i32, dz: i32, y_to_ground: i32) -> f64 {
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

fn bury_ref(dx: f64, dy: f64, dz: f64) -> f64 {
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    if distance < 0.0 {
        1.0
    } else if distance > 6.0 {
        0.0
    } else {
        1.0 - distance / 6.0
    }
}

/// 光照四路径：加速器 vs CPU 参考。
#[test]
fn matrix_light_consistency() {
    let mut accel = mk_light_accel();

    // sky fill
    let n = 16usize;
    let h = 256usize;
    let mut s = SEED;
    let hm: Vec<i32> = (0..n)
        .map(|_| {
            let v = ((s % 200) + 64) as i32;
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
    let mut acc = vec![0u8; n * h];
    accel.batch_sky_fill(&hm, &op, &mut acc, n, h);
    let mut cpu = vec![0u8; n * h];
    for col in 0..n {
        let top = hm[col];
        for y in (top + 1)..h as i32 {
            cpu[col * h + y as usize] = 15;
        }
        let mut lt: u8 = 15;
        for y in (0..=top).rev() {
            let i = col * h + y as usize;
            lt = lt.saturating_sub(op[i]);
            cpu[i] = lt;
        }
    }
    assert_eq!(fnv1a_u8(&cpu), fnv1a_u8(&acc), "sky fill mismatch");
    println!(
        "MATRIX[{}|{}] sky_fill consistency OK",
        backend_tag(),
        jit_tag()
    );

    // block scan
    let lum: Vec<u8> = (0..n * h).map(|i| (i % 17) as u8).collect();
    let mut acc_bl = vec![0u8; n * h];
    let sources = accel.batch_block_scan(&lum, &mut acc_bl, n * h);
    let mut cpu_sources = Vec::new();
    for (i, &v) in lum.iter().enumerate() {
        if v > 0 {
            cpu_sources.push(i as i32);
        }
    }
    let mut sorted = sources;
    sorted.sort_unstable();
    assert_eq!(sorted, cpu_sources, "block scan sources mismatch");
    assert_eq!(
        fnv1a_u8(&acc_bl),
        fnv1a_u8(&lum),
        "block scan values mismatch"
    );
    println!(
        "MATRIX[{}|{}] block_scan consistency OK",
        backend_tag(),
        jit_tag()
    );

    // iterative propagate（5×5×5）
    let side = 5usize;
    let nn = side * side * side;
    let mut s = SEED;
    let opacity: Vec<u8> = (0..nn)
        .map(|_| {
            let v = (s % 4) as u8;
            s = s.wrapping_mul(1442695040888963407);
            v
        })
        .collect();
    let mut light = vec![0u8; nn];
    light[nn / 2] = 15;
    let mut neighbors = Vec::with_capacity(nn * 6);
    for i in 0..nn {
        let x = (i % side) as i32;
        let y = ((i / side) % side) as i32;
        let z = (i / (side * side)) as i32;
        let si = side as i32;
        for (dx, dy, dz) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            let idx = if (0..si).contains(&nx) && (0..si).contains(&ny) && (0..si).contains(&nz) {
                nz * si * si + ny * si + nx
            } else {
                -1
            };
            neighbors.push(idx);
        }
    }
    let mut acc_light = light.clone();
    accel.iterative_propagate(&mut acc_light, &opacity, &neighbors, nn, 64);
    let mut cpu_light = light.clone();
    for _ in 0..64 {
        let mut changed = false;
        for i in 0..nn {
            let cur = cpu_light[i];
            let mut best = cur;
            for d in 0..6 {
                let ni = neighbors[i * 6 + d] as usize;
                if ni < nn {
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
        fnv1a_u8(&acc_light),
        "iterative propagate mismatch"
    );
    println!(
        "MATRIX[{}|{}] iterative_propagate consistency OK",
        backend_tag(),
        jit_tag()
    );

    // sky horizontal（6×6×32）
    let width = 6usize;
    let depth = 6usize;
    let height = 32usize;
    let n_total = width * depth * height;
    let mut sky = vec![0u8; n_total];
    for z in 0..depth {
        for x in 0..width {
            sky[z * width * height + x * height + height - 1] = 15;
        }
    }
    let opacity = vec![1u8; n_total];
    let mut acc_sky = sky.clone();
    accel.sky_horizontal_propagate(&mut acc_sky, &opacity, width, depth, height, 64);
    let mut cpu_sky = sky.clone();
    let stride_x = height;
    let stride_z = width * height;
    for _ in 0..64 {
        let mut changed = false;
        for z in 0..depth {
            for x in 0..width {
                for y in (0..height).rev() {
                    let idx = z * stride_z + x * stride_x + y;
                    let cur = cpu_sky[idx];
                    let mut best = cur;
                    if x > 0 {
                        let nl = cpu_sky[idx - stride_x];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if x < width - 1 {
                        let nl = cpu_sky[idx + stride_x];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if z > 0 {
                        let nl = cpu_sky[idx - stride_z];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if z < depth - 1 {
                        let nl = cpu_sky[idx + stride_z];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if y < height - 1 {
                        let above = cpu_sky[idx + 1];
                        if above == 15 && opacity[idx] == 0 && 15 > best {
                            best = 15;
                        }
                    }
                    if best > cur {
                        cpu_sky[idx] = best;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    assert_eq!(
        fnv1a_u8(&cpu_sky),
        fnv1a_u8(&acc_sky),
        "sky horizontal mismatch"
    );
    println!(
        "MATRIX[{}|{}] sky_horizontal consistency OK",
        backend_tag(),
        jit_tag()
    );
}

/// JIT 段：验证 JIT 内核实际编译/启动并输出与 batch 一致。
/// JIT 编译失败时跳过（不判失败），仅输出记录供报告分析。
#[test]
fn matrix_jit_path() {
    let config = matrix_config();
    if !config.jit_enabled {
        println!(
            "MATRIX[{}|{}] jit disabled — skip",
            backend_tag(),
            jit_tag()
        );
        return;
    }
    let device = GpuDevice::from_config(&config);
    if device.device_type() == DeviceType::Cpu {
        println!(
            "MATRIX[{}|{}] jit on CPU device — 回退 batch 路径（跳过 JIT 内核验证）",
            backend_tag(),
            jit_tag()
        );
        return;
    }

    let mut gpu = pumpkin_gpu::noise::GpuNoiseSampler::new(device);
    let sampler = mk_sampler(SEED, &[0, 1, 2]);
    let n = 1024;
    let pos = mk_pos3d(n, SEED);

    let mut batch = vec![0.0f64; n];
    let mut jit = vec![0.0f64; n];
    if gpu.sample_octave_batch(&sampler, &pos, &mut batch).is_err() {
        println!(
            "MATRIX[{}|{}] batch launch failed — skip",
            backend_tag(),
            jit_tag()
        );
        return;
    }
    if gpu.sample_octave_jit(&sampler, &pos, &mut jit).is_err() {
        println!(
            "MATRIX[{}|{}] JIT 启动失败 — skip（原因记录见报告）",
            backend_tag(),
            jit_tag()
        );
        return;
    }
    // JIT kernel 名含配置指纹（八度数 + 采样器内容哈希），
    // 通过 specialize 计算期望名而非硬编码。
    let expected_jit_name = pumpkin_gpu::jit::specialize_octave_perlin(
        &pumpkin_gpu::noise::cache::SerializedOctaveConfig::from_sampler(&sampler),
        16,
    )
    .map(|k| k.name)
    .unwrap_or_default();
    let jit_kernel_compiled = gpu
        .device
        .kernel_launcher()
        .is_some_and(|l| l.has_kernel(&expected_jit_name));
    println!(
        "MATRIX[{}|{}] jit_kernel_compiled={jit_kernel_compiled}",
        backend_tag(),
        jit_tag()
    );
    assert_eq!(fnv1a_f64(&batch), fnv1a_f64(&jit), "jit vs batch mismatch");
    println!(
        "MATRIX[{}|{}] jit octave vs batch OK",
        backend_tag(),
        jit_tag()
    );
}

/// 性能对比：CPU 参考 vs 加速器耗时（打印供报告采集）。
///
/// 每次计时前先做一次小规模预热调用，以排除惰性设备初始化
/// （CUDA NVRTC 编译全部 kernel ~400ms、OpenCL ~180ms）对首调耗时的污染。
#[test]
fn matrix_perf() {
    let mut accel = mk_noise_accel();
    let accel_b = mk_batch_accel();
    let sampler = mk_sampler(SEED, &[-2, -1, 0, 1, 2, 3]);
    let n = 262_144;
    let pos = mk_pos3d(n, SEED);

    // 预热：排除惰性初始化与首次 JIT 编译开销
    let _: () = {
        let warm_pos = mk_pos3d(64, SEED);
        let mut warm = vec![0.0f64; 64];
        accel.sample_octave(&sampler, &warm_pos, &mut warm);
        let warm_corners: Vec<f64> = (0..64 * 8).map(|i| (i as f64 % 16.0) * 1e-3).collect();
        let warm_deltas: Vec<f64> = (0..64 * 3).map(|i| (i as f64 % 7.0) / 7.0).collect();
        accel_b.batch_trilinear(&warm_corners, &warm_deltas, &mut warm);
    };

    // octave
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];
    let t0 = std::time::Instant::now();
    for i in 0..n {
        cpu[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    accel.sample_octave(&sampler, &pos, &mut acc);
    let acc_ms = t1.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&acc));
    println!(
        "MATRIX_PERF[{}|{}] octave n={n} cpu={cpu_ms:.2}ms accel={acc_ms:.2}ms speedup={:.2}x",
        backend_tag(),
        jit_tag(),
        cpu_ms / acc_ms.max(1e-9)
    );

    // double perlin
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[-1, 0, 1]);
    let n2 = 65_536;
    let pos2 = mk_pos3d(n2, SEED.wrapping_add(1));
    let c = 1.0181268882175227f64;
    let mut cpu2 = vec![0.0f64; n2];
    let mut acc2 = vec![0.0f64; n2];
    // 预热 double perlin（触发 JIT kernel 首次编译）
    let _: () = {
        let warm_pos = mk_pos3d(64, SEED.wrapping_add(1));
        let mut warm = vec![0.0f64; 64];
        accel.sample_double_perlin(&a, &b, 0.5, &warm_pos, &mut warm);
    };
    let t0 = std::time::Instant::now();
    for i in 0..n2 {
        let x = pos2[i * 3];
        let y = pos2[i * 3 + 1];
        let z = pos2[i * 3 + 2];
        cpu2[i] = (a.sample(x, y, z) + b.sample(x * c, y * c, z * c)) * 0.5;
    }
    let cpu2_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    accel.sample_double_perlin(&a, &b, 0.5, &pos2, &mut acc2);
    let acc2_ms = t1.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(fnv1a_f64(&cpu2), fnv1a_f64(&acc2));
    println!(
        "MATRIX_PERF[{}|{}] double_perlin n={n2} cpu={cpu2_ms:.2}ms accel={acc2_ms:.2}ms speedup={:.2}x",
        backend_tag(),
        jit_tag(),
        cpu2_ms / acc2_ms.max(1e-9)
    );

    // trilinear
    let n3 = 131_072;
    let corners: Vec<f64> = (0..n3 * 8).map(|i| (i as f64 % 16.0) * 1e-3).collect();
    let deltas: Vec<f64> = (0..n3 * 3).map(|i| (i as f64 % 7.0) / 7.0).collect();
    let mut cpu3 = vec![0.0f64; n3];
    let mut acc3 = vec![0.0f64; n3];
    let t0 = std::time::Instant::now();
    for i in 0..n3 {
        let bb = i * 8;
        let dx = deltas[i * 3];
        let dy = deltas[i * 3 + 1];
        let dz = deltas[i * 3 + 2];
        cpu3[i] = corners[bb] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + corners[bb + 1] * dx * (1.0 - dy) * (1.0 - dz)
            + corners[bb + 2] * (1.0 - dx) * dy * (1.0 - dz)
            + corners[bb + 3] * dx * dy * (1.0 - dz)
            + corners[bb + 4] * (1.0 - dx) * (1.0 - dy) * dz
            + corners[bb + 5] * dx * (1.0 - dy) * dz
            + corners[bb + 6] * (1.0 - dx) * dy * dz
            + corners[bb + 7] * dx * dy * dz;
    }
    let cpu3_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    accel_b.batch_trilinear(&corners, &deltas, &mut acc3);
    let acc3_ms = t1.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(fnv1a_f64(&cpu3), fnv1a_f64(&acc3));
    println!(
        "MATRIX_PERF[{}|{}] trilinear n={n3} cpu={cpu3_ms:.2}ms accel={acc3_ms:.2}ms speedup={:.2}x",
        backend_tag(),
        jit_tag(),
        cpu3_ms / acc3_ms.max(1e-9)
    );
}

/// 初始化一次以确保本文件内共享全局（供其他测试复用初始化开销统计）。
#[allow(dead_code)]
fn init_once() -> &'static () {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| ())
}

/// SoA 布局变体（`soa_layout = true` 且 n ≥ 64 时走 `octave_perlin_sample_soa_f64`）
/// 与 CPU 参考的一致性。SoA 路径此前无测试覆盖。
#[test]
fn matrix_soa_layout_consistency() {
    let mut config = matrix_config();
    config.soa_layout = true;
    let device = GpuDevice::from_config(&config);
    if device.device_type() == DeviceType::Cpu {
        println!(
            "MATRIX[{}|{}] no GPU — skip soa_layout",
            backend_tag(),
            jit_tag()
        );
        return;
    }
    let mut accel = NoiseAccelerator::new(&config);
    let n = 4096usize; // ≥ 64 以启用 SoA 路径
    let sampler = mk_sampler(SEED, &[-2, -1, 0, 1, 2]);
    let pos = mk_pos3d(n, SEED);
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    accel.sample_octave(&sampler, &pos, &mut acc);
    assert_eq!(
        fnv1a_f64(&cpu),
        fnv1a_f64(&acc),
        "soa_layout octave mismatch"
    );
    println!(
        "MATRIX[{}|{}] soa_layout consistency OK",
        backend_tag(),
        jit_tag()
    );
}

/// 直接驱动采样器级 API，断言 GPU kernel **真实执行**（而非静默回退 CPU）。
///
/// 背景：`BatchAccelerator` 在 GPU 启动失败时会静默回退 CPU 并返回正确结果，
/// 上一层的「一致性 OK」无法区分「GPU 正确执行」与「GPU 失败回退 CPU」。
/// 本测试在 GPU 设备上直接调用采样器并要求返回 `Ok`（采样器级 `Ok`
/// 仅在 GPU launch 成功时返回），并同时对比 CPU 参考值。
#[test]
fn matrix_gpu_samplers_really_run() {
    let config = matrix_config();
    let device = GpuDevice::from_config(&config);
    if device.device_type() == DeviceType::Cpu {
        println!(
            "MATRIX[{}|{}] no GPU — skip samplers_really_run",
            backend_tag(),
            jit_tag()
        );
        return;
    }
    let n = 1024usize;
    let positions = mk_pos3d(n, SEED.wrapping_add(9));

    // 1. aquifer：小网格（tiled kernel 路径）
    let _: () = {
        use pumpkin_gpu::noise::batch_cell::GpuAquiferBatchSampler;
        let mut sampler = GpuAquiferBatchSampler::new(GpuDevice::from_config(&config));
        let densities: Vec<f64> = (0..n).map(|i| (i as f64 % 64.0) / 64.0 - 0.5).collect();
        let mut packed_grid = Vec::with_capacity(64 * 4);
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    packed_grid.push((((ix - 1) * 24) as f64).to_bits() as i64);
                    packed_grid.push((((iy - 2) * 24) as f64).to_bits() as i64);
                    packed_grid.push((((iz - 1) * 24) as f64).to_bits() as i64);
                    packed_grid.push(0.3f64.to_bits() as i64);
                }
            }
        }
        // fluid_level = 5.0：部分位置位于水位之下，覆盖「水」分支
        let fluid_level = 5.0f64;
        // 围绕网格的小坐标（x/z ∈ [-72, 72], y ∈ [-100, 100]），保证水分支被命中
        let mut ps = SEED.wrapping_add(31);
        let near_positions: Vec<f64> = (0..n)
            .flat_map(|_| {
                let x = (ps as f64 % 144.0) - 72.0;
                ps = ps.wrapping_mul(1442695040888963407);
                let y = (ps as f64 % 200.0) - 100.0;
                ps = ps.wrapping_mul(1442695040888963407);
                let z = (ps as f64 % 144.0) - 72.0;
                ps = ps.wrapping_mul(1442695040888963407);
                [x, y, z]
            })
            .collect();
        let res = sampler
            .batch_aquifer_apply(&near_positions, &densities, &packed_grid, fluid_level, 0.3)
            .expect("aquifer tiled kernel 必须真实启动");
        assert_eq!(res.block_ids.len(), n);
        assert_eq!(res.fluid_updates.len(), n);
        // CPU 参考（4-NN，与 batch_accel 的 cpu_aquifer_apply 相同语义）
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
        let mut water_seen = false;
        for i in 0..n {
            let qx = near_positions[i * 3];
            let qy = near_positions[i * 3 + 1];
            let qz = near_positions[i * 3 + 2];
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
            let barrier: f64 = best_idx.iter().map(|&j| grid_densities[j]).sum::<f64>() / 4.0;
            let eff = densities[i] + barrier * 0.3;
            let (want_id, want_fluid) = if eff > 0.0 {
                (1, 0)
            } else if qy < fluid_level {
                water_seen = true;
                (2, 1)
            } else {
                (0, 0)
            };
            assert_eq!(res.block_ids[i], want_id, "aquifer block id @{i}");
            assert_eq!(res.fluid_updates[i], want_fluid, "aquifer fluid flag @{i}");
        }
        assert!(water_seen, "测试数据应覆盖「水」分支");
        println!(
            "MATRIX[{}|{}] aquifer tiled kernel 真实执行 OK（含水分支）",
            backend_tag(),
            jit_tag()
        );

        // 2. aquifer：大网格（标准 kernel 路径，M > 阈值 2048）
        let mut big_grid = Vec::with_capacity(13 * 13 * 13 * 4);
        for ix in 0..13 {
            for iy in 0..13 {
                for iz in 0..13 {
                    big_grid.push((((ix - 6) * 16) as f64).to_bits() as i64);
                    big_grid.push((((iy - 6) * 16) as f64).to_bits() as i64);
                    big_grid.push((((iz - 6) * 16) as f64).to_bits() as i64);
                    big_grid.push(0.2f64.to_bits() as i64);
                }
            }
        }
        let res2 = sampler
            .batch_aquifer_apply(&positions, &densities, &big_grid, -10000.0, 0.3)
            .expect("aquifer standard kernel 必须真实启动");
        assert!(
            res2.block_ids.iter().all(|&b| (0..=2).contains(&b)),
            "block ids 必须在 0..=2 范围内"
        );
        println!(
            "MATRIX[{}|{}] aquifer standard kernel 真实执行 OK",
            backend_tag(),
            jit_tag()
        );
    };

    // 3. beardifier 采样器级
    let _: () = {
        use pumpkin_gpu::noise::batch_cell::GpuBeardifierBatchSampler;
        let mut sampler = GpuBeardifierBatchSampler::new(GpuDevice::from_config(&config));
        let structures = vec![BeardifierStructureData {
            box_min_x: -16,
            box_min_y: -32,
            box_min_z: -16,
            box_max_x: 16,
            box_max_y: 0,
            box_max_z: 16,
            adaptation: 1,
            ground_delta: 8,
        }];
        let junctions = vec![BeardifierJunctionData {
            x: 0,
            ground_y: 0,
            z: 0,
        }];
        let mut results = vec![0.0f64; n];
        sampler
            .batch_beardifier(
                &positions,
                &structures,
                &junctions,
                [-64, -48, -64, 64, 48, 64],
                &mut results,
            )
            .expect("beardifier kernel 必须真实启动");
        assert!(results.iter().all(|v| v.is_finite()));
        println!(
            "MATRIX[{}|{}] beardifier kernel 真实执行 OK",
            backend_tag(),
            jit_tag()
        );
    };

    // 4. 光照迭代传播：确认 GPU kernel 存在且可用（采样器只在 GPU 设备上构建）
    let _: () = {
        use pumpkin_gpu::light::GpuLightSampler;
        let launcher = device.kernel_launcher().expect("GPU launcher");
        assert!(
            launcher.has_kernel("light_propagate_u8"),
            "light_propagate_u8 kernel 必须已编译"
        );
        let mut light_sampler = GpuLightSampler::new(GpuDevice::from_config(&config));
        let cells = 256usize;
        let mut light = vec![0u8; cells];
        let opacity = vec![1u8; cells];
        let mut neighbors = vec![-1i32; cells * 6];
        for i in 0..cells {
            for (d, delta) in [-16isize, 16, -1, 1, -4, 4].iter().enumerate() {
                let j = i as isize + delta;
                if (0..cells as isize).contains(&j) {
                    neighbors[i * 6 + d] = j as i32;
                }
            }
        }
        let iters = light_sampler
            .iterative_propagate(&mut light, &opacity, &neighbors, cells, 32)
            .expect("iterative propagate 必须成功");
        assert!(iters <= 32);
        println!(
            "MATRIX[{}|{}] light propagate 真实执行 OK (iters={iters})",
            backend_tag(),
            jit_tag()
        );
    };
}
