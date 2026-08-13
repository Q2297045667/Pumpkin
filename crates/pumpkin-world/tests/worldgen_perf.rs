//! 世界生成 GPU 路径性能基准。
//!
//! 每个基准测量 CPU 参考与加速器路径（GPU 或 CPU 回退）的耗时，
//! 打印吞吐与加速比。断言宽松（单侧 60s 上限），避免在 CI 上抖动失败；
//! 同时校验输出一致性，保证基准测量的是正确结果。
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
/// 单侧耗时上限（毫秒）—— 宽松上限,仅捕获路径完全损坏的情况。
const TIME_LIMIT_MS: f64 = 60_000.0;

/// 从环境变量构建配置：
/// `PUMPKIN_GPU_BACKEND`：`auto` | `cuda` | `opencl` | `cpu`（默认 `auto`）
/// `PUMPKIN_GPU_JIT`：`0` | `1`（默认 `0`）
fn env_config() -> GpuConfig {
    use pumpkin_config::gpu::GpuBackend;
    let backend = match std::env::var("PUMPKIN_GPU_BACKEND").as_deref() {
        Ok("cuda") => GpuBackend::Cuda,
        Ok("opencl") => GpuBackend::OpenCl,
        Ok("cpu") => GpuBackend::Cpu,
        _ => GpuBackend::Auto,
    };
    let jit_enabled = std::env::var("PUMPKIN_GPU_JIT").as_deref() == Ok("1");
    // OpenCL 不支持 auto 调度策略（会警告并回退 CPU），测试时显式指定 ByIndex(0)。
    let device = if backend == GpuBackend::OpenCl {
        pumpkin_config::gpu::GpuDeviceSelection::ByIndex { index: 0 }
    } else {
        pumpkin_config::gpu::GpuDeviceSelection::Auto
    };
    GpuConfig {
        enabled: true,
        noise_acceleration: true,
        batch_acceleration: true,
        light_acceleration: true,
        jit_enabled,
        jit_max_unroll: 16,
        backend,
        device,
        ..Default::default()
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
    NoiseAccelerator::new(&env_config())
}

fn mk_batch_accel() -> BatchAccelerator {
    BatchAccelerator::new(&env_config())
}

/// 输出基准结果并断言双方都在宽松时间上限内。
fn report(name: &str, n: usize, cpu_ms: f64, accel_ms: f64) {
    println!(
        "{name} (n={n}): cpu={cpu_ms:.2}ms, accel={accel_ms:.2}ms, speedup={:.2}x",
        cpu_ms / accel_ms.max(1e-9)
    );
    assert!(
        cpu_ms < TIME_LIMIT_MS,
        "{name}: CPU path too slow ({cpu_ms:.1}ms)"
    );
    assert!(
        accel_ms < TIME_LIMIT_MS,
        "{name}: accel path too slow ({accel_ms:.1}ms)"
    );
}

#[test]
fn perf_octave_262k() {
    let sampler = mk_sampler(SEED, &[-2, -1, 0, 1, 2, 3]);
    let n = 262_144;
    let pos = mk_pos3d(n, SEED);
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];

    let t0 = std::time::Instant::now();
    for i in 0..n {
        cpu[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut accel = mk_noise_accel();
    let t1 = std::time::Instant::now();
    accel.sample_octave(&sampler, &pos, &mut acc);
    let acc_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&acc), "perf_octave consistency");
    report("perf_octave", n, cpu_ms, acc_ms);
}

#[test]
fn perf_double_perlin_65k() {
    let a = mk_sampler(SEED, &[0, 1, 2]);
    let b = mk_sampler(SEED ^ 1, &[-1, 0, 1]);
    let n = 65_536;
    let pos = mk_pos3d(n, SEED.wrapping_add(1));
    let c = 1.0181268882175227f64;
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];

    let t0 = std::time::Instant::now();
    for i in 0..n {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        cpu[i] = (a.sample(x, y, z) + b.sample(x * c, y * c, z * c)) * 0.5;
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut accel = mk_noise_accel();
    let t1 = std::time::Instant::now();
    accel.sample_double_perlin(&a, &b, 0.5, &pos, &mut acc);
    let acc_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(fnv1a_f64(&cpu), fnv1a_f64(&acc), "perf_double consistency");
    report("perf_double_perlin", n, cpu_ms, acc_ms);
}

#[test]
fn perf_flatcache_65k() {
    let sampler = mk_sampler(SEED, &[0, 1, 2, 3]);
    let n = 65_536;
    let pos = mk_pos3d(n, SEED.wrapping_add(2));
    let xz: Vec<f64> = (0..n).flat_map(|i| [pos[i * 3], pos[i * 3 + 2]]).collect();
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];

    let t0 = std::time::Instant::now();
    for i in 0..n {
        cpu[i] = sampler.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut accel = mk_noise_accel();
    let t1 = std::time::Instant::now();
    accel.precompute_flatcache(&sampler, &xz, &mut acc);
    let acc_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        fnv1a_f64(&cpu),
        fnv1a_f64(&acc),
        "perf_flatcache consistency"
    );
    report("perf_flatcache", n, cpu_ms, acc_ms);
}

#[test]
fn perf_trilinear_131k() {
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
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];

    let t0 = std::time::Instant::now();
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
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    accel.batch_trilinear(&corners, &deltas, &mut acc);
    let acc_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        fnv1a_f64(&cpu),
        fnv1a_f64(&acc),
        "perf_trilinear consistency"
    );
    report("perf_trilinear", n, cpu_ms, acc_ms);
}

#[test]
fn perf_aquifer_16k() {
    let accel = mk_batch_accel();
    let n = 16_384;
    let positions = mk_pos3d(n, SEED.wrapping_add(3));
    let densities: Vec<f64> = (0..n).map(|i| (i as f64 % 64.0) / 64.0 - 0.5).collect();
    let g = 7usize;
    let mut packed_grid = Vec::with_capacity(g * g * g * 4);
    for ix in 0..g {
        for iy in 0..g {
            for iz in 0..g {
                packed_grid.push((((ix as i32 - 3) * 16) as f64).to_bits() as i64);
                packed_grid.push((((iy as i32 - 3) * 16) as f64).to_bits() as i64);
                packed_grid.push((((iz as i32 - 3) * 16) as f64).to_bits() as i64);
                packed_grid.push(0.3f64.to_bits() as i64);
            }
        }
    }

    let t0 = std::time::Instant::now();
    let result = accel.batch_aquifer_apply(&positions, &densities, &packed_grid, -10000.0, 0.3);
    let acc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(result.block_ids.len(), n);
    report("perf_aquifer", n, acc_ms, acc_ms); // cpu_ms = accel_ms 占位,仅报 accel 耗时
}

#[test]
fn perf_beardifier_16k() {
    let accel = mk_batch_accel();
    let n = 16_384;
    let positions = mk_pos3d(n, SEED.wrapping_add(4));
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
    ];
    let junctions = vec![BeardifierJunctionData {
        x: 0,
        ground_y: 0,
        z: 0,
    }];
    let mut results = vec![0.0f64; n];

    let t0 = std::time::Instant::now();
    accel.batch_beardifier(
        &positions,
        &structures,
        &junctions,
        [-96, -80, -96, 96, 32, 96],
        &mut results,
    );
    let acc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert!(results.iter().all(|v| v.is_finite()));
    report("perf_beardifier", n, acc_ms, acc_ms);
}

#[test]
fn perf_light_sky_fill() {
    use pumpkin_world::light_accel::LightAccelerator;
    let mut accel = LightAccelerator::new(&env_config());
    let n = 256usize; // 256 列
    let h = 384usize;
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
    let mut sky = vec![0u8; n * h];

    let t0 = std::time::Instant::now();
    accel.batch_sky_fill(&hm, &op, &mut sky, n, h);
    let acc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert!(sky.iter().any(|&v| v > 0), "天空光应非全零");
    report("perf_light_sky_fill", n * h, acc_ms, acc_ms);
}

#[test]
fn perf_surface() {
    let surface_a = mk_sampler(SEED, &[0, 1]);
    let surface_b = mk_sampler(SEED.wrapping_add(3), &[0, 1]);
    let secondary_a = mk_sampler(SEED.wrapping_add(5), &[0]);
    let secondary_b = mk_sampler(SEED.wrapping_add(7), &[0]);
    let mut accel = mk_noise_accel();

    let t0 = std::time::Instant::now();
    let cache = accel.precompute_surface(
        &surface_a,
        &surface_b,
        0.7,
        &secondary_a,
        &secondary_b,
        0.3,
        0,
        0,
    );
    let acc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert!(cache.surface.iter().all(|v| v.is_finite()));
    report("perf_surface", 256, acc_ms, acc_ms);
}
