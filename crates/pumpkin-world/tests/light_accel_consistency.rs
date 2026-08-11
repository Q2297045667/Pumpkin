//! `LightAccelerator` 指纹一致性测试 + 基准。
//!
//! 固定种子 138782381985206，对比 CPU 光照与 GPU 光照路径的输出。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    unused_variables
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_world::light_accel::LightAccelerator;

const SEED: u64 = 138_782_381_985_206;

// ============================================================================
// 辅助函数
// ============================================================================

fn fnv1a_u8(d: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in d {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn mk_light_accel() -> LightAccelerator {
    LightAccelerator::new(&GpuConfig {
        enabled: true,
        light_acceleration: true,
        ..Default::default()
    })
}

// ============================================================================
// 天空光填充
// ============================================================================

#[test]
fn sky_fill_single_column() {
    let n = 1usize;
    let h = 16;
    let hm = vec![8i32];
    let op = vec![1u8; n * h];
    let mut cpu = vec![0u8; n * h];
    let mut gpu = vec![0u8; n * h];

    // CPU
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

    // GPU
    let mut accel = mk_light_accel();
    accel.batch_sky_fill(&hm, &op, &mut gpu, n, h);

    assert_eq!(fnv1a_u8(&cpu), fnv1a_u8(&gpu), "sky_fill_single");
}

#[test]
fn sky_fill_16x256() {
    let n = 16usize; // 16 columns = one chunk row
    let h = 256;
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

    let mut cpu = vec![0u8; n * h];
    let mut gpu = vec![0u8; n * h];

    // CPU
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

    // GPU
    let mut accel = mk_light_accel();
    accel.batch_sky_fill(&hm, &op, &mut gpu, n, h);

    assert_eq!(fnv1a_u8(&cpu), fnv1a_u8(&gpu), "sky_fill_16x256");
}

#[test]
fn sky_fill_no_opacity() {
    // 透明度全为零 → 天空光应全部为 15
    let n = 8usize;
    let h = 32;
    let hm = vec![16i32; n];
    let op = vec![0u8; n * h];
    let mut cpu = vec![0u8; n * h];
    let mut gpu = vec![0u8; n * h];

    for col in 0..n {
        for y in 0..h {
            cpu[col * h + y] = 15;
        }
    }

    let mut accel = mk_light_accel();
    accel.batch_sky_fill(&hm, &op, &mut gpu, n, h);

    assert_eq!(fnv1a_u8(&cpu), fnv1a_u8(&gpu), "sky_fill_no_opacity");
}

// ============================================================================
// 方块光扫描
// ============================================================================

#[test]
fn block_scan_consistency() {
    let n = 16384;
    let mut s = SEED;
    let lum: Vec<u8> = (0..n)
        .map(|_| {
            let v = (s % 16) as u8;
            s = s.wrapping_mul(1442695040888963407);
            if v > 14 { 0 } else { v }
        })
        .collect();

    let mut cpu_bl = vec![0u8; n];
    let mut gpu_bl = vec![0u8; n];

    // CPU
    let mut cpu_src = Vec::new();
    for i in 0..n {
        cpu_bl[i] = lum[i];
        if lum[i] > 0 {
            cpu_src.push(i as i32);
        }
    }

    // GPU
    let mut accel = mk_light_accel();
    let gpu_src = accel.batch_block_scan(&lum, &mut gpu_bl, n);

    assert_eq!(fnv1a_u8(&cpu_bl), fnv1a_u8(&gpu_bl), "block_scan_values");
    assert_eq!(cpu_src.len(), gpu_src.len(), "block_scan_count");
}

#[test]
fn block_scan_no_sources() {
    let n = 1024;
    let lum = vec![0u8; n];
    let mut cpu_bl = vec![0u8; n];
    let mut gpu_bl = vec![0u8; n];

    let mut accel = mk_light_accel();
    let gpu_src = accel.batch_block_scan(&lum, &mut gpu_bl, n);

    assert_eq!(fnv1a_u8(&cpu_bl), fnv1a_u8(&gpu_bl), "no_sources_values");
    assert!(gpu_src.is_empty(), "no_sources_empty");
}

// ============================================================================
// 光照传播
// ============================================================================

#[test]
fn propagate_small_grid() {
    let n = 64usize;
    let max_iters = 128;
    let mut s = SEED;

    // 生成 6-neighbor 连接表（稀疏随机图）
    let mut neighbors = Vec::with_capacity(n * 6);
    for i in 0..n {
        for _ in 0..6 {
            let ni = (s as usize % n) as i32;
            s = s.wrapping_mul(1442695040888963407);
            neighbors.push(ni);
        }
    }

    let op: Vec<u8> = (0..n)
        .map(|_| {
            let v = (s % 4) as u8;
            s = s.wrapping_mul(1442695040888963407);
            v
        })
        .collect();

    let init: Vec<u8> = (0..n)
        .map(|_| {
            let v = if s.is_multiple_of(5) { 15u8 } else { 0u8 };
            s = s.wrapping_mul(1442695040888963407);
            v
        })
        .collect();

    // CPU 传播
    let mut cpu_lt = init.clone();
    let mut cpu_iters = 0;
    for _ in 0..max_iters {
        let mut ch = false;
        for i in 0..n {
            let cur = cpu_lt[i];
            let mut best = cur;
            for d in 0..6 {
                let ni = neighbors[i * 6 + d] as usize;
                if ni < n {
                    let nl = cpu_lt[ni];
                    let no = op[ni];
                    let p = if nl > 1 + no { nl - 1 - no } else { 0 };
                    if p > best {
                        best = p;
                    }
                }
            }
            if best > cur {
                cpu_lt[i] = best;
                ch = true;
            }
        }
        cpu_iters += 1;
        if !ch {
            break;
        }
    }

    // GPU 传播
    let mut gpu_lt = init; // 直接移动 init（CPU 已 clone）
    let mut accel = mk_light_accel();
    let gpu_iters = accel.iterative_propagate(&mut gpu_lt, &op, &neighbors, n, max_iters);

    assert_eq!(fnv1a_u8(&cpu_lt), fnv1a_u8(&gpu_lt), "propagate_values");
    assert_eq!(cpu_iters, gpu_iters, "propagate_iters");
}

#[test]
fn propagate_empty() {
    let n = 0;
    let max_iters = 10;
    let mut lt = vec![];
    let op = vec![];
    let nb = vec![];
    let mut accel = mk_light_accel();
    let iters = accel.iterative_propagate(&mut lt, &op, &nb, n, max_iters);
    assert_eq!(iters, 0);
}

// ============================================================================
// 基准
// ============================================================================

#[test]
fn bench_sky_fill() {
    let n = 324; // 18x18 chunk area
    let h = 384;
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

    let mut cpu = vec![0u8; n * h];
    let mut gpu = vec![0u8; n * h];

    let t0 = std::time::Instant::now();
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
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut accel = mk_light_accel();
    let t1 = std::time::Instant::now();
    accel.batch_sky_fill(&hm, &op, &mut gpu, n, h);
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(fnv1a_u8(&cpu), fnv1a_u8(&gpu), "bench_sky");

    println!(
        "sky_fill(n={n},h={h}): cpu={cpu_ms:.1}ms, gpu={gpu_ms:.1}ms, speedup={:.2}x",
        cpu_ms / gpu_ms.max(1e-9)
    );
}
