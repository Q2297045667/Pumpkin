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

// ============================================================================
// 天空光水平传播
// ============================================================================

fn cpu_sky_horizontal(
    sky_light: &mut [u8],
    opacity: &[u8],
    width: usize,
    depth: usize,
    height: usize,
    max_iters: usize,
) -> usize {
    let stride_x = height;
    let stride_z = width * height;
    let mut iterations = 0;
    for _ in 0..max_iters {
        let mut changed = false;
        for z in 0..depth {
            for x in 0..width {
                for y in (0..height).rev() {
                    let idx = z * stride_z + x * stride_x + y;
                    let cur = sky_light[idx];
                    let mut best = cur;
                    if x > 0 {
                        let nl = sky_light[idx - stride_x];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if x < width - 1 {
                        let nl = sky_light[idx + stride_x];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if z > 0 {
                        let nl = sky_light[idx - stride_z];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if z < depth - 1 {
                        let nl = sky_light[idx + stride_z];
                        if nl > 1 && nl - 1 > best {
                            best = nl - 1;
                        }
                    }
                    if y < height - 1 {
                        let above = sky_light[idx + 1];
                        if above == 15 && opacity[idx] == 0 && 15 > best {
                            best = 15;
                        }
                    }
                    if best > cur {
                        sky_light[idx] = best;
                        changed = true;
                    }
                }
            }
        }
        iterations += 1;
        if !changed {
            break;
        }
    }
    iterations
}

#[test]
fn sky_horizontal_small_grid() {
    // 2x2 grid, 8 height: one tall column casts light to adjacent short column
    let width = 2usize;
    let depth = 2usize;
    let height = 8usize;
    let n = width * depth * height; // 32

    // Tall column at (0,0) top=4, short at (1,0) top=2
    let mut cpu_lt = vec![0u8; n];
    let mut gpu_lt = vec![0u8; n];
    let opacity = vec![0u8; n]; // all air

    // Initial vertical fill (simplified: all air, light 15 above top_y)
    let hm = [4i32, 2i32, 4i32, 4i32]; // 2x2: (0,0)=4, (1,0)=2, (0,1)=4, (1,1)=4
    for z in 0..depth {
        for x in 0..width {
            let col = z * width + x;
            let top = hm[col];
            for y in 0..height {
                let idx = z * (width * height) + x * height + y;
                if (y as i32) > top {
                    cpu_lt[idx] = 15;
                    gpu_lt[idx] = 15;
                }
            }
            let mut l: u8 = 15;
            for y in (0..=top).rev() {
                let idx = z * (width * height) + x * height + (y as usize);
                cpu_lt[idx] = l;
                gpu_lt[idx] = l;
                l = l.saturating_sub(1); // air attenuation
            }
        }
    }

    let cpu_iters = cpu_sky_horizontal(&mut cpu_lt, &opacity, width, depth, height, 32);

    let mut accel = mk_light_accel();
    let gpu_iters = accel.sky_horizontal_propagate(&mut gpu_lt, &opacity, width, depth, height, 32);

    assert_eq!(
        fnv1a_u8(&cpu_lt),
        fnv1a_u8(&gpu_lt),
        "sky_horizontal_small values mismatch"
    );
    assert_eq!(cpu_iters, gpu_iters, "sky_horizontal_small iters mismatch");
}

#[test]
fn sky_horizontal_flat_equal() {
    // All columns same height = no horizontal propagation
    let width = 4usize;
    let depth = 4usize;
    let height = 16usize;
    let n = width * depth * height;

    let mut cpu_lt = vec![15u8; n]; // all sky
    let mut gpu_lt = cpu_lt.clone();
    let opacity = vec![0u8; n];

    let cpu_iters = cpu_sky_horizontal(&mut cpu_lt, &opacity, width, depth, height, 32);
    let mut accel = mk_light_accel();
    let gpu_iters = accel.sky_horizontal_propagate(&mut gpu_lt, &opacity, width, depth, height, 32);

    assert_eq!(
        fnv1a_u8(&cpu_lt),
        fnv1a_u8(&gpu_lt),
        "sky_horizontal_flat values mismatch"
    );
    // Should converge immediately (1 iteration = 1 pass that found no changes)
    assert_eq!(cpu_iters, gpu_iters, "iters");
    assert_eq!(cpu_iters, 1, "flat equal should converge in 1 iter");
}

#[test]
fn sky_horizontal_chequerboard() {
    // Chequerboard pattern: alternating tall/short columns
    let width = 4usize;
    let depth = 4usize;
    let height = 64usize;
    let n = width * depth * height;

    let mut cpu_lt = vec![0u8; n];
    let mut gpu_lt = vec![0u8; n];
    let mut opacity = vec![0u8; n];

    // Generate checkerboard heightmap and initial vertical fill
    let mut s = SEED;
    for z in 0..depth {
        for x in 0..width {
            let col = z * width + x;
            let top = if (x + z) % 2 == 0 { 48i32 } else { 32i32 };
            for y in 0..height {
                let idx = z * (width * height) + x * height + y;
                // Add some opacity to make it interesting
                opacity[idx] = (s % 4) as u8;
                s = s.wrapping_mul(1442695040888963407);
                if (y as i32) > top {
                    cpu_lt[idx] = 15;
                    gpu_lt[idx] = 15;
                }
            }
            // Vertical fill (top-down with opacity)
            let mut l: i32 = 15;
            for y in (0..=top).rev() {
                let idx = z * (width * height) + x * height + (y as usize);
                let op = opacity[idx] as i32;
                l = (l - op).max(0);
                cpu_lt[idx] = l as u8;
                gpu_lt[idx] = l as u8;
            }
        }
    }

    let cpu_iters = cpu_sky_horizontal(&mut cpu_lt, &opacity, width, depth, height, 32);
    let mut accel = mk_light_accel();
    let gpu_iters = accel.sky_horizontal_propagate(&mut gpu_lt, &opacity, width, depth, height, 32);

    assert_eq!(
        fnv1a_u8(&cpu_lt),
        fnv1a_u8(&gpu_lt),
        "sky_horizontal_chequer values mismatch"
    );
    assert_eq!(cpu_iters, gpu_iters, "iters");
}

#[test]
fn sky_horizontal_18x18x384() {
    // Full chunk-sized test: 18x18 columns, 384 height
    // Matches the dimensions used in LightEngine::sky_horizontal_propagate
    let width = 18usize;
    let depth = 18usize;
    let height = 384usize;
    let n = width * depth * height;

    let mut s = SEED;
    let mut cpu_lt = vec![0u8; n];
    let mut gpu_lt = vec![0u8; n];
    let mut opacity = vec![0u8; n];

    for z in 0..depth {
        for x in 0..width {
            let col = z * width + x;
            let top = ((s as i32).rem_euclid(128) + 48).clamp(0, (height - 1) as i32);
            s = s.wrapping_mul(1442695040888963407);
            for y in 0..height {
                let idx = z * (width * height) + x * height + y;
                opacity[idx] = (s % 6) as u8;
                s = s.wrapping_mul(1442695040888963407);
                if (y as i32) > top {
                    cpu_lt[idx] = 15;
                    gpu_lt[idx] = 15;
                }
            }
            let mut l: i32 = 15;
            for y in (0..=top).rev() {
                let idx = z * (width * height) + x * height + (y as usize);
                let op = opacity[idx] as i32;
                l = (l - op).max(0);
                cpu_lt[idx] = l as u8;
                gpu_lt[idx] = l as u8;
            }
        }
    }

    let t0 = std::time::Instant::now();
    let cpu_iters = cpu_sky_horizontal(&mut cpu_lt, &opacity, width, depth, height, 32);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let mut accel = mk_light_accel();
    let gpu_iters = accel.sky_horizontal_propagate(&mut gpu_lt, &opacity, width, depth, height, 32);
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        fnv1a_u8(&cpu_lt),
        fnv1a_u8(&gpu_lt),
        "sky_horizontal_18x18x384 values mismatch"
    );
    assert_eq!(cpu_iters, gpu_iters, "iters");

    println!(
        "sky_horizontal(18x18x384): cpu={cpu_ms:.1}ms cpu_iters={cpu_iters}, gpu={gpu_ms:.1}ms gpu_iters={gpu_iters}"
    );
}
