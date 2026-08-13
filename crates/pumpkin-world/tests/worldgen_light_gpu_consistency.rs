//! 真 GPU 光照一致性测试。
//!
//! 光照 kernel（天空光填充 / 方块光扫描 / 迭代传播 / 水平传播）此前仅在 CPU
//! 回退模式下验证；本文件在真 GPU（CUDA / OpenCL）上对比 GPU 与 CPU 参考输出。
//! 无 GPU 时自动跳过。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_gpu::{DeviceType, GpuDevice};
use pumpkin_world::light_accel::LightAccelerator;

const SEED: u64 = 138_782_381_985_206;

fn fnv1a_u8(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 有真 GPU 时返回配置对应的加速器，否则返回 None（测试跳过）。
fn gpu_light_accel_or_none() -> Option<LightAccelerator> {
    let device = GpuDevice::init();
    if device.device_type() == DeviceType::Cpu {
        eprintln!("SKIP: 无可用 GPU 设备");
        None
    } else {
        Some(LightAccelerator::new(&GpuConfig {
            enabled: true,
            light_acceleration: true,
            ..Default::default()
        }))
    }
}

#[test]
fn gpu_sky_fill_vs_cpu() {
    let Some(mut accel) = gpu_light_accel_or_none() else {
        return;
    };
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

    let mut gpu = vec![0u8; n * h];
    accel.batch_sky_fill(&hm, &op, &mut gpu, n, h);

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
    assert_eq!(
        fnv1a_u8(&cpu),
        fnv1a_u8(&gpu),
        "真 GPU 天空光填充与 CPU 不一致"
    );
}

#[test]
fn gpu_block_scan_vs_cpu() {
    let Some(mut accel) = gpu_light_accel_or_none() else {
        return;
    };
    let n = 65536usize;
    let mut s = SEED;
    let lum: Vec<u8> = (0..n)
        .map(|_| {
            let v = (s % 16) as u8;
            s = s.wrapping_mul(1442695040888963407);
            v
        })
        .collect();

    let mut gpu_bl = vec![0u8; n];
    let gpu_sources = accel.batch_block_scan(&lum, &mut gpu_bl, n);

    let mut cpu_sources = Vec::new();
    for (i, &v) in lum.iter().enumerate() {
        if v > 0 {
            cpu_sources.push(i as i32);
        }
    }
    // GPU kernel 用原子计数收集源索引，顺序不固定；按集合比对。
    let mut gpu_sorted = gpu_sources;
    gpu_sorted.sort_unstable();
    assert_eq!(gpu_sorted, cpu_sources, "真 GPU 方块光扫描源列表不一致");
    assert_eq!(
        fnv1a_u8(&gpu_bl),
        fnv1a_u8(&lum),
        "真 GPU 方块光扫描值不一致"
    );
}

#[test]
fn gpu_iterative_propagate_vs_cpu() {
    let Some(mut accel) = gpu_light_accel_or_none() else {
        return;
    };
    // 5×5×5 网格
    let side = 5usize;
    let n = side * side * side;
    let mut s = SEED;
    let opacity: Vec<u8> = (0..n)
        .map(|_| {
            let v = (s % 4) as u8;
            s = s.wrapping_mul(1442695040888963407);
            v
        })
        .collect();
    let mut light = vec![0u8; n];
    light[n / 2] = 15; // 中心光源
    let mut neighbors = Vec::with_capacity(n * 6);
    for i in 0..n {
        let x = (i % side) as i32;
        let y = ((i / side) % side) as i32;
        let z = (i / (side * side)) as i32;
        let side_i = side as i32;
        for (dx, dy, dz) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            let idx = if (0..side_i).contains(&nx)
                && (0..side_i).contains(&ny)
                && (0..side_i).contains(&nz)
            {
                nz * side_i * side_i + ny * side_i + nx
            } else {
                -1
            };
            neighbors.push(idx);
        }
    }

    let mut gpu_light = light.clone();
    let gpu_iters = accel.iterative_propagate(&mut gpu_light, &opacity, &neighbors, n, 64);

    let mut cpu_light = light.clone();
    let mut cpu_iters = 0;
    for _ in 0..64 {
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
        cpu_iters += 1;
        if !changed {
            break;
        }
    }

    assert_eq!(
        fnv1a_u8(&cpu_light),
        fnv1a_u8(&gpu_light),
        "真 GPU 迭代传播与 CPU 不一致"
    );
    assert!(gpu_iters > 0 && cpu_iters > 0, "应发生传播迭代");
}

#[test]
fn gpu_sky_horizontal_vs_cpu() {
    let Some(mut accel) = gpu_light_accel_or_none() else {
        return;
    };
    let width = 6usize;
    let depth = 6usize;
    let height = 32usize;
    let n_total = width * depth * height;

    // 顶部一行光源 + 半透明方块
    let mut sky = vec![0u8; n_total];
    for z in 0..depth {
        for x in 0..width {
            sky[z * width * height + x * height + height - 1] = 15;
        }
    }
    let opacity = vec![1u8; n_total];

    let mut gpu_sky = sky.clone();
    let gpu_iters =
        accel.sky_horizontal_propagate(&mut gpu_sky, &opacity, width, depth, height, 64);

    // CPU 参考（与 LightAccelerator CPU 回退逻辑一致）
    let mut cpu_sky = sky.clone();
    let stride_x = height;
    let stride_z = width * height;
    let mut cpu_iters = 0;
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
                    // 向下级联：上方为 15 且当前为空气（透明度 0）时保持 15
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
        cpu_iters += 1;
        if !changed {
            break;
        }
    }

    assert_eq!(
        fnv1a_u8(&cpu_sky),
        fnv1a_u8(&gpu_sky),
        "真 GPU 天空光水平传播与 CPU 不一致"
    );
    assert!(gpu_iters > 0 && cpu_iters > 0, "应发生水平传播迭代");
}
