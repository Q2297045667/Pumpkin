//! CUDA persistent kernel（`light_propagate_u8_persistent`）一致性测试。
//!
//! 验证单次 cooperative launch 的迭代式距离场传播与 CPU 参考一致。
//! 仅当 CUDA 可用且 persistent 模式可启动时运行，否则跳过。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::needless_range_loop,
    clippy::cast_possible_wrap,
    clippy::doc_markdown
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::{GpuBackend, GpuConfig};
use pumpkin_world::light_accel::LightAccelerator;

fn fnv1a_u8(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// CPU 参考：迭代距离场传播。
fn cpu_propagate(light: &mut [u8], opacity: &[u8], neighbors: &[i32], n: usize, max_iters: usize) {
    for _ in 0..max_iters {
        let mut changed = false;
        for i in 0..n {
            let cur = light[i];
            let mut best = cur;
            for d in 0..6 {
                let n_idx = neighbors[i * 6 + d] as usize;
                if n_idx < n {
                    let nl = light[n_idx];
                    let n_op = opacity[n_idx];
                    let prop = if nl > 1 + n_op {
                        nl.saturating_sub(1 + n_op)
                    } else {
                        0
                    };
                    if prop > best {
                        best = prop;
                    }
                }
            }
            if best > cur {
                light[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

#[test]
fn persistent_propagate_matches_cpu() {
    let config = GpuConfig {
        enabled: true,
        light_acceleration: true,
        cuda: pumpkin_config::gpu::CudaConfig {
            persistent_kernels: true,
            ..Default::default()
        },
        backend: GpuBackend::Cuda,
        ..Default::default()
    };
    let mut accel = LightAccelerator::new(&config);
    if !accel.is_active() {
        println!("SKIP: CUDA 不可用，跳过 persistent kernel 测试");
        return;
    }

    // 16×16×16 网格（4096 单元，16 blocks × 256 threads）
    let size = 16usize;
    let n = size * size * size;
    let mut light = vec![0u8; n];
    let opacity: Vec<u8> = (0..n).map(|i| ((i * 7 + 3) % 4) as u8).collect();
    let mut neighbors = vec![-1i32; n * 6];
    for x in 0..size {
        for y in 0..size {
            for z in 0..size {
                let i = (x * size + y) * size + z;
                // 邻居偏移：x 轴 ±size²、y 轴 ±size、z 轴 ±1
                let deltas = [
                    (x > 0, -((size * size) as isize)),
                    (x < size - 1, (size * size) as isize),
                    (y > 0, -(size as isize)),
                    (y < size - 1, size as isize),
                    (z > 0, -1isize),
                    (z < size - 1, 1isize),
                ];
                for (d, (in_bounds, delta)) in deltas.iter().enumerate() {
                    if *in_bounds {
                        neighbors[i * 6 + d] = (i as isize + delta) as i32;
                    }
                }
            }
        }
    }
    // 放置一些光源
    for s in [0usize, 128, 4095] {
        light[s] = 15;
    }

    let mut gpu_light = light.clone();
    let iters = accel.iterative_propagate(&mut gpu_light, &opacity, &neighbors, n, 64);
    let mut cpu_light = light.clone();
    cpu_propagate(&mut cpu_light, &opacity, &neighbors, n, 64);

    assert_eq!(
        fnv1a_u8(&cpu_light),
        fnv1a_u8(&gpu_light),
        "persistent kernel 结果必须与 CPU 参考一致 (iters={iters})"
    );
    println!("persistent propagate OK: iters={iters}");
}
