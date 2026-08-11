//! Lighting GPU 加速指纹测试。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, unused_mut)]
#![cfg(feature = "gpu")]
use pumpkin_config::gpu::GpuConfig;
use pumpkin_world::light_accel::LightAccelerator;

const SEED: u64 = 138_782_381_985_206;
fn accel() -> LightAccelerator {
    LightAccelerator::new(&GpuConfig {
        enabled: true,
        light_acceleration: true,
        ..Default::default()
    })
}

fn make_hm_op_nb(n: usize, h: usize) -> (Vec<i32>, Vec<u8>, Vec<i32>) {
    let mut hm = vec![0i32; n];
    let mut op = vec![0u8; n * h];
    let mut s = SEED;
    for c in 0..n {
        hm[c] = ((s as u64 % 200) + 64) as i32;
        s = s.wrapping_mul(1442695040888963407);
        for y in 0..h {
            op[c * h + y] = s as u8 % 16;
            s = s.wrapping_mul(1442695040888963407);
        }
    }
    let nb: Vec<i32> = (0..n * 6)
        .map(|i| {
            if i % 6 == 0 && i / 6 % h > 0 {
                ((i / 6) as i32) - (h as i32)
            } else if i % 6 == 1 && i / 6 % h < h - 1 {
                ((i / 6) as i32) + (h as i32)
            } else if i % 6 == 2 && (i / 6) as i32 > 0 {
                (i / 6) as i32 - 1
            } else if i % 6 == 3 && ((i / 6) as i32) < n as i32 - 1 {
                (i / 6) as i32 + 1
            } else {
                -1i32
            }
        })
        .collect();
    (hm, op, nb)
}

#[test]
fn sky_fill_consistency() {
    let (hm, op, _) = make_hm_op_nb(324, 384);
    let mut cpu = vec![0u8; 324 * 384];
    let mut gpu = vec![0u8; 324 * 384];
    for col in 0..324 {
        let t = hm[col] as i32;
        for y in (t + 1)..384 {
            cpu[col * 384 + y as usize] = 15;
        }
        let mut lt: u8 = 15;
        for y in (0..=t).rev() {
            let i = col * 384 + y as usize;
            lt = lt.saturating_sub(op[i]);
            cpu[i] = lt;
        }
    }
    accel().batch_sky_fill(&hm, &op, &mut gpu, 324, 384);
    assert_eq!(cpu, gpu, "sky_fill mismatch");
}

#[test]
fn block_scan_consistency() {
    let n = 65536;
    let mut lum = vec![0u8; n];
    let mut s = SEED;
    for i in 0..n {
        lum[i] = s as u8 % 16;
        if lum[i] > 14 {
            lum[i] = 0;
        }
        s = s.wrapping_mul(1442695040888963407);
    }
    let mut cpu_bl = vec![0u8; n];
    let mut gpu_bl = vec![0u8; n];
    let mut sc = Vec::new();
    for i in 0..n {
        cpu_bl[i] = lum[i];
        if lum[i] > 0 {
            sc.push(i as i32);
        }
    }
    let sg = accel().batch_block_scan(&lum, &mut gpu_bl, n);
    assert_eq!(cpu_bl, gpu_bl, "block_scan");
    assert_eq!(sc, sg, "sources");
}

#[test]
fn propagate_consistency() {
    let (_, op, nb) = make_hm_op_nb(1024, 64);
    let mut _lt = vec![15u8; 1024];
    let mut cpu_lt = vec![15u8; 1024];
    let mut gpu_lt = vec![15u8; 1024];
    // CPU iterative distance field
    let mut it = 0;
    loop {
        let mut ch = false;
        for i in 0..1024 {
            let cur = cpu_lt[i];
            let op_v = op[i];
            let mut best = cur;
            for d in 0..6 {
                let ni = nb[i * 6 + d] as usize;
                if ni < 1024 {
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
        it += 1;
        if !ch {
            break;
        }
    }
    let gi = accel().iterative_propagate(&mut gpu_lt, &op, &nb, 1024, 50);
    assert_eq!(cpu_lt, gpu_lt, "propagate");
    assert!(gi <= it, "GPU it={gi} should <= CPU it={it}");
}

#[test]
fn perf_propagate() {
    let (_, op, nb) = make_hm_op_nb(65536, 64);
    let mut _lt = vec![15u8; 65536];
    let n_iter = 10u32;
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let mut l2 = vec![15u8; 65536];
        let mut ch = true;
        while ch {
            ch = false;
            for i in 0..65536 {
                let cur = l2[i];
                let _op_v = op[i];
                let mut best = cur;
                for d in 0..6 {
                    let ni = nb[i * 6 + d] as usize;
                    if ni < 65536 {
                        let nl = l2[ni];
                        let p = if nl > 1 + op[ni] { nl - 1 - op[ni] } else { 0 };
                        if p > best {
                            best = p;
                        }
                    }
                }
                if best > cur {
                    l2[i] = best;
                    ch = true;
                }
            }
        }
    }
    let cm = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..n_iter {
        let mut gpu = vec![15u8; 65536];
        accel().iterative_propagate(&mut gpu, &op, &nb, 65536, 50);
    }
    let gm = t1.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;
    println!(
        "Light propagate: cpu={cm:.1}ms, gpu={gm:.1}ms, speedup={:.2}x",
        cm / gm
    );
    assert!(gm < 5000.0, "batch propagation should complete within 5s");
}
