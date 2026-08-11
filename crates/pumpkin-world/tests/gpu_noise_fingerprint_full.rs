//! 噪声阶段完整指纹测试 — 含三线性插值和 FlatCache。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, unused_mut)]
#![cfg(feature = "gpu")]
use pumpkin_config::gpu::GpuConfig;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::noise_accel::NoiseAccelerator;

const SEED: u64 = 138_782_381_985_206;
fn mk_pos3(n: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(n * 3);
    let mut s = SEED;
    for _ in 0..n {
        p.push((s.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        p.push((s as f64) * 1e-8);
    }
    p
}
fn mk_sampler(o: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(SEED);
    let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(o);
    let mut g = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
}
fn fnv1a(d: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in d {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
fn accel() -> NoiseAccelerator {
    NoiseAccelerator::new(&GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    })
}

#[test]
fn trilinear_fingerprint() {
    let n = 1024;
    let mut c = vec![0f64; n * 8];
    let mut d = vec![0f64; n * 3];
    let mut s = SEED;
    for i in 0..n * 8 {
        c[i] = (s.wrapping_mul(6364136223846793005) as f64) * 1e-10;
        s = s.wrapping_mul(1442695040888963407);
    }
    for i in 0..n * 3 {
        d[i] = (s as f64 % 1.0).abs();
        s = s.wrapping_mul(1442695040888963407);
    }
    let mut cpu = vec![0f64; n];
    let mut gpu = vec![0f64; n];
    for i in 0..n {
        let b = i * 8;
        let dx = d[i * 3];
        let dy = d[i * 3 + 1];
        let dz = d[i * 3 + 2];
        cpu[i] = c[b] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + c[b + 1] * dx * (1.0 - dy) * (1.0 - dz)
            + c[b + 2] * (1.0 - dx) * dy * (1.0 - dz)
            + c[b + 3] * dx * dy * (1.0 - dz)
            + c[b + 4] * (1.0 - dx) * (1.0 - dy) * dz
            + c[b + 5] * dx * (1.0 - dy) * dz
            + c[b + 6] * (1.0 - dx) * dy * dz
            + c[b + 7] * dx * dy * dz;
    }
    accel().batch_trilinear(&c, &d, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "trilinear hash mismatch");
}

#[test]
fn flatcache_fingerprint() {
    let s = mk_sampler(&[0, 1, 2, 3]);
    let n = 1024;
    let mut xz = vec![0f64; n * 2];
    let mut seed = SEED;
    for i in 0..n * 2 {
        xz[i] = (seed.wrapping_mul(6364136223846793005) as f64) * 1e-8;
        seed = seed.wrapping_mul(1442695040888963407);
    }
    let mut cpu = vec![0f64; n];
    let mut gpu = vec![0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(xz[i * 2], 0.0, xz[i * 2 + 1]);
    }
    accel().precompute_flatcache(&s, &xz, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "flatcache hash mismatch");
}

#[test]
fn all_noise_types() {
    let n = 1024;
    let p3 = mk_pos3(n);
    let p2: Vec<f64> = p3
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 != 1)
        .map(|(_, &v)| v)
        .collect();
    let mut cpu = vec![0f64; n];
    let mut gpu = vec![0f64; n];
    let s1 = mk_sampler(&[0, 1, 2]);
    for i in 0..n {
        cpu[i] = s1.sample(p3[i * 3], p3[i * 3 + 1], p3[i * 3 + 2]);
    }
    accel().sample_octave(&s1, &p3, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "octave");
    let sa = mk_sampler(&[0, 1, 2]);
    let sb = mk_sampler(&[1, 2, 3]);
    let amp = 0.5;
    let c = 1.0181268882175227;
    for i in 0..n {
        cpu[i] = (sa.sample(p3[i * 3], p3[i * 3 + 1], p3[i * 3 + 2])
            + sb.sample(p3[i * 3] * c, p3[i * 3 + 1] * c, p3[i * 3 + 2] * c))
            * amp;
    }
    accel().sample_double_perlin(&sa, &sb, amp, &p3, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "double_perlin");
    let s3 = mk_sampler(&[0, 1]);
    for i in 0..n {
        cpu[i] = s3.sample(p2[i * 2] * 0.25, 0.0, p2[i * 2 + 1] * 0.25) * 4.0;
    }
    accel().sample_shift_a(&s3, &p2, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "shift_a");
    let s4 = mk_sampler(&[0, 1]);
    for i in 0..n {
        cpu[i] = s4.sample(p2[i * 2 + 1] * 0.25, 0.0, p2[i * 2] * 0.25) * 4.0;
    }
    accel().sample_shift_b(&s4, &p2, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "shift_b");
}
