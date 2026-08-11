//! 噪声阶段 GPU 加速指纹测试 (遗留兼容)。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, unused_mut)]
#![cfg(feature = "gpu")]
use pumpkin_config::gpu::GpuConfig;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::noise_accel::NoiseAccelerator;

const SEED: u64 = 138_782_381_985_206;
fn mk_sampler(seed: u64, o: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(o);
    let mut g = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
}
fn mk_positions(n: usize) -> Vec<f64> {
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
fn single_octave() {
    let s = mk_sampler(SEED, &[0]);
    let p = mk_positions(256);
    let n = 256;
    let mut cpu = vec![0f64; n];
    let mut gpu = vec![0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    accel().sample_octave(&s, &p, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "single_octave");
}
#[test]
fn multi_octave() {
    let s = mk_sampler(SEED, &[0, 1, 2, 3]);
    let p = mk_positions(1024);
    let n = 1024;
    let mut cpu = vec![0f64; n];
    let mut gpu = vec![0f64; n];
    for i in 0..n {
        cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    accel().sample_octave(&s, &p, &mut gpu);
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "multi_octave");
}
#[test]
fn large() {
    let s = mk_sampler(SEED, &[-2, -1, 0, 1, 2, 3]);
    let p = mk_positions(16384);
    let n = 16384;
    let mut cpu = vec![0f64; n];
    let mut gpu = vec![0f64; n];
    let t0 = std::time::Instant::now();
    for i in 0..n {
        cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    }
    let ct = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    accel().sample_octave(&s, &p, &mut gpu);
    let gt = t1.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "large");
    #[allow(clippy::print_stdout)]
    {
        println!("large: cpu={ct:.1}ms gpu={gt:.1}ms ({:.1}x)", ct / gt);
    }
}
#[test]
fn various_octaves() {
    let cs: &[(&[i32], &str)] = &[
        (&[0], "1oct"),
        (&[0, 1, 2], "3oct"),
        (&[0, 1, 2, 3], "4oct"),
        (&[-2, -1, 0, 1, 2, 3], "6oct"),
        (&[-4, -3, -2, -1, 0, 1, 2, 3], "8oct"),
    ];
    let n = 2048;
    let p = mk_positions(n);
    let mut cpu = vec![0f64; n];
    let mut gpu = vec![0f64; n];
    let mut a = accel();
    for (o, l) in cs {
        let s = mk_sampler(SEED, o);
        for i in 0..n {
            cpu[i] = s.sample(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        }
        a.sample_octave(&s, &p, &mut gpu);
        assert_eq!(fnv1a(&cpu), fnv1a(&gpu), "{}", *l);
    }
}
#[test]
fn empty() {
    let s = mk_sampler(SEED, &[0, 1]);
    let mut r = vec![];
    accel().sample_octave(&s, &[], &mut r);
    assert!(r.is_empty());
}
