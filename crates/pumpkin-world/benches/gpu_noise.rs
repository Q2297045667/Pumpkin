//! GPU 噪声加速基准测试。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    unused_variables
)]
#![cfg(feature = "gpu")]
use criterion::{Criterion, criterion_group, criterion_main};
use pumpkin_config::gpu::GpuConfig;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::noise_accel::NoiseAccelerator;
use std::hint::black_box;

const TEST_SEED: u64 = 138_782_381_985_206;

fn mk_sampler(seed: u64, octaves: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let (start, amplitudes) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
    let mut rand_gen = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut rand_gen, start, &amplitudes, false)
}

fn mk_positions(n: usize) -> Vec<f64> {
    let mut positions = Vec::with_capacity(n * 3);
    let mut seed = TEST_SEED;
    for _ in 0..n {
        let x = (seed.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8;
        seed = seed.wrapping_mul(1442695040888963407);
        let y = (seed as f64) * 1e-8;
        seed = seed.wrapping_mul(1442695040888963407);
        let z = (seed as f64) * 1e-8;
        positions.push(x);
        positions.push(y);
        positions.push(z);
    }
    positions
}

fn bench_single(c: &mut Criterion) {
    let sampler = mk_sampler(TEST_SEED, &[0, 1, 2]);
    let pos = mk_positions(1);
    c.bench_function("noise_cpu_single", |b| {
        b.iter(|| {
            black_box(sampler.sample(black_box(pos[0]), black_box(pos[1]), black_box(pos[2])));
        });
    });
}

fn bench_batch_1024(c: &mut Criterion) {
    let sampler = mk_sampler(TEST_SEED, &[0, 1, 2, 3]);
    let positions = mk_positions(1024);
    let mut results = vec![0.0f64; 1024];
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    };
    let mut accel = NoiseAccelerator::new(&config);
    c.bench_function("noise_batch_1024", |b| {
        b.iter(|| {
            accel.sample_octave(
                black_box(&sampler),
                black_box(&positions),
                black_box(&mut results),
            );
            black_box(&results);
        });
    });
}

fn bench_batch_16384(c: &mut Criterion) {
    let sampler = mk_sampler(TEST_SEED, &[-2, -1, 0, 1, 2, 3]);
    let positions = mk_positions(16384);
    let mut results = vec![0.0f64; 16384];
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    };
    let mut accel = NoiseAccelerator::new(&config);
    c.bench_function("noise_batch_16384", |b| {
        b.iter(|| {
            accel.sample_octave(
                black_box(&sampler),
                black_box(&positions),
                black_box(&mut results),
            );
            black_box(&results);
        });
    });
}

fn bench_multi_octave(c: &mut Criterion) {
    let octave_configs = [
        (&[0][..], "1oct"),
        (&[0, 1, 2][..], "3oct"),
        (&[0, 1, 2, 3][..], "4oct"),
        (&[-2, -1, 0, 1, 2, 3][..], "6oct"),
        (&[-4, -3, -2, -1, 0, 1, 2, 3][..], "8oct"),
    ];
    let n = 4096;
    let positions = mk_positions(n);
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        ..Default::default()
    };
    let mut accel = NoiseAccelerator::new(&config);
    for (octaves, label) in &octave_configs {
        let sampler = mk_sampler(TEST_SEED, octaves);
        let mut results = vec![0.0f64; n];
        c.bench_function(&format!("noise_compare_{label}"), |b| {
            b.iter(|| {
                accel.sample_octave(
                    black_box(&sampler),
                    black_box(&positions),
                    black_box(&mut results),
                );
                black_box(&results);
            });
        });
    }
}

criterion_group!(
    benches,
    bench_single,
    bench_batch_1024,
    bench_batch_16384,
    bench_multi_octave
);
criterion_main!(benches);
