#![cfg(feature = "gpu")]

use criterion::{Criterion, criterion_group, criterion_main};
use pumpkin_config::gpu::GpuConfig;
use pumpkin_gpu::GpuDevice;
use pumpkin_gpu::noise::GpuNoiseSampler;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

const SEED: u64 = 138_782_381_985_206;

fn sampler() -> OctavePerlinNoiseSampler {
    let random = Xoroshiro::from_seed(SEED);
    let (octaves, amplitudes) = OctavePerlinNoiseSampler::calculate_amplitudes(&[0, 1, 2]);
    let mut generator = RandomGenerator::Xoroshiro(random);
    OctavePerlinNoiseSampler::new(&mut generator, octaves, &amplitudes, false)
}

fn positions() -> Vec<f64> {
    (0..4096)
        .flat_map(|i| {
            let x = i as f64 * 0.03125;
            [x, x * 0.5 - 7.0, x * -0.25 + 3.0]
        })
        .collect()
}

fn benchmark_cpu_noise(c: &mut Criterion) {
    let config = GpuConfig {
        enabled: false,
        ..GpuConfig::default()
    };
    let device = GpuDevice::from_config(&config);
    let mut sampler_gpu = GpuNoiseSampler::new(device);
    let noise = sampler();
    let pos = positions();
    let mut result = vec![0.0; pos.len() / 3];

    c.bench_function("cpu_noise_fingerprint_batch", |b| {
        b.iter(|| {
            assert!(
                sampler_gpu
                    .sample_octave_batch(&noise, &pos, &mut result)
                    .is_ok()
            );
            std::hint::black_box(&result);
        });
    });
}

criterion_group!(benches, benchmark_cpu_noise);
criterion_main!(benches);
