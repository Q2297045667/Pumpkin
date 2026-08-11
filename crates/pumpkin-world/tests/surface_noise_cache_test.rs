//! Surface 阶段噪声批量优化指纹测试。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    unused_imports
)]
use pumpkin_config::gpu::GpuConfig;
use pumpkin_data::chunk::DoublePerlinNoiseParameters;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::generation::noise::perlin::DoublePerlinNoiseSampler;
use pumpkin_world::generation::surface::CachedSurfaceNoise;
use pumpkin_world::noise_accel::NoiseAccelerator;

const SEED: u64 = 138_782_381_985_206;

fn make_dp(seed: u64, p: &DoublePerlinNoiseParameters) -> DoublePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let mut g = RandomGenerator::Xoroshiro(r);
    DoublePerlinNoiseSampler::from_params(&mut g, p, false)
}

fn fnv(d: &[f64]) -> u64 {
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
fn cached_vs_direct() {
    let a = make_dp(SEED, &DoublePerlinNoiseParameters::SURFACE);
    let b = make_dp(SEED + 1, &DoublePerlinNoiseParameters::SURFACE);
    let c = make_dp(SEED + 2, &DoublePerlinNoiseParameters::SURFACE_SECONDARY);
    let d = make_dp(SEED + 3, &DoublePerlinNoiseParameters::SURFACE_SECONDARY);
    let amp = a.amplitude();
    let samp = c.amplitude();
    let cc = 1.0181268882175227f64;
    let mut ds = vec![0.0f64; 256];
    let mut dd = vec![0.0f64; 256];
    for lx in 0i32..16 {
        for lz in 0i32..16 {
            let x = lx as f64;
            let z = lz as f64;
            let i = (lx * 16 + lz) as usize;
            ds[i] = (a.first_sampler().sample(x, 0.0, z)
                + b.first_sampler().sample(x * cc, 0.0, z * cc))
                * amp;
            dd[i] = (c.first_sampler().sample(x, 0.0, z)
                + d.first_sampler().sample(x * cc, 0.0, z * cc))
                * samp;
        }
    }
    let mut accel = accel();
    let cached = accel.precompute_surface(
        a.first_sampler(),
        b.first_sampler(),
        amp,
        c.first_sampler(),
        d.first_sampler(),
        samp,
        0,
        0,
    );
    assert_eq!(fnv(&ds), fnv(cached.surface.as_slice()), "surface");
    assert_eq!(fnv(&dd), fnv(cached.secondary.as_slice()), "secondary");
}

#[test]
fn perf() {
    let a = make_dp(SEED, &DoublePerlinNoiseParameters::SURFACE);
    let c = make_dp(SEED + 2, &DoublePerlinNoiseParameters::SURFACE_SECONDARY);
    let amp = a.amplitude();
    let samp = c.amplitude();
    let n = 1000u32;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = CachedSurfaceNoise::compute_cpu(
            a.first_sampler(),
            a.first_sampler(),
            amp,
            c.first_sampler(),
            c.first_sampler(),
            samp,
            0,
            0,
        );
    }
    let dm = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let mut accel = accel();
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        let _ = accel.precompute_surface(
            a.first_sampler(),
            a.first_sampler(),
            amp,
            c.first_sampler(),
            c.first_sampler(),
            samp,
            0,
            0,
        );
    }
    let bm = t1.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!(
        "Surface: direct={dm:.3}ms, batched={bm:.3}ms, speedup={:.2}x",
        dm / bm
    );
    assert!(bm <= dm * 1.1);
}
