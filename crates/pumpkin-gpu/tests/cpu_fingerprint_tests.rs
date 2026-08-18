#![cfg(all(feature = "pumpkin-util", feature = "pumpkin-config"))]
#![allow(clippy::expect_used, clippy::panic, clippy::print_stdout)]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_gpu::GpuDevice;
use pumpkin_gpu::light::GpuLightSampler;
use pumpkin_gpu::noise::GpuNoiseSampler;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

const SEED: u64 = 138_782_381_985_206;

fn cpu_device() -> GpuDevice {
    let config = GpuConfig {
        enabled: false,
        ..GpuConfig::default()
    };
    GpuDevice::from_config(&config)
}

fn sampler(octaves: &[i32]) -> OctavePerlinNoiseSampler {
    let random = Xoroshiro::from_seed(SEED);
    let (octave_data, amplitudes) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
    let mut generator = RandomGenerator::Xoroshiro(random);
    OctavePerlinNoiseSampler::new(&mut generator, octave_data, &amplitudes, false)
}

fn fingerprint(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        let mixed = value.to_bits().wrapping_add(0x9e37_79b9_7f4a_7c15);
        (hash ^ mixed).wrapping_mul(0x1000_0000_01b3)
    })
}

fn positions_3d() -> Vec<f64> {
    (0..128)
        .flat_map(|i| {
            let x = i as f64 * 0.125 - 8.0;
            [x, x * -0.25 + 1.5, x * 0.5 - 3.0]
        })
        .collect()
}

fn positions_xz() -> Vec<f64> {
    (0..128)
        .flat_map(|i| {
            let x = i as f64 * 0.125 - 8.0;
            [x, x * 0.5 - 3.0]
        })
        .collect()
}

#[test]
fn cpu_noise_functions_have_stable_fingerprints() {
    let mut gpu = GpuNoiseSampler::new(cpu_device());
    let noise = sampler(&[0, 1, 2]);
    let positions = positions_3d();
    let xz = positions_xz();
    let mut octave = vec![0.0; positions.len() / 3];
    let mut double = vec![0.0; positions.len() / 3];
    let mut shift_a = vec![0.0; xz.len() / 2];
    let mut shift_b = vec![0.0; xz.len() / 2];

    gpu.sample_octave_batch(&noise, &positions, &mut octave)
        .expect("CPU octave sampling");
    gpu.sample_double_perlin_batch(&noise, &noise, 0.75, &positions, &mut double)
        .expect("CPU double Perlin sampling");
    gpu.sample_shift_a_batch(&noise, &xz, &mut shift_a)
        .expect("CPU Shift A sampling");
    gpu.sample_shift_b_batch(&noise, &xz, &mut shift_b)
        .expect("CPU Shift B sampling");

    assert_eq!(fingerprint(&octave), 13_173_505_367_063_298_189);
    assert_eq!(fingerprint(&double), 17_211_247_475_703_052_160);
    assert_eq!(fingerprint(&shift_a), 14_873_412_606_967_129_933);
    assert_eq!(fingerprint(&shift_b), 2_878_627_917_599_469_674);
}

#[test]
fn cpu_sky_light_propagates_across_all_columns() {
    let device = cpu_device();
    let mut light = GpuLightSampler::new(device);
    let width = 3;
    let depth = 2;
    let height = 2;
    let mut sky = vec![0; width * depth * height];
    let opacity = vec![0; sky.len()];
    sky[height - 1] = 15;

    let iterations = light
        .sky_horizontal_propagate(&mut sky, &opacity, width, depth, height, 16)
        .expect("CPU sky propagation");

    assert!(iterations > 0);
    assert!(sky.iter().any(|&value| value > 0));
    assert!(sky[height - 1 + height] > 0);
    assert!(sky[height - 1 + 2 * height] > 0);
    assert!(sky[height - 1 + depth * width * height / 2] > 0);
}

#[test]
fn cpu_sky_fill_clamps_heightmap_values() {
    let mut light = GpuLightSampler::new(cpu_device());
    light
        .batch_sky_fill(&[0], &[], &mut [], 1, 0)
        .expect("zero-height sky fill");
    let heightmap = [-2, 0, 99];
    let max_height = 3;
    let opacity = vec![0; heightmap.len() * max_height];
    let mut sky = vec![0; opacity.len()];

    light
        .batch_sky_fill(&heightmap, &opacity, &mut sky, heightmap.len(), max_height)
        .expect("CPU sky fill");

    assert_eq!(&sky[0..3], &[15, 15, 15]);
    assert_eq!(&sky[3..6], &[15, 15, 15]);
    assert_eq!(&sky[6..9], &[15, 15, 15]);
}

#[test]
fn cpu_light_paths_cover_empty_and_boundary_batches() {
    let mut light = GpuLightSampler::new(cpu_device());
    let mut block_light = Vec::new();
    assert!(
        light
            .batch_block_scan(&[], &mut block_light, 0)
            .expect("empty block scan")
            .is_empty()
    );

    let mut sky = vec![0; 2];
    let opacity = vec![0; 2];
    assert_eq!(
        light
            .sky_horizontal_propagate(&mut sky, &opacity, 1, 1, 2, 0)
            .expect("zero-iteration sky propagation"),
        0
    );

    let mut light_values = vec![15, 0, 0, 0];
    let opacity = vec![0; 4];
    let neighbors = vec![
        -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1,
    ];
    let iterations = light
        .iterative_propagate(&mut light_values, &opacity, &neighbors, 4, 16)
        .expect("CPU distance-field propagation");
    assert!(iterations > 0);
    assert_eq!(light_values, vec![15, 14, 13, 12]);
}

#[test]
fn cpu_interpolation_and_noise_are_repeatable_at_stress_size() {
    let mut gpu = GpuNoiseSampler::new(cpu_device());
    let noise = sampler(&[0, 1, 2]);
    let positions: Vec<f64> = (0..4096)
        .flat_map(|i| {
            let x = i as f64 * 0.03125;
            [x, x * 0.25 - 2.0, x * -0.5 + 4.0]
        })
        .collect();
    let mut first = vec![0.0; positions.len() / 3];
    let mut second = vec![0.0; positions.len() / 3];
    gpu.sample_octave_batch(&noise, &positions, &mut first)
        .expect("first stress batch");
    gpu.sample_octave_batch(&noise, &positions, &mut second)
        .expect("second stress batch");
    assert_eq!(first, second);
    assert!(first.iter().all(|value| value.is_finite()));

    let corners = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let deltas = [0.5, 0.25, 0.75];
    let mut result = [0.0];
    gpu.batch_trilinear(&corners, &deltas, &mut result)
        .expect("CPU trilinear interpolation");
    assert!((result[0] - 4.0).abs() < f64::EPSILON);
}
