//! JIT 特化与标准路径一致性测试。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::doc_markdown
)]
#![cfg(feature = "pumpkin-util")]

use pumpkin_gpu::jit;
use pumpkin_gpu::noise::cache::SerializedOctaveConfig;
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

const SEED: u64 = 138_782_381_985_206;

fn mk_sampler(octaves: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(SEED);
    let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
    let mut g = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
}

/// JIT 特化的 kernel 源码应该能被生成且包含展开的逻辑
#[test]
fn jit_source_generation_small() {
    let sampler = mk_sampler(&[0, 1, 2]);
    let config = SerializedOctaveConfig::from_sampler(&sampler);
    assert!(jit::should_jit_specialize(config.num_octaves(), 16));
    let kernel = jit::specialize_octave_perlin(&config, 16);
    assert!(kernel.is_some());
    let k = kernel.unwrap();
    assert!(k.name.contains("jit_m3"));
    // JIT kernel 不应包含原始循环
    assert!(!k.source.contains("for (int o"));
}

/// 八度数 > max_unroll 时不应生成 JIT kernel
#[test]
fn jit_skip_large_octaves() {
    let sampler = mk_sampler(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let config = SerializedOctaveConfig::from_sampler(&sampler);
    // max_unroll=5, octaves=9 > 5 → 应跳过
    assert!(!jit::should_jit_specialize(config.num_octaves(), 5));
    assert!(jit::specialize_octave_perlin(&config, 5).is_none());
    // max_unroll=16, octaves=9 ≤ 16 → 应生成
    assert!(jit::should_jit_specialize(config.num_octaves(), 16));
    assert!(jit::specialize_octave_perlin(&config, 16).is_some());
}

/// max_unroll=1 边界
#[test]
fn jit_max_unroll_one() {
    let sampler = mk_sampler(&[0]);
    let config = SerializedOctaveConfig::from_sampler(&sampler);
    assert!(jit::should_jit_specialize(config.num_octaves(), 1));
    assert!(jit::specialize_octave_perlin(&config, 1).is_some());

    let sampler2 = mk_sampler(&[0, 1]);
    let config2 = SerializedOctaveConfig::from_sampler(&sampler2);
    assert!(!jit::should_jit_specialize(config2.num_octaves(), 1));
}
