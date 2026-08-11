//! JIT 编译测试。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pumpkin_gpu::jit;
use pumpkin_gpu::noise::cache::{SerializedOctave, SerializedOctaveConfig};

#[test]
fn jit_specialize_small_octaves() {
    let config = SerializedOctaveConfig {
        octaves: vec![
            SerializedOctave {
                permutation: [0u8; 256],
                x_origin: 0.0,
                y_origin: 0.0,
                z_origin: 0.0,
                amplitude: 1.0,
                persistence: 0.5,
                lacunarity: 2.0,
            },
            SerializedOctave {
                permutation: [1u8; 256],
                x_origin: 1.0,
                y_origin: 1.0,
                z_origin: 1.0,
                amplitude: 0.5,
                persistence: 0.5,
                lacunarity: 2.0,
            },
        ]
        .into_boxed_slice(),
        max_value: 2.0,
    };

    assert!(jit::should_jit_specialize(config.num_octaves()));
    let result = jit::specialize_octave_perlin(&config);
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(kernel.name.contains("jit_m2"));
    assert!(kernel.source.contains("sample_no_fade_core"));
    // 验证循环被展开（源中不应有 "for (int o"）
    assert!(!kernel.source.contains("for (int o"));
}

#[test]
fn jit_skip_large_octaves() {
    let octaves: Vec<SerializedOctave> = (0..20)
        .map(|i| SerializedOctave {
            permutation: [i as u8; 256],
            x_origin: 0.0,
            y_origin: 0.0,
            z_origin: 0.0,
            amplitude: 1.0,
            persistence: 0.5,
            lacunarity: 2.0,
        })
        .collect();

    let config = SerializedOctaveConfig {
        octaves: octaves.into_boxed_slice(),
        max_value: 10.0,
    };

    assert!(!jit::should_jit_specialize(config.num_octaves()));
    assert!(jit::specialize_octave_perlin(&config).is_none());
}

#[test]
fn jit_source_contains_amplitudes() {
    let config = SerializedOctaveConfig {
        octaves: vec![SerializedOctave {
            permutation: [0u8; 256],
            x_origin: 1.5,
            y_origin: 2.5,
            z_origin: 3.5,
            amplitude: 1.0,
            persistence: 0.5,
            lacunarity: 2.0,
        }]
        .into_boxed_slice(),
        max_value: 1.0,
    };

    let kernel = jit::specialize_octave_perlin(&config).unwrap();
    // 验证烘焙了振幅值
    assert!(kernel.source.contains("1"));
    // 验证烘焙了 origin
    assert!(kernel.source.contains("1.5"));
    assert!(kernel.source.contains("2.5"));
    assert!(kernel.source.contains("3.5"));
}

#[test]
fn should_jit_specialize_bounds() {
    assert!(!jit::should_jit_specialize(0));
    assert!(jit::should_jit_specialize(1));
    assert!(jit::should_jit_specialize(16));
    assert!(!jit::should_jit_specialize(17));
    assert!(!jit::should_jit_specialize(32));
}

#[test]
fn jit_kernel_name_includes_octave_count() {
    for m in 1..=16 {
        let octaves: Vec<SerializedOctave> = (0..m)
            .map(|i| SerializedOctave {
                permutation: [i as u8; 256],
                x_origin: 0.0,
                y_origin: 0.0,
                z_origin: 0.0,
                amplitude: 1.0,
                persistence: 0.5,
                lacunarity: 2.0,
            })
            .collect();
        let config = SerializedOctaveConfig {
            octaves: octaves.into_boxed_slice(),
            max_value: m as f64,
        };
        let kernel = jit::specialize_octave_perlin(&config).unwrap();
        assert!(
            kernel.name.contains(&format!("jit_m{m}")),
            "expected jit_m{m} in name, got {}",
            kernel.name
        );
    }
}
