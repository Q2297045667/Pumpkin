//! JIT 编译测试。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "pumpkin-util")]

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

    assert!(jit::should_jit_specialize(config.num_octaves(), 16));
    let result = jit::specialize_octave_perlin(&config, 16);
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

    assert!(!jit::should_jit_specialize(config.num_octaves(), 16));
    assert!(jit::specialize_octave_perlin(&config, 16).is_none());
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

    let kernel = jit::specialize_octave_perlin(&config, 16).unwrap();
    // 验证烘焙了振幅值
    assert!(kernel.source.contains('1'));
    // 验证烘焙了 origin
    assert!(kernel.source.contains("1.5"));
    assert!(kernel.source.contains("2.5"));
    assert!(kernel.source.contains("3.5"));
}

#[test]
fn should_jit_specialize_bounds() {
    assert!(!jit::should_jit_specialize(0, 16));
    assert!(jit::should_jit_specialize(1, 16));
    assert!(jit::should_jit_specialize(16, 16));
    assert!(!jit::should_jit_specialize(17, 16));
    assert!(!jit::should_jit_specialize(32, 16));
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
        let kernel = jit::specialize_octave_perlin(&config, 16).unwrap();
        assert!(
            kernel.name.contains(&format!("jit_m{m}")),
            "expected jit_m{m} in name, got {}",
            kernel.name
        );
    }
}

// ============================================================================
// FlatCache JIT 特化
// ============================================================================

fn flatcache_config(m: usize) -> SerializedOctaveConfig {
    let octaves: Vec<SerializedOctave> = (0..m)
        .map(|i| SerializedOctave {
            permutation: [i as u8; 256],
            x_origin: 1.5,
            y_origin: 2.5,
            z_origin: 3.5,
            amplitude: 1.0,
            persistence: 0.5,
            lacunarity: 2.0,
        })
        .collect();
    SerializedOctaveConfig {
        octaves: octaves.into_boxed_slice(),
        max_value: m as f64,
    }
}

#[test]
fn jit_specialize_flatcache_generates() {
    let config = flatcache_config(3);
    let kernel = jit::specialize_flatcache(&config, 16).expect("should specialize");
    // kernel 名 = 基名 + 八度数 + 配置内容指纹（防止不同采样器共用同一 JIT kernel）
    assert_eq!(
        kernel.name,
        format!(
            "flatcache_precompute_f64_jit_m3_h{:016x}",
            config.fingerprint()
        )
    );
    assert!(kernel.source.contains("sample_no_fade_core"));
    // 循环完全展开
    assert!(!kernel.source.contains("for (int o"));
    // 2D 输入 + y 固定为 0
    assert!(kernel.source.contains("pos[i*2]"));
    assert!(kernel.source.contains("0.0"));
}

#[test]
fn jit_specialize_flatcache_skips_large() {
    let config = flatcache_config(17);
    assert!(jit::specialize_flatcache(&config, 16).is_none());
    assert!(jit::specialize_flatcache(&config, 32).is_some());
}

#[test]
fn jit_specialize_double_perlin_generates() {
    let c1 = flatcache_config(2);
    let c2 = flatcache_config(3);
    let kernel = jit::specialize_double_perlin(&c1, &c2, 0.5, 16).expect("should specialize");
    assert_eq!(
        kernel.name,
        format!(
            "double_perlin_sample_f64_jit_m2_3_h{:016x}",
            c1.fingerprint().wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ c2.fingerprint()
                ^ 0.5f64.to_bits()
        )
    );
    assert!(kernel.source.contains("sample_no_fade_core"));
    assert!(!kernel.source.contains("for (int o"));
}

#[test]
fn jit_specialize_double_perlin_skips_large() {
    let c1 = flatcache_config(17);
    let c2 = flatcache_config(2);
    assert!(jit::specialize_double_perlin(&c1, &c2, 0.5, 16).is_none());
}

// ============================================================================
// Shift JIT 特化
// ============================================================================

#[test]
fn jit_specialize_shift_a_generates() {
    let config = flatcache_config(2);
    let kernel = jit::specialize_shift("shift_a", &config, 16).expect("should specialize");
    assert_eq!(
        kernel.name,
        format!("shift_a_sample_f64_jit_m2_h{:016x}", config.fingerprint())
    );
    assert!(kernel.source.contains("sample_no_fade_core"));
    // ShiftA: x = pos[i*2]*0.25, z = pos[i*2+1]*0.25, 输出 * 4.0
    assert!(kernel.source.contains("pos[i*2] * 0.25"));
    assert!(kernel.source.contains("res[i] = sum * 4.0"));
    assert!(!kernel.source.contains("for (int o"));
}

#[test]
fn jit_specialize_shift_b_swaps_coordinates() {
    let config = flatcache_config(3);
    let kernel = jit::specialize_shift("shift_b", &config, 16).expect("should specialize");
    assert_eq!(
        kernel.name,
        format!("shift_b_sample_f64_jit_m3_h{:016x}", config.fingerprint())
    );
    // ShiftB: z = pos[i*2]*0.25, x = pos[i*2+1]*0.25
    assert!(kernel.source.contains("pos[i*2] * 0.25"));
    assert!(kernel.source.contains("pos[i*2+1] * 0.25"));
    assert!(kernel.source.contains("res[i] = sum * 4.0"));
}

#[test]
fn jit_specialize_shift_skips_large() {
    let config = flatcache_config(17);
    assert!(jit::specialize_shift("shift_a", &config, 16).is_none());
    assert!(jit::specialize_shift("shift_b", &config, 16).is_none());
}
