//! 端到端 GPU 管线指纹测试（lib 测试，可访问 `pub(crate)` 的私有 API）。
//!
//! 与 `tests/worldgen_pipeline_fingerprint.rs` 不同：本测试向 router 注入
//! **真实 `beardifier` 结构**，使 `CellCache` 填充路径中的 `Beardifier::fill`
//! 实际调用 GPU beardifier kernel，从而端到端验证 GPU kernel 与 vanilla
//! 数值逐位一致（若不一致，指纹将随后端/运行环境漂移）。

use crate::generation::GlobalRandomConfig;
use crate::generation::noise::router::chunk_density_function::{
    ChunkNoiseFunctionBuilderOptions, ChunkNoiseFunctionSampleOptions, SampleAction, WrapperData,
};
use crate::generation::noise::router::chunk_noise_router::ChunkNoiseRouter;
use crate::generation::noise::router::density_function::beardifier::{
    BeardifierJunction, BeardifierStructure, TerrainAdaptation,
};
use crate::generation::noise::router::proto_noise_router::ProtoNoiseRouters;
use crate::generation::noise::{ChunkIndexMapper, InterpolationIndexMapper};
use pumpkin_config::gpu::GpuConfig;
use pumpkin_data::noise_router::OVERWORLD_BASE_NOISE_ROUTER;
use pumpkin_util::math::block_box::BlockBox;
use pumpkin_util::math::vector3::Vector3;

fn fnv1a_hash_f64(values: impl Iterator<Item = f64>) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = FNV_OFFSET;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn ensure_gpu_init() {
    static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        let config = GpuConfig {
            enabled: true,
            noise_acceleration: true,
            batch_acceleration: true,
            light_acceleration: true,
            ..Default::default()
        };
        crate::gpu::init_gpu_config(config);
    });
}

/// 构建带真实 beardifier 结构的 router。
fn build_router(seed: u64) -> ChunkNoiseRouter<'static> {
    let random_config = GlobalRandomConfig::new(seed, false);
    let proto_routers = ProtoNoiseRouters::generate(&OVERWORLD_BASE_NOISE_ROUTER, &random_config);

    // 结构位于 CellCache 填充区域（mapper: x/z ∈ [0,16)、y ∈ [0,64)）内，
    // 使 GPU beardifier kernel 在 cell cache 填充路径中被实际调用。
    let builder_options = ChunkNoiseFunctionBuilderOptions::new(
        4,
        8,
        48,
        4,
        0,
        0,
        4,
        vec![
            BeardifierStructure {
                bounding_box: BlockBox {
                    min: Vector3::new(2, 8, 2),
                    max: Vector3::new(10, 24, 10),
                },
                terrain_adaptation: TerrainAdaptation::BeardThin,
                ground_level_delta: 8,
            },
            BeardifierStructure {
                bounding_box: BlockBox {
                    min: Vector3::new(6, 32, 6),
                    max: Vector3::new(14, 48, 14),
                },
                terrain_adaptation: TerrainAdaptation::Bury,
                ground_level_delta: 4,
            },
        ],
        vec![
            BeardifierJunction {
                x: 5,
                ground_y: 16,
                z: 5,
            },
            BeardifierJunction {
                x: 10,
                ground_y: 40,
                z: 10,
            },
        ],
        Some(BlockBox {
            min: Vector3::new(-16, -16, -16),
            max: Vector3::new(32, 80, 32),
        }),
    );

    let noise: &'static _ = Box::leak(Box::new(proto_routers.noise));
    ChunkNoiseRouter::generate(noise, &builder_options)
}

/// 带真实 beardifier 结构的完整密度管线（GPU 加速器已初始化）指纹稳定。
///
/// 指纹覆盖：`FlatCache` GPU 预计算（构造时）→ `CellCache` 填充（GPU beardifier kernel）
/// → 插值器填充 → GPU 三线性插值 → 密度采样。
#[test]
fn density_pipeline_with_beardifier_fingerprint_stable() {
    ensure_gpu_init();

    let mut results = Vec::new();
    for seed in [0u64, 42] {
        let mut router = build_router(seed);

        // CellCache 填充（ChunkIndexMapper 数学与生成器一致）
        let mapper = ChunkIndexMapper {
            start_x: 0,
            start_y: 0,
            start_z: 0,
            horizontal_cell_block_count: 4,
            vertical_cell_block_count: 8,
        };
        let mut cell_options = ChunkNoiseFunctionSampleOptions::new(
            true,
            SampleAction::CellCaches(WrapperData::new(0, 0, 0, 4, 8)),
            0,
            0,
            0,
        );
        router.fill_cell_caches(&mapper, &mut cell_options);

        // 插值器缓冲填充
        let interp_mapper = InterpolationIndexMapper {
            x: 4,
            z: 8,
            minimum_cell_y: 0,
            vertical_cell_block_count: 8,
        };
        let mut interp_options =
            ChunkNoiseFunctionSampleOptions::new(true, SampleAction::SkipCellCaches, 0, 0, 0);
        router.fill_interpolator_buffers(true, 0, &interp_mapper, &mut interp_options);
        router.fill_interpolator_buffers(false, 0, &interp_mapper, &mut interp_options);

        // GPU 三线性插值
        router.interpolate_xyz(0.5, 0.25, 0.75);

        // 密度采样
        let options = ChunkNoiseFunctionSampleOptions::new(
            false,
            SampleAction::CellCaches(WrapperData::new(0, 0, 0, 4, 8)),
            0,
            0,
            0,
        );
        for x in (-32..32).step_by(7) {
            for y in (-64..320).step_by(23) {
                for z in (-32..32).step_by(9) {
                    let pos = Vector3::new(x, y, z);
                    results.push(router.final_density(&pos, &options));
                    results.push(router.vein_toggle(&pos, &options));
                }
            }
        }
    }

    let hash = fnv1a_hash_f64(results.into_iter());
    assert_ne!(hash, 0, "密度管线指纹不应为零");
    assert_eq!(
        hash, 0x6584_9dac_e60e_6b25,
        "带 beardifier 结构的密度管线指纹改变"
    );
}

/// 同一管线（真实 beardifier 结构）两次运行指纹一致。
#[test]
fn density_pipeline_with_beardifier_deterministic() {
    ensure_gpu_init();

    let run = |results: &mut Vec<f64>| {
        let mut router = build_router(7);
        let mapper = ChunkIndexMapper {
            start_x: 0,
            start_y: 0,
            start_z: 0,
            horizontal_cell_block_count: 4,
            vertical_cell_block_count: 8,
        };
        let mut cell_options = ChunkNoiseFunctionSampleOptions::new(
            true,
            SampleAction::CellCaches(WrapperData::new(0, 0, 0, 4, 8)),
            0,
            0,
            0,
        );
        router.fill_cell_caches(&mapper, &mut cell_options);
        let interp_mapper = InterpolationIndexMapper {
            x: 4,
            z: 8,
            minimum_cell_y: 0,
            vertical_cell_block_count: 8,
        };
        let mut interp_options =
            ChunkNoiseFunctionSampleOptions::new(true, SampleAction::SkipCellCaches, 0, 0, 0);
        router.fill_interpolator_buffers(true, 0, &interp_mapper, &mut interp_options);
        router.fill_interpolator_buffers(false, 0, &interp_mapper, &mut interp_options);
        router.interpolate_xyz(0.25, 0.5, 0.125);

        let options = ChunkNoiseFunctionSampleOptions::new(
            false,
            SampleAction::CellCaches(WrapperData::new(0, 0, 0, 4, 8)),
            0,
            0,
            0,
        );
        for x in (-16..16).step_by(5) {
            for y in (-64..256).step_by(31) {
                for z in (-16..16).step_by(7) {
                    let pos = Vector3::new(x, y, z);
                    results.push(router.final_density(&pos, &options));
                    results.push(router.depth(&pos, &options));
                    results.push(router.erosion(&pos, &options));
                }
            }
        }
    };

    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    run(&mut r1);
    run(&mut r2);
    assert!(r1.iter().all(|&v| v.is_finite()), "管线输出必须有限");
    assert_eq!(
        fnv1a_hash_f64(r1.into_iter()),
        fnv1a_hash_f64(r2.into_iter()),
        "两次运行指纹必须一致"
    );
}
