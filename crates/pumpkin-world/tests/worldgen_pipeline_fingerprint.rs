//! 世界生成端到端管线指纹测试。
//!
//! 初始化全局 GPU 加速器后，驱动完整密度管线：
//! FlatCache 预计算（路由器构造时）→ CellCache 填充 → 插值器缓冲填充 →
//! 三线性插值 → 密度采样，验证跨多次运行指纹稳定、输出有限，
//! 并钉住指纹常量以捕获回归。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_mut,
    clippy::print_stdout,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::GpuConfig;
use pumpkin_data::noise_router::OVERWORLD_BASE_NOISE_ROUTER;
use pumpkin_util::math::vector3::Vector3;
use pumpkin_world::generation::GlobalRandomConfig;
use pumpkin_world::generation::noise::router::chunk_density_function::{
    ChunkNoiseFunctionBuilderOptions, ChunkNoiseFunctionSampleOptions, SampleAction, WrapperData,
};
use pumpkin_world::generation::noise::router::chunk_noise_router::ChunkNoiseRouter;
use pumpkin_world::generation::noise::router::density_function::IndexToNoisePos;
use pumpkin_world::generation::noise::router::proto_noise_router::ProtoNoiseRouters;

/// 与 `ChunkIndexMapper` 数学映射一致的测试 mapper。
///
/// 注：`ChunkNoiseFunctionSampleOptions` 的 `fill_index`/`action` 字段为 `pub(crate)`，
/// 集成测试无法更新。overworld（1.21.x）router 的 CellCache DAG 不含嵌套 CellCache /
/// `CacheOnce` 包装，填充期间不依赖这两个字段，因此位置数学一致即可。
struct TestChunkMapper {
    start_x: i32,
    start_y: i32,
    start_z: i32,
    horizontal_cell_block_count: usize,
    vertical_cell_block_count: usize,
}

impl IndexToNoisePos for TestChunkMapper {
    fn at(
        &self,
        index: usize,
        _sample_options: Option<&mut ChunkNoiseFunctionSampleOptions>,
    ) -> Vector3<i32> {
        let cell_z_position = index % self.horizontal_cell_block_count;
        let xy_portion = index / self.horizontal_cell_block_count;
        let cell_x_position = xy_portion % self.horizontal_cell_block_count;
        let cell_y_position =
            self.vertical_cell_block_count - 1 - (xy_portion / self.horizontal_cell_block_count);

        Vector3::new(
            self.start_x + (cell_x_position * self.horizontal_cell_block_count) as i32,
            self.start_y + (cell_y_position * self.vertical_cell_block_count) as i32,
            self.start_z + (cell_z_position * self.horizontal_cell_block_count) as i32,
        )
    }
}

/// 与 `InterpolationIndexMapper` 数学映射一致的测试 mapper。
struct TestInterpMapper {
    x: i32,
    z: i32,
    minimum_cell_y: i32,
    vertical_cell_block_count: i32,
}

impl IndexToNoisePos for TestInterpMapper {
    fn at(
        &self,
        index: usize,
        _sample_options: Option<&mut ChunkNoiseFunctionSampleOptions>,
    ) -> Vector3<i32> {
        let y = (index as i32 + self.minimum_cell_y) * self.vertical_cell_block_count;
        Vector3::new(self.x, y, self.z)
    }
}

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

/// 初始化全局 GPU 加速器（仅一次，多个测试共享）。
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
        pumpkin_world::gpu::init_gpu_config(config);
    });
}

fn build_router(seed: u64) -> ChunkNoiseRouter<'static> {
    let random_config = GlobalRandomConfig::new(seed, false);
    let proto_routers = ProtoNoiseRouters::generate(&OVERWORLD_BASE_NOISE_ROUTER, &random_config);
    let builder_options =
        ChunkNoiseFunctionBuilderOptions::new(4, 8, 48, 4, 0, 0, 4, Vec::new(), Vec::new(), None);
    // 泄漏 proto_routers.noise 以获得 'static 生命周期（测试进程内一次性开销）
    let noise: &'static _ = Box::leak(Box::new(proto_routers.noise));
    ChunkNoiseRouter::generate(noise, &builder_options)
}

/// 完整密度管线（填充 + 插值 + 采样）指纹稳定且有限。
#[test]
fn density_pipeline_fingerprint_stable() {
    ensure_gpu_init();

    let mut results = Vec::new();
    for seed in [0u64, 42] {
        let mut router = build_router(seed);

        // 1) CellCache 填充（GPU 入口在 DAG 复杂时回退 CPU DAG 求值）
        let mapper = TestChunkMapper {
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

        // 2) 插值器缓冲填充（start 与 end 两个方向）
        let interp_mapper = TestInterpMapper {
            x: 4,
            z: 8,
            minimum_cell_y: 0,
            vertical_cell_block_count: 8,
        };
        let mut interp_options =
            ChunkNoiseFunctionSampleOptions::new(true, SampleAction::SkipCellCaches, 0, 0, 0);
        router.fill_interpolator_buffers(true, 0, &interp_mapper, &mut interp_options);
        router.fill_interpolator_buffers(false, 0, &interp_mapper, &mut interp_options);

        // 3) 三线性插值（GPU batch trilinear 入口，复杂 DAG 时回退逐分量 CPU）
        router.interpolate_xyz(0.5, 0.25, 0.75);

        // 4) 密度采样
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
                    results.push(router.vein_ridged(&pos, &options));
                    results.push(router.vein_gap(&pos, &options));
                }
            }
        }
    }

    let hash = fnv1a_hash_f64(results.into_iter());
    assert_ne!(hash, 0, "密度管线指纹不应为零");
    // 钉住指纹：任何改变密度管线数值结果的改动都会在此失败。
    assert_eq!(
        hash, 0x2053_6d90_1ad7_1490,
        "密度管线指纹改变（GPU 加速器初始化后的端到端密度值）"
    );
}

/// 同一管线的两次运行产生相同指纹（确定性验证，不依赖硬编码常量）。
#[test]
fn density_pipeline_deterministic_twice() {
    ensure_gpu_init();

    let run = |results: &mut Vec<f64>| {
        let mut router = build_router(1);
        let mapper = TestChunkMapper {
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
        let interp_mapper = TestInterpMapper {
            x: 4,
            z: 8,
            minimum_cell_y: 0,
            vertical_cell_block_count: 8,
        };
        let mut interp_options =
            ChunkNoiseFunctionSampleOptions::new(true, SampleAction::SkipCellCaches, 0, 0, 0);
        router.fill_interpolator_buffers(true, 0, &interp_mapper, &mut interp_options);
        router.fill_interpolator_buffers(false, 0, &interp_mapper, &mut interp_options);
        router.interpolate_xyz(0.5, 0.25, 0.75);

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
