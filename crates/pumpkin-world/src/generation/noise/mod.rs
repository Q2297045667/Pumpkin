pub mod aquifer_sampler;
pub mod ore_sampler;
pub mod perlin;
pub mod router;

use pumpkin_data::{Block, BlockState, chunk_gen_settings::GenerationShapeConfig};
use pumpkin_util::{math::vector3::Vector3, random::xoroshiro128::XoroshiroSplitter};

use crate::generation::{
    noise::{
        aquifer_sampler::{
            AquiferSampler, AquiferSamplerImpl, SeaLevelAquiferSampler, WorldAquiferSampler,
        },
        ore_sampler::OreVeinSampler,
    },
    proto_chunk::StandardChunkFluidLevelSampler,
    section_coords,
};

use super::{
    GlobalRandomConfig, biome_coords,
    noise::router::{
        chunk_density_function::{
            ChunkNoiseFunctionBuilderOptions, ChunkNoiseFunctionSampleOptions, SampleAction,
            WrapperData,
        },
        chunk_noise_router::ChunkNoiseRouter,
        density_function::IndexToNoisePos,
        proto_noise_router::ProtoNoiseRouter,
        surface_height_sampler::SurfaceHeightEstimateSampler,
    },
};

pub const LAVA_BLOCK: Block = Block::LAVA;
pub const WATER_BLOCK: Block = Block::WATER;

pub const CHUNK_DIM: u8 = 16;

pub enum BlockStateSampler {
    Aquifer(AquiferSampler),
    Ore(OreVeinSampler),
}

impl BlockStateSampler {
    pub fn sample(
        &mut self,
        router: &mut ChunkNoiseRouter,
        ore_random_deriver: &XoroshiroSplitter,
        pos: &Vector3<i32>,
        sample_options: &ChunkNoiseFunctionSampleOptions,
        height_estimator: &mut SurfaceHeightEstimateSampler,
    ) -> Option<&'static BlockState> {
        match self {
            Self::Aquifer(aquifer) => {
                aquifer
                    .apply(router, pos, sample_options, height_estimator)
                    .0
            }
            Self::Ore(ore) => ore.sample(router, ore_random_deriver, pos, sample_options),
        }
    }
}

pub struct ChainedBlockStateSampler {
    pub(crate) samplers: Box<[BlockStateSampler]>,
}

impl ChainedBlockStateSampler {
    #[must_use]
    pub const fn new(samplers: Box<[BlockStateSampler]>) -> Self {
        Self { samplers }
    }

    fn sample(
        &mut self,
        router: &mut ChunkNoiseRouter,
        ore_random_deriver: &XoroshiroSplitter,
        pos: &Vector3<i32>,
        sample_options: &ChunkNoiseFunctionSampleOptions,
        height_estimator: &mut SurfaceHeightEstimateSampler,
    ) -> Option<&'static BlockState> {
        for sampler in &mut self.samplers {
            if let Some(state) = sampler.sample(
                router,
                ore_random_deriver,
                pos,
                sample_options,
                height_estimator,
            ) {
                return Some(state);
            }
        }
        None
    }
}

struct InterpolationIndexMapper {
    x: i32,
    z: i32,

    minimum_cell_y: i32,
    vertical_cell_block_count: i32,
}

impl IndexToNoisePos for InterpolationIndexMapper {
    fn at(
        &self,
        index: usize,
        sample_data: Option<&mut ChunkNoiseFunctionSampleOptions>,
    ) -> Vector3<i32> {
        if let Some(sample_data) = sample_data {
            sample_data.cache_result_unique_id += 1;
            sample_data.fill_index = index;
        }

        let y = (index as i32 + self.minimum_cell_y) * self.vertical_cell_block_count;

        // TODO: Change this when Blender is implemented
        Vector3::new(self.x, y, self.z)
    }
}

struct ChunkIndexMapper {
    start_x: i32,
    start_y: i32,
    start_z: i32,

    horizontal_cell_block_count: usize,
    vertical_cell_block_count: usize,
}

impl IndexToNoisePos for ChunkIndexMapper {
    fn at(
        &self,
        index: usize,
        sample_options: Option<&mut ChunkNoiseFunctionSampleOptions>,
    ) -> Vector3<i32> {
        // Matches vanilla mathematical index mapping (yInCell, xInCell, zInCell)
        let cell_z_position = index % self.horizontal_cell_block_count;
        let xy_portion = index / self.horizontal_cell_block_count;
        let cell_x_position = xy_portion % self.horizontal_cell_block_count;
        let cell_y_position =
            self.vertical_cell_block_count - 1 - (xy_portion / self.horizontal_cell_block_count);

        if let Some(sample_options) = sample_options {
            sample_options.fill_index = index;
            if let SampleAction::CellCaches(wrapper_data) = &mut sample_options.action {
                wrapper_data.update_position(cell_x_position, cell_y_position, cell_z_position);
            }
        }

        // TODO: Change this when Blender is implemented
        Vector3::new(
            self.start_x + cell_x_position as i32,
            self.start_y + cell_y_position as i32,
            self.start_z + cell_z_position as i32,
        )
    }
}

pub struct ChunkNoiseGenerator<'a> {
    pub state_sampler: ChainedBlockStateSampler,
    generation_shape: &'a GenerationShapeConfig,
    start_cell_pos_x: i32,
    start_cell_pos_z: i32,
    horizontal_cell_count: usize,

    vertical_cell_count: usize,
    minimum_cell_y: i32,

    cache_fill_unique_id: u64,
    cache_result_unique_id: u64,

    pub router: ChunkNoiseRouter<'a>,

    /// GPU 批量加速器 —— 为 Cell Cache、Aquifer、Beardifier、Vein 提供批量采样。
    /// 仅在 `gpu` feature 启用且 GPU 可用时存在。
    #[cfg(feature = "gpu")]
    batch_accel: Option<&'a crate::batch_accel::BatchAccelerator>,

    /// GPU 矿脉批量预计算结果缓存。
    /// Key: 展平的本地块索引 (`local_x` + `local_y` * `stride_x` + `local_z` * `stride_x` * `stride_y`)
    /// Value: 矿脉类型码 (0=无, 1=矿石, 2=粗矿, 3=围岩)
    #[cfg(feature = "gpu")]
    gpu_vein_cache: Option<Vec<i32>>,

    /// GPU 批量预计算 cell cache 结果（整个 chunk 所有 cell 一次性 GPU 填充）。
    /// 布局：`[cell_flat_index * hb² * vb + local_index]`
    #[cfg(feature = "gpu")]
    gpu_cell_cache: Option<Vec<f64>>,
}

impl<'a> ChunkNoiseGenerator<'a> {
    #[expect(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        noise_router_base: &'a ProtoNoiseRouter,
        random_config: &GlobalRandomConfig,
        horizontal_cell_count: usize,
        start_block_x: i32,
        start_block_z: i32,
        generation_shape: &'a GenerationShapeConfig,
        level_sampler: StandardChunkFluidLevelSampler,
        aquifers: bool,
        ore_veins: bool,
        beardifier_structures: Vec<
            crate::generation::noise::router::density_function::beardifier::BeardifierStructure,
        >,
        beardifier_junctions: Vec<
            crate::generation::noise::router::density_function::beardifier::BeardifierJunction,
        >,
        affected_box: Option<pumpkin_util::math::block_box::BlockBox>,
    ) -> Self {
        let start_cell_pos_x =
            start_block_x / generation_shape.horizontal_cell_block_count() as i32;
        let start_cell_pos_z =
            start_block_z / generation_shape.horizontal_cell_block_count() as i32;

        let horizontal_biome_end = biome_coords::from_block(
            horizontal_cell_count as i32 * generation_shape.horizontal_cell_block_count() as i32,
        );
        let vertical_cell_count = (generation_shape.height as usize)
            / (generation_shape.vertical_cell_block_count() as usize);
        let minimum_cell_y =
            (generation_shape.min_y as i32) / (generation_shape.vertical_cell_block_count() as i32);

        let vertical_cell_block_count = generation_shape.vertical_cell_block_count();
        let horizontal_cell_block_count = generation_shape.horizontal_cell_block_count();

        let builder_options = ChunkNoiseFunctionBuilderOptions::new(
            horizontal_cell_block_count as usize,
            vertical_cell_block_count as usize,
            vertical_cell_count,
            horizontal_cell_count,
            biome_coords::from_block(start_block_x),
            biome_coords::from_block(start_block_z),
            horizontal_biome_end as usize,
            beardifier_structures,
            beardifier_junctions,
            affected_box,
        );

        let aquifer_sampler = if aquifers {
            let section_x = section_coords::block_to_section(start_block_x);
            let section_z = section_coords::block_to_section(start_block_z);
            AquiferSampler::Aquifer(WorldAquiferSampler::new(
                section_x,
                section_z,
                &random_config.aquifer_random_deriver,
                generation_shape.min_y,
                generation_shape.height,
                level_sampler,
            ))
        } else {
            AquiferSampler::SeaLevel(SeaLevelAquiferSampler::new(level_sampler))
        };

        let samplers: Box<[BlockStateSampler]> = if ore_veins {
            Box::new([
                BlockStateSampler::Aquifer(aquifer_sampler),
                BlockStateSampler::Ore(OreVeinSampler),
            ])
        } else {
            Box::new([BlockStateSampler::Aquifer(aquifer_sampler)])
        };
        let state_sampler = ChainedBlockStateSampler::new(samplers);

        let router = ChunkNoiseRouter::generate(noise_router_base, &builder_options);

        #[cfg(feature = "gpu")]
        let batch_accel = crate::gpu::get_batch_accel();

        Self {
            state_sampler,
            generation_shape,
            start_cell_pos_x,
            start_cell_pos_z,
            horizontal_cell_count,
            vertical_cell_count,
            minimum_cell_y,

            cache_fill_unique_id: 0,
            cache_result_unique_id: 0,

            router,

            #[cfg(feature = "gpu")]
            batch_accel,

            #[cfg(feature = "gpu")]
            gpu_vein_cache: None,

            #[cfg(feature = "gpu")]
            gpu_cell_cache: None,
        }
    }

    #[inline]
    pub fn sample_start_density(&mut self) {
        self.cache_result_unique_id = 0;
        self.sample_density(true, self.start_cell_pos_x);
    }

    #[inline]
    pub fn sample_end_density(&mut self, cell_x: i32) {
        self.sample_density(false, self.start_cell_pos_x + cell_x + 1);
    }

    fn sample_density(&mut self, start: bool, current_x: i32) {
        let x = current_x * self.horizontal_cell_block_count() as i32;

        for cell_z in 0..=self.horizontal_cell_count {
            let current_cell_z_pos = self.start_cell_pos_z + cell_z as i32;
            let z = current_cell_z_pos * self.horizontal_cell_block_count() as i32;
            self.cache_fill_unique_id += 1;

            let mapper = InterpolationIndexMapper {
                x,
                z,
                minimum_cell_y: self.minimum_cell_y,
                vertical_cell_block_count: self.vertical_cell_block_count() as i32,
            };

            let mut options = ChunkNoiseFunctionSampleOptions::new(
                false,
                SampleAction::CellCaches(WrapperData::new(
                    0,
                    0,
                    0,
                    self.horizontal_cell_block_count() as usize,
                    self.vertical_cell_block_count() as usize,
                )),
                self.cache_result_unique_id,
                self.cache_fill_unique_id,
                0,
            );

            self.fill_interpolator_buffers(start, cell_z, &mapper, &mut options);
            self.cache_result_unique_id = options.cache_result_unique_id;
        }
        self.cache_fill_unique_id += 1;
    }

    #[inline]
    fn fill_interpolator_buffers(
        &mut self,
        start: bool,
        cell_z: usize,
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        self.router
            .fill_interpolator_buffers(start, cell_z, mapper, sample_options);
    }

    #[inline]
    pub fn interpolate_x(&mut self, delta: f64) {
        self.router.interpolate_x(delta);
    }

    #[inline]
    pub fn interpolate_y(&mut self, delta: f64) {
        self.router.interpolate_y(delta);
    }

    #[inline]
    pub fn interpolate_z(&mut self, delta: f64) {
        self.cache_result_unique_id += 1;
        self.router.interpolate_z(delta);
    }

    /// GPU-accelerated combined trilinear interpolation.
    ///
    /// Replaces `interpolate_y` → `interpolate_x` → `interpolate_z` with a single
    /// GPU batch call. Falls back to the sequential CPU path when GPU is unavailable.
    #[inline]
    pub fn interpolate_xyz(&mut self, delta_y: f64, delta_x: f64, delta_z: f64) {
        self.cache_result_unique_id += 1;
        self.router.interpolate_xyz(delta_y, delta_x, delta_z);
    }

    #[inline]
    pub fn swap_buffers(&mut self) {
        self.router.swap_buffers();
    }

    pub fn on_sampled_cell_corners(&mut self, cell_x: i32, cell_y: i32, cell_z: i32) {
        self.router
            .on_sampled_cell_corners(cell_y as usize, cell_z as usize);
        self.cache_fill_unique_id += 1;

        // GPU 批量预计算路径：直接从 chunk 级缓存复制当前 cell 数据
        #[cfg(feature = "gpu")]
        if let Some(ref chunk_cache) = self.gpu_cell_cache {
            let hb = self.horizontal_cell_block_count() as usize;
            let vb = self.vertical_cell_block_count() as usize;
            let hc = self.horizontal_cell_count;
            let vc = self.vertical_cell_count as usize;
            let ppc = hb * hb * vb; // positions per cell
            let total_cells = hc * hc * vc;
            let per_cache = ppc * total_cells;
            let n_caches = self.router.cell_cache_count();
            // 展平 cell 索引：cy → cx → cz（与 precompute 收集顺序一致）
            let cell_flat = cell_y as usize * hc * hc + cell_x as usize * hc + cell_z as usize;
            let base = cell_flat * ppc;
            if chunk_cache.len() == per_cache * n_caches && base + ppc <= per_cache {
                for cache_index in 0..n_caches {
                    let start = cache_index * per_cache + base;
                    let slice = &chunk_cache[start..start + ppc];
                    self.router.copy_to_cell_cache(cache_index, slice);
                }
                self.cache_fill_unique_id += 1;
                return;
            }
        }

        // CPU 或单 cell GPU 回退路径
        let start_x = (self.start_cell_pos_x + cell_x) * self.horizontal_cell_block_count() as i32;
        let start_y = (cell_y + self.minimum_cell_y) * self.vertical_cell_block_count() as i32;
        let start_z = (self.start_cell_pos_z + cell_z) * self.horizontal_cell_block_count() as i32;

        let mapper = ChunkIndexMapper {
            start_x,
            start_y,
            start_z,
            horizontal_cell_block_count: self.horizontal_cell_block_count() as usize,
            vertical_cell_block_count: self.vertical_cell_block_count() as usize,
        };

        let mut sample_options = ChunkNoiseFunctionSampleOptions::new(
            true,
            SampleAction::CellCaches(WrapperData::new(
                0,
                0,
                0,
                self.horizontal_cell_block_count() as usize,
                self.vertical_cell_block_count() as usize,
            )),
            self.cache_result_unique_id,
            self.cache_fill_unique_id,
            0,
        );

        self.router.fill_cell_caches(&mapper, &mut sample_options);
        self.cache_fill_unique_id += 1;
    }

    #[expect(clippy::too_many_arguments)]
    pub fn sample_block_state(
        &mut self,
        ore_random_deriver: &XoroshiroSplitter,
        start_x: i32,
        start_y: i32,
        start_z: i32,
        cell_x: i32,
        cell_y: i32,
        cell_z: i32,
        height_estimator: &mut SurfaceHeightEstimateSampler,
    ) -> Option<&'static BlockState> {
        let pos = Vector3::new(start_x + cell_x, start_y + cell_y, start_z + cell_z);

        // GPU 矿脉批量预计算：若缓存命中则直接返回矿脉结果，绕过 CPU DAG。
        #[cfg(feature = "gpu")]
        if let Some(ref cache) = self.gpu_vein_cache {
            let h_blocks = self.horizontal_cell_block_count() as i32;
            let v_blocks = self.vertical_cell_block_count() as i32;
            let total_x = h_blocks * self.horizontal_cell_count as i32;
            let total_y = v_blocks * self.vertical_cell_count as i32;
            // 展平索引布局与 precompute_gpu_veins 一致: X → Y → Z
            let local_x = pos.x - (self.start_cell_pos_x * h_blocks);
            let local_y = pos.y - (self.minimum_cell_y * v_blocks);
            let local_z = pos.z - (self.start_cell_pos_z * h_blocks);
            if local_x >= 0 && local_y >= 0 && local_z >= 0 {
                let idx = (local_x + local_y * total_x + local_z * total_x * total_y) as usize;
                if idx < cache.len() {
                    let vein_type = cache[idx];
                    if vein_type != 0 {
                        // 根据 Y 坐标确定矿脉种类（与 OreVeinSampler 一致）
                        let block_y = pos.y;
                        let vein_def: &ore_sampler::VeinType = if (0..=50).contains(&block_y) {
                            &ore_sampler::vein_type::COPPER
                        } else {
                            &ore_sampler::vein_type::IRON
                        };
                        return match vein_type {
                            1 => Some(vein_def.ore.default_state),
                            2 => Some(vein_def.raw_ore.default_state),
                            3 => Some(vein_def.stone.default_state),
                            _ => None,
                        };
                    }
                }
            }
        }

        let options = ChunkNoiseFunctionSampleOptions::new(
            false,
            SampleAction::CellCaches(WrapperData::new(
                cell_x as usize,
                cell_y as usize,
                cell_z as usize,
                self.horizontal_cell_block_count() as usize,
                self.vertical_cell_block_count() as usize,
            )),
            self.cache_result_unique_id,
            self.cache_fill_unique_id,
            0,
        );

        self.state_sampler.sample(
            &mut self.router,
            ore_random_deriver,
            &pos,
            &options,
            height_estimator,
        )
    }

    /// GPU 批量矿脉预计算。
    ///
    /// 收集当前区块所有块位置，一次性调用 GPU `batch_vein_sample`，
    /// 将结果写入内部缓存。后续 `sample_block_state` 调用将直接查表
    /// 绕过 CPU DAG 矿脉噪声计算。
    ///
    /// 仅在 `gpu` feature 启用且 `batch_accel` 可用时生效。
    #[cfg(feature = "gpu")]
    pub fn precompute_gpu_veins(&mut self) {
        let Some(accel) = self.batch_accel else {
            return;
        };

        let h_blocks = self.horizontal_cell_block_count() as i32;
        let v_blocks = self.vertical_cell_block_count() as i32;
        let h_cells = self.horizontal_cell_count as i32;
        let v_cells = self.vertical_cell_count as i32;
        let min_cell_y = self.minimum_cell_y;

        let total_x = (h_blocks * h_cells) as usize;
        let total_y = (v_blocks * v_cells) as usize;
        let total_z = (h_blocks * h_cells) as usize;
        let total_blocks = total_x * total_y * total_z;

        if total_blocks == 0 {
            return;
        }

        // 收集所有块位置（展平为 f64 数组，布局 X → Y → Z）
        let mut positions = Vec::with_capacity(total_blocks * 3);
        let start_block_x = self.start_cell_pos_x * h_blocks;
        let start_block_z = self.start_cell_pos_z * h_blocks;
        let start_block_y = min_cell_y * v_blocks;

        for z in 0..total_z as i32 {
            for y in 0..total_y as i32 {
                for x in 0..total_x as i32 {
                    positions.push((start_block_x + x) as f64);
                    positions.push((start_block_y + y) as f64);
                    positions.push((start_block_z + z) as f64);
                }
            }
        }

        let vein_params = self.router.build_vein_params();
        let mut vein_results = vec![0i32; total_blocks];
        accel.batch_vein_sample(&positions, &vein_params, &mut vein_results);
        self.gpu_vein_cache = Some(vein_results);
    }

    /// 清除 GPU 矿脉缓存（在区块切换时调用）。
    #[cfg(feature = "gpu")]
    pub fn invalidate_vein_cache(&mut self) {
        self.gpu_vein_cache = None;
    }

    /// GPU 批量预计算 Cell Cache（合并 N 次调用为 1 次）。
    ///
    /// 收集当前区块所有 cell 的所有角点位置，一次性调用 GPU `batch_fill_cell_caches`，
    /// 将结果写入内部缓存。后续 `on_sampled_cell_corners` 调用将直接从缓存复制
    /// 绕过 GPU kernel launch。
    #[cfg(feature = "gpu")]
    pub fn precompute_gpu_cell_caches(&mut self) {
        let Some(accel) = self.batch_accel else {
            return;
        };
        // 仅支持 DAG 根为独立 Noise 的 router（其余回退 CPU 路径）
        let Some(specs) = self.router.build_cell_cache_fill_specs() else {
            return;
        };
        let n_caches = specs.len();

        let hb = self.horizontal_cell_block_count() as i32;
        let vb = self.vertical_cell_block_count() as i32;
        let hc = self.horizontal_cell_count as i32;
        let vc = self.vertical_cell_count as i32;

        let ppc = (hb * hb * vb) as usize; // positions per cell
        let total = ppc * (hc * hc * vc) as usize;
        if total == 0 {
            return;
        }

        let mut positions = Vec::with_capacity(total * 3);
        let start_bx = self.start_cell_pos_x * hb;
        let start_bz = self.start_cell_pos_z * hb;
        let start_by = self.minimum_cell_y * vb;

        // 收集所有 cell 角点位置（布局与 ChunkIndexMapper 一致）
        for cy in 0..vc {
            let by = start_by + cy * vb;
            for cx in 0..hc {
                let bx = start_bx + cx * hb;
                for cz in 0..hc {
                    let bz = start_bz + cz * hb;
                    for ly in 0..vb {
                        for lx in 0..hb {
                            for lz in 0..hb {
                                positions.push((bx + lx) as f64);
                                positions.push((by + ly) as f64);
                                positions.push((bz + lz) as f64);
                            }
                        }
                    }
                }
            }
        }

        // 布局：[cache_index][cell 按生成顺序][局部索引]
        let mut results = vec![0.0f64; total * n_caches];
        accel.batch_fill_cell_caches_vanilla(&positions, &specs, &mut results);
        self.gpu_cell_cache = Some(results);
    }

    #[inline]
    #[must_use]
    pub const fn horizontal_cell_block_count(&self) -> u8 {
        self.generation_shape.horizontal_cell_block_count()
    }

    #[inline]
    #[must_use]
    pub const fn vertical_cell_block_count(&self) -> u8 {
        self.generation_shape.vertical_cell_block_count()
    }

    /// Aka `bottom_y`
    #[inline]
    #[must_use]
    pub const fn min_y(&self) -> i8 {
        self.generation_shape.min_y
    }

    #[inline]
    #[must_use]
    pub const fn height(&self) -> u16 {
        self.generation_shape.height
    }
}

#[cfg(test)]
mod test {
    // TODO: Add test to verify the height estimator has no interpolators or cell caches
}
