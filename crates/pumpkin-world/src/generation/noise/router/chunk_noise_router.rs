use pumpkin_data::noise_router::WrapperType;
use pumpkin_util::math::vector3::Vector3;

use crate::generation::biome_coords;

use super::{
    chunk_density_function::{
        Cache2D, CacheOnce, CellCache, ChunkNoiseFunctionBuilderOptions,
        ChunkNoiseFunctionSampleOptions, ChunkSpecificNoiseFunctionComponent, DensityInterpolator,
        FlatCache, SampleAction,
    },
    density_function::{
        IndexToNoisePos, NoiseFunctionComponentRange, PassThrough,
        StaticIndependentChunkNoiseFunctionComponentImpl,
    },
    proto_noise_router::{
        DependentProtoNoiseFunctionComponent, IndependentProtoNoiseFunctionComponent,
        ProtoNoiseFunctionComponent, ProtoNoiseRouter,
    },
};

pub trait StaticChunkNoiseFunctionComponentImpl {
    fn sample(
        &self,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        pos: &Vector3<i32>,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64;

    fn fill(
        &self,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        array.iter_mut().enumerate().for_each(|(index, value)| {
            let pos = mapper.at(index, Some(sample_options));
            *value = self.sample(component_stack, &pos, sample_options);
        });
    }
}

pub trait MutableChunkNoiseFunctionComponentImpl {
    fn sample(
        &mut self,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        pos: &Vector3<i32>,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64;

    fn fill(
        &mut self,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        array.iter_mut().enumerate().for_each(|(index, value)| {
            let pos = mapper.at(index, Some(sample_options));
            *value = self.sample(component_stack, &pos, sample_options);
        });
    }
}

pub enum ChunkNoiseFunctionComponent<'a> {
    Independent(&'a IndependentProtoNoiseFunctionComponent),
    Dependent(&'a DependentProtoNoiseFunctionComponent),
    Chunk(ChunkSpecificNoiseFunctionComponent),
    PassThrough(PassThrough),
}

impl NoiseFunctionComponentRange for ChunkNoiseFunctionComponent<'_> {
    #[inline]
    fn min(&self) -> f64 {
        match self {
            Self::Independent(independent) => independent.min(),
            Self::Dependent(dependent) => dependent.min(),
            Self::Chunk(chunk) => chunk.min(),
            Self::PassThrough(pass_through) => pass_through.min(),
        }
    }

    #[inline]
    fn max(&self) -> f64 {
        match self {
            Self::Independent(independent) => independent.max(),
            Self::Dependent(dependent) => dependent.max(),
            Self::Chunk(chunk) => chunk.max(),
            Self::PassThrough(pass_through) => pass_through.max(),
        }
    }
}

impl MutableChunkNoiseFunctionComponentImpl for ChunkNoiseFunctionComponent<'_> {
    #[inline]
    fn sample(
        &mut self,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        pos: &Vector3<i32>,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        match self {
            Self::Independent(independent) => independent.sample(pos),
            Self::Dependent(dependent) => dependent.sample(component_stack, pos, sample_options),
            Self::Chunk(chunk) => chunk.sample(component_stack, pos, sample_options),
            Self::PassThrough(pass_through) => ChunkNoiseFunctionComponent::sample_from_stack(
                &mut component_stack[..=pass_through.input_index()],
                pos,
                sample_options,
            ),
        }
    }

    #[inline]
    fn fill(
        &mut self,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        match self {
            Self::Independent(independent) => independent.fill(array, mapper),
            Self::Dependent(dependent) => {
                dependent.fill(component_stack, array, mapper, sample_options);
            }
            Self::Chunk(chunk) => chunk.fill(component_stack, array, mapper, sample_options),
            Self::PassThrough(pass_through) => ChunkNoiseFunctionComponent::fill_from_stack(
                &mut component_stack[..=pass_through.input_index()],
                array,
                mapper,
                sample_options,
            ),
        }
    }
}

impl ChunkNoiseFunctionComponent<'_> {
    #[inline]
    pub fn sample_from_stack(
        component_stack: &mut [ChunkNoiseFunctionComponent],
        pos: &Vector3<i32>,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        let Some((top_component, component_stack)) = component_stack.split_last_mut() else {
            return 0.0;
        };
        if !sample_options.populating_caches
            && let ChunkNoiseFunctionComponent::Chunk(
                ChunkSpecificNoiseFunctionComponent::DensityInterpolator(interp),
            ) = top_component
        {
            return interp.result;
        }
        top_component.sample(component_stack, pos, sample_options)
    }

    pub fn fill_from_stack(
        component_stack: &mut [ChunkNoiseFunctionComponent],
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        if let Some((top_component, component_stack)) = component_stack.split_last_mut() {
            top_component.fill(component_stack, array, mapper, sample_options);
        }
    }
}

pub struct ChunkNoiseDensityFunction<'a> {
    pub(crate) component_stack: &'a mut [ChunkNoiseFunctionComponent<'a>],
}

impl ChunkNoiseDensityFunction<'_> {
    #[inline]
    pub fn sample(
        &mut self,
        pos: &Vector3<i32>,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        ChunkNoiseFunctionComponent::sample_from_stack(self.component_stack, pos, sample_options)
    }

    #[inline]
    fn fill(
        &mut self,
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        ChunkNoiseFunctionComponent::fill_from_stack(
            self.component_stack,
            array,
            mapper,
            sample_options,
        );
    }
}

macro_rules! sample_function {
    ($name:ident) => {
        #[inline]
        pub fn $name(
            &mut self,
            pos: &Vector3<i32>,
            sample_options: &ChunkNoiseFunctionSampleOptions,
        ) -> f64 {
            ChunkNoiseFunctionComponent::sample_from_stack(
                &mut self.component_stack[..=self.$name],
                pos,
                sample_options,
            )
        }
    };
}

pub struct ChunkNoiseRouter<'a> {
    barrier_noise: usize,
    fluid_level_floodedness_noise: usize,
    fluid_level_spread_noise: usize,
    lava_noise: usize,
    erosion: usize,
    depth: usize,
    final_density: usize,
    vein_toggle: usize,
    vein_ridged: usize,
    vein_gap: usize,
    component_stack: Box<[ChunkNoiseFunctionComponent<'a>]>,
    interpolator_indices: Box<[usize]>,
    cell_indices: Box<[usize]>,

    /// 缓存的 GPU CellFillParams（预计算 DAG 提取）。
    #[cfg(feature = "gpu")]
    cached_cell_params: std::cell::OnceCell<pumpkin_gpu::noise::batch_cell::CellFillParams>,
    /// 缓存的 GPU 插值器填充参数。
    #[cfg(feature = "gpu")]
    cached_interp_params: std::cell::OnceCell<pumpkin_gpu::noise::batch_cell::CellFillParams>,
}

impl ChunkNoiseRouter<'_> {
    sample_function!(barrier_noise);
    sample_function!(fluid_level_floodedness_noise);
    sample_function!(fluid_level_spread_noise);
    sample_function!(lava_noise);
    sample_function!(erosion);
    sample_function!(depth);
    sample_function!(final_density);
    sample_function!(vein_toggle);
    sample_function!(vein_ridged);
    sample_function!(vein_gap);
}

impl<'a> ChunkNoiseRouter<'a> {
    #[must_use]
    #[expect(clippy::too_many_lines)]
    pub fn generate(
        base: &'a ProtoNoiseRouter,
        build_options: &ChunkNoiseFunctionBuilderOptions,
    ) -> Self {
        let mut component_stack =
            Vec::<ChunkNoiseFunctionComponent>::with_capacity(base.full_component_stack.len());
        let mut cell_cache_indices = Vec::new();
        let mut interpolator_indices = Vec::new();

        for (component_index, base_component) in base.full_component_stack.iter().enumerate() {
            let chunk_component = match base_component {
                ProtoNoiseFunctionComponent::Dependent(dependent) => {
                    ChunkNoiseFunctionComponent::Dependent(dependent)
                }
                ProtoNoiseFunctionComponent::Independent(independent) => {
                    ChunkNoiseFunctionComponent::Independent(independent)
                }
                ProtoNoiseFunctionComponent::PassThrough(pass_through) => {
                    ChunkNoiseFunctionComponent::PassThrough(pass_through.clone())
                }
                ProtoNoiseFunctionComponent::Beardifier(_) => {
                    ChunkNoiseFunctionComponent::Chunk(
                        ChunkSpecificNoiseFunctionComponent::Beardifier(
                            crate::generation::noise::router::density_function::beardifier::Beardifier::new(
                                build_options.beardifier_structures.clone(),
                                build_options.beardifier_junctions.clone(),
                                build_options.affected_box,
                            ),
                        ),
                    )
                }
                ProtoNoiseFunctionComponent::Wrapper(wrapper) => {
                    let min_value = component_stack[wrapper.input_index].min();
                    let max_value = component_stack[wrapper.input_index].max();

                    match wrapper.wrapper_type {
                        WrapperType::Interpolated => {
                            interpolator_indices.push(component_index);
                            ChunkNoiseFunctionComponent::Chunk(
                                ChunkSpecificNoiseFunctionComponent::DensityInterpolator(
                                    DensityInterpolator::new(
                                        wrapper.input_index,
                                        min_value,
                                        max_value,
                                        build_options,
                                    ),
                                ),
                            )
                        }
                        WrapperType::CellCache => {
                            cell_cache_indices.push(component_index);
                            ChunkNoiseFunctionComponent::Chunk(
                                ChunkSpecificNoiseFunctionComponent::CellCache(CellCache::new(
                                    wrapper.input_index,
                                    min_value,
                                    max_value,
                                    build_options,
                                )),
                            )
                        }
                        WrapperType::CacheOnce => ChunkNoiseFunctionComponent::Chunk(
                            ChunkSpecificNoiseFunctionComponent::CacheOnce(CacheOnce::new(
                                wrapper.input_index,
                                min_value,
                                max_value,
                            )),
                        ),
                        WrapperType::Cache2D => ChunkNoiseFunctionComponent::Chunk(
                            ChunkSpecificNoiseFunctionComponent::Cache2D(Cache2D::new(
                                wrapper.input_index,
                                min_value,
                                max_value,
                            )),
                        ),
                        WrapperType::CacheFlat => {
                            let mut flat_cache = FlatCache::new(
                                wrapper.input_index,
                                min_value,
                                max_value,
                                build_options.start_biome_x,
                                build_options.start_biome_z,
                                build_options.horizontal_biome_end,
                            );

                            // GPU 批量预计算 FlatCache
                            #[cfg(feature = "gpu")]
                            let gpu_did_fill = crate::gpu::get_noise_accel()
                                .as_mut()
                                .and_then(|accel| {
                                    let h_end = build_options.horizontal_biome_end as i32;
                                    let n_cols = ((h_end + 1) * (h_end + 1)) as usize;
                                    let start_bx = biome_coords::to_block(build_options.start_biome_x);
                                    let start_bz = biome_coords::to_block(build_options.start_biome_z);
                                    let mut pos_3d = Vec::with_capacity(n_cols * 3);
                                    for bz in 0..=h_end {
                                        for bx in 0..=h_end {
                                            pos_3d.push((start_bx + biome_coords::to_block(bx)) as f64);
                                            pos_3d.push(0.0f64);
                                            pos_3d.push((start_bz + biome_coords::to_block(bz)) as f64);
                                        }
                                    }
                                    let sampler_info = extract_flatcache_sampler(
                                        &component_stack[..=wrapper.input_index],
                                    );
                                    sampler_info.map(|sampler| {
                                        let mut results = vec![0.0f64; n_cols];
                                        accel.sample_octave(sampler, &pos_3d, &mut results);
                                        for bi in 0..=build_options.horizontal_biome_end {
                                            for bj in 0..=build_options.horizontal_biome_end {
                                                let idx = bi * (build_options.horizontal_biome_end + 1) + bj;
                                                flat_cache.cache[idx] = results[idx];
                                            }
                                        }
                                        true
                                    })
                                })
                                .unwrap_or(false);
                            #[cfg(not(feature = "gpu"))]
                            let gpu_did_fill = false;

                            // CPU 回退
                            if !gpu_did_fill {
                                let sample_options = ChunkNoiseFunctionSampleOptions::new(
                                    false,
                                    SampleAction::SkipCellCaches,
                                    0,
                                    0,
                                    0,
                                );

                                for biome_x_position in 0..=build_options.horizontal_biome_end {
                                    let absolute_biome_x_position =
                                        build_options.start_biome_x + biome_x_position as i32;
                                    let block_x_position =
                                        biome_coords::to_block(absolute_biome_x_position);

                                    for biome_z_position in 0..=build_options.horizontal_biome_end {
                                        let absolute_biome_z_position =
                                            build_options.start_biome_z + biome_z_position as i32;
                                        let block_z_position =
                                            biome_coords::to_block(absolute_biome_z_position);

                                        let pos = Vector3::new(block_x_position, 0, block_z_position);

                                        let sample = ChunkNoiseFunctionComponent::sample_from_stack(
                                            &mut component_stack[..=wrapper.input_index],
                                            &pos,
                                            &sample_options,
                                        );

                                        let cache_index = flat_cache
                                            .xz_to_index_const(biome_x_position, biome_z_position);
                                        flat_cache.cache[cache_index] = sample;
                                    }
                                }
                            }

                            ChunkNoiseFunctionComponent::Chunk(
                                ChunkSpecificNoiseFunctionComponent::FlatCache(flat_cache),
                            )
                        }
                    }
                }
            };
            component_stack.push(chunk_component);
        }

        Self {
            barrier_noise: base.barrier_noise,
            fluid_level_floodedness_noise: base.fluid_level_floodedness_noise,
            fluid_level_spread_noise: base.fluid_level_spread_noise,
            lava_noise: base.lava_noise,
            erosion: base.erosion,
            depth: base.depth,
            final_density: base.final_density,
            vein_toggle: base.vein_toggle,
            vein_ridged: base.vein_ridged,
            vein_gap: base.vein_gap,
            component_stack: component_stack.into_boxed_slice(),
            interpolator_indices: interpolator_indices.into_boxed_slice(),
            cell_indices: cell_cache_indices.into_boxed_slice(),
            #[cfg(feature = "gpu")]
            cached_cell_params: std::cell::OnceCell::new(),
            #[cfg(feature = "gpu")]
            cached_interp_params: std::cell::OnceCell::new(),
        }
    }

    pub fn fill_cell_caches(
        &mut self,
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        // GPU 批量加速：预填充所有 Cell Cache
        // 通过 DAG 提取 perlin 配置，调用 GPU kernel 批量计算噪声密度。
        #[cfg(feature = "gpu")]
        if let Some(accel) = crate::gpu::get_batch_accel() {
            let params = self.build_cell_fill_params();
            if !params.perlin_configs.is_empty() && !params.num_octaves.is_empty() {
                // 从 mapper 提取所有世界坐标
                let total_positions = self
                    .cell_indices
                    .first()
                    .and_then(|&ci| {
                        self.component_stack.get(ci).and_then(|c| {
                            if let ChunkNoiseFunctionComponent::Chunk(
                                ChunkSpecificNoiseFunctionComponent::CellCache(cc),
                            ) = c
                            {
                                Some(cc.cache.len())
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or(0);

                if total_positions > 0 {
                    let mut positions = Vec::with_capacity(total_positions * 3);
                    for idx in 0..total_positions {
                        let pos = mapper.at(idx, Some(sample_options));
                        positions.push(pos.x as f64);
                        positions.push(pos.y as f64);
                        positions.push(pos.z as f64);
                    }

                    let mut gpu_results = vec![0.0f64; total_positions];
                    accel.batch_fill_cell_caches(&positions, &params, &mut gpu_results);

                    // 将 GPU 结果写回 CellCache 数组
                    let indices = &self.cell_indices;
                    let components = &mut self.component_stack;
                    for cell_cache_index in indices {
                        let (_, component) = components.split_at_mut(*cell_cache_index);
                        if let Some(ChunkNoiseFunctionComponent::Chunk(
                            ChunkSpecificNoiseFunctionComponent::CellCache(cell_cache),
                        )) = component.first_mut()
                            && cell_cache.cache.len() == gpu_results.len()
                        {
                            cell_cache.cache.copy_from_slice(&gpu_results);
                        }
                    }

                    // 回填噪声缓存：将 GPU 计算中使用的 Perlin 采样结果
                    // 写入线程本地缓存，加速后续 CPU DAG 求值。
                    if let Some(&cell_idx) = self.cell_indices.first() {
                        self.backfill_noise_cache(&positions, cell_idx);
                    }

                    return;
                }
            }
        }

        // CPU 回退：DAG 逐位求值
        let indices = &self.cell_indices;
        let components = &mut self.component_stack;
        for cell_cache_index in indices {
            let (component_stack, component) = components.split_at_mut(*cell_cache_index);

            let Some(ChunkNoiseFunctionComponent::Chunk(chunk)) = component.first_mut() else {
                tracing::error!("Expected ChunkNoiseFunctionComponent::Chunk");
                continue;
            };
            let ChunkSpecificNoiseFunctionComponent::CellCache(cell_cache) = chunk else {
                tracing::error!("Expected ChunkSpecificNoiseFunctionComponent::CellCache");
                continue;
            };

            ChunkNoiseFunctionComponent::fill_from_stack(
                &mut component_stack[..=cell_cache.input_index],
                &mut cell_cache.cache,
                mapper,
                sample_options,
            );
        }
    }

    pub fn fill_interpolator_buffers(
        &mut self,
        start: bool,
        cell_z: usize,
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        // GPU 批量加速：预填充所有插值器缓冲区。
        #[cfg(feature = "gpu")]
        if let Some(accel) = crate::gpu::get_batch_accel() {
            let params = self.build_interpolator_fill_params();
            if !params.perlin_configs.is_empty() && !params.num_octaves.is_empty() {
                // 从第一个插值器获取缓冲区大小
                let total_positions = self
                    .interpolator_indices
                    .first()
                    .and_then(|&ii| {
                        self.component_stack.get(ii).and_then(|c| {
                            if let ChunkNoiseFunctionComponent::Chunk(
                                ChunkSpecificNoiseFunctionComponent::DensityInterpolator(di),
                            ) = c
                            {
                                Some(di.vertical_cell_count + 1)
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or(0);

                if total_positions > 0 {
                    let mut positions = Vec::with_capacity(total_positions * 3);
                    for idx in 0..total_positions {
                        let pos = mapper.at(idx, Some(sample_options));
                        positions.push(pos.x as f64);
                        positions.push(pos.y as f64);
                        positions.push(pos.z as f64);
                    }

                    let mut gpu_results = vec![0.0f64; total_positions];
                    accel.batch_fill_interpolators(&positions, &params, &mut gpu_results);

                    // 将 GPU 结果写回 DensityInterpolator 缓冲区
                    let indices = &self.interpolator_indices;
                    let components = &mut self.component_stack;
                    for interpolator_index in indices {
                        let (_, component) = components.split_at_mut(*interpolator_index);
                        if let Some(ChunkNoiseFunctionComponent::Chunk(
                            ChunkSpecificNoiseFunctionComponent::DensityInterpolator(di),
                        )) = component.first_mut()
                        {
                            let start_index = di.yz_to_buf_index(0, cell_z);
                            let buf_len = di.vertical_cell_count + 1;
                            if gpu_results.len() == buf_len {
                                let buf = if start {
                                    &mut di.start_buffer[start_index..start_index + buf_len]
                                } else {
                                    &mut di.end_buffer[start_index..start_index + buf_len]
                                };
                                buf.copy_from_slice(&gpu_results);
                            }
                        }
                    }

                    // 回填噪声缓存
                    if let Some(&interp_idx) = self.interpolator_indices.first() {
                        self.backfill_noise_cache(&positions, interp_idx);
                    }

                    return;
                }
            }
        }

        // CPU 回退：DAG 逐位求值
        let indices = &self.interpolator_indices;
        let components = &mut self.component_stack;
        for interpolator_index in indices {
            let (component_stack, component) = components.split_at_mut(*interpolator_index);

            let Some(ChunkNoiseFunctionComponent::Chunk(chunk)) = component.first_mut() else {
                tracing::error!("Expected ChunkNoiseFunctionComponent::Chunk");
                continue;
            };
            let ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) =
                chunk
            else {
                tracing::error!(
                    "Expected ChunkSpecificNoiseFunctionComponent::DensityInterpolator"
                );
                continue;
            };

            let start_index = density_interpolator.yz_to_buf_index(0, cell_z);
            let buf = if start {
                &mut density_interpolator.start_buffer
                    [start_index..=start_index + density_interpolator.vertical_cell_count]
            } else {
                &mut density_interpolator.end_buffer
                    [start_index..=start_index + density_interpolator.vertical_cell_count]
            };

            ChunkNoiseFunctionComponent::fill_from_stack(
                &mut component_stack[..=density_interpolator.input_index],
                buf,
                mapper,
                sample_options,
            );
        }
    }

    pub fn interpolate_x(&mut self, delta: f64) {
        let indices = &self.interpolator_indices;
        let components = &mut self.component_stack;
        for interpolator_index in indices {
            let ChunkNoiseFunctionComponent::Chunk(chunk) = &mut components[*interpolator_index]
            else {
                tracing::error!("Expected ChunkNoiseFunctionComponent::Chunk");
                continue;
            };

            let ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) =
                chunk
            else {
                tracing::error!(
                    "Expected ChunkSpecificNoiseFunctionComponent::DensityInterpolator"
                );
                continue;
            };

            density_interpolator.interpolate_x(delta);
        }
    }

    /// 回填线程本地噪声缓存。
    ///
    /// 在 GPU 批量计算完成后，对每个 Perlin 噪声采样器，
    /// 用 CPU 计算所有位置的噪声值并写入线程本地缓存。
    /// 后续 `OctavePerlinNoiseSampler::sample` 调用将命中缓存，
    /// 避免重复计算。
    #[cfg(feature = "gpu")]
    fn backfill_noise_cache(&self, positions: &[f64], start_index: usize) {
        use std::collections::HashSet;

        let mut visited = HashSet::new();
        let samplers = self.collect_noise_samplers(start_index, &mut visited);
        if samplers.is_empty() {
            return;
        }

        let n = positions.len() / 3;
        if n == 0 {
            return;
        }

        // 收集所有采样器-位置对的噪声值，合并写入一次缓存
        let mut cache_entries: std::collections::HashMap<(u64, i64, i64, i64), f64> =
            std::collections::HashMap::with_capacity(samplers.len() * n);

        for info in &samplers {
            if info.sampler_id == 0 {
                continue;
            }
            for idx in 0..n {
                let x = positions[idx * 3];
                let y = positions[idx * 3 + 1];
                let z = positions[idx * 3 + 2];
                let value = info.sampler.sample(x, y, z);
                let ix = (x * 1_000_000.0) as i64;
                let iy = (y * 1_000_000.0) as i64;
                let iz = (z * 1_000_000.0) as i64;
                cache_entries.insert((info.sampler_id, ix, iy, iz), value);
            }
        }

        if !cache_entries.is_empty() {
            pumpkin_util::noise::perlin::set_noise_cache(cache_entries);
        }
    }

    pub fn interpolate_y(&mut self, delta: f64) {
        let indices = &self.interpolator_indices;
        let components = &mut self.component_stack;
        for interpolator_index in indices {
            let ChunkNoiseFunctionComponent::Chunk(chunk) = &mut components[*interpolator_index]
            else {
                tracing::error!("Expected ChunkNoiseFunctionComponent::Chunk");
                continue;
            };

            let ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) =
                chunk
            else {
                tracing::error!(
                    "Expected ChunkSpecificNoiseFunctionComponent::DensityInterpolator"
                );
                continue;
            };

            density_interpolator.interpolate_y(delta);
        }
    }

    pub fn interpolate_z(&mut self, delta: f64) {
        let indices = &self.interpolator_indices;
        let components = &mut self.component_stack;
        for interpolator_index in indices {
            let ChunkNoiseFunctionComponent::Chunk(chunk) = &mut components[*interpolator_index]
            else {
                tracing::error!("Expected ChunkNoiseFunctionComponent::Chunk");
                continue;
            };
            let ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) =
                chunk
            else {
                tracing::error!(
                    "Expected ChunkSpecificNoiseFunctionComponent::DensityInterpolator"
                );
                continue;
            };

            density_interpolator.interpolate_z(delta);
        }
    }

    /// GPU-accelerated combined trilinear interpolation.
    ///
    /// Replaces the three separate `interpolate_y` → `interpolate_x` →
    /// `interpolate_z` calls with a single GPU batch trilinear operation.
    /// Falls back to sequential CPU interpolation when GPU is unavailable.
    pub fn interpolate_xyz(&mut self, delta_y: f64, delta_x: f64, delta_z: f64) {
        #[cfg(feature = "gpu")]
        if let Some(accel) = crate::gpu::get_batch_accel() {
            let n = self.interpolator_indices.len();
            if n > 0 {
                // Collect 8 corner values (first_pass) from all interpolators
                let mut corners = Vec::with_capacity(n * 8);
                for idx in &self.interpolator_indices {
                    if let ChunkNoiseFunctionComponent::Chunk(
                        ChunkSpecificNoiseFunctionComponent::DensityInterpolator(di),
                    ) = &self.component_stack[*idx]
                    {
                        // Map DensityInterpolator corner layout to standard trilinear:
                        // X=0→start, X=1→end; Y=0→y, Y=1→y+1; Z=0→z, Z=1→z+1
                        let fp = &di.first_pass;
                        corners.extend_from_slice(&[
                            fp[0], // (X=start, Y=y,   Z=z)
                            fp[4], // (X=end,   Y=y,   Z=z)
                            fp[2], // (X=start, Y=y+1, Z=z)
                            fp[6], // (X=end,   Y=y+1, Z=z)
                            fp[1], // (X=start, Y=y,   Z=z+1)
                            fp[5], // (X=end,   Y=y,   Z=z+1)
                            fp[3], // (X=start, Y=y+1, Z=z+1)
                            fp[7], // (X=end,   Y=y+1, Z=z+1)
                        ]);
                    }
                }

                // Replicate deltas for all interpolators
                let mut deltas = Vec::with_capacity(n * 3);
                for _ in 0..n {
                    deltas.push(delta_x);
                    deltas.push(delta_y);
                    deltas.push(delta_z);
                }

                let mut results = vec![0.0f64; n];
                accel.batch_trilinear(&corners, &deltas, &mut results);

                // Write results back to interpolator.result
                for (i, idx) in self.interpolator_indices.iter().enumerate() {
                    if let ChunkNoiseFunctionComponent::Chunk(
                        ChunkSpecificNoiseFunctionComponent::DensityInterpolator(di),
                    ) = &mut self.component_stack[*idx]
                    {
                        di.result = results[i];
                    }
                }
                return;
            }
        }

        // CPU fallback: sequential interpolation (original path)
        self.interpolate_y(delta_y);
        self.interpolate_x(delta_x);
        self.interpolate_z(delta_z);
    }

    pub fn on_sampled_cell_corners(&mut self, cell_y_position: usize, cell_z_position: usize) {
        let indices = &self.interpolator_indices;
        let components = &mut self.component_stack;
        for interpolator_index in indices {
            let ChunkNoiseFunctionComponent::Chunk(chunk) = &mut components[*interpolator_index]
            else {
                tracing::error!("Expected ChunkNoiseFunctionComponent::Chunk");
                continue;
            };
            let ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) =
                chunk
            else {
                tracing::error!(
                    "Expected ChunkSpecificNoiseFunctionComponent::DensityInterpolator"
                );
                continue;
            };

            density_interpolator.on_sampled_cell_corners(cell_y_position, cell_z_position);
        }
    }

    /// 将预计算的 cell 数据批量复制到所有 `CellCache` 实例。
    /// 由 chunk 级批量 GPU 填充后调用，替代逐 cell 的 kernel launch。
    pub fn copy_to_cell_caches(&mut self, data: &[f64]) {
        let indices = &self.cell_indices;
        let components = &mut self.component_stack;
        for cell_cache_index in indices {
            let (_, component) = components.split_at_mut(*cell_cache_index);
            if let Some(ChunkNoiseFunctionComponent::Chunk(
                ChunkSpecificNoiseFunctionComponent::CellCache(cell_cache),
            )) = component.first_mut()
            {
                debug_assert_eq!(cell_cache.cache.len(), data.len());
                cell_cache.cache.copy_from_slice(data);
            }
        }
    }

    pub fn swap_buffers(&mut self) {
        let indices = &self.interpolator_indices;
        let components = &mut self.component_stack;
        for interpolator_index in indices {
            let ChunkNoiseFunctionComponent::Chunk(chunk) = &mut components[*interpolator_index]
            else {
                tracing::error!("Expected ChunkNoiseFunctionComponent::Chunk");
                continue;
            };
            let ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) =
                chunk
            else {
                tracing::error!(
                    "Expected ChunkSpecificNoiseFunctionComponent::DensityInterpolator"
                );
                continue;
            };

            density_interpolator.swap_buffers();
        }
    }

    // ========================================================================
    // DAG context-driven CellFillParams extraction (GPU feature)
    // ========================================================================

    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn build_cell_fill_params(&self) -> pumpkin_gpu::noise::batch_cell::CellFillParams {
        self.cached_cell_params
            .get_or_init(|| self.compute_cell_fill_params())
            .clone()
    }

    /// 实际计算 CellFillParams（DAG 遍历）。
    #[cfg(feature = "gpu")]
    fn compute_cell_fill_params(&self) -> pumpkin_gpu::noise::batch_cell::CellFillParams {
        use pumpkin_gpu::noise::batch_cell::CellFillParams;
        use std::collections::HashSet;

        let mut perlin_configs = Vec::<f64>::new();
        let mut num_octaves = Vec::<i32>::new();
        let mut sampler_types = Vec::<i32>::new();

        // 从第一个 cell cache 开始遍历 DAG
        if let Some(&cell_idx) = self.cell_indices.first() {
            let mut visited = HashSet::new();
            let samplers = self.collect_noise_samplers(cell_idx, &mut visited);

            for info in &samplers {
                num_octaves.push(info.num_octaves);
                sampler_types.push(info.sampler_type);

                // 编码：num_octaves + amplitudes + lacunarities + origins
                perlin_configs.push(info.num_octaves as f64);
                perlin_configs.extend_from_slice(&info.amplitudes);
                perlin_configs.extend_from_slice(&info.lacunarities);
                perlin_configs.extend_from_slice(&info.origins);
            }
        }

        CellFillParams {
            perlin_configs,
            num_octaves,
            sampler_types,
        }
    }

    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn build_interpolator_fill_params(&self) -> pumpkin_gpu::noise::batch_cell::CellFillParams {
        self.cached_interp_params
            .get_or_init(|| self.compute_interpolator_fill_params())
            .clone()
    }

    /// 实际计算插值器填充参数（DAG 遍历）。
    #[cfg(feature = "gpu")]
    fn compute_interpolator_fill_params(&self) -> pumpkin_gpu::noise::batch_cell::CellFillParams {
        use pumpkin_gpu::noise::batch_cell::CellFillParams;
        use std::collections::HashSet;

        let mut perlin_configs = Vec::<f64>::new();
        let mut num_octaves = Vec::<i32>::new();
        let mut sampler_types = Vec::<i32>::new();

        if let Some(&interp_idx) = self.interpolator_indices.first() {
            let mut visited = HashSet::new();
            let samplers = self.collect_noise_samplers(interp_idx, &mut visited);

            for info in &samplers {
                num_octaves.push(info.num_octaves);
                sampler_types.push(info.sampler_type);

                let xz_scale = info.xz_scale.unwrap_or(1.0);
                let y_scale = info.y_scale.unwrap_or(1.0);

                for o in 0..info.num_octaves as usize {
                    perlin_configs.push(info.amplitudes[o]);
                    perlin_configs.push(info.lacunarities[o]);
                    perlin_configs.push(info.origins[o * 3]);
                    perlin_configs.push(info.origins[o * 3 + 1]);
                    perlin_configs.push(info.origins[o * 3 + 2]);
                    perlin_configs.push(xz_scale);
                    perlin_configs.push(y_scale);
                    perlin_configs.push(0.0); // reserved
                }
            }
        }

        CellFillParams {
            perlin_configs,
            num_octaves,
            sampler_types,
        }
    }

    /// Extract vein noise parameters from the DAG for GPU-accelerated vein sampling.
    ///
    /// Walks from `vein_toggle`, `vein_ridged`, and `vein_gap` component indices
    /// to extract perlin configurations. Uses the same 8-double-per-octave
    /// encoding as interpolator fills.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn build_vein_params(&self) -> pumpkin_gpu::noise::batch_cell::VeinParams {
        use pumpkin_gpu::noise::batch_cell::VeinParams;
        use std::collections::HashSet;

        fn extract_segment_config(router: &ChunkNoiseRouter, start_idx: usize) -> Vec<f64> {
            let mut visited = HashSet::new();
            let samplers = router.collect_noise_samplers(start_idx, &mut visited);
            let mut config = Vec::new();
            for info in &samplers {
                let xz = info.xz_scale.unwrap_or(1.0);
                let ys = info.y_scale.unwrap_or(1.0);
                for o in 0..info.num_octaves as usize {
                    config.push(info.amplitudes[o]);
                    config.push(info.lacunarities[o]);
                    config.push(info.origins[o * 3]);
                    config.push(info.origins[o * 3 + 1]);
                    config.push(info.origins[o * 3 + 2]);
                    config.push(xz);
                    config.push(ys);
                    config.push(0.0); // reserved
                }
            }
            config
        }

        VeinParams {
            toggle_config: extract_segment_config(self, self.vein_toggle),
            ridged_config: extract_segment_config(self, self.vein_ridged),
            gap_config: extract_segment_config(self, self.vein_gap),
        }
    }

    /// Recursively walk the component stack from `start_index` collecting all
    /// reachable leaf noise samplers.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_lines)]
    fn collect_noise_samplers(
        &self,
        start_index: usize,
        visited: &mut std::collections::HashSet<usize>,
    ) -> Vec<NoiseSamplerInfo<'_>> {
        if !visited.insert(start_index) {
            return Vec::new();
        }

        let Some(component) = self.component_stack.get(start_index) else {
            return Vec::new();
        };

        match component {
            ChunkNoiseFunctionComponent::Independent(independent) => {
                match independent {
                    IndependentProtoNoiseFunctionComponent::Noise(noise) => {
                        let info = extract_double_perlin_config(
                            noise.sampler.first_sampler(),
                            noise.sampler.amplitude(),
                            0, // Noise
                            Some(noise.data.xz_scale),
                            Some(noise.data.y_scale),
                        );
                        vec![info]
                    }
                    IndependentProtoNoiseFunctionComponent::ShiftA(shift_a) => {
                        let info = extract_double_perlin_config(
                            shift_a.sampler.first_sampler(),
                            shift_a.sampler.amplitude() * 4.0, // ShiftA multiplies by 4
                            1,                                 // ShiftA
                            None,
                            None,
                        );
                        vec![info]
                    }
                    IndependentProtoNoiseFunctionComponent::ShiftB(shift_b) => {
                        let info = extract_double_perlin_config(
                            shift_b.sampler.first_sampler(),
                            shift_b.sampler.amplitude() * 4.0, // ShiftB multiplies by 4
                            2,                                 // ShiftB
                            None,
                            None,
                        );
                        vec![info]
                    }
                    IndependentProtoNoiseFunctionComponent::InterpolatedNoise(interp) => {
                        // InterpolatedNoise has 3 OctavePerlinNoiseSamplers.
                        // For GPU, extract the main `noise` sampler.
                        let info = extract_interpolated_noise_config(interp);
                        vec![info]
                    }
                    // Non-noise independent components → no perlin configs
                    IndependentProtoNoiseFunctionComponent::Constant(_)
                    | IndependentProtoNoiseFunctionComponent::EndIsland(_)
                    | IndependentProtoNoiseFunctionComponent::ClampedYGradient(_) => Vec::new(),
                }
            }
            ChunkNoiseFunctionComponent::Dependent(dependent) => {
                match dependent {
                    DependentProtoNoiseFunctionComponent::Linear(linear) => {
                        self.collect_noise_samplers(linear.input_index, visited)
                    }
                    DependentProtoNoiseFunctionComponent::Unary(unary) => {
                        self.collect_noise_samplers(unary.input_index, visited)
                    }
                    DependentProtoNoiseFunctionComponent::Binary(binary) => {
                        let mut result = self.collect_noise_samplers(binary.input1_index, visited);
                        result.extend(self.collect_noise_samplers(binary.input2_index, visited));
                        result
                    }
                    DependentProtoNoiseFunctionComponent::ShiftedNoise(shifted) => {
                        let mut result =
                            self.collect_noise_samplers(shifted.input_x_index, visited);
                        result.extend(self.collect_noise_samplers(shifted.input_y_index, visited));
                        result.extend(self.collect_noise_samplers(shifted.input_z_index, visited));
                        // Also collect the shifted noise's own sampler
                        let info = extract_double_perlin_config(
                            shifted.sampler.first_sampler(),
                            shifted.sampler.amplitude(),
                            0, // treat as Noise
                            Some(shifted.data.xz_scale),
                            Some(shifted.data.y_scale),
                        );
                        result.push(info);
                        result
                    }
                    DependentProtoNoiseFunctionComponent::IntervalSelect(is) => {
                        let mut result = self.collect_noise_samplers(is.input_index, visited);
                        for &idx in is.functions_indices {
                            result.extend(self.collect_noise_samplers(idx, visited));
                        }
                        result
                    }
                    DependentProtoNoiseFunctionComponent::RangeChoice(rc) => {
                        let mut result = self.collect_noise_samplers(rc.input_index, visited);
                        result.extend(self.collect_noise_samplers(rc.when_in_index, visited));
                        result.extend(self.collect_noise_samplers(rc.when_out_index, visited));
                        result
                    }
                    DependentProtoNoiseFunctionComponent::FindTopSurface(fts) => {
                        let mut result = self.collect_noise_samplers(fts.density_index(), visited);
                        result
                            .extend(self.collect_noise_samplers(fts.upper_bound_index(), visited));
                        result
                    }
                    DependentProtoNoiseFunctionComponent::Clamp(clamp) => {
                        self.collect_noise_samplers(clamp.input_index, visited)
                    }
                    DependentProtoNoiseFunctionComponent::Spline(spline_fn) => {
                        self.collect_noise_samplers(spline_fn.spline().input_index, visited)
                    }
                }
            }
            ChunkNoiseFunctionComponent::Chunk(chunk) => match chunk {
                ChunkSpecificNoiseFunctionComponent::CellCache(cc) => {
                    self.collect_noise_samplers(cc.input_index, visited)
                }
                ChunkSpecificNoiseFunctionComponent::DensityInterpolator(di) => {
                    self.collect_noise_samplers(di.input_index, visited)
                }
                ChunkSpecificNoiseFunctionComponent::CacheOnce(co) => {
                    self.collect_noise_samplers(co.input_index, visited)
                }
                ChunkSpecificNoiseFunctionComponent::Cache2D(c2d) => {
                    self.collect_noise_samplers(c2d.input_index, visited)
                }
                ChunkSpecificNoiseFunctionComponent::FlatCache(fc) => {
                    self.collect_noise_samplers(fc.input_index, visited)
                }
                ChunkSpecificNoiseFunctionComponent::Beardifier(_) => Vec::new(),
            },
            ChunkNoiseFunctionComponent::PassThrough(pass_through) => {
                self.collect_noise_samplers(pass_through.input_index(), visited)
            }
        }
    }
}

/// 尝试从 DAG 组件栈中提取简单的 `OctavePerlinNoiseSampler` 用于 `FlatCache` GPU 加速。
///
/// 如果组件栈解析到单一的 `Noise`、`ShiftA`、`ShiftB` 或 `InterpolatedNoise`
/// 采样器，返回其 `first_sampler`。如果 DAG 更复杂（含 Dependent 组件等），
/// 返回 None 让调用方回退到 CPU 路径。
#[cfg(feature = "gpu")]
fn extract_flatcache_sampler<'a>(
    stack: &'a [ChunkNoiseFunctionComponent<'a>],
) -> Option<&'a pumpkin_util::noise::perlin::OctavePerlinNoiseSampler> {
    // 从栈顶向下查找，看是否直接指向 Independent(Noise/ShiftA/ShiftB/InterpolatedNoise)
    let top = stack.last()?;
    match top {
        ChunkNoiseFunctionComponent::Independent(independent) => match independent {
            IndependentProtoNoiseFunctionComponent::Noise(n) => Some(n.sampler.first_sampler()),
            IndependentProtoNoiseFunctionComponent::ShiftA(s) => Some(s.sampler.first_sampler()),
            IndependentProtoNoiseFunctionComponent::ShiftB(s) => Some(s.sampler.first_sampler()),
            IndependentProtoNoiseFunctionComponent::InterpolatedNoise(i) => Some(&i.noise),
            _ => None,
        },
        _ => None,
    }
}

// ============================================================================
// Helper types & functions for DAG-driven CellFillParams (GPU feature)
// ============================================================================

/// Noise sampler configuration extracted from the DAG.
#[cfg(feature = "gpu")]
struct NoiseSamplerInfo<'a> {
    /// Reference to the actual sampler (for noise cache backfill).
    sampler: &'a pumpkin_util::noise::perlin::OctavePerlinNoiseSampler,
    /// Unique sampler identifier for noise cache lookups.
    sampler_id: u64,
    /// 0=Noise, 1=ShiftA, 2=ShiftB, 3=InterpolatedNoise
    sampler_type: i32,
    /// Number of octaves in the `OctavePerlinNoiseSampler`
    num_octaves: i32,
    /// Per-octave amplitudes (already multiplied by persistence and parent amplitude)
    amplitudes: Vec<f64>,
    /// Per-octave lacunarities
    lacunarities: Vec<f64>,
    /// Flattened per-octave origins [x0, y0, z0, x1, y1, z1, ...]
    origins: Vec<f64>,
    /// XZ scale from `NoiseData` (for interpolator encoding)
    xz_scale: Option<f64>,
    /// Y scale from `NoiseData` (for interpolator encoding)
    y_scale: Option<f64>,
}

/// Extract per-octave configuration from an [`OctavePerlinNoiseSampler`].
///
/// The `parent_amplitude` is the multiplier from the parent
/// `DoublePerlinNoiseSampler` (or Shift multiplier).
/// Amplitudes are pre-multiplied by persistence so the GPU kernel only
/// needs `amp * sample_perlin(…)` without additional scaling.
#[cfg(feature = "gpu")]
fn extract_double_perlin_config(
    octave_sampler: &pumpkin_util::noise::perlin::OctavePerlinNoiseSampler,
    parent_amplitude: f64,
    sampler_type: i32,
    xz_scale: Option<f64>,
    y_scale: Option<f64>,
) -> NoiseSamplerInfo<'_> {
    let num_octaves = octave_sampler.samplers.len() as i32;
    let mut amplitudes = Vec::with_capacity(num_octaves as usize);
    let mut lacunarities = Vec::with_capacity(num_octaves as usize);
    let mut origins = Vec::with_capacity(num_octaves as usize * 3);

    for sd in &octave_sampler.samplers {
        // GPU kernel multiplies amp * sample_perlin, so bake persistence and
        // parent amplitude into the stored amplitude.
        amplitudes.push(sd.amplitude * sd.persistence * parent_amplitude);
        lacunarities.push(sd.lacunarity);
        origins.push(sd.sampler.x_origin());
        origins.push(sd.sampler.y_origin());
        origins.push(sd.sampler.z_origin());
    }

    NoiseSamplerInfo {
        sampler: octave_sampler,
        sampler_id: octave_sampler.sampler_id,
        sampler_type,
        num_octaves,
        amplitudes,
        lacunarities,
        origins,
        xz_scale,
        y_scale,
    }
}

/// Extract configuration from an [`InterpolatedNoiseSampler`].
///
/// Uses the `noise` field (the primary 8-octave sampler).
#[cfg(feature = "gpu")]
fn extract_interpolated_noise_config(
    interp: &super::density_function::noise::InterpolatedNoiseSampler,
) -> NoiseSamplerInfo<'_> {
    let num_octaves = interp.noise.samplers.len() as i32;
    let mut amplitudes = Vec::with_capacity(num_octaves as usize);
    let mut lacunarities = Vec::with_capacity(num_octaves as usize);
    let mut origins = Vec::with_capacity(num_octaves as usize * 3);

    for sd in &interp.noise.samplers {
        // InterpolatedNoise uses the noise sampler directly within
        // a complex weighted sum; amplitude here is per-octave weight.
        amplitudes.push(sd.amplitude * sd.persistence);
        lacunarities.push(sd.lacunarity);
        origins.push(sd.sampler.x_origin());
        origins.push(sd.sampler.y_origin());
        origins.push(sd.sampler.z_origin());
    }

    NoiseSamplerInfo {
        sampler: &interp.noise,
        sampler_id: interp.noise.sampler_id,
        sampler_type: 3, // InterpolatedNoise
        num_octaves,
        amplitudes,
        lacunarities,
        origins,
        xz_scale: None,
        y_scale: None,
    }
}
