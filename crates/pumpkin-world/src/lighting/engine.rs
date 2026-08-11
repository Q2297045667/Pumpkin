use crate::chunk_system::Chunk;
use crate::chunk_system::generation_cache::Cache;
use crate::generation::height_limit::HeightLimitView;
use crate::generation::proto_chunk::GenerationCache;
use crate::lighting::storage::{get_block_light, get_sky_light, set_block_light, set_sky_light};
use pumpkin_config::lighting::LightingEngineConfig;
use pumpkin_data::BlockDirection;
use pumpkin_util::HeightMap;
use pumpkin_util::math::position::BlockPos;
use pumpkin_util::math::vector3::Vector3;
use std::collections::VecDeque;
//use std::time::Instant;

// These are hit on every neighbour of every propagated block, so the hash
// function dominates. Use rustc-hash's fast hasher instead of the std default
// (SipHash), which is what "Fast" was meant to imply.
type FastHashSet<K> = rustc_hash::FxHashSet<K>;
type FastHashMap<K, V> = rustc_hash::FxHashMap<K, V>;

/// Trait to unify Block and Sky light logic
pub trait LightProvider {
    fn get_light(cache: &Cache, pos: BlockPos) -> u8;
    fn set_light(cache: &mut Cache, pos: BlockPos, level: u8);
    fn propagate_level(current_level: u8, opacity: u8, dir: BlockDirection) -> u8;
}

pub struct BlockLightProvider;
impl LightProvider for BlockLightProvider {
    fn get_light(cache: &Cache, pos: BlockPos) -> u8 {
        get_block_light(cache, pos)
    }
    fn set_light(cache: &mut Cache, pos: BlockPos, level: u8) {
        set_block_light(cache, pos, level);
    }
    fn propagate_level(current_level: u8, opacity: u8, _dir: BlockDirection) -> u8 {
        current_level.saturating_sub(opacity.max(1))
    }
}

pub struct SkyLightProvider;
impl LightProvider for SkyLightProvider {
    fn get_light(cache: &Cache, pos: BlockPos) -> u8 {
        get_sky_light(cache, pos)
    }
    fn set_light(cache: &mut Cache, pos: BlockPos, level: u8) {
        set_sky_light(cache, pos, level);
    }
    fn propagate_level(current_level: u8, opacity: u8, dir: BlockDirection) -> u8 {
        if current_level == 15 && dir == BlockDirection::Down && opacity == 0 {
            return 15;
        }

        current_level.saturating_sub(opacity.max(1))
    }
}

#[derive(Clone, Copy)]
pub struct PropagationEntry {
    pos: BlockPos,
    skip_direction: Option<BlockDirection>, // direction from which the light came, used to prevent back-propagation
}

pub struct LightPropagator<P: LightProvider> {
    pub(crate) queue: VecDeque<PropagationEntry>,
    pub(crate) visited: FastHashSet<BlockPos>,
    pub(crate) decrease_queue: VecDeque<(BlockPos, u8)>,
    _marker: std::marker::PhantomData<P>,
}

impl<P: LightProvider> LightPropagator<P> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: VecDeque::with_capacity(4096),
            visited: FastHashSet::default(),
            decrease_queue: VecDeque::new(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn clear(&mut self) {
        self.queue.clear();
        self.visited.clear();
        self.decrease_queue.clear();
    }

    /// Core Propagation Logic (BFS).
    ///
    /// Reads and writes light directly through the light storage (a fast array
    /// lookup) instead of maintaining a separate hashed shadow cache and batched
    /// write buffer; the storage is the single source of truth.
    pub fn propagate(&mut self, cache: &mut Cache) {
        // Cache metadata for bounds checking
        let cache_x = cache.x;
        let cache_z = cache.z;
        let cache_size = cache.size;
        let min_y = cache.bottom_y() as i32;
        let max_y = min_y + cache.height() as i32;

        while let Some(entry) = self.queue.pop_front() {
            let pos = entry.pos;

            let current_light = P::get_light(cache, pos);
            if current_light <= 1 {
                continue;
            }

            for dir in BlockDirection::all() {
                // Skip the direction we came from (if specified)
                if let Some(skip_dir) = entry.skip_direction
                    && dir == skip_dir
                {
                    continue;
                }

                let neighbor_pos = pos.offset(dir.to_offset());

                // Skip if already visited (critical early-exit optimization)
                if self.visited.contains(&neighbor_pos) {
                    continue;
                }

                // Skip neighbor if it's outside world bounds
                if neighbor_pos.0.y < min_y || neighbor_pos.0.y >= max_y {
                    continue;
                }

                let (cx, _rel) = neighbor_pos.chunk_and_chunk_relative_position();
                let rel_x = cx.x - cache_x;
                let rel_z = cx.y - cache_z;
                if rel_x < 0 || rel_x >= cache_size || rel_z < 0 || rel_z >= cache_size {
                    continue;
                }

                // Get block opacity
                let state = cache.get_block_state(&neighbor_pos.0);
                let opacity = state.to_state().opacity;

                let new_level = P::propagate_level(current_light, opacity, dir);
                let neighbor_light = P::get_light(cache, neighbor_pos);

                if new_level > neighbor_light {
                    P::set_light(cache, neighbor_pos, new_level);

                    // Add to propagation queue if bright enough
                    if new_level > 1 && self.visited.insert(neighbor_pos) {
                        self.queue.push_back(PropagationEntry {
                            pos: neighbor_pos,
                            skip_direction: Some(dir.opposite()),
                        });
                    }
                }
            }
        }
    }

    /// Handle light removal
    pub fn process_decrease_queue(&mut self, cache: &mut Cache) {
        {
            // Cache metadata for bounds checking
            let cache_x = cache.x;
            let cache_z = cache.z;
            let cache_size = cache.size;

            while let Some((pos, old_val)) = self.decrease_queue.pop_front() {
                for dir in BlockDirection::all() {
                    let neighbor_pos = pos.offset(dir.to_offset());

                    // Bounds check
                    let (cx, _rel) = neighbor_pos.chunk_and_chunk_relative_position();
                    let rel_x = cx.x - cache_x;
                    let rel_z = cx.y - cache_z;

                    if rel_x < 0 || rel_x >= cache_size || rel_z < 0 || rel_z >= cache_size {
                        continue;
                    }

                    let neighbor_light = P::get_light(cache, neighbor_pos);
                    if neighbor_light == 0 {
                        continue;
                    }

                    let state = cache.get_block_state(&neighbor_pos.0);
                    let opacity = state.to_state().opacity;

                    let predicted = P::propagate_level(old_val, opacity, dir);

                    if neighbor_light == predicted || neighbor_light < old_val {
                        // Darken
                        P::set_light(cache, neighbor_pos, 0);
                        self.decrease_queue
                            .push_back((neighbor_pos, neighbor_light));
                    } else if neighbor_light >= old_val {
                        // Re-illuminate from this bright neighbor
                        self.queue.push_back(PropagationEntry {
                            pos: neighbor_pos,
                            skip_direction: None,
                        });
                        self.visited.insert(neighbor_pos);
                    }
                }
            }
        }

        self.propagate(cache); // Re-propagate from survivors
    }
}

pub type BlockLightPropagator = LightPropagator<BlockLightProvider>;
pub type SkyLightPropagator = LightPropagator<SkyLightProvider>;

impl<P: LightProvider> Default for LightPropagator<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockLightPropagator {
    pub fn propagate_light(&mut self, cache: &mut Cache) {
        self.clear();

        //let scan_start = Instant::now();

        let min_y = cache.bottom_y() as i32;
        let max_y = min_y + cache.height() as i32;
        let center_x = cache.x + (cache.size / 2);
        let center_z = cache.z + (cache.size / 2);

        let start_x = center_x * 16 - 1;
        let start_z = center_z * 16 - 1;
        let end_x = start_x + 18;
        let end_z = start_z + 18;

        for y in min_y..max_y {
            for z in start_z..end_z {
                for x in start_x..end_x {
                    let pos_vec = Vector3::new(x, y, z);
                    let state = cache.get_block_state(&pos_vec);
                    let emission = state.to_state().luminance;
                    if emission > 0 {
                        let pos = BlockPos(pos_vec);
                        set_block_light(cache, pos, emission);
                        if self.visited.insert(pos) {
                            // Block light propagates in all directions
                            self.queue.push_back(PropagationEntry {
                                pos,
                                skip_direction: None,
                            });
                        }
                    }
                }
            }
        }
        //let scan_elapsed = scan_start.elapsed();
        //let propagate_start = Instant::now();

        self.propagate(cache);

        //let propagate_elapsed = propagate_start.elapsed();
        //log::info!("Block light timing - Scan: {:?}, Propagate: {:?}", scan_elapsed, propagate_elapsed);
    }
}

impl SkyLightPropagator {
    #[expect(clippy::too_many_lines)]
    pub fn convert_light(&mut self, cache: &mut Cache) {
        self.clear();

        //let scan_start = Instant::now();

        let center_x = cache.x + (cache.size / 2);
        let center_z = cache.z + (cache.size / 2);
        let start_x = center_x * 16 - 1;
        let start_z = center_z * 16 - 1;
        let end_x = start_x + 18;
        let end_z = start_z + 18;

        let bottom_y = cache.bottom_y() as i32;
        let max_y = bottom_y + cache.height() as i32;

        // Pre-allocate with exact size needed
        let capacity = ((end_x - start_x) * (end_z - start_z)) as usize;
        let mut surface_heights =
            FastHashMap::with_capacity_and_hasher(capacity, rustc_hash::FxBuildHasher);

        // Process in Z-outer, X-inner order for better cache locality
        for z in start_z..end_z {
            let chunk_z = z >> 4;
            let local_z = (z & 15) as usize;

            for x in start_x..end_x {
                let chunk_x = x >> 4;
                let local_x = (x & 15) as usize;

                // Get heightmap (top solid blocks)
                let top_y = cache.get_top_y(&HeightMap::WorldSurface, x, z);
                surface_heights.insert((x, z), top_y);

                // Get chunk index once per column
                let rel_x = chunk_x - cache.x;
                let rel_z = chunk_z - cache.z;

                if rel_x < 0 || rel_x >= cache.size || rel_z < 0 || rel_z >= cache.size {
                    continue;
                }

                let chunk_idx = (rel_x * cache.size + rel_z) as usize;

                // Fill everything above heightmap with 15 immediately
                for y in (top_y + 1)..max_y {
                    let section_idx = ((y - bottom_y) >> 4) as usize;
                    let local_y = (y & 15) as usize;

                    // Direct array access - skip all function call overhead
                    match &mut cache.chunks[chunk_idx] {
                        Chunk::Proto(c) => {
                            if section_idx < c.light.sky_light.len() {
                                c.light.sky_light[section_idx].set(local_x, local_y, local_z, 15);
                            }
                        }
                        Chunk::Level(c) => {
                            let mut light_engine = c
                                .light_engine
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            if section_idx < light_engine.sky_light.len() {
                                light_engine.sky_light[section_idx]
                                    .set(local_x, local_y, local_z, 15);
                            }
                        }
                    }
                }

                // Only iterate from top_y DOWN - not from max_y
                let mut light: i32 = 15;

                for y in (bottom_y..=top_y).rev() {
                    let section_idx = ((y - bottom_y) >> 4) as usize;
                    let local_y = (y & 15) as usize;

                    // Get block opacity
                    let opacity = {
                        let pos_vec = Vector3::new(x, y, z);
                        let state = cache.get_block_state(&pos_vec);
                        state.to_state().opacity
                    } as i32;

                    // Reduce light by opacity
                    light = light.saturating_sub(opacity);

                    // Set the light value directly
                    let light_val = if light <= 0 { 0 } else { light as u8 };

                    match &mut cache.chunks[chunk_idx] {
                        Chunk::Proto(c) => {
                            if section_idx < c.light.sky_light.len() {
                                c.light.sky_light[section_idx]
                                    .set(local_x, local_y, local_z, light_val);
                            }
                        }
                        Chunk::Level(c) => {
                            let mut light_engine = c
                                .light_engine
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            if section_idx < light_engine.sky_light.len() {
                                light_engine.sky_light[section_idx]
                                    .set(local_x, local_y, local_z, light_val);
                            }
                        }
                    }

                    // Early exit when light hits 0
                    if light <= 0 {
                        break;
                    }
                }
            }
        }

        // Enqueue horizontal propagation
        for z in start_z..end_z {
            for x in start_x..end_x {
                let top_y = surface_heights[&(x, z)];

                let north_top = surface_heights.get(&(x, z - 1)).copied().unwrap_or(top_y);
                let south_top = surface_heights.get(&(x, z + 1)).copied().unwrap_or(top_y);
                let west_top = surface_heights.get(&(x - 1, z)).copied().unwrap_or(top_y);
                let east_top = surface_heights.get(&(x + 1, z)).copied().unwrap_or(top_y);

                // We must check up to the highest neighbor to catch the "air sources"
                let max_check_y = top_y
                    .max(north_top)
                    .max(south_top)
                    .max(west_top)
                    .max(east_top);

                for y in (bottom_y..=max_check_y).rev() {
                    let pos = BlockPos(Vector3::new(x, y, z));
                    let light = get_sky_light(cache, pos);

                    // Use continue, or only break if we are safely below all possible side-light
                    if light == 0 {
                        if y <= top_y {
                            break;
                        }
                        continue;
                    }

                    let is_at_surface = y == top_y;
                    let below_neighbor =
                        y < north_top || y < south_top || y < west_top || y < east_top;

                    if (is_at_surface || below_neighbor) && self.visited.insert(pos) {
                        let skip_dir = (y >= top_y).then_some(BlockDirection::Up);

                        self.queue.push_back(PropagationEntry {
                            pos,
                            skip_direction: skip_dir,
                        });
                    }
                }
            }
        }

        //let propagate_start = Instant::now();

        self.propagate(cache);

        //let propagate_elapsed = propagate_start.elapsed();
        //let scan_elapsed = scan_start.elapsed();
        //log::info!("Sky light timing - Scan: {:?}, Propagate: {:?}", scan_elapsed, propagate_elapsed);
    }
}

pub struct LightEngine {
    block_light: BlockLightPropagator,
    sky_light: SkyLightPropagator,
}

impl LightEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_light: BlockLightPropagator::new(),
            sky_light: SkyLightPropagator::new(),
        }
    }

    /// 初始化光照（天空光 + 方块光）。
    ///
    /// GPU 加速（需 `gpu` feature + `light_acceleration = true`）：
    /// 1. 天空光垂直填充 → `sky_light_fill_u8` kernel（256 列并行）
    /// 2. 天空光水平传播 → `sky_light_horizontal_propagate_u8` kernel
    /// 3. GPU 不可用时自动回退 CPU 路径。
    pub fn initialize_light(&mut self, cache: &mut Cache, config: &LightingEngineConfig) {
        if *config != LightingEngineConfig::Default {
            return;
        }

        let should_skip = {
            let center_chunk = cache.get_center_chunk();
            center_chunk.stage >= crate::chunk_system::chunk_state::StagedChunkEnum::Lighting
        };
        if should_skip {
            return;
        }

        // 尝试 GPU 加速天空光填充；若不可用则回退到 CPU 路径。
        #[cfg(feature = "gpu")]
        let gpu_did_sky = crate::gpu::get_light_accel()
            .is_some_and(|mut light_accel| Self::try_gpu_sky_fill(cache, &mut light_accel));
        #[cfg(not(feature = "gpu"))]
        let gpu_did_sky = false;

        if gpu_did_sky {
            // 尝试 GPU 水平传播；不可用时回退 CPU。
            #[cfg(feature = "gpu")]
            let gpu_did_horiz = crate::gpu::get_light_accel().is_some_and(|mut light_accel| {
                Self::try_gpu_sky_horizontal(cache, &mut light_accel)
            });
            #[cfg(not(feature = "gpu"))]
            let gpu_did_horiz = false;

            if !gpu_did_horiz {
                // CPU 回退：源检测 + BFS 水平传播
                Self::sky_horizontal_propagate(self, cache);
            }
        } else {
            // CPU 路径：完整天空光初始化。
            self.sky_light.convert_light(cache);
        }

        self.block_light.propagate_light(cache);

        // GPU 加速方块光传播（实验性）：在 CPU 完成基本传播后尝试 GPU 优化。
        #[cfg(feature = "gpu")]
        if let Some(mut light_accel) = crate::gpu::get_light_accel() {
            Self::try_gpu_block_propagate(cache, &mut light_accel);
        }

        self.block_light.clear();
        self.sky_light.clear();
    }

    /// 在 GPU 天空光填充后执行水平传播（替代 `convert_light` 的第二步）。
    fn sky_horizontal_propagate(&mut self, cache: &mut Cache) {
        let center_x = cache.x + (cache.size / 2);
        let center_z = cache.z + (cache.size / 2);
        let start_x = center_x * 16 - 1;
        let start_z = center_z * 16 - 1;
        let end_x = start_x + 18;
        let end_z = start_z + 18;
        let bottom_y = cache.bottom_y() as i32;

        let mut surface_heights: FastHashMap<(i32, i32), i32> =
            FastHashMap::with_capacity_and_hasher(
                ((end_x - start_x) * (end_z - start_z)) as usize,
                rustc_hash::FxBuildHasher,
            );
        for z in start_z..end_z {
            for x in start_x..end_x {
                let top_y = cache.get_top_y(&HeightMap::WorldSurface, x, z);
                surface_heights.insert((x, z), top_y);
            }
        }

        for z in start_z..end_z {
            for x in start_x..end_x {
                let top_y = surface_heights[&(x, z)];
                let north_top = surface_heights.get(&(x, z - 1)).copied().unwrap_or(top_y);
                let south_top = surface_heights.get(&(x, z + 1)).copied().unwrap_or(top_y);
                let west_top = surface_heights.get(&(x - 1, z)).copied().unwrap_or(top_y);
                let east_top = surface_heights.get(&(x + 1, z)).copied().unwrap_or(top_y);
                let max_check_y = top_y
                    .max(north_top)
                    .max(south_top)
                    .max(west_top)
                    .max(east_top);
                for y in (bottom_y..=max_check_y).rev() {
                    let pos = BlockPos(Vector3::new(x, y, z));
                    let light = get_sky_light(cache, pos);
                    if light == 0 {
                        if y <= top_y {
                            break;
                        }
                        continue;
                    }
                    let is_at_surface = y == top_y;
                    let below_neighbor =
                        y < north_top || y < south_top || y < west_top || y < east_top;
                    if (is_at_surface || below_neighbor) && self.sky_light.visited.insert(pos) {
                        let skip_dir = (y >= top_y).then_some(BlockDirection::Up);
                        self.sky_light.queue.push_back(PropagationEntry {
                            pos,
                            skip_direction: skip_dir,
                        });
                    }
                }
            }
        }

        self.sky_light.propagate(cache);
        self.sky_light.clear();
    }

    /// 尝试使用 GPU 加速器批量填充天空光。
    /// 返回 `true` 表示 GPU 路径成功执行。
    #[cfg(feature = "gpu")]
    fn try_gpu_sky_fill(
        cache: &mut Cache,
        light_accel: &mut crate::light_accel::LightAccelerator,
    ) -> bool {
        let center_idx = (cache.size / 2) * cache.size + (cache.size / 2);
        let center_x = cache.x + (cache.size / 2);
        let center_z = cache.z + (cache.size / 2);
        let start_x = center_x * 16;
        let start_z = center_z * 16;
        let n_cols = 256usize;
        let height = cache.height() as usize;

        let mut hm = vec![0i32; n_cols];
        let mut opacity = vec![0u8; n_cols * height];
        let mut sky_light_out = vec![0u8; n_cols * height];

        for lx in 0..16 {
            for lz in 0..16 {
                let col_idx = (lx * 16 + lz) as usize;
                let world_x = start_x + lx;
                let world_z = start_z + lz;
                let top_y = cache.get_top_y(&HeightMap::WorldSurface, world_x, world_z);
                hm[col_idx] = top_y;

                let bottom_y = cache.bottom_y() as i32;
                for y in bottom_y..(bottom_y + height as i32) {
                    let local_y = (y - bottom_y) as usize;
                    let pos_vec = Vector3::new(world_x, y, world_z);
                    let state = cache.get_block_state(&pos_vec);
                    opacity[col_idx * height + local_y] = state.to_state().opacity;
                }
            }
        }

        light_accel.batch_sky_fill(&hm, &opacity, &mut sky_light_out, n_cols, height);

        let chunk = &mut cache.chunks[center_idx as usize];
        match chunk {
            crate::chunk_system::Chunk::Proto(proto) => {
                for lx in 0..16 {
                    for lz in 0..16 {
                        let col_idx = (lx * 16 + lz) as usize;
                        for ly in 0..height {
                            let light_val = sky_light_out[col_idx * height + ly];
                            if light_val == 0 {
                                continue;
                            }
                            let section_idx = ly >> 4;
                            let local_y = ly & 15;
                            if section_idx < proto.light.sky_light.len() {
                                proto.light.sky_light[section_idx].set(
                                    lx as usize,
                                    local_y,
                                    lz as usize,
                                    light_val,
                                );
                            }
                        }
                    }
                }
                true
            }
            crate::chunk_system::Chunk::Level(level) => {
                let mut light_engine = level
                    .light_engine
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for lx in 0..16 {
                    for lz in 0..16 {
                        let col_idx = (lx * 16 + lz) as usize;
                        for ly in 0..height {
                            let light_val = sky_light_out[col_idx * height + ly];
                            if light_val == 0 {
                                continue;
                            }
                            let section_idx = ly >> 4;
                            let local_y = ly & 15;
                            if section_idx < light_engine.sky_light.len() {
                                light_engine.sky_light[section_idx].set(
                                    lx as usize,
                                    local_y,
                                    lz as usize,
                                    light_val,
                                );
                            }
                        }
                    }
                }
                true
            }
        }
    }

    /// 尝试使用 GPU 加速器执行天空光水平传播 + 向下级联。
    /// 返回 `true` 表示 GPU 路径成功执行。
    #[cfg(feature = "gpu")]
    fn try_gpu_sky_horizontal(
        cache: &mut Cache,
        light_accel: &mut crate::light_accel::LightAccelerator,
    ) -> bool {
        let center_x = cache.x + (cache.size / 2);
        let center_z = cache.z + (cache.size / 2);
        let start_x = center_x * 16 - 1;
        let start_z = center_z * 16 - 1;
        let width = 18usize;
        let depth = 18usize;
        let height = cache.height() as usize;
        let bottom_y = cache.bottom_y() as i32;
        let n_total = width * depth * height;

        let mut sky_light_data = vec![0u8; n_total];
        let mut opacity_data = vec![0u8; n_total];

        // Extract sky light and opacity for 18×18 area (center 16×16 + 1 border)
        for lz in 0..depth {
            for lx in 0..width {
                let world_x = start_x + lx as i32;
                let world_z = start_z + lz as i32;
                let col_idx = lz * width + lx;
                let col_base = col_idx * height;
                for ly in 0..height {
                    let world_y = bottom_y + ly as i32;
                    let idx = col_base + ly;
                    let pos = BlockPos(Vector3::new(world_x, world_y, world_z));
                    sky_light_data[idx] = get_sky_light(cache, pos);
                    let state = cache.get_block_state(&pos.0);
                    opacity_data[idx] = state.to_state().opacity;
                }
            }
        }

        let max_iters = 32;
        light_accel.sky_horizontal_propagate(
            &mut sky_light_data,
            &opacity_data,
            width,
            depth,
            height,
            max_iters,
        );

        // Write results back
        for lz in 0..depth {
            for lx in 0..width {
                let world_x = start_x + lx as i32;
                let world_z = start_z + lz as i32;
                let col_idx = lz * width + lx;
                let col_base = col_idx * height;
                for ly in 0..height {
                    let world_y = bottom_y + ly as i32;
                    let idx = col_base + ly;
                    let new_val = sky_light_data[idx];
                    if new_val > 0 {
                        let pos = BlockPos(Vector3::new(world_x, world_y, world_z));
                        set_sky_light(cache, pos, new_val);
                    }
                }
            }
        }

        true
    }

    pub fn update_block_light(
        &mut self,
        cache: &mut Cache,
        pos: BlockPos,
        old_luminance: u8,
        new_luminance: u8,
    ) {
        // Decrease Logic
        if old_luminance > new_luminance {
            let current_light = get_block_light(cache, pos);
            if current_light > 0 {
                self.block_light
                    .decrease_queue
                    .push_back((pos, current_light));
                set_block_light(cache, pos, 0);
            }
        }

        // Increase Logic
        if new_luminance > 0 {
            set_block_light(cache, pos, new_luminance);
            if self.block_light.visited.insert(pos) {
                self.block_light.queue.push_back(PropagationEntry {
                    pos,
                    skip_direction: None,
                });
            }
        }
    }

    pub fn run_light_updates(&mut self, cache: &mut Cache) {
        if !self.block_light.decrease_queue.is_empty() {
            self.block_light.process_decrease_queue(cache);
        }
        if !self.block_light.queue.is_empty() {
            self.block_light.propagate(cache);
            self.block_light.visited.clear();
        }
        if !self.sky_light.decrease_queue.is_empty() {
            self.sky_light.process_decrease_queue(cache);
        }
        if !self.sky_light.queue.is_empty() {
            self.sky_light.propagate(cache);
            self.sky_light.visited.clear();
        }
    }

    /// GPU 加速方块光传播（实验性）。
    ///
    /// 尝试通过 `batch_block_scan` + `iterative_propagate` 加速光源扫描和 BFS
    /// 传播。适用于整块重新计算场景，GPU 不可用时自动回退。
    #[cfg(feature = "gpu")]
    fn try_gpu_block_propagate(
        cache: &mut Cache,
        light_accel: &mut crate::light_accel::LightAccelerator,
    ) {
        let center_x = cache.x + (cache.size / 2);
        let center_z = cache.z + (cache.size / 2);
        let start_x = center_x * 16 - 1;
        let start_z = center_z * 16 - 1;
        let min_y = cache.bottom_y() as i32;
        let max_y = min_y + cache.height() as i32;
        let height = (max_y - min_y) as usize;
        let width = 18usize;
        let area = width * width * height;

        if area == 0 {
            return;
        }

        // 展平 3D 区域为 1D 数组（GPU kernel 所需格式）
        let mut luminances = vec![0u8; area];
        let mut opacities = vec![0u8; area];
        let mut bl = vec![0u8; area];

        for y_idx in 0..height {
            let y = min_y + y_idx as i32;
            for z_idx in 0..width {
                let z = start_z + z_idx as i32;
                for x_idx in 0..width {
                    let x = start_x + x_idx as i32;
                    let idx = (y_idx * width + z_idx) * width + x_idx;
                    let pos_vec = Vector3::new(x, y, z);
                    let state = cache.get_block_state(&pos_vec);
                    luminances[idx] = state.to_state().luminance;
                    opacities[idx] = state.to_state().opacity;
                }
            }
        }

        // GPU 批量方块光扫描
        let sources = light_accel.batch_block_scan(&luminances, &mut bl, area);

        if sources.is_empty() {
            return; // 无光源，跳过
        }

        // 构建 6-邻接索引数组（-1 表示越界）
        let w = width as i32;
        let h = height as i32;
        let mut neighbors = vec![-1i32; area * 6];
        for y_idx in 0..height {
            let yi = y_idx as i32;
            for z_idx in 0..width {
                let zi = z_idx as i32;
                for x_idx in 0..width {
                    let xi = x_idx as i32;
                    let idx = (y_idx * width + z_idx) * width + x_idx;
                    let base = idx * 6;
                    // +Y (up)
                    neighbors[base] = if yi + 1 < h {
                        ((yi + 1) * w + zi) * w + xi
                    } else {
                        -1
                    };
                    // -Y (down)
                    neighbors[base + 1] = if yi > 0 {
                        ((yi - 1) * w + zi) * w + xi
                    } else {
                        -1
                    };
                    // +Z (south)
                    neighbors[base + 2] = if zi + 1 < w {
                        (yi * w + zi + 1) * w + xi
                    } else {
                        -1
                    };
                    // -Z (north)
                    neighbors[base + 3] = if zi > 0 {
                        (yi * w + zi - 1) * w + xi
                    } else {
                        -1
                    };
                    // +X (east)
                    neighbors[base + 4] = if xi + 1 < w {
                        (yi * w + zi) * w + xi + 1
                    } else {
                        -1
                    };
                    // -X (west)
                    neighbors[base + 5] = if xi > 0 {
                        (yi * w + zi) * w + xi - 1
                    } else {
                        -1
                    };
                }
            }
        }

        // GPU 迭代距离场传播
        let max_iters = height + width; // 保守上限
        light_accel.iterative_propagate(&mut bl, &opacities, &neighbors, area, max_iters);

        // 将 GPU 结果写回缓存
        for y_idx in 0..height {
            let y = min_y + y_idx as i32;
            for z_idx in 0..width {
                let z = start_z + z_idx as i32;
                for x_idx in 0..width {
                    let x = start_x + x_idx as i32;
                    let idx = (y_idx * width + z_idx) * width + x_idx;
                    let pos = BlockPos(Vector3::new(x, y, z));
                    set_block_light(cache, pos, bl[idx]);
                }
            }
        }
    }
}

impl Default for LightEngine {
    fn default() -> Self {
        Self::new()
    }
}
