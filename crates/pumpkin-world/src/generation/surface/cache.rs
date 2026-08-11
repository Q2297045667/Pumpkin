//! Surface 阶段噪声缓存。
//!
//! 预计算 256 个列的表面噪声和次级噪声值，
//! 支持 CPU 批量（缓存友好）和 GPU 批量（一次传输）两种模式。

use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

/// 预计算的表面噪声缓存（16×16 列）。
#[derive(Clone)]
pub struct CachedSurfaceNoise {
    /// `surface_noise` 采样结果 (`DoublePerlin`)，索引 = `local_x` * 16 + `local_z`
    pub surface: Box<[f64; 256]>,
    /// `secondary_noise` 采样结果 (`DoublePerlin`)，索引同上
    pub secondary: Box<[f64; 256]>,
}

impl CachedSurfaceNoise {
    /// 创建空缓存（全零）
    #[must_use]
    pub fn new_empty() -> Self {
        Self {
            surface: Box::new([0.0f64; 256]),
            secondary: Box::new([0.0f64; 256]),
        }
    }

    /// CPU 路径：逐列计算并填充缓存
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cpu(
        surface_noise_a: &OctavePerlinNoiseSampler,
        surface_noise_b: &OctavePerlinNoiseSampler,
        surface_amplitude: f64,
        secondary_a: &OctavePerlinNoiseSampler,
        secondary_b: &OctavePerlinNoiseSampler,
        secondary_amplitude: f64,
        start_x: i32,
        start_z: i32,
    ) -> Self {
        let mut cache = Self::new_empty();
        let c = 1.0181268882175227f64;
        for local_x in 0..16 {
            for local_z in 0..16 {
                let x = (start_x + local_x) as f64;
                let z = (start_z + local_z) as f64;
                let idx = (local_x * 16 + local_z) as usize;

                // DoublePerlin: (first(x,0,z) + second(x*c, 0, z*c)) * amplitude
                cache.surface[idx] = (surface_noise_a.sample(x, 0.0, z)
                    + surface_noise_b.sample(x * c, 0.0, z * c))
                    * surface_amplitude;

                cache.secondary[idx] = (secondary_a.sample(x, 0.0, z)
                    + secondary_b.sample(x * c, 0.0, z * c))
                    * secondary_amplitude;
            }
        }
        cache
    }

    /// 获取指定列的表面噪声值
    #[must_use]
    pub fn surface_at(&self, local_x: i32, local_z: i32) -> f64 {
        self.surface[(local_x * 16 + local_z) as usize]
    }

    /// 获取指定列的次级噪声值
    #[must_use]
    pub fn secondary_at(&self, local_x: i32, local_z: i32) -> f64 {
        self.secondary[(local_x * 16 + local_z) as usize]
    }
}

impl Default for CachedSurfaceNoise {
    fn default() -> Self {
        Self::new_empty()
    }
}
