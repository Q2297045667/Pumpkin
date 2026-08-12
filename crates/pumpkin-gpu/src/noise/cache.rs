//! 噪声采样缓存。
//!
//! 提供 CPU/GPU 之间共享的预计算数据。

use rustc_hash::FxHashMap;

#[cfg(feature = "pumpkin-util")]
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

/// 单个八度的采样器配置（已序列化，可直接上传到 GPU）。
#[derive(Clone, Debug)]
pub struct SerializedOctave {
    pub permutation: [u8; 256],
    pub x_origin: f64,
    pub y_origin: f64,
    pub z_origin: f64,
    pub amplitude: f64,
    pub persistence: f64,
    pub lacunarity: f64,
}

/// 完整的八度噪声配置（多个八度）。
#[derive(Clone, Debug)]
pub struct SerializedOctaveConfig {
    pub octaves: Box<[SerializedOctave]>,
    pub max_value: f64,
}

impl SerializedOctaveConfig {
    /// 从 `OctavePerlinNoiseSampler` 提取并序列化配置。
    #[cfg(feature = "pumpkin-util")]
    #[must_use]
    pub fn from_sampler(sampler: &OctavePerlinNoiseSampler) -> Self {
        let octaves: Box<[SerializedOctave]> = sampler
            .samplers
            .iter()
            .map(|data| SerializedOctave {
                permutation: *data.sampler.permutation(),
                x_origin: data.sampler.x_origin(),
                y_origin: data.sampler.y_origin(),
                z_origin: data.sampler.z_origin(),
                amplitude: data.amplitude(),
                persistence: data.persistence(),
                lacunarity: data.lacunarity(),
            })
            .collect();

        Self {
            octaves,
            max_value: sampler.max_value(),
        }
    }

    #[must_use]
    pub fn num_octaves(&self) -> usize {
        self.octaves.len()
    }

    #[must_use]
    pub fn packed_permutations(&self) -> Vec<u8> {
        let count = self.num_octaves();
        let mut packed = Vec::with_capacity(count * 256);
        for octave in self.octaves.iter() {
            packed.extend_from_slice(&octave.permutation);
        }
        packed
    }

    #[must_use]
    pub fn packed_amplitudes(&self) -> Vec<f64> {
        // GPU kernel 按 CPU 路径的结合顺序计算：(amplitude * sample) * persistence。
        // persistence 通过 packed_persistences 单独上传，不能预乘。
        self.octaves.iter().map(|o| o.amplitude).collect()
    }

    #[must_use]
    pub fn packed_persistences(&self) -> Vec<f64> {
        self.octaves.iter().map(|o| o.persistence).collect()
    }

    #[must_use]
    pub fn packed_lacunarities(&self) -> Vec<f64> {
        self.octaves.iter().map(|o| o.lacunarity).collect()
    }

    #[must_use]
    pub fn packed_origins(&self) -> Vec<f64> {
        let count = self.num_octaves();
        let mut packed = Vec::with_capacity(count * 3);
        for octave in self.octaves.iter() {
            packed.push(octave.x_origin);
            packed.push(octave.y_origin);
            packed.push(octave.z_origin);
        }
        packed
    }
}

/// 噪声采样器缓存。
pub struct NoiseCache {
    configs: parking_lot::Mutex<FxHashMap<u64, SerializedOctaveConfig>>,
}

impl NoiseCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            configs: parking_lot::Mutex::new(FxHashMap::default()),
        }
    }

    /// 获取或创建采样器的序列化配置。
    #[cfg(feature = "pumpkin-util")]
    #[must_use]
    pub fn get_or_insert(
        &self,
        key: u64,
        sampler: &OctavePerlinNoiseSampler,
    ) -> parking_lot::MutexGuard<'_, FxHashMap<u64, SerializedOctaveConfig>> {
        let mut guard = self.configs.lock();
        guard
            .entry(key)
            .or_insert_with(|| SerializedOctaveConfig::from_sampler(sampler));
        guard
    }

    #[must_use]
    pub fn get(&self, key: u64) -> Option<SerializedOctaveConfig> {
        self.configs.lock().get(&key).cloned()
    }

    pub fn clear(&self) {
        self.configs.lock().clear();
    }
}

impl Default for NoiseCache {
    fn default() -> Self {
        Self::new()
    }
}
