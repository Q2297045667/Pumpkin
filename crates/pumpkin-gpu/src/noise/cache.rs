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

    /// 序列化配置的内容指纹（FNV-1a 64）。
    ///
    /// 覆盖置换表与全部八度参数。JIT kernel 名称以此指纹区分不同采样器——
    /// 两个采样器即使八度数相同，只要内容不同就得到不同的 JIT kernel 名，
    /// 避免常量烘焙的 kernel 被错误复用（见 jit.rs 的历史 bug）。
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for octave in self.octaves.iter() {
            for &b in octave.permutation.iter() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
            for v in [
                octave.x_origin,
                octave.y_origin,
                octave.z_origin,
                octave.amplitude,
                octave.persistence,
                octave.lacunarity,
            ] {
                for b in v.to_bits().to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
        h
    }
}

/// 噪声采样器缓存。
///
/// 缓存键为调用方提供的指针地址，但条目额外记录采样器内容指纹：
/// 采样器被 drop 后其地址可能被新采样器复用，仅靠地址会在复用场景下
/// 返回过期配置（错误数值）。指纹校验确保命中条目与当前采样器内容一致，
/// 不一致时透明替换。
pub struct NoiseCache {
    configs: parking_lot::Mutex<FxHashMap<u64, (u64, SerializedOctaveConfig)>>,
}

/// 计算采样器内容指纹（无分配的 FNV-1a 64 位哈希）。
///
/// 覆盖置换表与全部八度参数；两个内容不同的采样器不会产生相同指纹
/// （碰撞概率约 2^-64）。
#[cfg(feature = "pumpkin-util")]
#[must_use]
pub fn config_fingerprint(sampler: &OctavePerlinNoiseSampler) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for data in sampler.samplers.iter() {
        for &b in data.sampler.permutation() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
        for v in [
            data.sampler.x_origin(),
            data.sampler.y_origin(),
            data.sampler.z_origin(),
            data.amplitude(),
            data.persistence(),
            data.lacunarity(),
        ] {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

impl NoiseCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            configs: parking_lot::Mutex::new(FxHashMap::default()),
        }
    }

    /// 获取或创建采样器的序列化配置（返回克隆）。
    ///
    /// 命中时校验内容指纹，不一致（地址复用导致的过期条目）则重新序列化。
    #[cfg(feature = "pumpkin-util")]
    #[must_use]
    pub fn get_or_insert(
        &self,
        key: u64,
        sampler: &OctavePerlinNoiseSampler,
    ) -> SerializedOctaveConfig {
        let fingerprint = config_fingerprint(sampler);
        let mut guard = self.configs.lock();
        let stale = guard.get(&key).is_none_or(|(fp, _)| *fp != fingerprint);
        if stale {
            guard.insert(
                key,
                (fingerprint, SerializedOctaveConfig::from_sampler(sampler)),
            );
        }
        guard.get(&key).map_or_else(
            || SerializedOctaveConfig::from_sampler(sampler),
            |(_, c)| c.clone(),
        )
    }

    #[must_use]
    pub fn get(&self, key: u64) -> Option<SerializedOctaveConfig> {
        self.configs.lock().get(&key).map(|(_, c)| c.clone())
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

#[cfg(all(test, feature = "pumpkin-util"))]
mod tests {
    use super::*;
    use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

    fn mk_sampler(octaves: &[i32]) -> OctavePerlinNoiseSampler {
        let r = Xoroshiro::from_seed(42);
        let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
        let mut g = RandomGenerator::Xoroshiro(r);
        OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
    }

    #[test]
    fn get_or_insert_is_idempotent() {
        let cache = NoiseCache::new();
        let sampler = mk_sampler(&[0, 1, 2]);

        let first = cache.get_or_insert(7, &sampler);
        // 第二次查询应返回相同的序列化配置（缓存命中，不重建）
        let second = cache.get_or_insert(7, &sampler);
        assert_eq!(first.num_octaves(), second.num_octaves());
        assert_eq!(first.packed_permutations(), second.packed_permutations());
        assert_eq!(first.packed_amplitudes(), second.packed_amplitudes());
        assert_eq!(first.packed_origins(), second.packed_origins());
    }

    #[test]
    fn keys_are_isolated() {
        let cache = NoiseCache::new();
        let a = mk_sampler(&[0]);
        let b = mk_sampler(&[0, 1, 2, 3]);
        assert_eq!(cache.get_or_insert(1, &a).num_octaves(), 1);
        assert_eq!(cache.get_or_insert(2, &b).num_octaves(), 4);
        assert_eq!(cache.get(1).map(|c| c.num_octaves()), Some(1));
        assert_eq!(cache.get(2).map(|c| c.num_octaves()), Some(4));

        cache.clear();
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_none());
    }

    /// 地址复用场景：同一 key 先后指向不同内容的采样器时，
    /// 缓存必须返回与当前采样器匹配的配置（指纹校验）。
    #[test]
    fn stale_entry_replaced_on_address_reuse() {
        let cache = NoiseCache::new();
        // 模拟地址复用：同一 key，采样器内容不同
        let key = 99u64;
        let first_slot = mk_sampler(&[0, 1, 2]);
        let second_slot = mk_sampler(&[4, 5]);
        let first = cache.get_or_insert(key, &first_slot);
        let second = cache.get_or_insert(key, &second_slot);
        assert_eq!(first.num_octaves(), 3);
        assert_eq!(second.num_octaves(), 2);
        assert_ne!(
            first.packed_permutations(),
            second.packed_permutations(),
            "不同采样器的配置必须不同"
        );
        // 再次查询返回第二个采样器的配置
        let third = cache.get(key).expect("cached");
        assert_eq!(third.num_octaves(), 2);
        assert_eq!(third.packed_permutations(), second.packed_permutations());
    }
}
