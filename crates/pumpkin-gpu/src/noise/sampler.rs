//! GPU 噪声采样器。

use crate::GpuDevice;
use crate::common::DeviceError;
use crate::noise::cache::{NoiseCache, SerializedOctaveConfig};

#[cfg(feature = "pumpkin-util")]
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;

/// GPU 加速的噪声采样器。
pub struct GpuNoiseSampler {
    device: GpuDevice,
    cache: NoiseCache,
}

impl GpuNoiseSampler {
    #[must_use]
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            cache: NoiseCache::new(),
        }
    }

    /// 批量采样八度 Perlin 噪声。
    #[cfg(feature = "pumpkin-util")]
    pub fn batch_sample_octave(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        assert_eq!(positions.len(), n * 3, "positions 长度必须为 N*3");
        if n == 0 {
            return Ok(());
        }

        let device_type = self.device.device_type();
        match device_type {
            crate::DeviceType::Cpu => {
                cpu_batch_octave(sampler, positions, results);
                Ok(())
            }
            _ => self.gpu_batch_octave(sampler, positions, results),
        }
    }

    /// GPU 端批量采样。
    #[cfg(feature = "pumpkin-util")]
    fn gpu_batch_octave(
        &mut self,
        sampler: &OctavePerlinNoiseSampler,
        positions: &[f64],
        results: &mut [f64],
    ) -> Result<(), DeviceError> {
        let n = results.len();
        let key = std::ptr::from_ref(sampler) as u64;
        let guard = self.cache.get_or_insert(key, sampler);
        let config = guard
            .get(&key)
            .cloned()
            .unwrap_or_else(|| SerializedOctaveConfig::from_sampler(sampler));
        drop(guard);

        let mut d_positions = self.device.alloc_f64(n * 3)?;
        let d_results = self.device.alloc_f64(n)?;
        let mut d_permutations = self.device.alloc_u8(config.num_octaves() * 256)?;
        let mut d_amplitudes = self.device.alloc_f64(config.num_octaves())?;
        let mut d_lacunarities = self.device.alloc_f64(config.num_octaves())?;
        let mut d_origins = self.device.alloc_f64(config.num_octaves() * 3)?;

        self.device.copy_to_device(&mut d_positions, positions)?;
        self.device
            .copy_to_device(&mut d_permutations, &config.packed_permutations())?;
        self.device
            .copy_to_device(&mut d_amplitudes, &config.packed_amplitudes())?;
        self.device
            .copy_to_device(&mut d_lacunarities, &config.packed_lacunarities())?;
        self.device
            .copy_to_device(&mut d_origins, &config.packed_origins())?;

        let launched = match self.device.kernel_launcher() {
            Some(launcher) if launcher.has_kernel("octave_perlin_sample_f64") => {
                launcher.launch(crate::common::kernel::KernelLaunch {
                    name: "octave_perlin_sample_f64",
                    global_work_size: [n, 1, 1],
                    local_work_size: Some([256, 1, 1]),
                    args: vec![
                        crate::common::kernel::KernelArg::I32(n as i32),
                        crate::common::kernel::KernelArg::I32(config.num_octaves() as i32),
                    ],
                    buffers: vec![],
                })?;
                launcher.synchronize()?;
                true
            }
            _ => false,
        };

        if launched {
            self.device.copy_from_device(&d_results, results)?;
        } else {
            tracing::debug!("GPU kernel 不可用，CPU 回退");
            cpu_batch_octave(sampler, positions, results);
        }

        self.device.free(d_positions)?;
        self.device.free(d_results)?;
        self.device.free(d_permutations)?;
        self.device.free(d_amplitudes)?;
        self.device.free(d_lacunarities)?;
        self.device.free(d_origins)?;
        Ok(())
    }
}

/// CPU 回退：批量采样。
#[cfg(feature = "pumpkin-util")]
pub fn cpu_batch_octave(
    sampler: &OctavePerlinNoiseSampler,
    positions: &[f64],
    results: &mut [f64],
) {
    for i in 0..results.len() {
        results[i] = sampler.sample(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    }
}

#[cfg(all(test, feature = "pumpkin-util"))]
mod tests {
    use super::*;
    use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};

    fn make_sampler() -> OctavePerlinNoiseSampler {
        let rand = Xoroshiro::from_seed(42);
        let (start, amplitudes) = OctavePerlinNoiseSampler::calculate_amplitudes(&[0, 1, 2]);
        let mut rand_gen = RandomGenerator::Xoroshiro(rand);
        OctavePerlinNoiseSampler::new(&mut rand_gen, start, &amplitudes, false)
    }

    #[test]
    fn batch_matches_single() {
        let sampler = make_sampler();
        let device = GpuDevice::init();
        let mut gpu_sampler = GpuNoiseSampler::new(device);
        let n = 512;
        let mut positions = Vec::with_capacity(n * 3);
        for i in 0..n {
            positions.push(i as f64 * 1.5);
            positions.push(i as f64 * 2.3);
            positions.push(i as f64 * 3.1);
        }
        let mut gpu_results = vec![0.0_f64; n];
        let mut cpu_results = vec![0.0_f64; n];
        cpu_batch_octave(&sampler, &positions, &mut cpu_results);
        gpu_sampler
            .batch_sample_octave(&sampler, &positions, &mut gpu_results)
            .unwrap();
        for i in 0..n {
            assert!(
                (cpu_results[i] - gpu_results[i]).abs() < 1e-15,
                "mismatch at {i}: cpu={}, gpu={}",
                cpu_results[i],
                gpu_results[i]
            );
        }
    }

    #[test]
    fn empty_batch() {
        let sampler = make_sampler();
        let device = GpuDevice::init();
        let mut gpu_sampler = GpuNoiseSampler::new(device);
        gpu_sampler
            .batch_sample_octave(&sampler, &[], &mut [])
            .unwrap();
    }
}
