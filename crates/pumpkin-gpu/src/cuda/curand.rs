//! cuRAND 随机数生成（CPU PRNG 回退实现）。
//!
//! cuRAND 的 PRNG 算法与 CPU 的 Xoroshiro128 不同，
//! 会产生不同的随机数序列。仅在非地形生成场景（粒子效果、实体 AI）中可用。
//!
//! 默认禁用，需显式设置配置 `[gpu.cuda]` 中的 `use_curand = true`。
//!
//! # 实现说明
//!
//! 当前使用 **SplitMix64** 确定性 PRNG 作为 CPU 端实现。
//! 该算法通过单个 `u64` 状态生成高质量伪随机数，
//! 与 cuRAND 库 API 无关，不依赖 CUDA 硬件。
//!
//! 序列特点：
//! - 确定性：相同种子 → 相同序列
//! - 均匀分布：f64 输出在 [0, 1) 区间均匀分布
//! - 周期：2^64
//! - 与 Xoroshiro128 序列 **不同**，不可互换使用

use crate::common::DeviceError;

/// cuRAND 生成器（SplitMix64 实现）。
///
/// # 使用限制
///
/// ⚠️ **警告**：生成的随机数序列与 CPU 路径的 Xoroshiro128 不同。
/// 地形生成、方块放置等需要确定性的场景 **必须** 使用 CPU 路径。
///
/// 适用场景：
/// - 粒子效果
/// - 实体 AI 随机行为
/// - 视觉效果
/// - 非确定性事件
pub struct CuRandGenerator {
    state: u64,
}

impl CuRandGenerator {
    /// 初始化 CuRand 生成器。
    ///
    /// # Arguments
    ///
    /// * `seed` — 64 位种子值，相同种子保证相同序列
    ///
    /// ⚠️ **警告**：生成的随机数序列与 CPU 路径不同。
    /// 地形一致性不保证。
    pub fn new(seed: u64) -> Result<Self, DeviceError> {
        tracing::warn!("cuRAND 已启用 — 随机数序列与 CPU 路径不同，地形一致性不保证 (seed={seed})");
        Ok(Self { state: seed })
    }

    /// 生成一批均匀分布的 `f64` 随机数，范围 `[0, 1)`。
    ///
    /// 使用 SplitMix64 算法生成确定性伪随机序列。
    ///
    /// # Arguments
    ///
    /// * `n` — 要生成的随机数数量
    /// * `output` — 输出缓冲区，长度必须 ≥ `n`
    ///
    /// # Panics
    ///
    /// 如果 `output.len() < n`。
    pub fn generate_uniform_f64(
        &mut self,
        n: usize,
        output: &mut [f64],
    ) -> Result<(), DeviceError> {
        assert!(
            output.len() >= n,
            "output buffer too small: {} < {n}",
            output.len()
        );

        for item in output.iter_mut().take(n) {
            *item = splitmix64_next_f64(&mut self.state);
        }
        Ok(())
    }

    /// 生成一批 `u64` 随机整数。
    ///
    /// 使用 SplitMix64 算法生成确定性伪随机序列。
    ///
    /// # Arguments
    ///
    /// * `n` — 要生成的随机数数量
    /// * `output` — 输出缓冲区，长度必须 ≥ `n`
    ///
    /// # Panics
    ///
    /// 如果 `output.len() < n`。
    pub fn generate_uniform_u64(
        &mut self,
        n: usize,
        output: &mut [u64],
    ) -> Result<(), DeviceError> {
        assert!(
            output.len() >= n,
            "output buffer too small: {} < {n}",
            output.len()
        );

        for item in output.iter_mut().take(n) {
            *item = splitmix64_next_u64(&mut self.state);
        }
        Ok(())
    }

    /// 检查生成器是否已初始化。
    #[must_use]
    pub const fn is_available(&self) -> bool {
        true
    }

    /// 返回当前内部状态（用于调试/复现）。
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }
}

// ============================================================================
// SplitMix64 PRNG 实现
// ============================================================================

/// SplitMix64 核心：推进状态并返回下一个 `u64`。
///
/// 算法来源：Guy L. Steele, Doug Lea, Christine H. Flood
/// "Fast Splittable Pseudorandom Number Generators" (2014)
#[inline]
fn splitmix64_next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 使用 SplitMix64 生成 `[0, 1)` 区间的 `f64`。
///
/// 方法：取高 53 位作为尾数，乘以 `2^-53`。
/// 这与 `std` 中 `rand` crate 的 `f64` 生成方式一致。
#[inline]
fn splitmix64_next_f64(state: &mut u64) -> f64 {
    let raw = splitmix64_next_u64(state);
    // 取高 53 位（舍弃低 11 位）
    (raw >> 11) as f64 * 1.110_223_024_625_156_5e-16 // 2^-53
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_produces_same_sequence_f64() {
        let mut g1 = CuRandGenerator::new(42).unwrap();
        let mut g2 = CuRandGenerator::new(42).unwrap();

        let mut out1 = [0.0f64; 100];
        let mut out2 = [0.0f64; 100];

        g1.generate_uniform_f64(100, &mut out1).unwrap();
        g2.generate_uniform_f64(100, &mut out2).unwrap();

        assert_eq!(out1, out2, "same seed should produce identical sequences");
    }

    #[test]
    fn same_seed_produces_same_sequence_u64() {
        let mut g1 = CuRandGenerator::new(12345).unwrap();
        let mut g2 = CuRandGenerator::new(12345).unwrap();

        let mut out1 = vec![0u64; 50];
        let mut out2 = vec![0u64; 50];

        g1.generate_uniform_u64(50, &mut out1).unwrap();
        g2.generate_uniform_u64(50, &mut out2).unwrap();

        assert_eq!(out1, out2, "same seed should produce identical sequences");
    }

    #[test]
    fn different_seeds_produce_different_sequences_f64() {
        let mut g1 = CuRandGenerator::new(42).unwrap();
        let mut g2 = CuRandGenerator::new(99).unwrap();

        let mut out1 = [0.0f64; 100];
        let mut out2 = [0.0f64; 100];

        g1.generate_uniform_f64(100, &mut out1).unwrap();
        g2.generate_uniform_f64(100, &mut out2).unwrap();

        assert_ne!(
            out1, out2,
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn different_seeds_produce_different_sequences_u64() {
        let mut g1 = CuRandGenerator::new(0xDEAD).unwrap();
        let mut g2 = CuRandGenerator::new(0xBEEF).unwrap();

        let mut out1 = vec![0u64; 50];
        let mut out2 = vec![0u64; 50];

        g1.generate_uniform_u64(50, &mut out1).unwrap();
        g2.generate_uniform_u64(50, &mut out2).unwrap();

        assert_ne!(
            out1, out2,
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn uniform_f64_is_within_range() {
        let mut g = CuRandGenerator::new(123).unwrap();
        let mut out = vec![0.0f64; 10_000];

        g.generate_uniform_f64(10_000, &mut out).unwrap();

        for &v in &out {
            assert!(
                (0.0..1.0).contains(&v),
                "f64 uniform value {v} out of [0, 1) range"
            );
        }
    }

    #[test]
    fn uniform_f64_has_reasonable_mean() {
        let mut g = CuRandGenerator::new(7).unwrap();
        let n = 100_000usize;
        let mut out = vec![0.0f64; n];

        g.generate_uniform_f64(n, &mut out).unwrap();

        let mean: f64 = out.iter().sum::<f64>() / n as f64;
        // 对于 [0,1) 均匀分布，期望均值为 0.5
        // 允许 ±0.02 的统计误差
        assert!(
            (mean - 0.5).abs() < 0.02,
            "uniform f64 mean {mean} deviates too far from 0.5 (n={n})"
        );
    }

    #[test]
    fn uniform_f64_has_no_all_zero() {
        let mut g = CuRandGenerator::new(42).unwrap();
        let mut out = [0.0f64; 200];

        g.generate_uniform_f64(200, &mut out).unwrap();

        let zero_count = out.iter().filter(|&&v| v == 0.0).count();
        assert!(
            zero_count < 5,
            "too many zeros ({zero_count}/200) — PRNG may be broken"
        );
    }

    #[test]
    fn state_advances_correctly() {
        let mut g = CuRandGenerator::new(42).unwrap();
        let s0 = g.state();

        let mut out = [0.0f64; 5];
        g.generate_uniform_f64(5, &mut out).unwrap();

        let s1 = g.state();
        assert_ne!(s0, s1, "state should change after generating values");
    }

    #[test]
    fn is_available_returns_true() {
        let g = CuRandGenerator::new(0).unwrap();
        assert!(g.is_available());
    }

    #[test]
    fn deterministic_after_interleaved_calls() {
        // 验证 f64 和 u64 生成不会互相干扰序列
        let mut g1 = CuRandGenerator::new(100).unwrap();
        let mut g2 = CuRandGenerator::new(100).unwrap();

        let mut f64_buf1 = [0.0f64; 4];
        let mut u64_buf1 = [0u64; 4];

        g1.generate_uniform_f64(2, &mut f64_buf1[..2]).unwrap();
        g1.generate_uniform_u64(2, &mut u64_buf1[..2]).unwrap();
        g1.generate_uniform_f64(2, &mut f64_buf1[2..]).unwrap();
        g1.generate_uniform_u64(2, &mut u64_buf1[2..]).unwrap();

        let mut f64_buf2 = [0.0f64; 4];
        let mut u64_buf2 = [0u64; 4];

        g2.generate_uniform_f64(2, &mut f64_buf2[..2]).unwrap();
        g2.generate_uniform_u64(2, &mut u64_buf2[..2]).unwrap();
        g2.generate_uniform_f64(2, &mut f64_buf2[2..]).unwrap();
        g2.generate_uniform_u64(2, &mut u64_buf2[2..]).unwrap();

        assert_eq!(f64_buf1, f64_buf2);
        assert_eq!(u64_buf1, u64_buf2);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn panics_on_output_too_small_f64() {
        let mut g = CuRandGenerator::new(1).unwrap();
        let mut out = [0.0f64; 3];
        // Request 5 but buffer only has 3
        g.generate_uniform_f64(5, &mut out).unwrap();
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn panics_on_output_too_small_u64() {
        let mut g = CuRandGenerator::new(1).unwrap();
        let mut out = [0u64; 2];
        g.generate_uniform_u64(5, &mut out).unwrap();
    }
}
