//! cuRAND 随机数生成（⚠️ **STUB** — 未实现，破坏地形一致性）。
//!
//! cuRAND 的 PRNG 算法与 CPU 的 Xoroshiro128 不同，
//! 会产生不同的随机数序列。仅在非地形生成场景（粒子效果、实体 AI）中可用。
//!
//! 默认禁用，需显式设置 `cudarc.use_curand = true`。
//! **当前为存根实现** — 所有方法返回零值。

use crate::common::DeviceError;

/// cuRAND 状态管理器（占位 — 需 GPU 硬件验证）。
pub struct CuRandGenerator {
    _initialized: bool,
}

impl CuRandGenerator {
    /// 初始化 cuRAND 生成器。
    ///
    /// ⚠️ **警告**：生成的随机数序列与 CPU 路径不同。
    pub fn new(_seed: u64) -> Result<Self, DeviceError> {
        tracing::warn!("cuRAND 已启用 — 随机数序列与 CPU 路径不同，地形一致性不保证");
        // cuRAND API 调用：curandCreateGenerator + curandSetPseudoRandomGeneratorSeed
        // 当前使用占位实现，待 GPU 硬件验证
        Ok(Self { _initialized: true })
    }

    /// 生成一批均匀分布的 f64 随机数。
    #[allow(unused_variables)]
    pub fn generate_uniform_f64(
        &mut self,
        n: usize,
        output: &mut [f64],
    ) -> Result<(), DeviceError> {
        // curandGenerateUniformDouble(generator, output, n)
        // 当前占位：填充 0
        output.fill(0.0);
        Ok(())
    }

    /// 检查 cuRAND 是否可用。
    #[must_use]
    pub const fn is_available(&self) -> bool {
        self._initialized
    }
}
