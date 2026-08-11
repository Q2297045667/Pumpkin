//! 噪声采样 GPU 加速模块 — 完整实现。

pub mod batch_sampler;
pub mod cache;
pub mod kernels;
pub mod kernels_extra;
pub mod kernels_light;
pub mod sampler;

pub use batch_sampler::GpuNoiseSampler;
pub use cache::NoiseCache;
