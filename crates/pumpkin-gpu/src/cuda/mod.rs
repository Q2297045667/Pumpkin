//! CUDA 后端（基于 [`cudarc`]）。
//!
//! # 实现状态
//!
//! | 功能 | 状态 | 说明 |
//! |------|------|------|
//! | 驱动初始化 | ✅ | cudarc driver API |
//! | NVRTC kernel 编译 | ✅ | CUDA .cu kernel 源码（原始 cuModuleLoadData） |
//! | GPU 内存分配 (标准) | ✅ | `cuMemAlloc` 原始驱动 API |
//! | GPU 内存分配 (零拷贝) | ✅ | `cuMemHostAlloc(DEVICEMAP)` 映射内存 |
//! | HtoD/DtoH 拷贝 | ✅ | `memcpy_htod` / `memcpy_dtoh` |
//! | Kernel 启动 | ⚠️ | `LaunchArgs` builder 框架就绪 |
//! | 设备选择 ByIndex | ✅ | |
//! | CPU 回退 | ✅ | 内存分配/传输失败自动回退 |

mod context;
pub mod curand;
pub(crate) mod kernel;
mod memory;

pub(crate) use context::cuda_driver_available;

use crate::common::{DeviceError, GpuBuffer, KernelLauncher};
use std::sync::Arc;

/// CUDA 后端实现。
pub struct CudaBackend {
    pub(crate) stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) name: String,
    pub(crate) launcher: kernel::CudaKernelLauncher,
    /// 零拷贝阈值（字节）
    zero_copy_threshold_bytes: usize,
    /// 是否启用 cuRAND（配置 `use_curand = true`）
    pub(crate) use_curand: bool,
}

// SAFETY: CudaBackend's internal state is Send by cudarc specification.
unsafe impl Send for CudaBackend {}

impl CudaBackend {
    pub fn try_init(
        device_index: Option<usize>,
        flags: Option<&[String]>,
        zero_copy_threshold_kb: usize,
        persistent_enabled: bool,
        compile_ptx: Option<&str>,
        use_curand: bool,
    ) -> Result<Self, DeviceError> {
        let idx = device_index.unwrap_or(0);

        // 初始化 CUDA 驱动
        let ctx =
            context::init_cuda(idx).map_err(|e| DeviceError::InitFailed(format!("CUDA: {e}")))?;

        // 获取设备名称和默认流（可能在无 GPU 时失败）
        let name = ctx
            .name()
            .unwrap_or_else(|_| String::from("Unknown CUDA Device"));
        let stream = ctx.default_stream();

        tracing::info!("CUDA 设备: {name}");
        if zero_copy_threshold_kb > 0 {
            tracing::info!(
                "CUDA 零拷贝: 小于 {} KB 的缓冲区使用映射内存（cuMemHostAlloc DEVICEMAP）",
                zero_copy_threshold_kb
            );
        } else {
            tracing::debug!("CUDA 零拷贝已禁用（阈值 = 0）");
        }
        if use_curand {
            tracing::warn!("⚠️ cuRAND 已启用 — 随机数序列与 CPU 不同，地形一致性不保证");
        }

        let mut launcher = kernel::CudaKernelLauncher::new();
        launcher.init(&ctx, stream.clone(), flags, persistent_enabled, compile_ptx);
        Ok(Self {
            stream,
            name,
            launcher,
            zero_copy_threshold_bytes: zero_copy_threshold_kb.saturating_mul(1024),
            use_curand,
        })
    }

    /// 创建 cuRAND 生成器（SplitMix64 确定性实现）。
    ///
    /// ⚠️ 仅在配置 `use_curand = true` 时可用；随机数序列与 CPU 的
    /// Xoroshiro128 不同，**不得用于地形生成**（会破坏一致性）。
    /// 适用场景：粒子效果、实体 AI 等非确定性内容。
    pub fn create_curand(&self, seed: u64) -> Result<curand::CuRandGenerator, DeviceError> {
        if !self.use_curand {
            return Err(DeviceError::Unsupported(
                "cuRAND 未启用（配置 [gpu.cuda] use_curand = true 后可用）".into(),
            ));
        }
        curand::CuRandGenerator::new(seed)
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    pub fn alloc_f64(&self, len: usize) -> Result<GpuBuffer<f64>, DeviceError> {
        memory::CudaMemory::alloc_f64(&self.stream, len, self.zero_copy_threshold_bytes)
    }

    pub fn alloc_i32(&self, len: usize) -> Result<GpuBuffer<i32>, DeviceError> {
        memory::CudaMemory::alloc_i32(&self.stream, len, self.zero_copy_threshold_bytes)
    }

    pub fn alloc_u8(&self, len: usize) -> Result<GpuBuffer<u8>, DeviceError> {
        memory::CudaMemory::alloc_u8(&self.stream, len, self.zero_copy_threshold_bytes)
    }

    pub fn copy_to_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        &self,
        buffer: &mut GpuBuffer<T>,
        data: &[T],
    ) -> Result<(), DeviceError> {
        memory::CudaMemory::copy_to_device::<T>(&self.stream, buffer, data)
    }

    pub fn copy_from_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        &self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), DeviceError> {
        memory::CudaMemory::copy_from_device::<T>(&self.stream, buffer, data)
    }

    pub fn free<T: bytemuck::Pod>(&self, buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        memory::CudaMemory::free(buffer)
    }

    pub fn kernel_launcher(&self) -> Option<&dyn KernelLauncher> {
        Some(&self.launcher)
    }

    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        self.launcher.compile_jit_kernel(jit_kernel)
    }

    /// 按需编译单个预注册 kernel（延迟加载）。
    ///
    /// 从全局 CUDA 源码注册表查找源码并编译；失败仅记录日志，
    /// 上层 `try_launch_kernel` 会回退到 CPU 路径。
    pub fn compile_kernel_by_name(&self, name: &str) {
        self.launcher.compile_kernel_by_name(name);
    }
}
