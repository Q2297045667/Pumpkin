//! # pumpkin-gpu
//!
//! GPU 加速计算模块，为 Pumpkin 世界生成提供可选的 GPU 后端支持。
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(
    clippy::missing_const_for_fn,
    clippy::separated_literal_suffix,
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::double_must_use,
    clippy::missing_fields_in_debug,
    clippy::tuple_array_conversions,
    clippy::explicit_iter_loop,
    clippy::needless_pass_by_ref_mut,
    clippy::wildcard_imports,
    clippy::new_without_default,
    clippy::collapsible_if,
    clippy::ptr_as_ptr,
    clippy::let_and_return,
    clippy::used_underscore_binding,
    clippy::module_name_repetitions,
    clippy::unnecessary_wraps
)]
//! - `common` — 平台无关的共享类型、错误类型、缓冲区抽象和后端枚举
//! - `cuda`   — CUDA 后端 ([`cudarc`])，通过 `features = ["cuda"]` 启用
//! - `opencl` — OpenCL 后端 ([`opencl3`])，通过 `features = ["opencl"]` 启用
//! - `cpu`    — CPU 回退实现，**始终可用**
//!
//! ## 功能标志
//!
//! | 标志 | 说明 |
//! |------|------|
//! | *(无)* | 仅 CPU 回退 |
//! | `cuda` | 启用 CUDA 后端 |
//! | `opencl` | 启用 OpenCL 后端 |
//! | `gpu` | 同时启用 `cuda` + `opencl`（推荐的生产标志） |
//!
//! ## 设备选择策略
//!
//! 初始化时按以下优先级尝试：
//! 1. CUDA（如果编译了且驱动可用）
//! 2. OpenCL（如果编译了且平台可用）
//! 3. CPU 回退（始终可用，作为兜底）
//!
//! ## 使用示例
//!
//! ```ignore
//! use pumpkin_gpu::{GpuDevice, DeviceType};
//!
//! let device = GpuDevice::init();
//! match device.device_type() {
//!     DeviceType::Cuda => println!("使用 CUDA"),
//!     DeviceType::OpenCl => println!("使用 OpenCL"),
//!     DeviceType::Cpu => println!("使用 CPU 回退"),
//! }
//! ```

pub mod common;
pub mod compile;
pub mod cpu;
#[cfg(feature = "pumpkin-util")]
pub mod jit;
pub mod light;
pub mod logging;
#[cfg(feature = "pumpkin-util")]
pub mod noise;

// Re-export backend types for internal use
#[allow(unused_imports)]
pub(crate) use cpu::CpuBackend;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;

use common::{BackendImpl, DeviceError, GpuBuffer, kernel::KernelLauncher};

/// 标识当前激活的计算设备类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// NVIDIA CUDA GPU
    Cuda,
    /// OpenCL 设备（AMD / Intel / NVIDIA）
    OpenCl,
    /// CPU 回退
    Cpu,
}

/// 统一 GPU 设备句柄，对上层隐藏具体后端实现。
///
/// 初始化时自动探测可用的最佳后端，失败则使用 CPU 回退。
pub struct GpuDevice {
    device_type: DeviceType,
    backend: BackendImpl,
}

impl GpuDevice {
    /// 使用默认配置初始化 GPU 设备。
    ///
    /// 按 CUDA → OpenCL → CPU 的顺序尝试，第一个成功即停止。
    /// 所有初始化失败都会通过 [`tracing::warn`] 输出日志。
    #[must_use]
    pub fn init() -> Self {
        let device = Self::init_internal(None, None, None, false, None, None, 4, 1, false);
        device.log_startup();
        device
    }

    /// 输出 GPU 启动状态日志。
    pub fn log_startup(&self) {
        logging::log_gpu_startup(self.device_type, self.device_name());
    }

    /// 使用指定的 GPU 配置初始化设备。
    ///
    /// 需要 `gpu` feature（该 feature 已包含 `dep:pumpkin-config`）。
    #[cfg(feature = "pumpkin-config")]
    #[must_use]
    pub fn from_config(config: &pumpkin_config::gpu::GpuConfig) -> Self {
        if !config.enabled {
            tracing::info!("GPU 加速已在配置中禁用，使用 CPU 回退");
            crate::logging::log_fallback(
                &crate::logging::FallbackReason::ConfigDisabled,
                "GpuDevice::from_config",
            );
            return Self {
                device_type: DeviceType::Cpu,
                backend: BackendImpl::Cpu(crate::cpu::CpuBackend::new()),
            };
        }

        let forced_backend = match config.backend {
            pumpkin_config::gpu::GpuBackend::Cpu => Some(DeviceType::Cpu),
            #[cfg(feature = "cuda")]
            pumpkin_config::gpu::GpuBackend::Cuda => Some(DeviceType::Cuda),
            #[cfg(feature = "opencl")]
            pumpkin_config::gpu::GpuBackend::OpenCl => Some(DeviceType::OpenCl),
            pumpkin_config::gpu::GpuBackend::Auto => None,
        };

        let (device_index, device_name_filter, prefer_integrated) = match &config.device {
            pumpkin_config::gpu::GpuDeviceSelection::ByIndex { index } => {
                (Some(*index), None, false)
            }
            pumpkin_config::gpu::GpuDeviceSelection::ByName { name } => {
                (None, Some(name.as_str()), false)
            }
            pumpkin_config::gpu::GpuDeviceSelection::Integrated => (None, None, true),
            pumpkin_config::gpu::GpuDeviceSelection::Auto => (None, None, false),
        };

        let device = Self::init_internal(
            forced_backend,
            device_index,
            device_name_filter,
            prefer_integrated,
            Some(&config.cudarc.flags),
            Some(&config.opencl3.flags),
            config.cudarc.zero_copy_threshold_kb,
            config.opencl3.pipeline_queues,
            config.cudarc.persistent_kernels,
        );

        // 将配置项注入全局 OnceLock，供各模块读取
        #[cfg(feature = "pumpkin-util")]
        {
            crate::jit::set_jit_max_unroll(config.jit_max_unroll);
            crate::noise::batch_cell::set_aquifer_tile_threshold(
                config.opencl3.local_mem_tile_threshold,
            );
            crate::noise::batch_sampler::set_soa_layout(config.soa_layout);
        };

        device.log_startup();
        device
    }

    /// 内部初始化逻辑。
    #[allow(clippy::too_many_lines)]
    fn init_internal(
        forced_backend: Option<DeviceType>,
        device_index: Option<usize>,
        device_name_filter: Option<&str>,
        prefer_integrated: bool,
        cuda_flags: Option<&[String]>,
        opencl_flags: Option<&[String]>,
        zero_copy_threshold_kb: usize,
        pipeline_queues: usize,
        persistent_kernels: bool,
    ) -> Self {
        // 如果强制指定了后端，仅尝试该后端
        if let Some(forced) = forced_backend {
            match forced {
                #[cfg(feature = "cuda")]
                DeviceType::Cuda => {
                    match crate::cuda::CudaBackend::try_init(
                        device_index,
                        cuda_flags,
                        zero_copy_threshold_kb,
                        persistent_kernels,
                    ) {
                        Ok(backend) => {
                            tracing::info!("GPU 加速已启用: CUDA 后端（强制指定）");
                            return Self {
                                device_type: DeviceType::Cuda,
                                backend: BackendImpl::Cuda(backend),
                            };
                        }
                        Err(e) => {
                            tracing::error!("强制指定的 CUDA 后端不可用 ({e}), 回退到 CPU");
                            crate::logging::log_fallback(
                                &crate::logging::FallbackReason::InitFailed(e.to_string()),
                                "GpuDevice::init_internal::forced_cuda",
                            );
                        }
                    }
                }
                #[cfg(feature = "opencl")]
                DeviceType::OpenCl => {
                    if crate::opencl::is_opencl_available() {
                        match crate::opencl::OpenClBackend::try_init(
                            device_index,
                            device_name_filter,
                            prefer_integrated,
                            opencl_flags,
                            pipeline_queues,
                        ) {
                            Ok(backend) => {
                                tracing::info!("GPU 加速已启用: OpenCL 后端（强制指定）");
                                return Self {
                                    device_type: DeviceType::OpenCl,
                                    backend: BackendImpl::OpenCl(backend),
                                };
                            }
                            Err(e) => {
                                tracing::error!("强制指定的 OpenCL 后端不可用 ({e}), 回退到 CPU");
                                crate::logging::log_fallback(
                                    &crate::logging::FallbackReason::InitFailed(e.to_string()),
                                    "GpuDevice::init_internal::forced_opencl",
                                );
                            }
                        }
                    } else {
                        tracing::error!("强制指定 OpenCL 但驱动未安装，回退到 CPU");
                        crate::logging::log_fallback(
                            &crate::logging::FallbackReason::DriverNotFound,
                            "GpuDevice::init_internal::forced_opencl",
                        );
                    }
                }
                // 如果编译时未包含对应后端，但运行时强制指定了，回退 CPU
                #[cfg(not(feature = "cuda"))]
                DeviceType::Cuda => {
                    tracing::error!("CUDA 未编译，回退到 CPU");
                    crate::logging::log_fallback(
                        &crate::logging::FallbackReason::InitFailed(
                            "CUDA backend not compiled".into(),
                        ),
                        "GpuDevice::init_internal::forced_cuda_not_compiled",
                    );
                }
                #[cfg(not(feature = "opencl"))]
                DeviceType::OpenCl => {
                    tracing::error!("OpenCL 未编译，回退到 CPU");
                    crate::logging::log_fallback(
                        &crate::logging::FallbackReason::InitFailed(
                            "OpenCL backend not compiled".into(),
                        ),
                        "GpuDevice::init_internal::forced_opencl_not_compiled",
                    );
                }
                DeviceType::Cpu => {
                    tracing::info!("GPU 加速已配置为 CPU 模式");
                    return Self {
                        device_type: DeviceType::Cpu,
                        backend: BackendImpl::Cpu(crate::cpu::CpuBackend::new()),
                    };
                }
            }
            tracing::info!("GPU 加速未启用，使用 CPU 回退");
            return Self {
                device_type: DeviceType::Cpu,
                backend: BackendImpl::Cpu(crate::cpu::CpuBackend::new()),
            };
        }

        // Auto 模式：按 CUDA → OpenCL → CPU 探测
        #[cfg(feature = "cuda")]
        {
            // 预检 CUDA 驱动 — 避免在无 GPU 系统上触发 segfault
            if crate::cuda::cuda_driver_available() {
                match crate::cuda::CudaBackend::try_init(
                    device_index,
                    cuda_flags,
                    zero_copy_threshold_kb,
                    persistent_kernels,
                ) {
                    Ok(backend) => {
                        tracing::info!("GPU 加速已启用: CUDA 后端初始化成功");
                        return Self {
                            device_type: DeviceType::Cuda,
                            backend: BackendImpl::Cuda(backend),
                        };
                    }
                    Err(e) => {
                        tracing::warn!("CUDA 后端初始化失败 ({e}), 尝试 OpenCL...");
                        crate::logging::log_fallback(
                            &crate::logging::FallbackReason::InitFailed(e.to_string()),
                            "GpuDevice::init_internal::auto_cuda",
                        );
                    }
                }
            } else {
                tracing::debug!("NVIDIA CUDA 驱动未安装，跳过 CUDA 后端");
            }
        }

        #[cfg(feature = "opencl")]
        {
            // 使用 catch_unwind 保护 — OpenCL DLL 可能损坏导致 segfault
            let opencl_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if !crate::opencl::is_opencl_available() {
                    return Err(DeviceError::InitFailed("OpenCL 驱动未安装".into()));
                }
                crate::opencl::OpenClBackend::try_init(
                    device_index,
                    device_name_filter,
                    prefer_integrated,
                    opencl_flags,
                    pipeline_queues,
                )
            }));
            match opencl_result {
                Ok(Ok(backend)) => {
                    tracing::info!("GPU 加速已启用: OpenCL 后端初始化成功");
                    return Self {
                        device_type: DeviceType::OpenCl,
                        backend: BackendImpl::OpenCl(backend),
                    };
                }
                Ok(Err(e)) => {
                    tracing::warn!("OpenCL 后端初始化失败 ({e}), 回退到 CPU...");
                    crate::logging::log_fallback(
                        &crate::logging::FallbackReason::InitFailed(e.to_string()),
                        "GpuDevice::init_internal::auto_opencl",
                    );
                }
                Err(_) => {
                    tracing::warn!("OpenCL 后端初始化时发生崩溃, 回退到 CPU...");
                    crate::logging::log_fallback(
                        &crate::logging::FallbackReason::DriverNotFound,
                        "GpuDevice::init_internal::auto_opencl_crash",
                    );
                }
            }
        }

        tracing::info!("GPU 加速未启用，使用 CPU 回退");
        crate::logging::log_fallback(
            &crate::logging::FallbackReason::DriverNotFound,
            "GpuDevice::init_internal::auto",
        );
        Self {
            device_type: DeviceType::Cpu,
            backend: BackendImpl::Cpu(crate::cpu::CpuBackend::new()),
        }
    }

    /// 返回当前使用的设备类型。
    #[must_use]
    pub const fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// 返回设备名称（如 "NVIDIA GeForce RTX 4090" 或 "CPU Fallback"）。
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.backend.device_name()
    }

    /// 在设备上分配一段 `f64` 缓冲区。
    ///
    /// # Errors
    /// 当设备内存不足或分配失败时返回 [`DeviceError`]。
    pub fn alloc_f64(&self, len: usize) -> Result<GpuBuffer<f64>, DeviceError> {
        self.backend.alloc_f64(len)
    }

    /// 在设备上分配一段 `i32` 缓冲区。
    ///
    /// # Errors
    /// 当设备内存不足或分配失败时返回 [`DeviceError`]。
    pub fn alloc_i32(&self, len: usize) -> Result<GpuBuffer<i32>, DeviceError> {
        self.backend.alloc_i32(len)
    }

    /// 在设备上分配一段 `u8` 缓冲区。
    ///
    /// # Errors
    /// 当设备内存不足或分配失败时返回 [`DeviceError`]。
    pub fn alloc_u8(&self, len: usize) -> Result<GpuBuffer<u8>, DeviceError> {
        self.backend.alloc_u8(len)
    }

    /// 将数据从主机复制到设备。
    ///
    /// # Errors
    /// 当缓冲区大小不匹配或传输失败时返回错误。
    pub fn copy_to_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        &self,
        buffer: &mut GpuBuffer<T>,
        data: &[T],
    ) -> Result<(), DeviceError> {
        self.backend.copy_to_device(buffer, data)
    }

    /// 将数据从设备复制到主机。
    ///
    /// # Errors
    /// 当传输失败时返回错误。
    pub fn copy_from_device<T: bytemuck::Pod + cudarc::driver::DeviceRepr>(
        &self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), DeviceError> {
        self.backend.copy_from_device(buffer, data)
    }

    /// 释放设备缓冲区。
    pub fn free<T: bytemuck::Pod>(&self, buffer: GpuBuffer<T>) -> Result<(), DeviceError> {
        self.backend.free(buffer)
    }

    /// 获取 Kernel 启动器（仅 GPU 后端返回 Some）。
    #[must_use]
    pub fn kernel_launcher(&self) -> Option<&dyn KernelLauncher> {
        self.backend.kernel_launcher()
    }

    /// 尝试启动 GPU kernel，成功返回 true。
    /// 代理到 `BackendImpl::try_launch_kernel`。
    pub(crate) fn try_launch_kernel(
        &self,
        name: &str,
        n: usize,
        args: Vec<crate::common::kernel::KernelArg<'_>>,
        gpu_buffers: Vec<crate::common::kernel::GpuBufferRef<'_>>,
    ) -> bool {
        self.backend.try_launch_kernel(name, n, args, gpu_buffers)
    }

    /// 编译一个 JIT 特化 kernel。
    ///
    /// 仅在 GPU 后端（CUDA / OpenCL）下有效。
    /// 当 kernel 编译成功时，后续的 `kernel_launcher()` 会识别该 kernel。
    ///
    /// # Errors
    /// 编译失败或当前为 CPU 后端时返回错误。
    #[cfg(feature = "pumpkin-util")]
    pub fn compile_jit_kernel(
        &mut self,
        jit_kernel: &crate::jit::JitSpecializedKernel,
    ) -> Result<(), DeviceError> {
        self.backend.compile_jit_kernel(jit_kernel)
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device_type", &self.device_type)
            .field("device_name", &self.device_name())
            .finish()
    }
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::init()
    }
}

// Safety: GpuDevice 的所有字段均实现了 Send。
unsafe impl Send for GpuDevice {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_initializes() {
        let device = GpuDevice::init();
        let name = device.device_name();
        assert!(!name.is_empty());
    }

    #[test]
    fn alloc_and_free_f64() {
        let device = GpuDevice::init();
        let buf = device.alloc_f64(1024).expect("allocation");
        assert_eq!(buf.len(), 1024);
        device.free(buf).expect("free");
    }

    #[test]
    fn alloc_and_free_i32() {
        let device = GpuDevice::init();
        let buf = device.alloc_i32(256).expect("allocation");
        assert_eq!(buf.len(), 256);
        device.free(buf).expect("free");
    }

    #[test]
    fn alloc_and_free_u8() {
        let device = GpuDevice::init();
        let buf = device.alloc_u8(512).expect("allocation");
        assert_eq!(buf.len(), 512);
        device.free(buf).expect("free");
    }

    #[test]
    fn copy_to_and_from_device_f64() {
        let device = GpuDevice::init();
        let len = 128;
        let mut buf = device.alloc_f64(len).expect("alloc");
        let src: Vec<f64> = (0..len).map(|i| i as f64 * 1.5).collect();
        device.copy_to_device(&mut buf, &src).expect("copy_to");
        let mut dst = vec![0.0_f64; len];
        device.copy_from_device(&buf, &mut dst).expect("copy_from");
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            assert!((s - d).abs() < 1e-12, "mismatch at index {i}");
        }
        device.free(buf).expect("free");
    }

    #[test]
    fn copy_to_and_from_device_u8() {
        let device = GpuDevice::init();
        let len = 256;
        let mut buf = device.alloc_u8(len).expect("alloc");
        let src: Vec<u8> = (0..len).map(|i| (i % 16) as u8).collect();
        device.copy_to_device(&mut buf, &src).expect("copy_to");
        let mut dst = vec![0_u8; len];
        device.copy_from_device(&buf, &mut dst).expect("copy_from");
        assert_eq!(src, dst);
        device.free(buf).expect("free");
    }

    #[test]
    fn zero_length_allocation() {
        let device = GpuDevice::init();
        let buf = device.alloc_f64(0).expect("zero alloc");
        assert_eq!(buf.len(), 0);
        device.free(buf).expect("free");
    }

    #[test]
    fn buffer_bounds_check() {
        let device = GpuDevice::init();
        let len = 128;
        let mut buf = device.alloc_f64(len).expect("alloc");
        let src = vec![42.0_f64; len];
        device.copy_to_device(&mut buf, &src).expect("exact fit");
        let oversized = vec![42.0_f64; len + 1];
        assert!(device.copy_to_device(&mut buf, &oversized).is_err());
        let mut dst = vec![0.0_f64; len + 1];
        assert!(device.copy_from_device(&buf, &mut dst).is_err());
        device.free(buf).expect("free");
    }
}
