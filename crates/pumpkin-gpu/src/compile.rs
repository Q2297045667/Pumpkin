//! GPU Kernel 编译与加载。
//!
//! 提供 CUDA (NVRTC) 和 OpenCL 两种后端的 kernel 编译、缓存和启动功能。

#[cfg(any(feature = "cuda", feature = "opencl"))]
use crate::common::DeviceError;

#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels;
#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels_cell;
#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels_extra;
#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels_light;

/// 编译好的 kernel 元数据。
pub struct CompiledKernel {
    pub name: String,
    pub source: String,
}

/// 返回所有已知 kernel 的名称和源码。
#[must_use]
pub fn all_kernel_sources() -> Vec<CompiledKernel> {
    #[cfg(feature = "pumpkin-util")]
    {
        vec![
            CompiledKernel {
                name: "octave_perlin_sample_f64".into(),
                source: kernels::OCTAVE_PERLIN_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "double_perlin_sample_f64".into(),
                source: kernels::DOUBLE_PERLIN_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "shift_a_sample_f64".into(),
                source: kernels::SHIFT_A_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "shift_b_sample_f64".into(),
                source: kernels::SHIFT_B_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "shifted_noise_sample_f64".into(),
                source: kernels::SHIFTED_NOISE_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "interpolated_noise_sample_f64".into(),
                source: kernels::INTERPOLATED_NOISE_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "vein_noise_sample_f64".into(),
                source: kernels::VEIN_NOISE_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "batch_density_sample_f64".into(),
                source: kernels::DENSITY_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "cell_cache_fill_f64".into(),
                source: kernels_cell::CELL_CACHE_FILL_CL.into(),
            },
            CompiledKernel {
                name: "interpolator_fill_f64".into(),
                source: kernels_cell::INTERPOLATOR_FILL_CL.into(),
            },
            CompiledKernel {
                name: "aquifer_batch_f64".into(),
                source: kernels_cell::AQUIFER_BATCH_CL.into(),
            },
            CompiledKernel {
                name: "aquifer_batch_tiled_f64".into(),
                source: kernels_cell::AQUIFER_BATCH_TILED_CL.into(),
            },
            CompiledKernel {
                name: "beardifier_batch_f64".into(),
                source: kernels_cell::BEARDIFIER_BATCH_CL.into(),
            },
            CompiledKernel {
                name: "vein_batch_f64".into(),
                source: kernels_cell::VEIN_BATCH_CL.into(),
            },
            CompiledKernel {
                name: "trilinear_interpolate_f64".into(),
                source: kernels_extra::TRILINEAR_INTERPOLATE_CL.into(),
            },
            CompiledKernel {
                name: "flatcache_precompute_f64".into(),
                source: kernels_extra::FLATCACHE_PRECOMPUTE_CL.into(),
            },
            CompiledKernel {
                name: "sky_light_fill_u8".into(),
                source: kernels_light::SKY_LIGHT_FILL_CL.into(),
            },
            CompiledKernel {
                name: "block_light_scan_u8".into(),
                source: kernels_light::BLOCK_LIGHT_SCAN_CL.into(),
            },
            CompiledKernel {
                name: "light_propagate_u8".into(),
                source: kernels_light::LIGHT_PROPAGATE_CL.into(),
            },
        ]
    }
    #[cfg(not(feature = "pumpkin-util"))]
    {
        vec![]
    }
}

// ============================================================================
// CUDA (NVRTC)
// ============================================================================

#[cfg(feature = "cuda")]
pub mod cuda_compile {
    use super::*;
    use std::collections::HashMap;

    pub struct CudaKernelCompiler {
        pub compiled: HashMap<String, cudarc::driver::CudaFunction>,
    }

    impl CudaKernelCompiler {
        pub fn new() -> Self {
            Self {
                compiled: HashMap::default(),
            }
        }

        pub fn compile_all(
            &mut self,
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            flags: &[String],
        ) -> Result<(), DeviceError> {
            for kernel in all_kernel_sources() {
                if kernel.source.is_empty() {
                    continue;
                }
                match Self::compile_one(ctx, &kernel.name, &kernel.source, flags) {
                    Ok(func) => {
                        self.compiled.insert(kernel.name.clone(), func);
                        tracing::info!("CUDA NVRTC: compiled '{}'", kernel.name);
                    }
                    Err(e) => {
                        tracing::warn!("CUDA NVRTC: failed '{}': {e}", kernel.name);
                        crate::logging::log_fallback(
                            &crate::logging::FallbackReason::KernelCompileFailed(format!(
                                "CUDA NVRTC '{}': {e}",
                                kernel.name
                            )),
                            "cuda_compile::compile_all",
                        );
                    }
                }
            }
            Ok(())
        }

        /// 编译一个 JIT 特化 kernel。
        ///
        /// JIT kernel 源码中不包含 `PERLIN_CORE_CL` 辅助函数，
        /// 此处将其拼接后再通过 NVRTC 编译。
        pub fn compile_jit_kernel(
            &mut self,
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            jit_kernel: &crate::jit::JitSpecializedKernel,
        ) -> Result<(), DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CL, jit_kernel.source);
            let ptx = cudarc::nvrtc::compile_ptx(full_source).map_err(|e| {
                let msg = format!("JIT NVRTC '{}': {e:?}", jit_kernel.name);
                crate::logging::log_fallback(
                    &crate::logging::FallbackReason::KernelCompileFailed(msg.clone()),
                    "cuda_compile::compile_jit_kernel",
                );
                DeviceError::KernelError(msg)
            })?;
            let module = ctx.load_module(ptx).map_err(|e| {
                DeviceError::KernelError(format!("JIT load '{}': {e:?}", jit_kernel.name))
            })?;
            let func = module.load_function(&jit_kernel.name).map_err(|e| {
                DeviceError::KernelError(format!("JIT load_fn '{}': {e:?}", jit_kernel.name))
            })?;
            self.compiled.insert(jit_kernel.name.clone(), func);
            tracing::info!("CUDA JIT: compiled '{}'", jit_kernel.name);
            Ok(())
        }

        fn compile_one(
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            name: &str,
            source: &str,
            _flags: &[String],
        ) -> Result<cudarc::driver::CudaFunction, DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CL, source);
            let ptx = cudarc::nvrtc::compile_ptx(full_source)
                .map_err(|e| DeviceError::KernelError(format!("NVRTC '{name}': {e:?}")))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| DeviceError::KernelError(format!("load '{name}': {e:?}")))?;
            let func = module
                .load_function(name)
                .map_err(|e| DeviceError::KernelError(format!("load_fn '{name}': {e:?}")))?;
            Ok(func)
        }

        pub fn has(&self, name: &str) -> bool {
            self.compiled.contains_key(name)
        }

        /// CUDA kernel 启动（需要 GPU 硬件）。
        ///
        /// 当前在非 CUDA 环境下返回 `Unsupported`。
        /// 在装有 CUDA 驱动的系统上，可通过 `CudaFunction` 调用 NVRTC 编译好的 PTX kernel。
        pub fn launch(&self, name: &str, n: usize) -> Result<(), DeviceError> {
            if !self.compiled.contains_key(name) {
                return Err(DeviceError::KernelError(format!("'{name}' not compiled")));
            }
            let _ = n;
            // CUDA kernel launch 需要 GPU 硬件验证参数类型和内存布局。
            // 连接 NVIDIA GPU 后取消注释以下代码即可启用：
            crate::logging::log_fallback(
                &crate::logging::FallbackReason::UnsupportedOperation(
                    "CUDA kernel launch not yet verified on GPU hardware".into(),
                ),
                "cuda_compile::launch",
            );
            Err(DeviceError::Unsupported(
                "CUDA kernel launch not yet verified on GPU hardware".into(),
            ))
        }
    }
}

// ============================================================================
// OpenCL
// ============================================================================

#[cfg(feature = "opencl")]
pub mod opencl_compile {
    use super::*;
    use opencl3::context::Context;
    use opencl3::kernel::Kernel;
    use opencl3::program::Program;
    use opencl3::types::cl_device_id;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// 已编译的 OpenCL kernel 条目。
    ///
    /// `Program` 通过 `Arc` 共享以保持存活（Kernel 依赖于它）。
    pub struct CompiledEntry {
        pub kernel: Kernel,
        #[allow(dead_code)]
        program: Arc<Program>,
    }

    pub struct OpenClKernelCompiler {
        compiled: HashMap<String, CompiledEntry>,
    }

    impl OpenClKernelCompiler {
        pub fn new() -> Self {
            Self {
                compiled: HashMap::default(),
            }
        }

        pub fn compile_all(
            &mut self,
            ctx: &Context,
            device_id: cl_device_id,
            flags: &[String],
        ) -> Result<(), DeviceError> {
            for kernel in all_kernel_sources() {
                if kernel.source.is_empty() {
                    continue;
                }
                match Self::compile_one(ctx, device_id, &kernel.name, &kernel.source, flags) {
                    Ok(entry) => {
                        self.compiled.insert(kernel.name.clone(), entry);
                        tracing::info!("OpenCL: compiled '{}'", kernel.name);
                    }
                    Err(e) => {
                        tracing::warn!("OpenCL: failed '{}': {e}", kernel.name);
                        crate::logging::log_fallback(
                            &crate::logging::FallbackReason::KernelCompileFailed(format!(
                                "OpenCL '{}': {e}",
                                kernel.name
                            )),
                            "opencl_compile::compile_all",
                        );
                    }
                }
            }
            Ok(())
        }

        /// 编译一个 JIT 特化 kernel。
        ///
        /// JIT kernel 源码中不包含 `PERLIN_CORE_CL` 辅助函数，
        /// 此处将其拼接后再通过 OpenCL 运行时编译。
        pub fn compile_jit_kernel(
            &mut self,
            ctx: &Context,
            device_id: cl_device_id,
            jit_kernel: &crate::jit::JitSpecializedKernel,
        ) -> Result<(), DeviceError> {
            let entry =
                Self::compile_one(ctx, device_id, &jit_kernel.name, &jit_kernel.source, &[])?;
            self.compiled.insert(jit_kernel.name.clone(), entry);
            tracing::info!("OpenCL JIT: compiled '{}'", jit_kernel.name);
            Ok(())
        }

        /// 编译单个 OpenCL kernel。
        ///
        /// 使用 `create_from_source` + `build` + `Kernel::create` 的 API 链。
        fn compile_one(
            ctx: &Context,
            device_id: cl_device_id,
            name: &str,
            source: &str,
            flags: &[String],
        ) -> Result<CompiledEntry, DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CL, source);
            let flag_str = flags.join(" ");

            let mut program = Program::create_from_source(ctx, &full_source).map_err(|e| {
                DeviceError::KernelError(format!("OpenCL create program '{name}': {e}"))
            })?;

            // SAFETY: device_id is valid and program was created from valid source
            program.build(&[device_id], &flag_str).map_err(|e| {
                // 尝试获取构建日志以提供更好的错误信息
                let _log = program.get_build_log(device_id);
                DeviceError::KernelError(format!("OpenCL build '{name}': {e}"))
            })?;

            let kernel = Kernel::create(&program, name).map_err(|e| {
                DeviceError::KernelError(format!("OpenCL create kernel '{name}': {e}"))
            })?;

            Ok(CompiledEntry {
                kernel,
                program: Arc::new(program),
            })
        }

        pub fn has(&self, name: &str) -> bool {
            self.compiled.contains_key(name)
        }

        /// 获取已编译 kernel 的引用。
        pub fn get_kernel(&self, name: &str) -> Option<&Kernel> {
            self.compiled.get(name).map(|e| &e.kernel)
        }
    }
}
