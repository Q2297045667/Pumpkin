//! GPU Kernel 编译与加载。
//!
//! 提供 CUDA (NVRTC) 和 OpenCL 两种后端的 kernel 编译、缓存和启动功能。

use std::collections::HashMap;
use std::sync::OnceLock;

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

/// 全局 kernel 源码注册表（用于延迟编译）。
static KERNEL_REGISTRY: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

/// 初始化全局 kernel 注册表。在设备初始化时调用一次。
///
/// 幂等：使用 [`OnceLock::get_or_init`]，重复调用不会重复泄漏源码字符串。
pub(crate) fn init_kernel_registry() {
    KERNEL_REGISTRY.get_or_init(|| {
        let mut map = HashMap::new();
        for k in all_kernel_sources() {
            // 将源码泄漏为 'static（编译时嵌入的字符串字面量本身是 'static）
            let source: &'static str = Box::leak(k.source.into_boxed_str());
            let name: &'static str = Box::leak(k.name.into_boxed_str());
            map.insert(name, source);
        }
        #[cfg(feature = "cuda")]
        for k in all_cuda_kernel_sources() {
            let source: &'static str = Box::leak(k.source.into_boxed_str());
            let name: &'static str = Box::leak(k.name.into_boxed_str());
            map.insert(name, source);
        }
        map
    });
}

/// 按名称查找 kernel 源码（用于延迟编译）。
#[must_use]
pub(crate) fn lookup_kernel_source(name: &str) -> Option<&'static str> {
    KERNEL_REGISTRY.get().and_then(|m| m.get(name).copied())
}

/// 编译好的 kernel 元数据。
pub(crate) struct CompiledKernel {
    pub name: String,
    pub source: String,
}

/// 返回所有已知 kernel 的名称和源码。
#[must_use]
pub(crate) fn all_kernel_sources() -> Vec<CompiledKernel> {
    #[cfg(feature = "pumpkin-util")]
    {
        vec![
            CompiledKernel {
                name: "octave_perlin_sample_f64".into(),
                source: kernels::OCTAVE_PERLIN_SAMPLE_CL.into(),
            },
            CompiledKernel {
                name: "octave_perlin_sample_soa_f64".into(),
                source: kernels::OCTAVE_PERLIN_SAMPLE_SOA_CL.into(),
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
            CompiledKernel {
                name: "sky_light_horizontal_propagate_u8".into(),
                source: kernels_light::SKY_LIGHT_HORIZONTAL_CL.into(),
            },
        ]
    }
    #[cfg(not(feature = "pumpkin-util"))]
    {
        vec![]
    }
}

/// 返回所有已知 CUDA kernel 的名称和源码（CUDA C++ 格式）。
#[must_use]
#[cfg(feature = "cuda")]
pub(crate) fn all_cuda_kernel_sources() -> Vec<CompiledKernel> {
    #[cfg(feature = "pumpkin-util")]
    {
        vec![
            CompiledKernel {
                name: "octave_perlin_sample_f64".into(),
                source: kernels::OCTAVE_PERLIN_SAMPLE_CU.into(),
            },
            CompiledKernel {
                name: "octave_perlin_sample_soa_f64".into(),
                source: kernels::OCTAVE_PERLIN_SAMPLE_SOA_CU.into(),
            },
            CompiledKernel {
                name: "double_perlin_sample_f64".into(),
                source: kernels::DOUBLE_PERLIN_SAMPLE_CU.into(),
            },
            CompiledKernel {
                name: "shift_a_sample_f64".into(),
                source: kernels::SHIFT_A_SAMPLE_CU.into(),
            },
            CompiledKernel {
                name: "shift_b_sample_f64".into(),
                source: kernels::SHIFT_B_SAMPLE_CU.into(),
            },
            CompiledKernel {
                name: "cell_cache_fill_f64".into(),
                source: kernels_cell::CELL_CACHE_FILL_CU.into(),
            },
            CompiledKernel {
                name: "interpolator_fill_f64".into(),
                source: kernels_cell::INTERPOLATOR_FILL_CU.into(),
            },
            CompiledKernel {
                name: "aquifer_batch_f64".into(),
                source: kernels_cell::AQUIFER_BATCH_CU.into(),
            },
            CompiledKernel {
                name: "aquifer_batch_tiled_f64".into(),
                source: kernels_cell::AQUIFER_BATCH_TILED_CU.into(),
            },
            CompiledKernel {
                name: "beardifier_batch_f64".into(),
                source: kernels_cell::BEARDIFIER_BATCH_CU.into(),
            },
            CompiledKernel {
                name: "vein_batch_f64".into(),
                source: kernels_cell::VEIN_BATCH_CU.into(),
            },
            CompiledKernel {
                name: "trilinear_interpolate_f64".into(),
                source: kernels_extra::TRILINEAR_INTERPOLATE_CU.into(),
            },
            CompiledKernel {
                name: "flatcache_precompute_f64".into(),
                source: kernels_extra::FLATCACHE_PRECOMPUTE_CU.into(),
            },
            CompiledKernel {
                name: "sky_light_fill_u8".into(),
                source: kernels_light::SKY_LIGHT_FILL_CU.into(),
            },
            CompiledKernel {
                name: "block_light_scan_u8".into(),
                source: kernels_light::BLOCK_LIGHT_SCAN_CU.into(),
            },
            CompiledKernel {
                name: "light_propagate_u8".into(),
                source: kernels_light::LIGHT_PROPAGATE_CU.into(),
            },
            CompiledKernel {
                name: "light_propagate_u8_persistent".into(),
                source: kernels_light::LIGHT_PROPAGATE_PERSISTENT_CU.into(),
            },
            CompiledKernel {
                name: "sky_light_horizontal_propagate_u8".into(),
                source: kernels_light::SKY_LIGHT_HORIZONTAL_CU.into(),
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
        compile_ptx_arch: Option<String>,
    }

    impl CudaKernelCompiler {
        pub fn new(compile_ptx: Option<&str>) -> Self {
            Self {
                compiled: HashMap::default(),
                compile_ptx_arch: compile_ptx.map(String::from),
            }
        }

        /// 构建常规 kernel 的 NVRTC CompileOptions。
        ///
        /// 仅包含架构目标和用户配置标志。
        /// 精度优先：默认禁用 FMA 融合与快速数学，保证与 CPU 路径逐位一致
        /// （用户可通过 flags 覆盖，后传入的选项优先）。
        fn build_compile_opts(&self, flags: &[String]) -> cudarc::nvrtc::CompileOptions {
            let mut opts = cudarc::nvrtc::CompileOptions::default();
            if let Some(ref arch) = self.compile_ptx_arch {
                opts.options.push(format!("--gpu-architecture={arch}"));
            }
            // 精度优先的默认选项
            opts.options.push("--fmad=false".into());
            opts.options.push("--ftz=false".into());
            opts.options.push("--prec-div=true".into());
            opts.options.push("--prec-sqrt=true".into());
            opts.options.push("--restrict".into());
            for flag in flags {
                opts.options.push(flag.clone());
            }
            opts
        }

        /// 构建 JIT 特化 kernel 的 NVRTC CompileOptions。
        ///
        /// JIT kernel 八度数 ≤ 16、循环完全展开、常量全部内联，
        /// 激进优化（`--opt-level=3`）。但保持 `--fmad=false` 等精度选项，
        /// 确保与 CPU 路径逐位一致。
        fn build_jit_compile_opts(&self) -> cudarc::nvrtc::CompileOptions {
            let mut opts = cudarc::nvrtc::CompileOptions::default();
            if let Some(ref arch) = self.compile_ptx_arch {
                opts.options.push(format!("--gpu-architecture={arch}"));
            }
            opts.options.push("--fmad=false".into());
            opts.options.push("--ftz=false".into());
            opts.options.push("--prec-div=true".into());
            opts.options.push("--prec-sqrt=true".into());
            opts.options.push("--opt-level=3".into());
            opts.options.push("--restrict".into());
            opts
        }

        pub fn compile_all(
            &mut self,
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            flags: &[String],
        ) -> Result<(), DeviceError> {
            let opts = self.build_compile_opts(flags);
            for kernel in all_cuda_kernel_sources() {
                if kernel.source.is_empty() {
                    continue;
                }
                match Self::compile_one(ctx, &kernel.name, &kernel.source, &opts) {
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
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CU, jit_kernel.source);
            // JIT 特化 kernel：使用激进优化（FMA + O3），不受配置 `--fmad=false` 约束。
            let opts = self.build_jit_compile_opts();
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(full_source, opts).map_err(|e| {
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
            opts: &cudarc::nvrtc::CompileOptions,
        ) -> Result<cudarc::driver::CudaFunction, DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CU, source);
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(full_source, opts.clone())
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

        /// 获取已编译的 `CudaFunction` 引用。
        #[must_use]
        pub fn get_function(&self, name: &str) -> Option<&cudarc::driver::CudaFunction> {
            self.compiled.get(name)
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
                let log = program.get_build_log(device_id).unwrap_or_default();
                if !log.is_empty() {
                    tracing::warn!(
                        "OpenCL build log for '{name}': {}",
                        &log[..log.len().min(500)]
                    );
                }
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
