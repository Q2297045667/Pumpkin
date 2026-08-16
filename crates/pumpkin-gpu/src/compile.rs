//! GPU Kernel 编译与加载。
//!
//! 提供 CUDA (NVRTC) 和 OpenCL 两种后端的 kernel 编译、缓存和启动功能。

#[cfg(any(feature = "cuda", feature = "opencl"))]
use std::collections::HashMap;
#[cfg(any(feature = "cuda", feature = "opencl"))]
use std::sync::OnceLock;

#[cfg(any(feature = "cuda", feature = "opencl"))]
use crate::common::DeviceError;

#[cfg(all(feature = "pumpkin-util", any(feature = "cuda", feature = "opencl")))]
use crate::noise::kernels;
#[cfg(all(feature = "pumpkin-util", any(feature = "cuda", feature = "opencl")))]
use crate::noise::kernels_cell;
#[cfg(all(feature = "pumpkin-util", any(feature = "cuda", feature = "opencl")))]
use crate::noise::kernels_extra;
#[cfg(all(feature = "pumpkin-util", any(feature = "cuda", feature = "opencl")))]
use crate::noise::kernels_light;

/// 全局 OpenCL kernel 源码注册表（用于延迟编译）。
#[cfg(feature = "opencl")]
static KERNEL_REGISTRY_CL: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
/// 全局 CUDA kernel 源码注册表（用于延迟编译）。
#[cfg(feature = "cuda")]
static KERNEL_REGISTRY_CU: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

/// 初始化全局 kernel 注册表。在设备初始化时调用一次。
///
/// 幂等：使用 [`OnceLock::get_or_init`]，重复调用不会重复泄漏源码字符串。
/// 两套注册表分开维护，避免 OpenCL 延迟编译时误取到 CUDA 源码。
#[cfg(feature = "pumpkin-util")]
pub(crate) fn init_kernel_registry() {
    #[cfg(feature = "opencl")]
    KERNEL_REGISTRY_CL.get_or_init(|| {
        let mut map = HashMap::new();
        for k in all_kernel_sources() {
            // 将源码泄漏为 'static（编译时嵌入的字符串字面量本身是 'static）
            let source: &'static str = Box::leak(k.source.into_boxed_str());
            let name: &'static str = Box::leak(k.name.into_boxed_str());
            map.insert(name, source);
        }
        map
    });
    #[cfg(feature = "cuda")]
    KERNEL_REGISTRY_CU.get_or_init(|| {
        let mut map = HashMap::new();
        for k in all_cuda_kernel_sources() {
            let source: &'static str = Box::leak(k.source.into_boxed_str());
            let name: &'static str = Box::leak(k.name.into_boxed_str());
            map.insert(name, source);
        }
        map
    });
}

/// 按名称查找 OpenCL kernel 源码（用于延迟编译）。
#[cfg(feature = "opencl")]
#[must_use]
pub(crate) fn lookup_opencl_kernel_source(name: &str) -> Option<&'static str> {
    KERNEL_REGISTRY_CL.get().and_then(|m| m.get(name).copied())
}

/// 按名称查找 CUDA kernel 源码（用于延迟编译）。
#[cfg(feature = "cuda")]
#[must_use]
pub(crate) fn lookup_cuda_kernel_source(name: &str) -> Option<&'static str> {
    KERNEL_REGISTRY_CU.get().and_then(|m| m.get(name).copied())
}

/// 编译好的 kernel 元数据。
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub(crate) struct CompiledKernel {
    pub name: String,
    pub source: String,
}

/// 返回所有已知 OpenCL kernel 的名称和源码（OpenCL C 格式）。
#[cfg(feature = "opencl")]
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

    /// 通过原始 NVRTC API 编译 CUDA 源码为 PTX 文本（以 NUL 结尾）。
    ///
    /// cudarc 的 `Ptx` 类型内容为私有（`PtxKind::Image`），外部无法提取
    /// PTX 字节；原始 `create_program` / `compile_program` / `get_ptx` 均为公开，
    /// 供零拷贝所需的 `cuModuleLoadData` 原始模块加载路径使用。
    fn nvrtc_compile_to_ptx(
        source: &str,
        name: &str,
        opts: &cudarc::nvrtc::CompileOptions,
    ) -> Result<Vec<i8>, DeviceError> {
        use cudarc::nvrtc;
        let src_c = std::ffi::CString::new(source)
            .map_err(|e| DeviceError::KernelError(format!("NVRTC src '{name}': {e}")))?;
        let name_c = std::ffi::CString::new(name)
            .map_err(|e| DeviceError::KernelError(format!("NVRTC name '{name}': {e}")))?;
        let prog = nvrtc::result::create_program(&src_c, Some(&name_c))
            .map_err(|e| DeviceError::KernelError(format!("NVRTC create '{name}': {e:?}")))?;
        // compile_program 接受选项切片（每个选项为字符串类类型）
        let opts_list = opts.options.clone();
        // SAFETY: prog 为有效 nvrtcProgram；opts_list 为合法选项字符串。
        if let Err(e) = unsafe { nvrtc::result::compile_program(prog, &opts_list) } {
            // SAFETY: prog 尚未销毁，可取日志。
            let log = unsafe { nvrtc::result::get_program_log(prog) }
                .ok()
                .map(|v| {
                    String::from_utf8_lossy(&v.iter().map(|&c| c as u8).collect::<Vec<u8>>())
                        .to_string()
                })
                .unwrap_or_default();
            // SAFETY: prog 有效。
            unsafe {
                let _ = nvrtc::result::destroy_program(prog);
            }
            return Err(DeviceError::KernelError(format!(
                "NVRTC '{name}': {e:?}; log: {log}"
            )));
        }
        // SAFETY: prog 有效且编译成功。
        let ptx = unsafe { nvrtc::result::get_ptx(prog) }
            .map_err(|e| DeviceError::KernelError(format!("NVRTC get_ptx '{name}': {e:?}")))?;
        // SAFETY: prog 有效。
        unsafe {
            let _ = nvrtc::result::destroy_program(prog);
        }
        Ok(ptx)
    }

    /// 已编译的 CUDA kernel（原始驱动句柄）。
    ///
    /// 零拷贝映射内存需要直接在 kernel 参数中传递 `CUdeviceptr` 值，
    /// 而 cudarc 的 `LaunchArgs.args` / `CudaSlice.cu_device_ptr` 均为私有，
    /// 公共 API 无法实现；因此 CUDA 后端的内存与启动层基于原始驱动 API
    /// （`cuModuleLoadData` / `cuLaunchKernel`）实现。
    pub struct RawCompiledKernel {
        pub function: cudarc::driver::sys::CUfunction,
        /// 模块句柄必须比函数句柄活得久。
        #[allow(dead_code)]
        module: cudarc::driver::sys::CUmodule,
    }

    // SAFETY: CUmodule / CUfunction 为驱动句柄，可跨线程使用（驱动 API 线程安全）。
    unsafe impl Send for RawCompiledKernel {}

    pub struct CudaKernelCompiler {
        pub compiled: HashMap<String, RawCompiledKernel>,
        compile_ptx_arch: Option<String>,
        /// 用户配置的额外 NVRTC 标志（后传入，优先级高于默认精度选项）。
        compile_flags: Vec<String>,
    }

    impl CudaKernelCompiler {
        pub fn new(compile_ptx: Option<&str>, flags: &[String]) -> Self {
            Self {
                compiled: HashMap::default(),
                compile_ptx_arch: compile_ptx.map(String::from),
                compile_flags: flags.to_vec(),
            }
        }

        /// 构建常规 kernel 的 NVRTC CompileOptions。
        ///
        /// 仅包含架构目标和用户配置标志。
        /// 精度优先：默认禁用 FMA 融合与快速数学，保证与 CPU 路径逐位一致
        /// （用户可通过 flags 覆盖，后传入的选项优先）。
        fn build_compile_opts(&self) -> cudarc::nvrtc::CompileOptions {
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
            for flag in &self.compile_flags {
                opts.options.push(flag.clone());
            }
            opts
        }

        /// 构建 JIT 特化 kernel 的 NVRTC CompileOptions。
        ///
        /// JIT kernel 八度数 ≤ 16、循环完全展开、常量全部内联。
        /// 保持 `--fmad=false` 等精度选项，确保与 CPU 路径逐位一致。
        /// NVRTC 设备码始终启用优化，无需（也不支持）`--opt-level` 选项。
        fn build_jit_compile_opts(&self) -> cudarc::nvrtc::CompileOptions {
            let mut opts = cudarc::nvrtc::CompileOptions::default();
            if let Some(ref arch) = self.compile_ptx_arch {
                opts.options.push(format!("--gpu-architecture={arch}"));
            }
            opts.options.push("--fmad=false".into());
            opts.options.push("--ftz=false".into());
            opts.options.push("--prec-div=true".into());
            opts.options.push("--prec-sqrt=true".into());
            opts.options.push("--restrict".into());
            opts
        }

        pub fn compile_all(
            &mut self,
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        ) -> Result<(), DeviceError> {
            let opts = self.build_compile_opts();
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

        /// 延迟编译单个预注册 kernel（与 `compile_all` 使用同一组编译选项）。
        ///
        /// 用于按需加载：初始化时编译失败的 kernel 或运行时注入的新 kernel
        /// 可在此处补编译；失败不阻断（返回 `Err`，由调用方记录）。
        pub fn compile_by_name(
            &mut self,
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            name: &str,
            source: &str,
        ) -> Result<(), DeviceError> {
            if self.compiled.contains_key(name) {
                return Ok(());
            }
            let opts = self.build_compile_opts();
            let func = Self::compile_one(ctx, name, source, &opts)?;
            self.compiled.insert(name.to_string(), func);
            tracing::info!("CUDA NVRTC: lazily compiled '{name}'");
            Ok(())
        }

        /// 编译一个 JIT 特化 kernel。
        ///
        /// JIT kernel 源码中不包含 `PERLIN_CORE_CU` 辅助函数，
        /// 此处将其拼接后再通过 NVRTC 编译。
        /// CUDA 方言源码（`jit_kernel.cuda_source`）与 OpenCL 方言
        /// 语义逐位一致，仅线程索引与函数签名语法不同。
        pub fn compile_jit_kernel(
            &mut self,
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            jit_kernel: &crate::jit::JitSpecializedKernel,
        ) -> Result<(), DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CU, jit_kernel.cuda_source);
            // JIT 特化 kernel：使用激进优化（FMA + O3），不受配置 `--fmad=false` 约束。
            let opts = self.build_jit_compile_opts();
            let ptx =
                nvrtc_compile_to_ptx(&full_source, &jit_kernel.name, &opts).inspect_err(|e| {
                    crate::logging::log_fallback(
                        &crate::logging::FallbackReason::KernelCompileFailed(e.to_string()),
                        "cuda_compile::compile_jit_kernel",
                    );
                })?;
            ctx.bind_to_thread().map_err(|e| {
                DeviceError::KernelError(format!("JIT bind '{}': {e:?}", jit_kernel.name))
            })?;
            // SAFETY: ptx 由 NVRTC 生成且以 NUL 结尾。
            let module = unsafe {
                cudarc::driver::result::module::load_data(ptx.as_ptr().cast::<std::ffi::c_void>())
            }
            .map_err(|e| {
                DeviceError::KernelError(format!("JIT load '{}': {e:?}", jit_kernel.name))
            })?;
            let fname = std::ffi::CString::new(jit_kernel.name.as_str())
                .map_err(|e| DeviceError::KernelError(format!("JIT name: {e}")))?;
            // SAFETY: module 为有效模块句柄；fname 为合法函数名。
            let function = unsafe { cudarc::driver::result::module::get_function(module, fname) }
                .map_err(|e| {
                DeviceError::KernelError(format!("JIT load_fn '{}': {e:?}", jit_kernel.name))
            })?;
            self.compiled.insert(
                jit_kernel.name.clone(),
                RawCompiledKernel { function, module },
            );
            tracing::info!("CUDA JIT: compiled '{}'", jit_kernel.name);
            Ok(())
        }

        fn compile_one(
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            name: &str,
            source: &str,
            opts: &cudarc::nvrtc::CompileOptions,
        ) -> Result<RawCompiledKernel, DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CU, source);
            let ptx = nvrtc_compile_to_ptx(&full_source, name, opts)?;
            // 绑定上下文后通过原始驱动 API 加载模块与函数（公共 API 无法传递
            // 任意设备指针，零拷贝 kernel 参数需要原始 CUfunction 句柄）。
            ctx.bind_to_thread()
                .map_err(|e| DeviceError::KernelError(format!("bind ctx '{name}': {e:?}")))?;
            // SAFETY: ptx 由 NVRTC 生成且以 NUL 结尾；缓冲在加载期间保持有效。
            let module = unsafe {
                cudarc::driver::result::module::load_data(ptx.as_ptr().cast::<std::ffi::c_void>())
            }
            .map_err(|e| DeviceError::KernelError(format!("load '{name}': {e:?}")))?;
            let fname = std::ffi::CString::new(name)
                .map_err(|e| DeviceError::KernelError(format!("name '{name}': {e}")))?;
            // SAFETY: module 为有效模块句柄；fname 为合法函数名。
            let function =
                unsafe { cudarc::driver::result::module::get_function(module, fname) }
                    .map_err(|e| DeviceError::KernelError(format!("load_fn '{name}': {e:?}")))?;
            Ok(RawCompiledKernel { function, module })
        }

        pub fn has(&self, name: &str) -> bool {
            self.compiled.contains_key(name)
        }

        /// 获取已编译 kernel 的原始 `CUfunction` 引用。
        #[must_use]
        pub fn get_function(&self, name: &str) -> Option<&RawCompiledKernel> {
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
        /// 用户配置的 OpenCL 构建标志（同时用于常规与 JIT 编译，保证数值一致）。
        compile_flags: Vec<String>,
    }

    impl OpenClKernelCompiler {
        pub fn new(flags: &[String]) -> Self {
            Self {
                compiled: HashMap::default(),
                compile_flags: flags.to_vec(),
            }
        }

        pub fn compile_all(
            &mut self,
            ctx: &Context,
            device_id: cl_device_id,
        ) -> Result<(), DeviceError> {
            for kernel in all_kernel_sources() {
                if kernel.source.is_empty() {
                    continue;
                }
                match Self::compile_one(
                    ctx,
                    device_id,
                    &kernel.name,
                    &kernel.source,
                    &self.compile_flags,
                ) {
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

        /// 延迟编译单个预注册 kernel（与 `compile_all` 使用同一组构建标志）。
        pub fn compile_by_name(
            &mut self,
            ctx: &Context,
            device_id: cl_device_id,
            name: &str,
            source: &str,
        ) -> Result<(), DeviceError> {
            if self.compiled.contains_key(name) {
                return Ok(());
            }
            let entry = Self::compile_one(ctx, device_id, name, source, &self.compile_flags)?;
            self.compiled.insert(name.to_string(), entry);
            tracing::info!("OpenCL: lazily compiled '{name}'");
            Ok(())
        }

        /// 编译一个 JIT 特化 kernel。
        ///
        /// JIT kernel 源码中不包含 `PERLIN_CORE_CL` 辅助函数，
        /// 此处将其拼接后再通过 OpenCL 运行时编译。
        /// 使用与常规 kernel 相同的精度标志，保证数值一致。
        pub fn compile_jit_kernel(
            &mut self,
            ctx: &Context,
            device_id: cl_device_id,
            jit_kernel: &crate::jit::JitSpecializedKernel,
        ) -> Result<(), DeviceError> {
            let entry = Self::compile_one(
                ctx,
                device_id,
                &jit_kernel.name,
                &jit_kernel.source,
                &self.compile_flags,
            )?;
            self.compiled.insert(jit_kernel.name.clone(), entry);
            tracing::info!("OpenCL JIT: compiled '{}'", jit_kernel.name);
            Ok(())
        }

        /// 编译单个 OpenCL kernel。
        ///
        /// 使用 `create_from_source` + `build` + `Kernel::create` 的 API 链。
        /// 源码头部注入 `#pragma OPENCL FP_CONTRACT OFF`：
        /// NVIDIA 的 OpenCL 编译器对 f64 默认开启 FMA 收缩（且拒绝
        /// `-fmad=false` 等 CUDA 风格标志），只有该标准 pragma 能关闭收缩，
        /// 保证与 CPU / CUDA 路径逐位一致。
        fn compile_one(
            ctx: &Context,
            device_id: cl_device_id,
            name: &str,
            source: &str,
            flags: &[String],
        ) -> Result<CompiledEntry, DeviceError> {
            let full_source = format!(
                "#pragma OPENCL FP_CONTRACT OFF\n{}\n\n{}",
                kernels::PERLIN_CORE_CL,
                source
            );
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

// ============================================================================
// Kernel 清单测试 — 保证 CUDA / OpenCL 功能对齐
// ============================================================================

#[cfg(all(test, feature = "pumpkin-util"))]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// CUDA 独有的 kernel 名单。
    /// OpenCL 列表之外的任何新增 CUDA kernel 都必须在此登记并给出理由，
    /// 否则本测试失败——强制两个后端功能对齐。
    const CUDA_ONLY_KERNELS: &[&str] = &[
        // Cooperative groups（grid-wide 栅栏）— OpenCL 1.2 无法表达。
        "light_propagate_u8_persistent",
    ];

    #[cfg(all(feature = "cuda", feature = "opencl"))]
    #[test]
    fn kernel_names_cuda_opencl_aligned() {
        let cl_kernels = all_kernel_sources();
        let cl: HashSet<&str> = cl_kernels.iter().map(|k| k.name.as_str()).collect();
        let cu_kernels = all_cuda_kernel_sources();
        let cu: HashSet<&str> = cu_kernels.iter().map(|k| k.name.as_str()).collect();

        // CUDA 必须覆盖所有 OpenCL kernel（功能对齐：OpenCL 有的 CUDA 必须有）
        for name in &cl {
            assert!(
                cu.contains(name),
                "CUDA kernel list missing '{name}' (present in OpenCL list)"
            );
        }

        // CUDA 额外 kernel 必须在豁免名单内
        for name in &cu {
            if !cl.contains(name) {
                assert!(
                    CUDA_ONLY_KERNELS.contains(name),
                    "CUDA-only kernel '{name}' not whitelisted — add it to CUDA_ONLY_KERNELS with a reason"
                );
            }
        }

        // 两个列表都不应为空（保证测试确实在比较实际清单）
        assert!(!cl.is_empty(), "OpenCL kernel list is empty");
        assert!(!cu.is_empty(), "CUDA kernel list is empty");
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn kernel_registry_is_idempotent_and_split() {
        init_kernel_registry();
        init_kernel_registry(); // 重复调用必须无副作用

        let cl_kernels = all_kernel_sources();
        let cl_names: HashSet<&str> = cl_kernels.iter().map(|k| k.name.as_str()).collect();
        for name in &cl_names {
            assert!(
                lookup_opencl_kernel_source(name).is_some(),
                "OpenCL registry missing '{name}'"
            );
        }

        #[cfg(feature = "cuda")]
        {
            let cu_kernels = all_cuda_kernel_sources();
            let cu_names: HashSet<&str> = cu_kernels.iter().map(|k| k.name.as_str()).collect();
            for name in &cu_names {
                assert!(
                    lookup_cuda_kernel_source(name).is_some(),
                    "CUDA registry missing '{name}'"
                );
            }
            // CUDA-only kernel 不得出现在 OpenCL 注册表中（防止误取 .cu 源码）
            for name in CUDA_ONLY_KERNELS {
                assert!(lookup_opencl_kernel_source(name).is_none());
            }
        }

        // 未知 kernel 应返回 None
        assert!(lookup_opencl_kernel_source("nonexistent_kernel").is_none());
    }
}
