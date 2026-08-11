//! GPU Kernel 编译与加载。

use crate::common::DeviceError;
use crate::noise::kernels;
use crate::noise::kernels_extra;

pub struct CompiledKernel {
    pub name: String,
    pub source: String,
}

pub fn all_kernel_sources() -> Vec<CompiledKernel> {
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
            name: "trilinear_interpolate_f64".into(),
            source: kernels_extra::TRILINEAR_INTERPOLATE_CL.into(),
        },
        CompiledKernel {
            name: "flatcache_precompute_f64".into(),
            source: kernels_extra::FLATCACHE_PRECOMPUTE_CL.into(),
        },
        CompiledKernel {
            name: "light_propagate_u8".into(),
            source: String::new(),
        },
    ]
}

// ========== CUDA (NVRTC) ==========

#[cfg(feature = "cuda")]
pub mod cuda_compile {
    use super::*;
    use std::collections::HashMap;

    pub struct CudaKernelCompiler {
        compiled: HashMap<String, cudarc::driver::CudaFunction>,
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
                    }
                }
            }
            Ok(())
        }

        fn compile_one(
            ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            name: &str,
            source: &str,
            flags: &[String],
        ) -> Result<cudarc::driver::CudaFunction, DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CL, source);
            let flag_strs: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();

            let ptx = cudarc::nvrtc::compile_ptx(full_source, name, flag_strs.as_slice())
                .map_err(|e| DeviceError::KernelError(format!("NVRTC '{name}': {e:?}")))?;

            let module = ctx
                .load_module(&ptx)
                .map_err(|e| DeviceError::KernelError(format!("load '{name}': {e:?}")))?;

            let func = module
                .get_function(name)
                .map_err(|e| DeviceError::KernelError(format!("get_fn '{name}': {e:?}")))?;

            Ok(func)
        }

        pub fn has(&self, name: &str) -> bool {
            self.compiled.contains_key(name)
        }

        pub fn launch(&self, name: &str, n: usize) -> Result<(), DeviceError> {
            let func = self
                .compiled
                .get(name)
                .ok_or_else(|| DeviceError::KernelError(format!("'{name}' not compiled")))?;
            let block_size: u32 = 256;
            let grid_size: u32 = ((n as u32) + block_size - 1) / block_size;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { func.launch(cfg, &[]) }
                .map_err(|e| DeviceError::LaunchFailed(format!("CUDA '{name}': {e:?}")))?;
            Ok(())
        }
    }
}

// ========== OpenCL ==========

#[cfg(feature = "opencl")]
pub mod opencl_compile {
    use super::*;
    use opencl3::context::Context;
    use opencl3::kernel::Kernel;
    use opencl3::program::Program;
    use std::collections::HashMap;

    pub struct OpenClKernelCompiler {
        kernels: HashMap<String, (Program, Kernel)>,
    }

    impl OpenClKernelCompiler {
        pub fn new() -> Self {
            Self {
                kernels: HashMap::default(),
            }
        }

        pub fn compile_all(
            &mut self,
            ctx: &Context,
            device_id: opencl3::types::cl_device_id,
            flags: &[String],
        ) -> Result<(), DeviceError> {
            for kernel in all_kernel_sources() {
                if kernel.source.is_empty() {
                    continue;
                }
                match Self::compile_one(ctx, device_id, &kernel.name, &kernel.source, flags) {
                    Ok((prog, k)) => {
                        self.kernels.insert(kernel.name.clone(), (prog, k));
                    }
                    Err(e) => {
                        tracing::warn!("OpenCL: failed '{}': {e}", kernel.name);
                    }
                }
            }
            Ok(())
        }

        fn compile_one(
            ctx: &Context,
            device_id: opencl3::types::cl_device_id,
            name: &str,
            source: &str,
            flags: &[String],
        ) -> Result<(Program, Kernel), DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CL, source);
            let flag_strs: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();

            // opencl3 0.12: create program from source string
            let program = Program::create_from_source(ctx, &full_source)
                .map_err(|e| DeviceError::KernelError(format!("OCL create program: {e}")))?;

            let build_err = program.build(device_id, flag_strs.as_slice());
            if let Err(e) = build_err {
                let log = program.build_log(device_id).unwrap_or_default();
                return Err(DeviceError::KernelError(format!(
                    "OCL build '{name}': {e}. Log: {log}"
                )));
            }
            let program = Program::create_and_build_from_source(
                ctx,
                &full_source,
                &[device_id],
                flag_strs.as_slice(),
            )
            .map_err(|e| DeviceError::KernelError(format!("OpenCL build '{name}': {e}")))?;

            let kernel = Kernel::create(&program, name).map_err(|e| {
                DeviceError::KernelError(format!("OpenCL create kernel '{name}': {e}"))
            })?;

            Ok((program, kernel))
        }

        pub fn has(&self, name: &str) -> bool {
            self.kernels.contains_key(name)
        }

        pub fn launch(&self, _name: &str, _n: usize) -> Result<(), DeviceError> {
            Err(DeviceError::Unsupported(
                "OpenCL kernel launch not yet implemented".into(),
            ))
        }
    }
}
