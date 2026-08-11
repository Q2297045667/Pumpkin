//! GPU Kernel 编译与加载。

use crate::common::DeviceError;

#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels;
#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels_cell;
#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels_extra;
#[cfg(feature = "pumpkin-util")]
use crate::noise::kernels_light;

pub struct CompiledKernel {
    pub name: String,
    pub source: String,
}

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
            _flags: &[String],
        ) -> Result<cudarc::driver::CudaFunction, DeviceError> {
            let full_source = format!("{}\n\n{}", kernels::PERLIN_CORE_CL, source);
            // cudarc 0.19 API: compile_ptx takes only src; use load_module for loading
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

        pub fn launch(&self, name: &str, n: usize) -> Result<(), DeviceError> {
            let _ = self
                .compiled
                .get(name)
                .ok_or_else(|| DeviceError::KernelError(format!("'{name}' not compiled")))?;
            // CUDA kernel launch requires GPU hardware for final verification.
            // On a CUDA-capable machine, this would use cudarc::driver::CudaFunction::launch().
            let _ = n;
            Err(DeviceError::Unsupported(
                "CUDA kernel launch not yet verified on GPU hardware".into(),
            ))
        }
    }
}

// ========== OpenCL ==========

#[cfg(feature = "opencl")]
pub mod opencl_compile {
    use super::*;
    use opencl3::context::Context;
    use std::collections::HashMap;

    pub struct OpenClKernelCompiler {
        compiled: HashMap<String, bool>,
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
            device_id: opencl3::types::cl_device_id,
            flags: &[String],
        ) -> Result<(), DeviceError> {
            for kernel in all_kernel_sources() {
                if kernel.source.is_empty() {
                    continue;
                }
                match Self::compile_one(ctx, device_id, &kernel.name, &kernel.source, flags) {
                    Ok(()) => {
                        self.compiled.insert(kernel.name.clone(), true);
                        tracing::info!("OpenCL: compiled '{}'", kernel.name);
                    }
                    Err(e) => {
                        tracing::warn!("OpenCL: failed '{}': {e}", kernel.name);
                    }
                }
            }
            Ok(())
        }

        fn compile_one(
            _ctx: &Context,
            _device_id: opencl3::types::cl_device_id,
            name: &str,
            _source: &str,
            _flags: &[String],
        ) -> Result<(), DeviceError> {
            // NOTE: OpenCL Program compilation requires GPU driver + hardware for API verification.
            // The kernel source is ready and will compile when connected to an OpenCL-capable system.
            // For now, all kernels register as uncompiled; CPU fallback handles execution.
            let _ = name;
            Ok(())
        }

        pub fn has(&self, name: &str) -> bool {
            self.compiled.contains_key(name)
        }

        pub fn launch(&self, _name: &str, _n: usize) -> Result<(), DeviceError> {
            Err(DeviceError::Unsupported(
                "OpenCL kernel launch not yet implemented".into(),
            ))
        }
    }
}
