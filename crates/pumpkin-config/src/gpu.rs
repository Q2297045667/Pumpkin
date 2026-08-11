//! GPU 加速配置选项。
//!
//! 仅在编译时启用 `gpu` feature 后才可用。
//! 当 feature 未启用时，此模块不会被编译，配置文件中也不会出现 GPU 相关选项。

use serde::{Deserialize, Serialize};

/// GPU 加速全局配置。
///
/// 控制是否启用 GPU 加速以及各子系统的加速开关。
///
/// # 示例 TOML
///
/// ```toml
/// [gpu]
/// enabled = true
/// noise_acceleration = true
/// light_acceleration = false
/// surface_acceleration = false
/// jit_enabled = true
/// backend = "auto"
/// ```
#[derive(Deserialize, Serialize, Clone, PartialEq, Eq, Debug, Default)]
#[serde(default)]
pub struct GpuConfig {
    /// 是否开启全局 GPU 加速。
    ///
    /// 当为 `false` 时，即使编译了 GPU 模块，也不会使用 GPU 加速。
    /// 默认值：`false`
    pub enabled: bool,

    /// 是否开启噪声算法加速（Perlin 噪声、密度函数计算）。
    ///
    /// 这是计算量最大的部分，推荐优先开启。
    /// 默认值：`false`
    pub noise_acceleration: bool,

    /// 是否开启光照加速（光照传播 BFS）。
    ///
    /// 默认值：`false`
    pub light_acceleration: bool,

    /// 是否开启地表加速（地表规则匹配与替换）。
    ///
    /// 默认值：`false`
    pub surface_acceleration: bool,

    /// 是否启用 JIT 专用内核。
    ///
    /// 开启后所有密度程序走 JIT 编译的专用内核路径，
    /// 支持 `CUDA`（通过 NVRTC）和 `OpenCL`（在线编译）。
    /// 默认值：`false`
    pub jit_enabled: bool,

    /// 是否启用 `SoA` 数据布局优化。
    /// 启用后位置数据以独立 X/Y/Z 数组格式上传，
    /// 改善 GPU 内存合并访问效率。
    /// 默认值：`false`（保持 `AoS` 交错格式以确保最大兼容性）
    pub soa_layout: bool,

    /// 是否启用 Local Memory Tiling 优化。
    /// 对于小网格（点 ≤ 2048），将 `packed_positions` 预加载到 local memory。
    /// 默认值：`false`（需要 GPU 硬件验证收益后开启）
    pub local_mem_tiling: bool,

    /// 后端选择策略。
    ///
    /// - `"auto"`：按 `CUDA` → `OpenCL` 顺序自动探测
    /// - `"cuda"`：强制使用 `CUDA`
    /// - `"opencl"`：强制使用 `OpenCL`
    /// - `"cpu"`：强制使用 CPU 回退
    ///
    /// 默认值：`"auto"`
    pub backend: GpuBackend,

    /// `CUDA` 特定配置。
    /// 仅当 backend 为 `"cuda"` 或 `"auto"`（探测到 `CUDA`）时生效。
    pub cudarc: CudaConfig,

    /// `OpenCL` 特定配置。
    /// 仅当 backend 为 `"opencl"` 或 `"auto"`（探测到 `OpenCL`）时生效。
    pub opencl3: OpenClConfig,

    /// 设备选择策略。
    pub device: GpuDeviceSelection,
}

impl GpuConfig {
    /// 验证配置的有效性。
    pub fn validate(&self) {
        if self.enabled
            && let GpuDeviceSelection::ByName { ref name } = self.device
        {
            assert!(!name.trim().is_empty(), "GPU device name must not be empty");
        }
    }
}

// ============================================================================
// Backend selection
// ============================================================================

/// 后端选择策略。
///
/// 控制初始化时优先使用哪个计算后端。
#[derive(Deserialize, Serialize, Clone, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    /// 自动探测：`CUDA` → `OpenCL` → CPU
    #[default]
    Auto,

    /// 强制使用 `CUDA`
    Cuda,

    /// 强制使用 `OpenCL`
    OpenCl,

    /// 强制使用 CPU 回退
    Cpu,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cuda => write!(f, "cuda"),
            Self::OpenCl => write!(f, "opencl"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

// ============================================================================
// CUDA configuration
// ============================================================================

/// `CUDA` (`NVRTC`) 编译选项配置。
///
/// 控制 PTX 编译时的架构目标和精度/性能权衡。
#[derive(Deserialize, Serialize, Clone, PartialEq, Eq, Debug)]
#[serde(default)]
pub struct CudaConfig {
    /// PTX 编译目标架构。
    ///
    /// - `"auto"`：自动检测当前 GPU 架构
    /// - `"compute_75"`：Turing 架构 (RTX 2060/2070/2080)
    /// - `"compute_86"`：Ampere 架构 (RTX 3060/3070/3080/3090)
    /// - `"compute_89"`：Ada Lovelace 架构 (RTX 4060/4070/4080/4090)
    /// - `"compute_120"`：Blackwell 架构 (RTX 5090)
    ///
    /// 默认值：`"auto"`
    pub compile_ptx: String,

    /// `NVRTC` 编译标志列表。
    ///
    /// 控制浮点精度、性能和优化行为。
    /// 默认值：精度优先的保守设置
    pub flags: Vec<String>,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            compile_ptx: String::from("auto"),
            flags: vec![
                String::from("--fmad=false"),
                String::from("--ftz=false"),
                String::from("--prec-div=true"),
                String::from("--prec-sqrt=true"),
            ],
        }
    }
}

// ============================================================================
// OpenCL configuration
// ============================================================================

/// `OpenCL` 编译选项配置。
///
/// 控制在线编译 Kernel 时的精度和优化行为。
#[derive(Deserialize, Serialize, Clone, PartialEq, Eq, Debug)]
#[serde(default)]
pub struct OpenClConfig {
    /// `OpenCL` 编译标志列表。
    ///
    /// 控制浮点精度和优化。
    /// 默认值：精度优先的保守设置（仅启用正确舍入的除法/平方根）
    pub flags: Vec<String>,
}

impl Default for OpenClConfig {
    fn default() -> Self {
        Self {
            flags: vec![String::from("-cl-fp32-correctly-rounded-divide-sqrt")],
        }
    }
}

// ============================================================================
// Device selection
// ============================================================================

/// 设备选择策略。
///
/// 控制在多 GPU 系统中如何选择计算设备。
#[derive(Deserialize, Serialize, Clone, PartialEq, Eq, Debug, Default)]
#[serde(tag = "strategy", rename_all = "lowercase")]
pub enum GpuDeviceSelection {
    /// 自动选择：优先独立 GPU，回退到集成显卡。
    /// 当 strategy 未指定或为 "auto" 时使用此变体。
    #[serde(rename = "auto")]
    #[default]
    Auto,

    /// 通过索引选择：0 = 第一个 GPU（通常为独立显卡），1 = 第二个 GPU。
    #[serde(rename = "index")]
    ByIndex {
        /// GPU 设备索引。
        #[serde(default = "default_device_index")]
        index: usize,
    },

    /// 通过名称匹配选择：大小写不敏感，匹配适配器名称的任意子串。
    /// 当出现重复内容时使用第一个匹配。
    #[serde(rename = "name")]
    ByName {
        /// GPU 适配器名称子串。
        name: String,
    },

    /// 优先使用集成显卡（适合低功耗笔记本）。
    #[serde(rename = "integrated")]
    Integrated,
}

const fn default_device_index() -> usize {
    0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_disabled() {
        let config = GpuConfig::default();
        assert!(!config.enabled);
        assert!(!config.noise_acceleration);
        assert!(!config.light_acceleration);
        assert!(!config.surface_acceleration);
        assert!(!config.jit_enabled);
        assert!(!config.soa_layout);
        assert!(!config.local_mem_tiling);
        assert_eq!(config.backend, GpuBackend::Auto);
    }

    #[test]
    fn device_selection_default() {
        let selection = GpuDeviceSelection::default();
        assert!(matches!(selection, GpuDeviceSelection::Auto));
    }

    #[test]
    fn cuda_config_default_flags() {
        let config = CudaConfig::default();
        assert_eq!(config.compile_ptx, "auto");
        assert!(config.flags.contains(&String::from("--fmad=false")));
        assert!(config.flags.contains(&String::from("--ftz=false")));
        assert!(config.flags.contains(&String::from("--prec-div=true")));
        assert!(config.flags.contains(&String::from("--prec-sqrt=true")));
    }

    #[test]
    fn opencl_config_default_flags() {
        let config = OpenClConfig::default();
        assert!(
            config
                .flags
                .contains(&String::from("-cl-fp32-correctly-rounded-divide-sqrt"))
        );
    }

    #[test]
    fn roundtrip_toml_default() {
        let config = GpuConfig::default();
        let toml_str = toml::to_string(&config).expect("serialize");
        let parsed: GpuConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.enabled, config.enabled);
        assert_eq!(parsed.backend, config.backend);
    }

    #[test]
    fn roundtrip_toml_full_enabled() {
        let config = GpuConfig {
            enabled: true,
            noise_acceleration: true,
            light_acceleration: true,
            surface_acceleration: false,
            jit_enabled: true,
            soa_layout: true,
            local_mem_tiling: false,
            backend: GpuBackend::Cuda,
            cudarc: CudaConfig {
                compile_ptx: String::from("compute_89"),
                flags: vec![String::from("--fmad=true"), String::from("--restrict")],
            },
            opencl3: OpenClConfig::default(),
            device: GpuDeviceSelection::ByIndex { index: 1 },
        };
        let toml_str = toml::to_string(&config).expect("serialize");
        let parsed: GpuConfig = toml::from_str(&toml_str).expect("deserialize");
        assert!(parsed.enabled);
        assert!(parsed.noise_acceleration);
        assert!(parsed.soa_layout);
        assert!(!parsed.local_mem_tiling);
        assert_eq!(parsed.backend, GpuBackend::Cuda);
        assert_eq!(parsed.cudarc.compile_ptx, "compute_89");
    }

    #[test]
    fn backend_display() {
        assert_eq!(GpuBackend::Auto.to_string(), "auto");
        assert_eq!(GpuBackend::Cuda.to_string(), "cuda");
        assert_eq!(GpuBackend::OpenCl.to_string(), "opencl");
        assert_eq!(GpuBackend::Cpu.to_string(), "cpu");
    }

    #[test]
    fn device_by_name_roundtrip() {
        let config = GpuDeviceSelection::ByName {
            name: String::from("GTX 1060"),
        };
        let toml_str = toml::to_string(&config).expect("serialize");
        let parsed: GpuDeviceSelection = toml::from_str(&toml_str).expect("deserialize");
        match parsed {
            GpuDeviceSelection::ByName { name } => {
                assert_eq!(name.to_lowercase(), "gtx 1060");
            }
            _ => panic!("expected ByName"),
        }
    }

    #[test]
    fn validate_empty_name_panics() {
        let config = GpuConfig {
            enabled: true,
            device: GpuDeviceSelection::ByName {
                name: String::new(),
            },
            ..Default::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            config.validate();
        }));
        assert!(result.is_err(), "validate should panic on empty name");
    }
}
