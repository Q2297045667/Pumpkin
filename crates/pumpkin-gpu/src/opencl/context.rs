//! OpenCL 上下文初始化和设备探测。
//!
//! 提供驱动可用性预检和最佳设备选择。

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use std::sync::OnceLock;

/// 缓存 OpenCL 驱动可用性探测结果，避免重复尝试加载驱动。
static OPENCL_PROBE_RESULT: OnceLock<bool> = OnceLock::new();

/// 预检 OpenCL 驱动是否可用。
///
/// 通过探测系统上的 `OpenCL.dll`（Windows）或 `libOpenCL.so`（Linux/macOS）
/// 是否存在来判断驱动是否安装。**不会加载任何 DLL**，仅做文件系统级别的检查。
///
/// 结果被缓存：首次调用后，后续调用直接返回缓存结果。
#[must_use]
pub fn is_opencl_available() -> bool {
    *OPENCL_PROBE_RESULT.get_or_init(probe_opencl_driver)
}

/// 执行实际的驱动文件探测。
fn probe_opencl_driver() -> bool {
    #[cfg(target_os = "windows")]
    {
        probe_windows_opencl()
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        probe_unix_opencl()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        // 未知平台：保守地认为 OpenCL 可能可用，让 opencl3 自己去尝试
        tracing::debug!("未知平台，跳过 OpenCL 预检，假设可用");
        true
    }
}

#[cfg(target_os = "windows")]
fn probe_windows_opencl() -> bool {
    // Windows 上 OpenCL.dll 通常位于以下位置：
    // - C:\Windows\System32\OpenCL.dll（64位系统）
    // - C:\Windows\SysWOW64\OpenCL.dll（32位兼容层）
    // - 或由 GPU 驱动（NVIDIA/AMD/Intel）安装在 System32 下
    let candidates = [
        r"C:\Windows\System32\OpenCL.dll",
        r"C:\Windows\SysWOW64\OpenCL.dll",
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            tracing::debug!("检测到 OpenCL 驱动: {path}");
            return true;
        }
    }

    // 也检查 PATH 环境变量中是否有 OpenCL.dll
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(';') {
            let candidate = std::path::Path::new(dir).join("OpenCL.dll");
            if candidate.exists() {
                tracing::debug!("检测到 OpenCL 驱动 (PATH): {}", candidate.display());
                return true;
            }
        }
    }

    // 检查 OPENCL_DYLIB_PATH 环境变量（opencl3 的备选路径机制）
    if let Ok(env_var) = std::env::var("OPENCL_DYLIB_PATH") {
        for lib_path in env_var.split(';') {
            if std::path::Path::new(lib_path).exists() {
                tracing::debug!("检测到 OpenCL 驱动 (OPENCL_DYLIB_PATH): {lib_path}");
                return true;
            }
        }
    }

    tracing::debug!("未检测到 OpenCL 驱动，将跳过 OpenCL 后端");
    false
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn probe_unix_opencl() -> bool {
    // Linux: libOpenCL.so.1 或 libOpenCL.so
    // macOS: /System/Library/Frameworks/OpenCL.framework/OpenCL
    #[cfg(target_os = "macos")]
    {
        let macos_path = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
        if std::path::Path::new(macos_path).exists() {
            tracing::debug!("检测到 OpenCL 框架: {macos_path}");
            return true;
        }
    }

    // 检查 OPENCL_DYLIB_PATH 环境变量
    if let Ok(env_var) = std::env::var("OPENCL_DYLIB_PATH") {
        for lib_path in env_var.split(':') {
            if std::path::Path::new(lib_path).exists() {
                tracing::debug!("检测到 OpenCL 驱动 (OPENCL_DYLIB_PATH): {lib_path}");
                return true;
            }
        }
    }

    let candidates = ["libOpenCL.so.1", "libOpenCL.so"];
    for name in &candidates {
        // 检查 ldconfig 缓存中的路径
        if let Ok(output) = std::process::Command::new("ldconfig").arg("-p").output() {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                if stdout.contains(name) {
                    tracing::debug!("检测到 OpenCL 驱动 (ldconfig): {name}");
                    return true;
                }
            }
        }
        // 检查常见路径
        for prefix in &["/usr/lib", "/usr/lib64", "/usr/local/lib"] {
            let candidate = std::path::Path::new(prefix).join(name);
            if candidate.exists() {
                tracing::debug!("检测到 OpenCL 驱动: {}", candidate.display());
                return true;
            }
        }
    }

    tracing::debug!("未检测到 OpenCL 驱动，将跳过 OpenCL 后端");
    false
}

/// 初始化 OpenCL 并选择最佳可用设备。返回 (context, queues, device, name)。
///
/// `pipeline_queues` 控制创建的 CommandQueue 数量。
/// 设为 1 时与之前行为一致。
/// 设为 N > 1 时创建 N 个队列用于流水线并行。
///
/// # Errors
///
/// 如果没有可用的 OpenCL 平台或设备，返回错误。
pub fn init_opencl(
    device_index: Option<usize>,
    device_name_filter: Option<&str>,
    prefer_integrated: bool,
    pipeline_queues: usize,
) -> Result<(Context, Vec<CommandQueue>, Device, String), String> {
    let device = find_best_device(device_index, device_name_filter, prefer_integrated)?;

    let name = device
        .name()
        .unwrap_or_else(|_| String::from("Unknown OpenCL Device"));

    let ctx = Context::from_device(&device).map_err(|e| format!("创建 OpenCL 上下文失败: {e}"))?;

    let n_queues = pipeline_queues.max(1);
    let mut queues = Vec::with_capacity(n_queues);
    for i in 0..n_queues {
        // `create_default` 的第二个参数是队列属性位掩码（非设备 ID），
        // 传 0 使用默认属性；队列绑定到 `Context::from_device` 的默认设备。
        let q = CommandQueue::create_default(&ctx, 0)
            .map_err(|e| format!("创建 OpenCL 命令队列 [{i}/{n_queues}] 失败: {e}"))?;
        queues.push(q);
    }

    if n_queues > 1 {
        tracing::info!("OpenCL: 已创建 {n_queues} 个命令队列（流水线模式）");
    }

    Ok((ctx, queues, device, name))
}

fn find_best_device(
    device_index: Option<usize>,
    device_name_filter: Option<&str>,
    prefer_integrated: bool,
) -> Result<Device, String> {
    let platforms =
        opencl3::platform::get_platforms().map_err(|e| format!("获取 OpenCL 平台失败: {e}"))?;

    if platforms.is_empty() {
        return Err("未检测到 OpenCL 平台".into());
    }

    // 收集所有设备：(device, name, is_gpu)
    let mut gpu_devices: Vec<(Device, String)> = Vec::new();
    let mut cpu_devices: Vec<(Device, String)> = Vec::new();

    for platform in &platforms {
        if let Ok(gpu_ids) = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
            for &id in &gpu_ids {
                let device = Device::new(id);
                if let Ok(name) = device.name() {
                    // 按名称过滤
                    if let Some(filter) = device_name_filter {
                        if !device_matches_name(&name, filter) {
                            tracing::debug!("  OpenCL GPU 设备 '{name}' 不匹配名称过滤 '{filter}'");
                            continue;
                        }
                    }
                    tracing::debug!("  OpenCL GPU 设备: {name}");
                    gpu_devices.push((device, name));
                }
            }
        }

        if let Ok(cpu_ids) = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_CPU) {
            for &id in &cpu_ids {
                let device = Device::new(id);
                if let Ok(name) = device.name() {
                    if let Some(filter) = device_name_filter {
                        if !device_matches_name(&name, filter) {
                            tracing::debug!("  OpenCL CPU 设备 '{name}' 不匹配名称过滤 '{filter}'");
                            continue;
                        }
                    }
                    tracing::debug!("  OpenCL CPU 设备: {name}");
                    cpu_devices.push((device, name));
                }
            }
        }
    }

    // ByIndex: 在扁平列表中按索引选择
    if let Some(idx) = device_index {
        let all_devices: Vec<(Device, String)> =
            gpu_devices.into_iter().chain(cpu_devices).collect();
        if idx >= all_devices.len() {
            return Err(format!(
                "设备索引 {} 超出范围（共 {} 个设备）",
                idx,
                all_devices.len()
            ));
        }
        #[allow(clippy::unwrap_used)]
        let (device, name) = all_devices.into_iter().nth(idx).unwrap();
        tracing::info!("OpenCL 按索引 {idx} 选择设备: {name}");
        return Ok(device);
    }

    // prefer_integrated: 在 GPU 设备中优先选择集成显卡
    if prefer_integrated && !gpu_devices.is_empty() {
        // 先找集成显卡
        for (device, name) in &gpu_devices {
            if is_integrated_gpu(name) {
                tracing::info!("OpenCL 优先选择集成显卡: {name}");
                return Ok(*device);
            }
        }
        // 未找到集成显卡，回退到第一个 GPU
        tracing::info!("OpenCL 未找到集成显卡，使用第一个 GPU");
        return Ok(gpu_devices[0].0);
    }

    // 默认：返回第一个 GPU，否则第一个 CPU
    if let Some((device, _)) = gpu_devices.into_iter().next() {
        return Ok(device);
    }

    if let Some((device, _)) = cpu_devices.into_iter().next() {
        return Ok(device);
    }

    Err("未找到可用的 OpenCL 设备".into())
}

/// 检查设备名称是否匹配给定的过滤字符串（大小写不敏感子串匹配）。
fn device_matches_name(device_name: &str, filter: &str) -> bool {
    device_name.to_lowercase().contains(&filter.to_lowercase())
}

/// 检查设备名称是否属于集成显卡。
fn is_integrated_gpu(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("intel")
        || lower.contains("uhd")
        || lower.contains("iris")
        || (lower.contains("radeon") && !lower.contains("radeon rx"))
        || lower.contains("hd graphics")
}
