//! CUDA 上下文初始化和设备探测。

use std::sync::Arc;

/// 初始化 CUDA 驱动并获取指定索引的设备。
///
/// 在调用任何 CUDA API 之前，先检查 NVIDIA 驱动 DLL 是否存在。
/// 如果驱动不存在，立即返回错误而不触发任何 CUDA 调用。
pub fn init_cuda(device_index: usize) -> Result<Arc<cudarc::driver::CudaContext>, String> {
    // 预检：检查 NVIDIA 驱动 DLL 是否存在
    // 避免在无驱动系统上调用 cudarc API 导致 segfault
    if !cuda_driver_available() {
        return Err("NVIDIA CUDA 驱动未安装".into());
    }

    let init_result = cudarc::driver::result::init();
    match init_result {
        Ok(()) => {
            tracing::debug!("CUDA 驱动初始化成功");
        }
        Err(e) => {
            return Err(format!("CUDA 驱动初始化失败: {e:?}"));
        }
    }

    let device_count = cudarc::driver::result::device::get_count()
        .map_err(|e| format!("无法获取 CUDA 设备数量: {e:?}"))? as usize;
    if device_count == 0 {
        return Err("未检测到 CUDA 设备".into());
    }
    if device_index >= device_count {
        return Err(format!(
            "CUDA 设备索引 {device_index} 超出范围 (共 {device_count} 个设备)"
        ));
    }

    let ctx = cudarc::driver::CudaContext::new(device_index)
        .map_err(|e| format!("无法创建 CUDA 上下文 (device {device_index}): {e:?}"))?;

    Ok(ctx)
}

/// 检查 NVIDIA CUDA 驱动是否已安装（文件系统检查，不调用任何 CUDA API）。
pub fn cuda_driver_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        // 检查 nvcuda.dll（NVIDIA 显示驱动自带的 CUDA 驱动）
        if std::path::Path::new(r"C:\Windows\System32\nvcuda.dll").exists() {
            return true;
        }
        // 检查是否有任何 cudart DLL（CUDA Toolkit）
        if let Ok(entries) = std::fs::read_dir(r"C:\Windows\System32") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("cudart") && name_str.ends_with(".dll") {
                    return true;
                }
            }
        }
        false
    }
    #[cfg(not(target_os = "windows"))]
    {
        // Linux: 检查 libcuda.so 或 libcudart.so
        std::path::Path::new("/usr/lib64/libcuda.so").exists()
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists()
            || std::path::Path::new("/usr/local/cuda/lib64/libcudart.so").exists()
    }
}
