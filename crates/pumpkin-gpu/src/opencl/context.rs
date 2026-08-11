//! OpenCL 上下文初始化和设备探测。

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;

/// 初始化 OpenCL 并选择最佳可用设备。返回 (context, queue, device, name)。
pub fn init_opencl() -> Result<(Context, CommandQueue, Device, String), String> {
    let device = find_best_device()?;

    let name = device
        .name()
        .unwrap_or_else(|_| String::from("Unknown OpenCL Device"));

    let ctx = Context::from_device(&device).map_err(|e| format!("创建 OpenCL 上下文失败: {e}"))?;

    let queue = CommandQueue::create_default(&ctx, device.id() as u64)
        .map_err(|e| format!("创建 OpenCL 命令队列失败: {e}"))?;

    Ok((ctx, queue, device, name))
}

fn find_best_device() -> Result<Device, String> {
    let platforms =
        opencl3::platform::get_platforms().map_err(|e| format!("获取 OpenCL 平台失败: {e}"))?;

    if platforms.is_empty() {
        return Err("未检测到 OpenCL 平台".into());
    }

    for platform in &platforms {
        let gpu_ids = platform
            .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("获取 GPU 设备失败: {e}"))?;

        for &id in &gpu_ids {
            let device = Device::new(id);
            if let Ok(name) = device.name() {
                tracing::debug!("  OpenCL GPU 设备: {name}");
                return Ok(device);
            }
        }

        let cpu_ids = platform
            .get_devices(opencl3::device::CL_DEVICE_TYPE_CPU)
            .map_err(|e| format!("获取 CPU 设备失败: {e}"))?;

        for &id in &cpu_ids {
            let device = Device::new(id);
            if let Ok(name) = device.name() {
                tracing::debug!("  OpenCL CPU 设备: {name}");
                return Ok(device);
            }
        }
    }

    Err("未找到可用的 OpenCL 设备".into())
}
