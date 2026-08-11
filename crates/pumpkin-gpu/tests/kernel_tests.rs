//! Kernel 相关测试。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::redundant_test_prefix
)]

use pumpkin_gpu::GpuDevice;

#[test]
fn test_kernel_launcher_exists() {
    let device = GpuDevice::init();
    // All backends should return a launcher
    assert!(device.kernel_launcher().is_some());
}

#[test]
fn test_synchronize_ok() {
    let device = GpuDevice::init();
    let launcher = device.kernel_launcher().expect("should have launcher");
    assert!(launcher.synchronize().is_ok());
}

#[test]
fn test_cpu_backend_no_kernel_launch_path() {
    // CPU 后端不再通过 KernelLaunch 路径处理 kernel。
    // has_kernel 始终返回 false，表示 CPU 路径不使用 kernel launcher。
    let device = GpuDevice::init();
    let launcher = device.kernel_launcher().expect("CPU should have launcher");
    // CPU backend: has_kernel always returns false (kernel launch not used)
    assert!(!launcher.has_kernel("octave_perlin_sample_f64"));
    assert!(!launcher.has_kernel("trilinear_interpolate_f64"));
    assert!(!launcher.has_kernel("light_propagate_u8"));
    assert!(!launcher.has_kernel("nonexistent"));
}
