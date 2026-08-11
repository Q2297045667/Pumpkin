//! 边界条件测试 — 空输入、单元素、极限值。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::approx_constant,
    clippy::semicolon_outside_block
)]

use pumpkin_gpu::GpuDevice;
use pumpkin_gpu::common::DeviceError;

/// 空输入不应崩溃
#[test]
fn alloc_zero_length() {
    let device = GpuDevice::init();
    for len in [0, 0, 0] {
        let buf = device.alloc_f64(len).expect("zero alloc");
        assert_eq!(buf.len(), 0);
        device.free(buf).expect("free");
    }
}

/// 单元素分配和传输
#[test]
fn single_element_roundtrip() {
    let device = GpuDevice::init();
    let len = 1;
    let mut buf = device.alloc_f64(len).expect("alloc");
    device
        .copy_to_device(&mut buf, &[std::f64::consts::PI])
        .expect("copy_to");
    let mut dst = vec![0.0f64; len];
    device.copy_from_device(&buf, &mut dst).expect("copy_from");
    assert!((dst[0] - std::f64::consts::PI).abs() < 1e-12);
    device.free(buf).expect("free");
}

/// 大缓冲区分配（应在内存允许时成功）
#[test]
fn large_allocation() {
    let device = GpuDevice::init();
    let len = 1_048_576; // 1M f64 = 8 MB
    match device.alloc_f64(len) {
        Ok(buf) => {
            assert_eq!(buf.len(), len);
            device.free(buf).expect("free");
        }
        Err(e) => {
            // 内存不足是预期路径
            assert!(matches!(e, DeviceError::OutOfMemory { .. }));
        }
    }
}

/// 多次分配-释放循环不应泄漏
#[test]
fn stress_alloc_free_1000() {
    let device = GpuDevice::init();
    for i in 0..1000 {
        let len = (i % 1024) + 1;
        if let Ok(buf) = device.alloc_f64(len) {
            assert_eq!(buf.len(), len);
            device.free(buf).expect("free");
        }
    }
}

/// 缓冲区大小不匹配应返回错误
#[test]
fn size_mismatch_various() {
    let device = GpuDevice::init();
    // f64
    {
        let len = 64;
        let mut buf = device.alloc_f64(len).expect("alloc");
        // 写入过多数据
        assert!(
            device
                .copy_to_device(&mut buf, &vec![0.0f64; len + 1])
                .is_err()
        );
        // 读取目标太小
        let mut dst = vec![0.0f64; len - 1];
        assert!(device.copy_from_device(&buf, &mut dst).is_err());
        device.free(buf).expect("free");
    };
    // i32
    {
        let len = 32;
        let mut buf = device.alloc_i32(len).expect("alloc");
        assert!(
            device
                .copy_to_device(&mut buf, &vec![0i32; len + 1])
                .is_err()
        );
        device.free(buf).expect("free");
    };
    // u8
    {
        let len = 128;
        let mut buf = device.alloc_u8(len).expect("alloc");
        assert!(
            device
                .copy_to_device(&mut buf, &vec![0u8; len + 1])
                .is_err()
        );
        device.free(buf).expect("free");
    }
}
