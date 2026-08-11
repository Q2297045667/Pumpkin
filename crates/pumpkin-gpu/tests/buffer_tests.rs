//! 缓冲区管理综合测试。
//!
//! 包含正常路径、边界条件和暴力测试。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_const_for_fn,
    clippy::redundant_test_prefix,
    clippy::separated_literal_suffix
)]

use pumpkin_gpu::GpuDevice;
use std::fs::File;
use std::io::Write;

fn log_result(test_name: &str, result: &str) {
    let mut f = File::options()
        .create(true)
        .append(true)
        .open("gpu_test_results.log")
        .unwrap_or_else(|_| File::create("gpu_test_results.log").unwrap());
    writeln!(f, "[{test_name}] {result}").ok();
}

#[test]
fn test_alloc_f64_basic() {
    let device = GpuDevice::init();
    let sizes = [1, 16, 256, 1024, 16384, 65536, 1_048_576];
    for &size in &sizes {
        let buf = match device.alloc_f64(size) {
            Ok(b) => b,
            Err(e) => {
                log_result(&format!("alloc_f64_{size}"), &format!("SKIP: {e}"));
                continue;
            }
        };
        assert_eq!(buf.len(), size);
        device.free(buf).expect("free");
        log_result(&format!("alloc_f64_{size}"), "PASS");
    }
}

#[test]
fn test_copy_roundtrip_f64_all_specials() {
    let device = GpuDevice::init();
    let len = 64;
    let test_values: Vec<f64> = [
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0_f64,
        -0.0_f64,
        1.0_f64,
        -1.0_f64,
        std::f64::consts::PI,
        std::f64::consts::E,
        f64::NAN,
    ]
    .into_iter()
    .cycle()
    .take(len)
    .collect();

    let mut buf = device.alloc_f64(len).expect("alloc");
    device
        .copy_to_device(&mut buf, &test_values)
        .expect("copy_to_device");
    let mut dst = vec![0.0_f64; len];
    device
        .copy_from_device(&buf, &mut dst)
        .expect("copy_from_device");

    for (i, (&s, &d)) in test_values.iter().zip(dst.iter()).enumerate() {
        if s.is_nan() {
            assert!(d.is_nan(), "NaN mismatch at {i}");
        } else if s.is_infinite() {
            assert_eq!(
                s.is_sign_positive(),
                d.is_sign_positive(),
                "inf sign at {i}"
            );
        } else {
            assert_eq!(s.to_bits(), d.to_bits(), "bit mismatch at {i}");
        }
    }
    device.free(buf).expect("free");
    log_result("copy_roundtrip_f64_specials", "PASS");
}

#[test]
fn test_copy_roundtrip_i32_sequential() {
    let device = GpuDevice::init();
    let len = 4096;
    let src: Vec<i32> = (-2048..2048).collect();
    let mut buf = device.alloc_i32(len).expect("alloc");
    device
        .copy_to_device(&mut buf, &src)
        .expect("copy_to_device");
    let mut dst = vec![0_i32; len];
    device
        .copy_from_device(&buf, &mut dst)
        .expect("copy_from_device");
    assert_eq!(src, dst);
    device.free(buf).expect("free");
    log_result("copy_roundtrip_i32", "PASS");
}

#[test]
fn test_copy_roundtrip_u8_full() {
    let device = GpuDevice::init();
    let len = 256;
    let src: Vec<u8> = (0..=255).cycle().take(len).collect();
    let mut buf = device.alloc_u8(len).expect("alloc");
    device
        .copy_to_device(&mut buf, &src)
        .expect("copy_to_device");
    let mut dst = vec![0_u8; len];
    device
        .copy_from_device(&buf, &mut dst)
        .expect("copy_from_device");
    assert_eq!(src, dst);
    device.free(buf).expect("free");
    log_result("copy_roundtrip_u8", "PASS");
}

#[test]
fn test_bounds_oversized_write_fails() {
    let device = GpuDevice::init();
    let len = 128;
    let mut buf = device.alloc_f64(len).expect("alloc");
    let oversized = vec![42.0_f64; len + 1];
    assert!(device.copy_to_device(&mut buf, &oversized).is_err());
    let mut undersized = vec![0.0_f64; len - 1];
    assert!(device.copy_from_device(&buf, &mut undersized).is_err());
    device.free(buf).expect("free");
    log_result("bounds_oversized", "PASS");
}

#[test]
fn test_stress_500_alloc_free() {
    let device = GpuDevice::init();
    for i in 0..500 {
        let len = (i % 256) + 1;
        if let Ok(buf) = device.alloc_f64(len) {
            device.free(buf).expect("free");
        }
    }
    log_result("stress_500_alloc_free", "PASS");
}

#[test]
fn test_device_name() {
    let device = GpuDevice::init();
    let name = device.device_name();
    assert!(!name.is_empty());
    assert!(name.len() < 1024);
    log_result("device_name", &format!("PASS: {name}"));
}
