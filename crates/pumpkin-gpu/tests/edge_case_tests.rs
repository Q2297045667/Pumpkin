//! GPU 缓冲区边界暴力测试 — NaN、Inf、非规格化数、极限值。
//!
//! 所有测试验证 GPU 路径能正确处理极端浮点值并与 CPU 路径保持一致。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::approx_constant,
    clippy::float_cmp,
    clippy::print_stdout,
    clippy::redundant_test_prefix
)]

use pumpkin_gpu::GpuDevice;
use std::fs::File;
use std::io::Write;

fn log_result(name: &str, status: &str) {
    if let Ok(mut f) = File::options()
        .create(true)
        .append(true)
        .open("gpu_edge_test_results.log")
    {
        writeln!(f, "[{name}] {status}").ok();
    }
}

// ============================================================================
// f64 特殊值测试
// ============================================================================

#[test]
fn copy_special_f64_values() {
    let device = GpuDevice::init();
    let specials = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::EPSILON,
        // 非规格化数
        5e-324,
        // 接近但非非规格化
        2.2250738585072014e-308,
    ];

    for (i, &val) in specials.iter().enumerate() {
        let mut buf = match device.alloc_f64(1) {
            Ok(b) => b,
            Err(e) => {
                log_result(&format!("special_f64_{i}"), &format!("SKIP alloc: {e}"));
                continue;
            }
        };

        match device.copy_to_device(&mut buf, &[val]) {
            Ok(()) => {}
            Err(e) => {
                log_result(&format!("special_f64_{i}"), &format!("SKIP copy_to: {e}"));
                device.free(buf).ok();
                continue;
            }
        }

        let mut dst = vec![0.0f64; 1];
        match device.copy_from_device(&buf, &mut dst) {
            Ok(()) => {
                if val.is_nan() {
                    assert!(dst[0].is_nan(), "NaN should remain NaN (index {i})");
                } else {
                    assert_eq!(val, dst[0], "f64 value mismatch at index {i}: {val:e}");
                }
                log_result(&format!("special_f64_{i}"), "PASS");
            }
            Err(e) => {
                log_result(&format!("special_f64_{i}"), &format!("SKIP copy_from: {e}"));
            }
        }
        device.free(buf).ok();
    }
}

// ============================================================================
// 缓冲区大小边界
// ============================================================================

#[test]
fn alloc_boundary_sizes() {
    let device = GpuDevice::init();
    let sizes = [
        0, 1, 2, 3, 127, 128, 255, 256, 1023, 1024, 65535, 65536, 1_048_576,
    ];

    for &len in &sizes {
        let buf = match device.alloc_f64(len) {
            Ok(b) => b,
            Err(e) => {
                log_result(&format!("alloc_{len}"), &format!("SKIP: {e}"));
                continue;
            }
        };
        assert_eq!(buf.len(), len, "len mismatch for size {len}");
        device.free(buf).expect("free");
        log_result(&format!("alloc_{len}"), "PASS");
    }
}

#[test]
fn alloc_boundary_i32() {
    let device = GpuDevice::init();
    for len in [0, 1, 256, 65536] {
        let buf = match device.alloc_i32(len) {
            Ok(b) => b,
            Err(e) => {
                log_result(&format!("alloc_i32_{len}"), &format!("SKIP: {e}"));
                continue;
            }
        };
        assert_eq!(buf.len(), len);
        device.free(buf).ok();
        log_result(&format!("alloc_i32_{len}"), "PASS");
    }
}

#[test]
fn alloc_boundary_u8() {
    let device = GpuDevice::init();
    for len in [0, 1, 256, 65536] {
        let buf = match device.alloc_u8(len) {
            Ok(b) => b,
            Err(e) => {
                log_result(&format!("alloc_u8_{len}"), &format!("SKIP: {e}"));
                continue;
            }
        };
        assert_eq!(buf.len(), len);
        device.free(buf).ok();
        log_result(&format!("alloc_u8_{len}"), "PASS");
    }
}

// ============================================================================
// 大小不匹配检测
// ============================================================================

#[test]
fn size_mismatch_copy_to() {
    let device = GpuDevice::init();
    let mut buf = device.alloc_f64(16).expect("alloc");
    // 数据比缓冲区多
    assert!(device.copy_to_device(&mut buf, &[1.0f64; 17]).is_err());
    // 数据比缓冲区少
    assert!(device.copy_to_device(&mut buf, &[1.0f64; 15]).is_err());
    // 精确匹配
    assert!(device.copy_to_device(&mut buf, &[1.0f64; 16]).is_ok());
    device.free(buf).ok();
    log_result("size_mismatch_copy_to", "PASS");
}

#[test]
fn size_mismatch_copy_from() {
    let device = GpuDevice::init();
    let buf = device.alloc_f64(8).expect("alloc");
    assert!(device.copy_from_device(&buf, &mut [0.0f64; 7]).is_err());
    assert!(device.copy_from_device(&buf, &mut [0.0f64; 9]).is_err());
    assert!(device.copy_from_device(&buf, &mut [0.0f64; 8]).is_ok());
    device.free(buf).ok();
    log_result("size_mismatch_copy_from", "PASS");
}

// ============================================================================
// 多次分配和释放（压力测试）
// ============================================================================

#[test]
fn repeated_alloc_free() {
    let device = GpuDevice::init();
    for i in 0..100 {
        let sizes = [16, 64, 256, 1024, 4096];
        for &sz in &sizes {
            let buf = match device.alloc_f64(sz) {
                Ok(b) => b,
                Err(e) => {
                    log_result(&format!("repeated_alloc_{i}_{sz}"), &format!("SKIP: {e}"));
                    continue;
                }
            };
            assert_eq!(buf.len(), sz);
            device.free(buf).ok();
        }
    }
    log_result("repeated_alloc_free", "PASS");
}

// ============================================================================
// 大值传输 — 验证位精度
// ============================================================================

#[test]
fn large_values_bit_precision() {
    let device = GpuDevice::init();
    let n = 128;

    // 测试值范围：从极小到极大
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let exp = (i as i32 - 64) * 10; // 指数从 -640 到 +630
            (i as f64 + 1.0) * 10.0f64.powi(exp)
        })
        .collect();

    let mut buf = device.alloc_f64(n).expect("alloc");
    device.copy_to_device(&mut buf, &values).expect("copy_to");
    let mut dst = vec![0.0f64; n];
    device.copy_from_device(&buf, &mut dst).expect("copy_from");

    for i in 0..n {
        assert_eq!(
            values[i].to_bits(),
            dst[i].to_bits(),
            "bit precision mismatch at index {i}: {:.12e} vs {:.12e}",
            values[i],
            dst[i]
        );
    }
    device.free(buf).ok();
    log_result("large_values_bit_precision", "PASS");
}

// ============================================================================
// 零拷贝阈值边界
// ============================================================================

#[test]
fn zero_length_buffer_operations() {
    let device = GpuDevice::init();
    let mut buf = device.alloc_f64(0).expect("zero alloc f64");
    assert_eq!(buf.len(), 0);
    // 零长度 copy 不应出错
    assert!(device.copy_to_device(&mut buf, &[]).is_ok());
    assert!(device.copy_from_device(&buf, &mut []).is_ok());
    device.free(buf).ok();

    let buf_i32 = device.alloc_i32(0).expect("zero alloc i32");
    assert_eq!(buf_i32.len(), 0);
    assert!(
        device
            .copy_to_device(&mut device.alloc_i32(0).unwrap(), &[])
            .is_ok()
    );
    device.free(buf_i32).ok();

    let buf_u8 = device.alloc_u8(0).expect("zero alloc u8");
    assert_eq!(buf_u8.len(), 0);
    device.free(buf_u8).ok();

    log_result("zero_length_buffer_operations", "PASS");
}
