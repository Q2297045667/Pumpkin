//! 新特性专项测试：CUDA 零拷贝、cuRAND 接入、OpenCL 调度策略回退。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::needless_range_loop,
    clippy::doc_markdown
)]
#![cfg(feature = "gpu")]

use pumpkin_config::gpu::{GpuBackend, GpuConfig};
use pumpkin_gpu::{DeviceType, GpuDevice};
use pumpkin_util::noise::perlin::OctavePerlinNoiseSampler;
use pumpkin_util::random::{RandomGenerator, xoroshiro128::Xoroshiro};
use pumpkin_world::noise_accel::NoiseAccelerator;

const SEED: u64 = 138_782_381_985_206;

fn fnv1a_f64(data: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn mk_sampler(seed: u64, octaves: &[i32]) -> OctavePerlinNoiseSampler {
    let r = Xoroshiro::from_seed(seed);
    let (s, a) = OctavePerlinNoiseSampler::calculate_amplitudes(octaves);
    let mut g = RandomGenerator::Xoroshiro(r);
    OctavePerlinNoiseSampler::new(&mut g, s, &a, false)
}

/// CUDA 零拷贝：阈值 > 0 时小缓冲区（排列表、收敛标志）走映射内存，
/// 端到端结果必须与 CPU 参考逐位一致。
#[test]
fn cuda_zero_copy_consistency() {
    let config = GpuConfig {
        enabled: true,
        noise_acceleration: true,
        cuda: pumpkin_config::gpu::CudaConfig {
            zero_copy_threshold_kb: 4,
            ..Default::default()
        },
        backend: GpuBackend::Cuda,
        ..Default::default()
    };
    let device = GpuDevice::from_config(&config);
    if device.device_type() != DeviceType::Cuda {
        println!("SKIP: CUDA 不可用");
        return;
    }
    let mut accel = NoiseAccelerator::new(&config);
    let n = 4096usize;
    let sampler = mk_sampler(SEED, &[-2, -1, 0, 1, 2]);
    let mut pos = Vec::with_capacity(n * 3);
    let mut s = SEED;
    for _ in 0..n {
        pos.push((s.wrapping_mul(6364136223846793005).wrapping_add(1) as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        pos.push((s as f64) * 1e-8);
        s = s.wrapping_mul(1442695040888963407);
        pos.push((s as f64) * 1e-8);
    }
    let mut cpu = vec![0.0f64; n];
    let mut acc = vec![0.0f64; n];
    for i in 0..n {
        cpu[i] = sampler.sample(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    accel.sample_octave(&sampler, &pos, &mut acc);
    assert_eq!(
        fnv1a_f64(&cpu),
        fnv1a_f64(&acc),
        "零拷贝模式下 octave 必须与 CPU 逐位一致"
    );
    println!("cuda zero-copy octave consistency OK");
}

/// cuRAND 接入：`use_curand = false`（默认）时 create_curand 必须报错；
/// 启用后返回可用生成器且序列确定性。
#[test]
fn curand_gated_by_config() {
    let device = GpuDevice::from_config(&GpuConfig {
        enabled: true,
        backend: GpuBackend::Cuda,
        ..Default::default()
    });
    if device.device_type() != DeviceType::Cuda {
        println!("SKIP: CUDA 不可用");
        return;
    }
    // 默认关闭 → 报错
    assert!(
        device.create_curand(42).is_err(),
        "use_curand = false 时 create_curand 必须报错"
    );

    let enabled = GpuDevice::from_config(&GpuConfig {
        enabled: true,
        cuda: pumpkin_config::gpu::CudaConfig {
            use_curand: true,
            ..Default::default()
        },
        backend: GpuBackend::Cuda,
        ..Default::default()
    });
    let mut g1 = enabled
        .create_curand(7)
        .expect("启用后 create_curand 必须可用");
    let mut g2 = enabled
        .create_curand(7)
        .expect("启用后 create_curand 必须可用");
    let mut a = [0.0f64; 64];
    let mut b = [0.0f64; 64];
    g1.generate_uniform_f64(64, &mut a).expect("gen a");
    g2.generate_uniform_f64(64, &mut b).expect("gen b");
    assert_eq!(a, b, "相同种子必须产生相同序列");
    println!("cuRAND config gating OK");
}

/// OpenCL 调度策略：`strategy = auto`（无法按性能自动选择）必须输出警告
/// 并回退 CPU；`ByIndex(0)` 则正常初始化 OpenCL。
#[test]
fn opencl_auto_strategy_falls_back_to_cpu() {
    let device = GpuDevice::from_config(&GpuConfig {
        enabled: true,
        backend: GpuBackend::OpenCl,
        ..Default::default()
    });
    // 默认 device 策略为 Auto → OpenCL 必须回退 CPU
    assert_eq!(
        device.device_type(),
        DeviceType::Cpu,
        "OpenCL + auto 策略必须回退 CPU"
    );

    // 显式 ByIndex(0) → 若有 OpenCL 设备则初始化成功
    let device = GpuDevice::from_config(&GpuConfig {
        enabled: true,
        backend: GpuBackend::OpenCl,
        device: pumpkin_config::gpu::GpuDeviceSelection::ByIndex { index: 0 },
        ..Default::default()
    });
    if pumpkin_gpu::opencl::is_opencl_available() {
        assert_eq!(
            device.device_type(),
            DeviceType::OpenCl,
            "OpenCL + ByIndex 策略应正常初始化"
        );
    }
    println!("OpenCL strategy fallback OK");
}
