//! OpenCL f64 FMA 收缩回归测试。
//!
//! 背景（Bug B）：NVIDIA 的 OpenCL 编译器对 f64 默认开启 FMA 收缩
//! （`a*b+c` → `fma(a,b,c)`），且拒绝 CUDA 风格的 `-fmad=false` 编译标志，
//! `-cl-opt-disable` / `-cl-mad-enable` 也无法关闭收缩。这导致长依赖链
//! 上 GPU 与 CPU 相差 1 ulp，破坏逐位一致性。
//!
//! 修复：`opencl_compile::compile_one` 向所有 kernel 源码注入标准 pragma
//! `#pragma OPENCL FP_CONTRACT OFF`（OpenCL 1.0+ 标准，NVIDIA 遵守）。
//!
//! 本测试通过 crate 自身的编译路径（`OpenClKernelCompiler::compile_by_name`）
//! 编译一个长依赖链 kernel，验证其输出与 CPU 参考逐位一致。
//! 无可用 OpenCL 设备时跳过。
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::doc_markdown,
    clippy::undocumented_unsafe_blocks,
    clippy::needless_raw_string_hashes,
    clippy::excessive_precision
)]
#![cfg(feature = "opencl")]

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::memory::Buffer;

const PROBE_SOURCE: &str = r#"
__kernel void fma_chain_probe(__global double* out, __global const double* in) {
    double a = in[0], b = in[1], c = in[2], d = in[3];
    // 长依赖链：FMA 收缩时与 CPU 左结合求值相差 1 ulp
    double du0 = a + b * (c - a);
    double du1 = c + b * (d - c);
    out[0] = du0 + a * (du1 - du0);
}
"#;

/// 在无 OpenCL 设备时返回 `None`。
fn try_opencl() -> Option<(Context, CommandQueue, opencl3::types::cl_device_id)> {
    let platforms = opencl3::platform::get_platforms().ok()?;
    let device_id = *platforms
        .iter()
        .find_map(|p| p.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU).ok())
        .filter(|devices| !devices.is_empty())?
        .first()?;
    let device = opencl3::device::Device::new(device_id);
    let ctx = Context::from_device(&device).ok()?;
    let queue = CommandQueue::create_default(&ctx, 0).ok()?;
    Some((ctx, queue, device_id))
}

#[test]
fn opencl_f64_no_fma_contraction_bitwise_cpu() {
    let Some((ctx, queue, device_id)) = try_opencl() else {
        println!("SKIP: 无可用 OpenCL GPU 设备");
        return;
    };

    // 通过 crate 编译路径编译（会注入 FP_CONTRACT OFF pragma）
    let mut compiler = pumpkin_gpu::compile::opencl_compile::OpenClKernelCompiler::new(&[
        "-cl-fp32-correctly-rounded-divide-sqrt".to_string(),
    ]);
    compiler
        .compile_by_name(&ctx, device_id, "fma_chain_probe", PROBE_SOURCE)
        .expect("compile fma_chain_probe");
    assert!(compiler.has("fma_chain_probe"));
    let kernel = compiler.get_kernel("fma_chain_probe").expect("kernel");

    let inputs = [
        0.37432345836450856f64,
        0.12345678901234567f64,
        -0.98765432198765432f64,
        0.55555555555555555f64,
    ];

    let mut out = [0.0f64; 1];
    let buf = unsafe {
        Buffer::<f64>::create(
            &ctx,
            opencl3::memory::CL_MEM_READ_WRITE,
            1,
            std::ptr::null_mut(),
        )
        .expect("out buf")
    };
    let mut in_buf = unsafe {
        Buffer::<f64>::create(
            &ctx,
            opencl3::memory::CL_MEM_READ_ONLY,
            4,
            std::ptr::null_mut(),
        )
        .expect("in buf")
    };
    let _: () = unsafe {
        queue
            .enqueue_write_buffer(&mut in_buf, opencl3::types::CL_TRUE, 0, &inputs, &[])
            .expect("write");
        kernel.set_arg(0, &buf).expect("arg0");
        kernel.set_arg(1, &in_buf).expect("arg1");
        let gws = [1usize];
        let lws = [1usize];
        queue
            .enqueue_nd_range_kernel(
                kernel.get(),
                1,
                std::ptr::null::<usize>(),
                gws.as_ptr(),
                lws.as_ptr(),
                &[],
            )
            .expect("run");
        queue
            .enqueue_read_buffer(&buf, opencl3::types::CL_TRUE, 0, &mut out, &[])
            .expect("read");
        queue.finish().expect("finish");
    };

    // CPU 参考：严格左结合（无 FMA 收缩）
    let [a, b, c, d] = inputs;
    let du0 = a + b * (c - a);
    let du1 = c + b * (d - c);
    let cpu = du0 + a * (du1 - du0);

    assert_eq!(
        out[0].to_bits(),
        cpu.to_bits(),
        "OpenCL 输出与 CPU 相差 {} ulp — FP_CONTRACT OFF pragma 未生效",
        out[0].to_bits().abs_diff(cpu.to_bits())
    );
}
