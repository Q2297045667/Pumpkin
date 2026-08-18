# CPUIntelCorei5-8300H_16GB_GTX1060_Linux7.1_2026-08-18 失败分析

## 失败摘要

| 问题 | 级别 | 影响 | 状态 |
|------|------|------|------|
| CUDA Aquifer tiled/standard kernel 启动失败 | error | CUDA batch 模块无法真实运行 Aquifer | 未完全修复；已加入 tiled 失败后重试 standard kernel，但本机 standard 也失败 |
| CUDA sky_fill 与 CPU hash 不一致 | error | CUDA 光照矩阵失败 | 已定位 root cause；OpenCL/CPU 已修，CUDA kernel 文件当前未能通过编辑工具落盘 |
| `gpu_backend_alignment` 未遵守矩阵后端变量 | warning | OpenCL JIT 报告中出现 auto/CUDA 误判 | 需修测试配置 |
| `jit_numerical_consistency` 强制 JIT 编译断言 | warning | CUDA JIT 失败时没有按跳过规则处理 | 需修测试跳过逻辑 |
| Criterion bench target 与 `cargo test --all-targets -- --nocapture` 不兼容 | warning | `pumpkin-gpu --all-targets` 命令误失败 | 应单独用 `cargo bench` 或 `--no-run` |

## 1. CUDA Aquifer 启动失败

- 失败命令：

```sh
PUMPKIN_GPU_BACKEND=cuda PUMPKIN_GPU_JIT=0 cargo test -p pumpkin-world --features gpu --test gpu_path_matrix -- --nocapture --test-threads=1
PUMPKIN_GPU_BACKEND=cuda PUMPKIN_GPU_JIT=1 cargo test -p pumpkin-world --features gpu --test gpu_path_matrix -- --nocapture --test-threads=1
```

- 失败位置：`crates/pumpkin-world/tests/gpu_path_matrix.rs:1017-1019`
- 失败信息：

```text
aquifer tiled kernel 必须真实启动: LaunchFailed("aquifer batch failed")
```

- 已做修复：`crates/pumpkin-gpu/src/noise/batch_cell.rs` 中增加 tiled kernel 启动失败后重试 `aquifer_batch_f64` 普通 kernel。
- 当前结果：本机 CUDA 后端 tiled 和 standard 都失败，因此仍返回 `LaunchFailed("aquifer batch failed")`。
- 当前诊断限制：`BackendImpl::try_launch_kernel` 过去吞掉底层 `DeviceError`，只返回 bool。本次已补 `tracing::debug!("GPU kernel '{name}' launch failed: {error}")`，但测试未安装 tracing subscriber，命令输出仍无法看到 CUDA driver 具体错误码。

### 建议修复方案

1. 将 `GpuDevice::try_launch_kernel` / `BackendImpl::try_launch_kernel` 从 `bool` 改为 `Result<bool, DeviceError>` 或新增 `try_launch_kernel_diagnostic`。
2. 在 `GpuAquiferBatchSampler::batch_aquifer_apply` 中保留最后一次底层错误：

```rust
let tiled = self.try_launch(...);
if let Err(error) = tiled {
    tracing::warn!("aquifer tiled failed: {error}");
    let standard = self.try_launch(...);
    if let Err(error) = standard {
        return Err(DeviceError::LaunchFailed(format!("aquifer batch failed: {error}")));
    }
}
```

3. 取得具体 CUDA 错误后再判断是：
   - NVRTC 编译失败；
   - dynamic shared memory 大小不合法；
   - 参数签名不匹配；
   - device pointer/context 绑定问题；
   - kernel 运行期非法访问。

## 2. CUDA sky_fill hash 不一致

- 失败命令：同 CUDA matrix。
- 失败位置：`crates/pumpkin-world/tests/gpu_path_matrix.rs:558`。
- 失败信息：

```text
assertion `left == right` failed: sky fill mismatch
left: 350374200474250085
right: 11240331588349629153
```

### Root Cause

测试 heightmap 的生成范围为 `64..263`，`max_height = 256`。原 kernel 使用：

```c
int top = heightmap[col];
for (int y = top; y >= 0; y--) {
    int idx = col * max_height + y;
    uchar op = opacity[idx];
    ...
}
```

当 `top >= max_height` 时会越界读取 `opacity`。CPU sampler 已经修复为：

```rust
let top = heightmap[col].clamp(-1, max_height.saturating_sub(1) as i32);
```

但世界层 CPU fallback 和测试 reference 原先没有 clamp，导致边界契约不一致；当 GPU 失败后 fallback 复用部分写入的 output，也会产生脏数据 hash。

### 已修复部分

- `crates/pumpkin-gpu/kernels/opencl/light_sky.cl`：加入 `clamp(heightmap[col], -1, max_height - 1)`。
- `crates/pumpkin-world/src/light_accel.rs`：CPU fallback 前 `sl.fill(0)`，并对 `hm[col]` clamp。
- `crates/pumpkin-world/tests/gpu_path_matrix.rs`：CPU reference clamp。
- `crates/pumpkin-world/tests/worldgen_light_gpu_consistency.rs`：CPU reference clamp。
- `crates/pumpkin-gpu/src/light.rs`：GPU 输出 buffer 启动前清零，CPU sampler 使用 clamp。

### 未完成部分

CUDA kernel `crates/pumpkin-gpu/kernels/cuda/light_sky.cu` 需要同样改为：

```c
int top = max(-1, min(heightmap[col], max_height - 1));
```

当前终端能列出该文件，但 Zed 文件编辑工具对该路径返回 `path not found`，因此未能通过本轮工具落盘。该修复完成后需重跑：

```sh
PUMPKIN_GPU_BACKEND=cuda PUMPKIN_GPU_JIT=0 cargo test -p pumpkin-world --features gpu --test gpu_path_matrix -- --nocapture --test-threads=1
PUMPKIN_GPU_BACKEND=cuda PUMPKIN_GPU_JIT=1 cargo test -p pumpkin-world --features gpu --test gpu_path_matrix -- --nocapture --test-threads=1
```

## 3. JIT 非矩阵测试误判

### `gpu_backend_alignment`

- 失败项：`gpu_kernel_launcher_registered`
- 失败位置：`crates/pumpkin-world/tests/gpu_backend_alignment.rs:167`
- 失败信息：

```text
核心 kernel 'octave_perlin_sample_f64' 未注册（编译失败将回退 CPU）
```

该测试没有使用 `PUMPKIN_GPU_BACKEND=opencl` 矩阵配置，而是默认 auto 初始化。本机 auto 优先 CUDA，因此 OpenCL JIT 报告中出现 CUDA/auto 的注册失败。

建议：复用 `gpu_path_matrix.rs` 的环境变量配置逻辑，或者在测试内明确读取 `PUMPKIN_GPU_BACKEND`。

### `jit_numerical_consistency`

- 失败项：`jit_gpu_backend_all_families_bitwise_cpu`
- 失败位置：`crates/pumpkin-world/tests/jit_numerical_consistency.rs:397`
- 失败信息：

```text
octave JIT kernel 必须真实编译
```

根据 README 规则，JIT 启动失败应直接跳过。该测试当前强制断言真实编译，和规则不一致。矩阵测试已证明 OpenCL JIT 可以真实编译并通过：

```text
MATRIX[opencl|jit-on] jit_kernel_compiled=true
MATRIX[opencl|jit-on] jit octave vs batch OK
```

建议：当后端为 CUDA 且 JIT 编译失败时打印 skip；当后端为 OpenCL 时保留强断言。

## 4. Benchmark target 运行方式

`cargo test -p pumpkin-gpu --all-features --all-targets -- --nocapture --test-threads=1` 会把 libtest 参数传给 Criterion bench target，导致：

```text
error: unexpected argument found
```

正确方式：

```sh
cargo test -p pumpkin-gpu --all-features --tests -- --nocapture --test-threads=1
cargo bench -p pumpkin-gpu --bench gpu_consistency --features gpu
```

本次 `cargo bench` 已通过，结果：

```text
cpu_noise_fingerprint_batch time: [758.76 us 764.04 us 770.94 us]
```

## 5. 当前可用结论

- OpenCL 后端可作为当前机器的主要 GPU 路径：矩阵、压力、指纹、光照、多种子、pipeline 全部通过。
- CUDA 后端不能作为完整世界生成 GPU 路径启用，至少需要修复 `light_sky.cu` heightmap clamp，并改造 launch 错误传播后定位 Aquifer 失败。
- 若生产配置使用 `backend = "auto"`，本机可能优先选 CUDA，从而踩到 CUDA 失败路径。临时建议将 `pumpkin.toml` 中 `[gpu] backend = "opencl"`，或在启动环境中指定 OpenCL，直到 CUDA 阻塞项修复。
