# Pumpkin GPU — CPU/MEM/GPU 全路径矩阵测试报告

**日期**: 2026-08-13
**环境**: Windows · CPU: Intel(R) Xeon(R) W-3345 @ 3.00GHz · GPU: NVIDIA Tesla T10(同时暴露 CUDA 与 OpenCL 平台)
**构建**: debug profile(未优化,`--features gpu`)
**配置**: `pumpkin.toml` → `backend = "auto"`(探测顺序 CUDA > OpenCL > CPU),CUDA flags `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`,OpenCL flags `-cl-fp32-correctly-rounded-divide-sqrt`

---

## 一、结论摘要

| 维度 | 结论 |
|------|------|
| **一致性** | CPU / CUDA / OpenCL 三种后端,噪声五族、三线性、批量三族、光照四族在 JIT 开/关两种模式下 **全部逐位一致**(fnv1a 哈希相等) |
| **CUDA 性能** | octave 262k **82–85×**,double perlin 65k **60–65×**,trilinear 131k **4–5×** |
| **OpenCL 性能** | octave 262k **55–62×**,double perlin 65k **34–41×**,trilinear 131k **2.1×** |
| **JIT** | CUDA 与 OpenCL 的 JIT 专用内核路径均已真实编译并运行(此前 CUDA JIT 静默失败、OpenCL JIT 数值错误),JIT 编译失败时按设计自动回退 batch → CPU |
| **本轮修复** | OpenCL f64 FMA 收缩(1 ulp 偏差)、OpenCL `kernel` 保留字编译失败、CUDA JIT 编译失败、JIT double perlin 数值错误、CPU 后端 sysinfo 重复枚举开销 |
| **已知限制** | CUDA 零拷贝未实现、cuRAND 未接入生产链路、OpenCL 无 persistent kernel(见 §8) |

---

## 二、本轮修复的问题(Bug 分析)

### 2.1 OpenCL f64 与 CPU 相差 1 ulp(Bug B)

**现象**: OpenCL 后端噪声数值与 CPU 逐位不一致,长依赖链(如 `du0 = a + b*(c-a); du1 = c + b*(d-c); du0 + a*(du1-du0)`)相差 ~1e-17(1 ulp)。

**根因**: NVIDIA 的 OpenCL 编译器对 f64 默认开启 FMA 收缩(`a*b+c` → `fma(a,b,c)`)。逐项排查结论:

- `-cl-opt-disable` **无效** —— 长依赖链仍差 1 ulp;
- `-cl-mad-enable` / `-cl-no-mad-enable` **被忽略**;
- 广告的 `cl_nv_compiler_options` 扩展**不接受** CUDA 风格标志 —— 注入 `-fmad=false` 等会直接导致编译失败(`Don't understand command line argument "-fmad=false"`),此前「注入标志后数值一致」是**假象**:编译失败 → 全部 kernel 缺失 → 回退 CPU 路径;
- `volatile` 中间值**可以**阻断收缩(验证通过),但侵入性大。

**修复**: 在 `opencl_compile::compile_one` 中向所有 OpenCL kernel 源码(常规 + JIT)头部注入标准 pragma **`#pragma OPENCL FP_CONTRACT OFF`**(OpenCL 1.0+ 标准,NVIDIA 编译器遵守——逐位微基准验证 `normal=true`)。

**验证**: 新增永久回归测试 `crates/pumpkin-gpu/tests/opencl_f64_precision.rs`,通过 crate 自身编译路径编译长依赖链 kernel,与 CPU 逐位比较,通过。

### 2.2 OpenCL `beardifier_batch.cl` 编译失败

`beard_contrib` 的参数名为 `kernel` —— **`kernel` 是 OpenCL C 保留字**,导致 `CL_BUILD_PROGRAM_FAILURE`(该 kernel 一直回退 CPU,GPU 路径未生效)。修复:参数重命名为 `kernel_table`。

### 2.3 JIT double perlin 数值错误(双重 maintain_precision)

`specialize_double_perlin` 对 `x*c` 提前应用了一次 `maintain_precision`,再在八度内对 `x2*lac` 应用第二次;而 CPU / batch kernel 只在 `(x*c)*lac` 上应用**一次**。修复:JIT 源码改为 `x2 = x*c`(原始值),与 batch kernel `maintain_precision(x*c*lac)` 语义一致。

### 2.4 CUDA JIT 静默失败(两个原因)

1. NVRTC 不接受 `--opt-level=3`(nvcc 专有选项,报 `NVRTC_ERROR_INVALID_OPTION`)→ 移除(NVRTC 设备码始终优化);
2. JIT 源码只生成 OpenCL C 方言(`__kernel` / `__global` / `get_global_id`),NVRTC 无法编译 → `JitSpecializedKernel` 新增 `cuda_source` 字段,所有特化函数同时生成 **CUDA C++ 方言**(`extern "C" __global__`, `blockIdx.x * blockDim.x + threadIdx.x`)。

**验证**: `matrix_jit_path` 输出 `jit_kernel_compiled=true`(此前为 `false`),且 JIT kernel 输出与 batch/CPU 逐位一致。

### 2.5 CPU 后端每次操作重复枚举硬件

`CpuBackend::new()` 与 `logging::log_gpu_startup` 每次调用都执行 `sysinfo::System::new_all()`(~100–250ms),而 `BatchAccelerator::ensure_device()` 在 CPU 模式下**每次操作**都重建设备,导致 CPU 回退路径出现 100–270ms 的纯开销(如 trilinear 131k 测得 96ms,实际计算仅 ~4ms)。修复:CPU 名缓存进 `OnceLock`(`cpu::cpu_name()`),启动日志复用同一缓存。修复后 CPU 后端 beardifier 98ms → 0.22ms,trilinear 96ms → 4.0ms。

### 2.6 OpenCL 命令队列初始化(Bug A,上一轮已修复,本轮回归确认)

`CommandQueue::create_default(&ctx, device.id() as u64)` 误把设备 ID 当「队列属性位掩码」,返回 `CL_INVALID_VALUE`。修复为 `create_default(&ctx, 0)`。这是 OpenCL 后端此前一直静默回退 CPU 的直接原因。

---

## 三、全路径矩阵测试结果(清理缓存后执行)

测试入口:`crates/pumpkin-world/tests/gpu_path_matrix.rs`(6 个测试 × 6 种组合,`--test-threads=1`)。

| 组合 | 噪声五族+三线性 | 批量三族 | 光照四族 | JIT 路径 | 设备报告 | 性能 | 结果 |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU · JIT off | ✅ | ✅ | ✅ | 跳过(设计) | CPU W-3345 | 1.0× | **6/6** |
| CPU · JIT on | ✅ | ✅ | ✅ | CPU 回退(设计) | CPU W-3345 | 1.0× | **6/6** |
| CUDA · JIT off | ✅ | ✅ | ✅ | 跳过(设计) | Tesla T10 | 82×/60×/4× | **6/6** |
| CUDA · JIT on | ✅ | ✅ | ✅ | ✅ 真实编译 | Tesla T10 | 85×/65×/5× | **6/6** |
| OpenCL · JIT off | ✅ | ✅ | ✅ | 跳过(设计) | Tesla T10 | 62×/41×/2× | **6/6** |
| OpenCL · JIT on | ✅ | ✅ | ✅ | ✅ 真实编译 | Tesla T10 | 55×/34×/2× | **6/6** |

一致性与 JIT 失败处理规则:

- 一致性断言为 **fnv1a 逐位哈希相等**(不是容差比较);
- JIT kernel 编译失败 → 该段跳过(不判失败),其余断言照常;
- 功能模块(kernel 编译/启动)失败 → 按设计回退 batch → CPU,一致性依旧成立。

---

## 四、性能对比(热数据,已预热排除惰性初始化)

来源:`gpu_path_matrix.rs::matrix_perf`(预热后计时)。

| 后端 | JIT | octave 262,144 | double perlin 65,536 | trilinear 131,072 |
|------|-----|----------------|----------------------|-------------------|
| CPU | off | 289.9ms 参考 / 275.5ms (1.05×) | 71.9 / 70.1 (1.03×) | 7.8 / 3.8 (2.02×) |
| CPU | on | 316.6 / 326.3 (0.97×) | 72.7 / 71.9 (1.01×) | 7.4 / 3.8 (~2×) |
| CUDA | off | 283.0 / **3.44ms (82.3×)** | 75.4 / **1.25ms (60.5×)** | 7.7 / **1.91ms (4.0×)** |
| CUDA | on | 299.1 / **3.51ms (85.2×)** | 81.5 / **1.26ms (64.8×)** | 9.4 / **1.85ms (5.1×)** |
| OpenCL | off | 300.4 / **4.87ms (61.7×)** | 74.8 / **1.84ms (40.7×)** | 7.3 / **3.53ms (2.1×)** |
| OpenCL | on | 286.9 / **5.20ms (55.2×)** | 71.3 / **2.09ms (34.1×)** | 7.4 / **3.44ms (2.1×)** |

冷启动数据(含一次性设备初始化,来源 `worldgen_perf`):

| 后端 | 初始化成本(首调) | 说明 |
|------|-----------------|------|
| CUDA | ~430–930ms | NVRTC 编译全部 15 个 kernel + 上下文创建,仅在 `BatchAccelerator`/`LightAccelerator` 首个操作发生(懒初始化);`NoiseAccelerator` 在构造时初始化 |
| OpenCL | ~180–530ms | 运行时编译全部 13 个 kernel |
| CPU | ~250ms(已修复至 <1ms) | 原为 sysinfo 硬件枚举(§2.5) |

### JIT 性能分析(如实记录)

在 n = 65k–262k 规模下,JIT 专用内核与 batch 内核**性能相当**(CUDA octave 3.51 vs 3.44ms;double 1.26 vs 1.25ms),原因是该规模下受显存带宽约束,循环开销可忽略。JIT 收益场景为:①小批量高八度(消除循环 + 间接访存);②CPU→GPU 调度延迟占比高的短调用。JIT 编译为一次性成本(内核名按八度数缓存,`has_kernel` 命中后不再编译)。OpenCL JIT 略慢于 batch(5.20 vs 4.87ms),因 JIT 路径每次调用新建缓冲区,而 batch 路径使用缓冲池——**后续优化项**:JIT 路径接入 buffer pool。

---

## 五、内存估算(CPU 主机内存 + GPU 显存)

| 操作 | 规模 | GPU 显存 | 主机内存 | 说明 |
|------|------|---------|---------|------|
| octave 采样 | 262,144 点 | ~8.4 MB | ~10.5 MB | pos 6.3MB + res 2.1MB + perms 1.5KB + 4 组配置表 |
| double perlin | 65,536 点 | ~2.3 MB | ~3.1 MB | pos 1.6MB + res 0.5MB + 2×perms + 2×配置表 |
| trilinear | 131,072 组 | ~12.6 MB | ~15.7 MB | corners 8.4MB + deltas 3.1MB + res 1.0MB |
| flatcache/shift | 65,536 列 | ~1.6 MB | ~2.6 MB | 2D 坐标 + res |
| aquifer | 16,384 点 | ~1.1 MB | ~2.2 MB | pos + 密度 + 打包网格(7³×4×8B ≈ 11KB) |
| beardifier | 16,384 点 | ~2.4 MB | ~3.9 MB | pos + 24³ 核表(110KB)+ structures + junctions |
| 天空光填充 | 256 列×384 高 | ~197 KB | ~394 KB | heightmap 1KB + opacity 98KB + sky 98KB |
| 光照传播 | 16³ 区块 | ~65 KB | ~130 KB | light + opacity + 6 邻接表 |

- 所有量级均远小于 Tesla T10 显存(16 GB),**无显存压力风险**;
- 缓冲池(`buffer_pool`)复用 octave/double/shift/flatcache 路径的缓冲区,避免重复分配;
- JIT 路径暂未接入缓冲池(见 §4 性能分析)。

---

## 六、功能对齐审计(CUDA vs OpenCL)

| 功能 | CUDA | OpenCL | 对齐方式 |
|------|:----:|:------:|---------|
| 八度 Perlin(octave) | ✅ | ✅ | 同一语义,各自方言,逐位一致 |
| 八度 SoA 变体 | ✅ | ✅ | `soa_layout = true` 且 n ≥ 64 时启用 |
| double perlin | ✅ | ✅ | 同上 |
| shift A / B | ✅ | ✅ | 同上 |
| flatcache | ✅ | ✅ | 同上 |
| trilinear | ✅ | ✅ | 同上 |
| aquifer(+tiled 局部内存变体) | ✅ | ✅ | `local_mem_tile_threshold` 控制 tiled 选择 |
| beardifier | ✅ | ✅ | 本轮修复 OpenCL 编译 |
| 天空光垂直填充 | ✅ | ✅ | |
| 方块光扫描 | ✅ | ✅ | |
| 迭代光照传播 | ✅(含 persistent) | ✅(仅迭代式) | OpenCL 无 persistent,见 §8.3 |
| 天空光水平传播 | ✅ | ✅ | |
| JIT 特化内核 | ✅(CUDA C++ 方言) | ✅(OpenCL C 方言) | 双方言同源生成,语义一致 |
| 失败回退 | ✅ → batch → CPU | ✅ → batch → CPU | 编译失败/启动失败均不阻断 |

CUDA 专属 kernel(`light_propagate_u8_persistent`)只注册在 CUDA 注册表;OpenCL 注册表无此条目,迭代式 `light_propagate_u8` 为双后端共同路径。`compile.rs` 内 `kernel_names_cuda_opencl_aligned` 测试断言双后端清单除该 CUDA 专属 kernel 外完全一致。

---

## 七、用户问题的专项回答

### 7.1 `GpuDevice::init()` 不初始化 kernel registry —— **已修复**

`GpuDevice::init()` 现在与 `from_config()` 一致,调用 `compile::init_kernel_registry()` 注入全局注册表,供延迟编译(按需加载)读取。回归测试 `lib.rs::tests::init_initializes_kernel_registry` 断言基础 kernel 已注册到 OpenCL 注册表、CUDA 专属 kernel 已注册到 CUDA 注册表。

### 7.2 OpenCL `light_propagate_u8_persistent` 为何不可行?这是什么,有什么影响

**是什么**: CUDA 专属的 persistent kernel —— 单次 `cudaLaunchCooperativeKernel` 启动后**在设备内循环迭代**距离场传播直至收敛(内部用 `__shared__` + `__syncthreads()` 做块内同步、`atomicInc` 计数器做**网格级软件栅栏**、`volatile` + `__threadfence_system()` 与主机通信),避免主机侧每轮迭代的 kernel 启动与同步开销。用于方块光/天空光的迭代传播。

**为何 OpenCL 不可行**:

1. **网格级软件栅栏需要 co-residency 保证**。persistent kernel 的关键是「所有 block 同时驻留」——`cudaLaunchCooperativeKernel` 显式保证这一点,OpenCL 无等价物,work-group 可能被调度器换出,自旋等待栅栏会**死锁**;
2. **主机可见的收敛标志 + 系统级内存栅栏**。OpenCL 1.2(NVIDIA 平台的实际版本)没有 `__threadfence_system`,需要 OpenCL 2.0 的 SVM + `memory_scope_all_svm_devices` 原子操作,NVIDIA OpenCL 平台不支持;
3. 即便用 `-cl-std=CL2.0`,NVIDIA 平台仍按 1.2 编译。

**影响**: 仅 CUDA 后端在 `persistent_kernels = true` 时可用单次启动优化(减少主机侧迭代轮转);OpenCL 与默认 CUDA 配置均走「主机循环 + 每轮启动 `light_propagate_u8` + 收敛标志回读」的迭代式路径,数值结果完全一致,仅每轮多一次启动/同步开销(区块规模 16³,实测可忽略)。

### 7.3 为何 CellCache GPU 算法用 `gen_perm_table` 而非 vanilla `ImprovedNoise`?有什么影响

**现状**: 当前实现**不**使用任何自造的 `gen_perm_table`。GPU 路径通过 `SerializedOctaveConfig::from_sampler` 直接提取 vanilla `OctavePerlinNoiseSampler` 内部**真实的 256 字节置换表**(`data.sampler.permutation()`),连同振幅/持续性/间隙/原点打包上传,与 CPU 路径共享同一张表 —— 这是逐位一致的根本保证。

**为什么不应用 `ImprovedNoise`**: vanilla 的 CellCache(基岩密度 `DoublePerlinNoiseSampler`)基于 `PerlinNoiseSampler`(经典 Perlin,16 梯度表 + 256 置换表),而 `ImprovedNoise` 是另一套算法(旧版地形高度图使用的 2D/3D improved noise,梯度集与置换方式都不同)。若 GPU 侧换成 `ImprovedNoise`,输出与 CPU 的 `DoublePerlinNoiseSampler` 完全不相等,**地形、生物群系、矿脉位置全部失配**——这正是必须对齐 vanilla 语义的原因。

**影响结论**: 用 vanilla 真实置换表(而非自造表)→ CPU/GPU 逐位一致,`cell_cache_vanilla` 指纹测试通过;若误用 `ImprovedNoise` 或自造表 → 世界生成不可复现、存档不兼容。

---

## 八、GPU 模块未实现功能清单

| 功能 | 状态 | 影响 |
|------|------|------|
| CUDA 零拷贝(`zero_copy_threshold_kb`) | ❌ 未实现(使用标准分配,启动日志提示) | 小缓冲区多一次 HtoD/DtoH 拷贝,功能与正确性无影响 |
| cuRAND(`use_curand`) | ⚠️ 已实现为 CPU SplitMix64,但**未接入任何生产链路**(无调用方) | 配置开启仅打印警告;不参与地形生成(符合「确定性优先」设计) |
| OpenCL persistent kernel | ❌ 不可行(§7.2) | 仅 CUDA 可选优化路径 |
| 缓冲池覆盖 | ⚠️ 噪声族已覆盖,JIT / trilinear / 光照路径未覆盖 | 高频调用时多若干次分配,见 §4 后续优化项 |
| 多命令队列流水线(`pipeline_queues`) | ✅ 已实现(轮转分配) | — |
| SoA 布局(`soa_layout`) | ✅ 已实现 | — |
| 含水层局部内存 tiled | ✅ 已实现(阈值控制) | — |
| kernel 注册表懒加载 | ✅ 已实现 | — |

---

## 九、移除的旧 API / 代码与清理

**已删除文件**(旧批量 kernel,被 `double_perlin_sample_f64` 直接复用取代,无任何引用残留):

- `crates/pumpkin-gpu/kernels/cuda/cell_cache.cu` / `opencl/cell_cache.cl`
- `crates/pumpkin-gpu/kernels/cuda/interpolator_fill.cu` / `opencl/interpolator_fill.cl`
- `crates/pumpkin-gpu/kernels/cuda/vein_batch.cu` / `opencl/vein_batch.cl`

**临时探针文件(本轮定位用,已全部删除)**: `zz_opencl_*.rs` × 7、`zz_cuda_jit.rs`。

**改动量评估**(工作树累计,43 个文件): +1,982 / −4,292 行(净 −2,310,主要来自旧 kernel 与旧测试移除)。本会话核心改动:

| 文件 | 改动 |
|------|------|
| `pumpkin-gpu/src/compile.rs` | OpenCL FP_CONTRACT pragma 注入;CUDA JIT 选项修复 + `cuda_source` |
| `pumpkin-gpu/src/jit.rs` | 双方言源码生成;double perlin 数值修复 |
| `pumpkin-gpu/src/opencl/mod.rs` / `context.rs` | 移除无效 `-fmad` 标志逻辑 |
| `pumpkin-gpu/src/cpu/mod.rs` / `logging.rs` | CPU 名 `OnceLock` 缓存 |
| `pumpkin-gpu/kernels/opencl/beardifier_batch.cl` | `kernel` → `kernel_table` |
| `pumpkin-gpu/tests/opencl_f64_precision.rs` | 新增 Bug B 回归测试 |
| `pumpkin-world/tests/gpu_path_matrix.rs` | 预热计时代;clippy 修复 |
| `pumpkin-world/tests/jit_numerical_consistency.rs` | 新增真实 GPU JIT 全族一致性测试 |
| `pumpkin-world/tests/worldgen_perf.rs` | 支持 `PUMPKIN_GPU_BACKEND` / `PUMPKIN_GPU_JIT` 环境变量 |
| `pumpkin-config/src/lib.rs` | 测试代码 clippy 修复 |

---

## 十、测试清单与结果

| 套件 | 数量 | 结果 | 说明 |
|------|-----:|:----:|------|
| `pumpkin-world` 单元测试 | 152 | ✅ | 含 `gpu_pipeline_test`(生产管线 7 项) |
| `gpu_path_matrix`(6 组合 × 6 测试) | 36 次执行 | ✅ | 全路径矩阵,见 §3 |
| `batch_fingerprint` | 6 | ✅ | CellCache/Aquifer/Beardifier 指纹(OpenCL 亦跑过) |
| `gpu_backend_alignment` | 4 | ✅ | 双后端对齐 |
| `gpu_noise_fingerprint(_full)` | 8 | ✅ | 噪声指纹 |
| `gpu_pipeline_integration` | 11 | ✅ | 管线集成 |
| `jit_numerical_consistency` | 9 | ✅ | 含新增真实 GPU JIT 全族测试 |
| `light_accel_consistency` / `light_fingerprint` | 17 | ✅ | 光照 |
| `noise_accel_consistency` | 15 | ✅ | 噪声一致性(OpenCL 亦跑过) |
| `surface_noise_cache_test` | 2 | ✅ | |
| `worldgen_bench` | 5 | ✅ | 基准 |
| `worldgen_cpu_fallback_consistency` | 5 | ✅ | CPU 回退一致性 |
| `worldgen_fingerprint` | 8 | ✅ | 世界生成指纹(OpenCL 亦跑过) |
| `worldgen_light_gpu_consistency` | 4 | ✅ | 光照 GPU 一致性(OpenCL 亦跑过) |
| `worldgen_multi_seed_consistency` | 10 | ✅ | 多种子(OpenCL 亦跑过) |
| `worldgen_perf` | 8 | ✅ | 性能基准(三后端均跑过) |
| `worldgen_pipeline_fingerprint` | 2 | ✅ | 管线指纹(OpenCL 亦跑过) |
| `worldgen_stress` | 11 | ✅ | 压力(OpenCL 亦跑过) |
| `pumpkin-gpu` 全部(含新增 opencl_f64_precision) | 87+ | ✅ | |

**质量门禁**: `cargo fmt --all --check` ✅ · `cargo clippy -p pumpkin-gpu -p pumpkin-world -p pumpkin-config -p pumpkin-util --features gpu --all-targets` 0 error ✅

---

## 十一、复现方法

```sh
# 六组合矩阵(清理缓存后)
cargo clean -p pumpkin-world -p pumpkin-gpu
for B in cuda opencl cpu; do
  for J in 0 1; do
    PUMPKIN_GPU_BACKEND=$B PUMPKIN_GPU_JIT=$J \
      cargo test -p pumpkin-world --features gpu --test gpu_path_matrix -- --nocapture --test-threads=1
  done
done

# 基准(支持同环境变量)
PUMPKIN_GPU_BACKEND=cuda cargo test -p pumpkin-world --features gpu --test worldgen_perf -- --nocapture

# Bug B 回归
cargo test -p pumpkin-gpu --features gpu --test opencl_f64_precision -- --nocapture
```

## 十二、后续建议(按优先级)

1. **JIT 路径接入缓冲池** —— 消除每调用分配,OpenCL JIT 可追上 batch(§4);
2. **trilinear/光照路径接入缓冲池** —— 高频小调用场景收益明显;
3. **CUDA 零拷贝** —— 实现 `zero_copy_threshold_kb` 的小缓冲 pinned/zero-copy 分配,或从配置中移除该占位项;
4. **cuRAND 决策** —— 明确「不接入地形生成」的定位并在配置文档标注,或移除该配置项;
5. **OpenCL 多队列流水线压力测试** —— `pipeline_queues > 1` 已实现但缺专项压力覆盖。
