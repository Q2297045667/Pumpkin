# Pumpkin — 全方位问题检查报告(CPU / 内存 / GPU)

**日期**: 2026-08-13
**环境**: Windows · CPU Intel(R) Xeon(R) W-3345 @ 3.00GHz · GPU NVIDIA Tesla T10(CUDA + OpenCL)
**构建**: debug profile,`--features gpu`
**范围**: 机械检查(clippy/fmt/check/typos/machete)+ GPU 模块逐函数 CPU 对比审计 + 测试补全

---

## 一、工具链机械检查

| 检查项 | 命令 | 结果 |
|--------|------|------|
| Clippy Lint | `cargo clippy --all-targets --all-features` | ✅ 0 error / 0 warning(仅第三方 `proc-macro-error2 v2.0.1` future-incompat 提示,非项目代码) |
| 代码格式化 | `cargo fmt --all -- --check` | ✅ 无问题通过 |
| 编译通过率 | `cargo check --all-targets --all-features` | ✅ 0 error(同上 future-incompat 提示) |
| 拼写检查 | `typos`(1.49.0) | ✅ 无问题通过 |
| 未使用依赖 | `cargo-machete`(0.9.2) | ✅ 未发现未使用依赖(含 feature/macro 间接引用分析) |

---

## 二、发现的 BUG(本轮修复,8 项)

### 🔴 BUG-1(Critical)· aquifer tiled kernel 语义完全错误

**文件**: `crates/pumpkin-gpu/kernels/cuda/aquifer_batch_tiled.cu`、`kernels/opencl/aquifer_batch_tiled.cl`

**现象**: 修复前 tiled kernel 与标准 kernel/CPU 回退的语义完全不同:
- 实心块输出 `block_ids=0`(应为 `1` 石头);
- 非实心块输出 `(int)fluid_level` = **-10000**(应为 `2` 水或 `0` 空气);
- 水分支完全缺失 `qy < fluid_level` 判定,`fluid_updates` 恒为 `q_density+barrier<=0`;
- 屏障计算 `sum*scale/4` 与标准 `(sum/4)*scale` 浮点舍入不同;
- **第 2/3 参数顺序与主机端相反**(`densities` 与 `packed_positions` 互换);
- 缺 `M<4` 早退,`best_idx` 初始化为 `{0,0,0,0}`。

**根因**: 该 kernel 是占位实现,从未真正执行过——启动层不支持 local/shared 内存参数(见 BUG-3),两个后端启动均失败后静默回退 CPU,因此错误从未暴露。

**修复**: 重写两个 tiled kernel,与标准 kernel/CPU 逐位一致(含水分支、参数顺序、求值顺序、M<4)。

### 🔴 BUG-2(Critical)· CUDA 标准 aquifer kernel 从未编译成功(静默回退)

**文件**: `crates/pumpkin-gpu/kernels/cuda/aquifer_batch.cu:35`

**现象**: 使用 `INFINITY` 宏——**NVRTC 不定义该宏**(报 `identifier "INFINITY" is undefined`),导致 CUDA 上 aquifer kernel 从未编译成功,`has_kernel=false` → 启动失败 → 静默回退 CPU。OpenCL 平台(标准头定义 `INFINITY`)正常。

**影响**: CUDA 的 aquifer 加速从未生效;一致性测试因 CPU 回退而"假通过"。

**修复**: 替换为远大于任何现实坐标距离平方的有限值 `1.0e300`(网格坐标为小整数,注释说明安全界)。

### 🔴 BUG-3(Critical)· 启动层不支持 local/shared 内存参数

**文件**: `crates/pumpkin-gpu/src/common/kernel.rs`、`src/cuda/kernel.rs:189`、`src/opencl/kernel.rs`

**现象**: tiled kernel 需要 CUDA `extern __shared__`(LaunchConfig.shared_mem_bytes 硬编码为 0 → `CUDA_ERROR_INVALID_VALUE`)与 OpenCL `__local` 参数(未 `set_arg_local_buffer` → `CL_INVALID_KERNEL_ARGS`)——即使 BUG-1 修复后也无法启动。

**修复**: `KernelLaunch` 新增 `local_mem_bytes: Vec<usize>`;CUDA launcher 汇总为 `shared_mem_bytes`;OpenCL launcher 依次 `set_arg_local_buffer`;`GpuAquiferBatchSampler` 按 M 计算并传入。同时补齐 OpenCL 缺失的 `KernelArg::U32/USize` 支持(与 CUDA 对齐)。

### 🔴 BUG-4(Critical)· JIT kernel 名称碰撞 → 错误地形

**文件**: `crates/pumpkin-gpu/src/jit.rs`(全部 4 个 specialize 函数)、`src/noise/cache.rs`

**现象**: JIT kernel 名只含八度数(如 `octave_perlin_sample_f64_jit_m3`),而振幅/间隙/原点等常量烘焙在源码中。两个**八度数相同但内容不同**的采样器(不同种子 → 不同置换表/原点)在同一个设备上先后调用时,第二个采样器直接复用第一个的 kernel,输出**错误数值**。实测复现:采样器 b 的 JIT 输出哈希与 CPU 参考不等(修复前)。

**影响**: `jit_enabled=true` 时,所有「同八度列表、不同种子」的密度函数(如不同生物群系的 surface 噪声)都会生成错误地形,且无任何报错。

**修复**: `SerializedOctaveConfig::fingerprint()`(FNV-1a 64,覆盖置换表+全部八度参数),kernel 名追加指纹后缀 `_h{:016x}`。永久回归测试 `jit_same_octave_count_different_seeds_no_collision`。

### 🔴 BUG-5(Critical)· CUDA persistent kernel 死锁(启用即挂死服务器)

**文件**: `crates/pumpkin-gpu/kernels/cuda/light_propagate_persistent.cu`、`src/light.rs`

**现象**(原实现 5 处致命缺陷):
1. **所有线程**都执行 `atomicInc`(而非每 block 一线程)→ ticket 语义错乱;
2. 收敛判定只读**单个 block** 的 `block_changed`,其余 block 的变更被忽略;
3. barrier 后缺 `__syncthreads()`,线程提前进入下一迭代读写 `light[]`,与仍在自旋的块竞态;
4. 计数器**重置为 0** 与下一迭代的 `atomicInc` 竞态 → 自旋块永远看不到 0 → 死锁;
5. 仅最后到达的 block 知道收敛结果并退出,其余 block 进入下一迭代后因计数器无人再递增而**永久自旋**(修复过程中实测复现:测试进程挂死)。

**影响**: `persistent_kernels = true` 时服务器直接挂死。

**修复**: 完全重写——单调计数器栅栏(永不重置,每迭代以 `(iter+1)*num_blocks` 为目标值)+ 每 block 发布 `changed_flags` + `__threadfence` + barrier 后**每个 block 独立扫描全部 flags** 决定收敛(消除跨 block 收敛标志竞态)+ barrier 前后 `__syncthreads` 保持 lockstep。同时 `light.rs`:persistent 启动失败自动回退迭代式 GPU 路径(而非整体报错);签名移除已无用的 `convergence_flag`。新增测试 `tests/light_persistent_consistency.rs`(cooperative launch 实测通过,与 CPU 参考一致)。

### 🟡 BUG-6(Warning)· block_light_scan 源索引顺序不确定

**文件**: `crates/pumpkin-gpu/kernels/opencl/light_block.cl:13`、`cuda/light_block.cu`、`src/light.rs:86`

**现象**: GPU 用 `atomic_add` 收集光源索引,顺序非确定;CPU 回退按坐标递增 push。返回值 `Vec<i32>` 顺序不一致。

**影响**: **无正确性影响**——距离场传播的收敛结果与处理顺序无关(测试 `matrix_light_consistency` 与 `light_accel_consistency` 均哈希一致);仅返回列表顺序不同。建议:若调用方未来依赖顺序,先排序或改用稳定计数。未修改(避免破坏逐位一致性)。

### 🟡 BUG-7(Warning)· light_propagate 的 `1 + op` u8 溢出

**文件**: `crates/pumpkin-gpu/kernels/cuda/light_propagate.cu:17`、`opencl/light_propagate.cl:17`、`src/light.rs:269`

**现象**: `nl > 1 + op` 中 `1 + op` 为 u8 运算:opacity=255 时 CPU debug 构建 panic、GPU 回绕为 0。

**影响**: **无实际影响**——生产路径 opacity 恒为 0..=15(见 `light_accel` 调用方)。建议:长期改为 `nl > 16` 或 i32 运算并加注释。未修改。

### 🟡 BUG-8(Warning)· 一致性测试存在「假通过」模式

**文件**: `crates/pumpkin-world/tests/gpu_path_matrix.rs`、`crates/pumpkin-gpu/src/noise/batch_cell.rs:435`

**现象**: `BatchAccelerator` 在 GPU 启动失败时静默回退 CPU 并返回正确结果——「一致性 OK」无法区分「GPU 正确执行」与「GPU 失败回退 CPU」。BUG-1/2/3 正是被此掩盖多年。原单测 `aquifer_gpu_unavailable_returns_error` 依赖「GPU 不可用」假设,在有 GPU 机器上因 kernel 损坏而"通过"。

**修复**: 新增 `matrix_gpu_samplers_really_run`(采样器级直接调用并要求 `Ok`,覆盖 aquifer tiled/standard/水分支、beardifier、light propagate;CPU 设备时跳过);单测改为确定性 CPU 设备 `aquifer_cpu_device_returns_error`。

---

## 三、GPU vs CPU 逐函数审计结论(全部对齐)

对每个 kernel 与 CPU 权威实现做了逐行运算顺序对比(结合矩阵测试哈希验证):

| Kernel | CPU 参考 | 审计结论 |
|--------|---------|---------|
| `sample_no_fade_core`(perlin_core .cl/.cu) | `PerlinNoiseSampler::sample` | ✅ 置换表索引、8 梯度、fade、lerp 结合顺序逐位一致 |
| `octave_perlin_sample_f64` | `OctavePerlinNoiseSampler::sample` | ✅ `(amp*sample)*pers` 顺序一致 |
| `double_perlin_sample_f64` | `DoublePerlinNoiseSampler` | ✅ `maintain_precision(x*c*lac)` 单次应用一致 |
| `shift_a/b` | `NoiseAccelerator` CPU 回退 | ✅ `x*0.25`、y=0、`*4.0` 顺序一致 |
| `flatcache_precompute` | 同上 | ✅ |
| `trilinear_interpolate` | `cpu_trilinear` | ✅ 8 项结合顺序一致 |
| `beardifier_batch` | vanilla `Beardifier::sample` 等价 | ✅ 24³ 核表 + 0.8/0.4 因子逐位一致 |
| `aquifer_batch(_tiled)` | `cpu_aquifer_apply` | ✅ 4-NN 插入顺序/求和顺序/水分支一致(本轮修复) |
| `sky_light_fill` | `LightAccelerator::batch_sky_fill` | ✅ 饱和减法语义一致 |
| `block_light_scan` | 同上 | ✅ 值一致;源索引顺序不同(见 BUG-6) |
| `light_propagate(_persistent)` | `iterative_propagate` CPU 回退 | ✅ 迭代式与 persistent 均与 CPU 收敛结果一致 |
| `sky_light_horizontal` | CPU 回退 | ✅ 收敛不动点一致(更新顺序不同不影响收敛值) |

**JIT 五族**(octave/double/shift_a/shift_b/flatcache)在 CUDA 与 OpenCL 真实 GPU 内核上与 CPU 逐位一致(哈希断言,`jit_gpu_backend_all_families_bitwise_cpu`)。

---

## 四、cudarc vs opencl3 功能对齐对比

| 能力 | CUDA(cudarc 0.19.8) | OpenCL(opencl3 0.12) | 对齐 |
|------|:---:|:---:|:---:|
| 设备探测/上下文/流(队列) | ✅ | ✅ | ✅ |
| NVRTC / 运行时编译 | ✅ | ✅ | ✅ |
| 延迟编译(按需加载) | ✅ | ✅ | ✅ |
| kernel 清单(除 persistent) | ✅ 15 个 | ✅ 13 个 | ✅(编译期断言 `kernel_names_cuda_opencl_aligned`) |
| persistent/cooperative launch | ✅(本轮修复) | ❌ 不可行(无共驻留保证,详见前报告 §7.2) | 设计差异 |
| 动态 shared/__local 内存 | ✅ `shared_mem_bytes`(本轮补齐) | ✅ `set_arg_local_buffer`(本轮补齐) | ✅ |
| 参数类型 I32/F64/Buffer | ✅ | ✅ | ✅ |
| 参数类型 U32/USize | ✅ | ✅(本轮补齐) | ✅ |
| 切片参数 | ❌ 不支持 | ❌ 不支持 | ✅ 一致 |
| 多队列流水线 | ❌ 无 | ✅ `pipeline_queues` | 设计差异(单向) |
| 零拷贝阈值 | ⚠️ 配置占位未实现 | ❌ 无 | 设计差异 |
| JIT 特化编译 | ✅ CUDA C++ 方言(本轮修复) | ✅ OpenCL C 方言 | ✅ |
| f64 精度控制 | `--fmad=false --ftz=false`(NVRTC 支持) | `#pragma OPENCL FP_CONTRACT OFF`(NVRTC 不支持 CUDA 风格标志,前报告已分析) | ✅ 结果逐位一致 |
| 失败回退 | ✅ → batch → CPU | ✅ → batch → CPU | ✅ |

**缺失功能指出**: ① CUDA 零拷贝(`zero_copy_threshold_kb`)仅占位;② OpenCL 无多设备/集成显卡之外的调度策略(仅有 index/name/integrated 选择);③ CUDA 侧无等价于 `pipeline_queues` 的多流流水线。均为设计缺口,不影响正确性,建议列入后续优化。

---

## 五、性能优化建议(在保证地形一致的前提下)

1. **JIT 路径接入 buffer pool**(`batch_sampler.rs::sample_octave_jit` 等每次调用新建缓冲区)——OpenCL JIT 目前比 batch 慢 6%(5.20 vs 4.87ms)即因此;
2. **trilinear/aquifer/beardifier/光照路径接入 buffer pool**——高频小调用时分配开销占比高;
3. **aquifer 4-NN 排序网络**:top-4 插入排序可用固定 4 元素排序网络替代,减少分支;
4. **`sky_light_horizontal` 每迭代回读 changed 标志引入同步开销**——可考虑双缓冲 ping-pong 减少传输(仅 CUDA 侧可进一步用 persistent 风格);
5. **CUDA persistent kernel 现在可用**:小网格(≤ 共驻留上限)光照传播可省去每迭代主机往返(`persistent_kernels = true`),本轮已修复并验证;
6. **cuRAND 定位澄清**:当前 SplitMix64 实现未接入生产链路——建议在配置文档标注「不用于地形生成」或移除该配置项(见前报告)。

---

## 六、测试补全清单(本轮新增/加强)

| 测试 | 文件 | 覆盖 |
|------|------|------|
| `matrix_gpu_samplers_really_run` | `pumpkin-world/tests/gpu_path_matrix.rs` | aquifer tiled/standard/**水分支**、beardifier、light propagate 的 GPU 真实执行断言(防假通过) |
| `matrix_soa_layout_consistency` | 同上 | SoA 变体(此前零覆盖) |
| `jit_same_octave_count_different_seeds_no_collision` | `pumpkin-world/tests/jit_numerical_consistency.rs` | JIT 名称碰撞回归 |
| `persistent_propagate_matches_cpu` | `pumpkin-world/tests/light_persistent_consistency.rs`(新) | CUDA persistent kernel 真实运行 + CPU 一致性 |
| `aquifer_cpu_device_returns_error` | `pumpkin-gpu/src/noise/batch_cell.rs` | 确定性 CPU 设备错误路径(替代依赖无 GPU 假设的旧测试) |
| jit_tests 名称断言更新 | `pumpkin-gpu/tests/jit_tests.rs` | 指纹命名断言 |

---

## 七、回归验证结果

| 套件 | 结果 |
|------|------|
| 六组合矩阵(cpu/cuda/opencl × JIT 开/关,每组 8 测试) | ✅ 48/48 |
| `pumpkin-gpu --features gpu`(单元 + 集成) | ✅ 79 通过 |
| `pumpkin-world --features gpu` 全套 | ✅ 152 单元 + 21 个集成套件全通过 |
| OpenCL 后端:指纹/一致性/压力/矩阵套件 | ✅ 全通过 |
| CPU 后端:回退一致性/指纹/矩阵套件 | ✅ 全通过 |
| `cargo clippy --all-targets --all-features` | ✅ 0 error |
| `cargo fmt --all -- --check` / `typos` / `cargo-machete` | ✅ 全部干净 |
| `cargo check --all-targets --all-features` | ✅ 0 error |

**纯 CPU 生成路径影响评估**: 本轮所有改动仅涉及 `pumpkin-gpu`/GPU 测试与 `pumpkin-world` 的 GPU 加速层;`pumpkin-util` 的 CPU 噪声、`pumpkin-world` 的纯 CPU 生成路径(非 `--features gpu` 构建)未做任何语义修改。CPU 后端全路径测试与指纹测试(多种子、管线指纹、回退一致性)全部通过,证明 CPU 生成内容不受影响。

---

## 八、遗留建议(按优先级)

1. 将 BUG-6/7 的防御性修改(源索引排序、opacity 溢出防护)纳入后续清理,当前均无实际影响;
2. CUDA 零拷贝实现或移除配置占位;
3. 生产环境默认 `jit_enabled = false`(当前默认),启用后注意同八度不同种子采样器的 kernel 数量(指纹命名已避免碰撞,但每配置一个 kernel,极端多配置场景下编译缓存会增长);
4. `persistent_kernels = true` 仅建议在网格较小(≤ 共驻留上限)时启用,大网格自动回退迭代式(已实现)。
