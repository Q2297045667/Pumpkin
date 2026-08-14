# GPU 模块基准报告规范(CPU / 内存 / GPU)

> 本目录存放 Pumpkin 世界生成 GPU 加速模块(noise / batch / light 三大加速器,后端 CUDA / OpenCL / CPU)的基准、压力、指纹与一致性测试报告。

---

## 1. 文件命名规范

格式:**`CPU{CPU型号}_{内存}GB_{GPU型号}_{系统}_{日期}.md`**

| 段 | 含义 | 示例 |
|----|------|------|
| CPU 段 | CPU 型号(去空格、去商标符号) | `XeonW3345`、`i9-14900K` |
| 内存段 | 物理内存容量(GB) | `32GB`、`64GB` |
| GPU 段 | GPU 型号(去空格);无独显/纯 CPU 环境写 `NoGPU` | `TeslaT10`、`RTX4090` |
| 系统段 | 操作系统 + 主版本 | `Windows10`、`Linux6.8` |
| 日期段 | 报告日期 `YYYY-MM-DD`,同日多份追加序号 | `2026-08-13`、`2026-08-13-2` |

示例(合规):

```
CPU_XeonW3345_32GB_TeslaT10_Windows10_2026-08-13.md
```

---

## 2. 报告章节模板

每份报告必须包含以下章节(顺序固定)。表格列含义见 §3。

### 2.1 硬件配置(必填)

```markdown
## 硬件配置

- **CPU**: {型号} @ {频率}
- **内存**: {容量} GB
- **GPU**: {厂商 型号}({架构});无则填「无(纯 CPU)」
- **显存**: {容量} MiB;无则填「无」
- **种子**: `{固定种子}`(全部测试使用同一固定种子,保证可复现)
- **时间**: {YYYY-MM-DD HH:MM}
```

### 2.2 测试环境(必填)

```markdown
## 测试环境

- **CUDA 版本**: {nvidia-smi 的 CUDA UMD Version};无 GPU 填「无」
- **OpenCL 版本**: {驱动对应 OpenCL 版本,注明来源}
- **操作系统**: {完整名称与位数}
- **内核版本**: {内核/built 号,Windows 用 `10.0.xxxxx`}
- **驱动版本**: {显卡驱动版本}
- **构建**: debug/release profile,feature 清单(必须含 `--features gpu`)
- **缓存状态**: 是否执行 `cargo clean`,清理范围与耗时
```

### 2.3 测试总览

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|

- 汇总所有套件通过数与矩阵组合数(例:矩阵 48/48、gpu 套件 78/78)。
- 必须含两行状态:**JIT 启动**(有无失败/跳过)、**功能模块启动**(有无失败)。

### 2.4 基准测试

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|

- 稳态数据(预热后,排除首次内核编译),来源 `matrix_perf`(`gpu_path_matrix.rs`)。
- 噪声类至少覆盖:octave 262k、double_perlin 65k、trilinear 131k。

### 2.5 噪声采样

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|

- 首次调用延迟(全新进程、含编译),来源 `worldgen_perf.rs`。
- 含编译的条目必须加注(`† = 含首次 NVRTC/OpenCL 编译`)。

### 2.6 压力测试

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|

- 来源 `worldgen_stress.rs`,逐条列出(大输入/极端坐标/边界/重复调用/大光照传播)。

### 2.7 一致性测试

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|

- 逐函数 CPU vs GPU 对比(哈希/逐位断言),覆盖噪声五族 + trilinear + aquifer + beardifier + cell_cache + SoA + 多种子 + 管线指纹。

### 2.8 指纹

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|

- 指纹套件汇总:`gpu_noise_fingerprint(_full)`、`light_fingerprint`、`batch_fingerprint`、`worldgen_fingerprint`、`worldgen_pipeline_fingerprint`。

### 2.9 光照引擎

| 项目 | CPU | GPU | 延迟 | 性能 | 结果 |
|------|-----|-----|------|------|------|

- sky_fill / block_scan / iterative_propagate(含 persistent)/ sky_horizontal,注明与 CPU 收敛结果一致性。

### 2.10 JIT 编译

| 项目 | CPU | GPU | 预热 | 性能 | 延迟 |
|------|-----|-----|------|------|------|

- CUDA 与 OpenCL 的 JIT 特化稳态性能 + 首次编译延迟对比;给出「该后端是否建议开 JIT」的结论。

### 2.11 JIT 一致性

| 项目 | CPU | GPU | 预热 | 性能 | 延迟 |
|------|-----|-----|------|------|------|

- `jit_numerical_consistency`、`gpu_backend_alignment`、`matrix_jit_path` 结果;JIT 五族与 batch/CPU 逐位一致;名称碰撞回归。

### 2.12 执行摘要与结论

- 总通过率、JIT/模块启动状态、`pumpkin.toml` 是否改动、CPU 生成内容影响声明、遗留建议列表。

---

## 3. 表格列含义

| 列 | 含义 |
|----|------|
| 项目 | 测试项名称,格式 `模块_规模`(如 `八度噪声 262k`) |
| CPU | CPU 参考/CPU 后端的结果:耗时(ms)或 ✅/❌ |
| GPU | GPU 端结果:耗时(ms),多后端写 `CUDA x / OpenCL y` |
| 结果 | `通过` / `一致`(哈希一致)/ `逐位一致` / `失败(原因)` |
| 性能 | 加速比 = CPU 耗时 ÷ GPU 耗时(`66.4x`);小负载写 `1.0x(无收益)` |
| 延迟 | GPU 端到端延迟(ms);一致性类无延迟写 `—` |
| 预热 | 仅 JIT 表:`冷启动`(含首次编译)或 `已预热`(稳态) |

---

## 4. 测试执行流程(标准步骤)

1. **清理缓存**(必做,保证基准不测到旧产物):

   ```sh
   cargo clean
   ```

2. **全路径矩阵**(cpu/cuda/opencl × JIT 开/关,共 6 组;捕获稳态性能):

   ```sh
   for backend in cuda opencl cpu; do
     for jit in 0 1; do
       PUMPKIN_GPU_BACKEND=$backend PUMPKIN_GPU_JIT=$jit \
         cargo test -p pumpkin-world --features gpu \
         --test gpu_path_matrix -- --nocapture --test-threads=1
     done
   done
   ```

3. **首次调用基准**(同 6 组,含编译开销):

   ```sh
   PUMPKIN_GPU_BACKEND=$backend PUMPKIN_GPU_JIT=$jit \
     cargo test -p pumpkin-world --features gpu \
     --test worldgen_perf -- --nocapture --test-threads=1
   ```

4. **压力 + 指纹 + 光照 + 多种子 + 基准**:

   ```sh
   cargo test -p pumpkin-world --features gpu \
     --test worldgen_stress --test worldgen_fingerprint \
     --test gpu_noise_fingerprint --test gpu_noise_fingerprint_full \
     --test light_fingerprint --test batch_fingerprint \
     --test worldgen_pipeline_fingerprint --test worldgen_bench \
     --test worldgen_light_gpu_consistency --test worldgen_multi_seed_consistency \
     -- --test-threads=1
   ```

5. **GPU crate 全套**:

   ```sh
   cargo test -p pumpkin-gpu --features gpu
   ```

> 环境变量:`PUMPKIN_GPU_BACKEND` = `auto|cuda|opencl|cpu`;`PUMPKIN_GPU_JIT` = `0|1`。
> 测试固定种子:`138_782_381_985_206`(`gpu_path_matrix.rs` / `worldgen_perf.rs` 内 `SEED` 常量)。

---

## 5. Release 构建流程(GPU 模块优化二进制)

### 5.1 构建命令

```sh
cargo build --release -p pumpkin --features gpu
```

- feature 链:`pumpkin` 的 `gpu = ["dep:pumpkin-gpu", "pumpkin-gpu/gpu", "pumpkin-world/gpu", "pumpkin-config/gpu"]`;
- 优化配置(workspace `Cargo.toml` 的 `[profile.release]`):`lto = true`、`codegen-units = 1`、`strip = "debuginfo"`;
- 产物:`target/release/pumpkin.exe`(Windows);调试符号在独立 `.pdb`。

### 5.2 产物记录字段(报告可选附注)

| 字段 | 示例 |
|------|------|
| 构建耗时 | 42m 56s |
| 二进制大小 | 84,128,768 字节 |
| PDB 大小 | 8,524,160 字节 |
| 告警 | 仅第三方 `proc-macro-error2` future-incompat 提示 |

### 5.3 GPU 链接验证(必做)

构建后必须确认 GPU 内核源码与配置键确实嵌入二进制(`strings`/`findstr` 对本 PE 不可靠,使用 PowerShell 字节扫描):

```sh
powershell.exe -NoProfile -Command '$b=[System.IO.File]::ReadAllBytes("target\release\pumpkin.exe"); $t=[System.Text.Encoding]::ASCII.GetString($b); foreach($s in @("aquifer_batch_tiled_f64","light_propagate_u8_persistent","octave_perlin_sample_soa_f64","sky_light_horizontal_propagate_u8","flatcache_precompute_f64","cuMemHostAlloc","use_curand","persistent_kernels")){ if($t.Contains($s)){"FOUND: $s"}else{"MISSING: $s"} }'
```

全部输出 `FOUND` 才算验证通过。

### 5.4 注意事项

- 服务器二进制无 CLI 参数,启动即绑定端口并生成 world/配置目录——**不要在仓库根目录冒烟运行**,部署目录中运行;
- CUDA/OpenCL 驱动为运行时动态加载,运行机器需安装对应驱动(如 NVIDIA 610.47);
- GPU 功能默认关闭(`[gpu] enabled = false`),部署时按需在 `pumpkin.toml` 开启;JIT/零拷贝/cuRAND 开关语义见各测试报告。

---

## 6. 系统信息采集命令

```sh
# GPU 型号 / 显存 / 驱动
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# CUDA UMD 版本(输出头部)
nvidia-smi
# 操作系统 / 内核版本 / 内存(Windows,经 PowerShell)
powershell.exe -NoProfile -Command \
  'Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture; (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'
```

> OpenCL 版本无法直接由 nvidia-smi 获得;按驱动版本对应关系填写并注明来源(如「驱动 610.47 → OpenCL 3.0,NVIDIA 465+ 驱动为 OpenCL 3.0」)。若环境有 `clinfo`,以 `clinfo` 实测为准。

---

## 7. 失败处理与验收标准

### 7.1 失败处理

| 情形 | 处理 |
|------|------|
| JIT 启动失败 | **直接跳过**,在报告「测试总览」中记录「JIT 启动失败,已跳过(原因)」 |
| 功能模块启动失败 | 分析原因,并将分析结果输出到本目录 `{报告同名}_FAILURE_ANALYSIS.md` |
| 一致性/指纹断言失败 | 定位 kernel 与 CPU 路径差异(运算顺序/平局语义/编译优化),修复后重跑全部矩阵 |

### 7.2 验收清单(报告发布前逐项确认)

- [ ] 文件名符合 §1 命名规范
- [ ] 硬件配置 / 测试环境字段全部填写(§2.1、§2.2)
- [ ] 六组合矩阵全部通过(48/48),JIT 无跳过(或跳过已记录)
- [ ] 基准表区分「稳态(预热)」与「首次调用(含编译)」并加注
- [ ] 所有一致性结论标注粒度:通过 / 一致 / 逐位一致
- [ ] 性能列写加速比,延迟列写 GPU 端到端延迟
- [ ] CPU 生成内容影响声明(纯 CPU 路径指纹测试通过)
- [ ] 遗留建议按优先级列出

---

## 8. 附录:测试套件 → 报告章节映射

| 测试套件 | 报告章节 |
|----------|---------|
| `gpu_path_matrix`(matrix_perf/matrix_device_report) | 基准测试、测试总览 |
| `worldgen_perf` | 噪声采样、基准测试(首调) |
| `worldgen_stress` | 压力测试 |
| `noise_accel_consistency` / `light_accel_consistency` / `worldgen_light_gpu_consistency` / `worldgen_multi_seed_consistency` | 一致性测试 |
| `gpu_noise_fingerprint(_full)` / `light_fingerprint` / `batch_fingerprint` / `worldgen_fingerprint` / `worldgen_pipeline_fingerprint` | 指纹 |
| `light_persistent_consistency` | 光照引擎 |
| `jit_numerical_consistency` / `gpu_backend_alignment` / `matrix_jit_path` | JIT 编译、JIT 一致性 |
| `worldgen_bench` / `gpu_features` / `gpu_pipeline_integration` | 测试总览、基准测试 |

---

## 9. 区块格式 GPU 加速选型规范

> 本规范固化 GPU 加速模块的区块存储格式选型决策、评估维度与验收标准。
> 分析对象:`crates/pumpkin-world/src/chunk/format/` 下的三种格式实现(anvil / linear / pump)。

### 9.1 适用范围与边界

本规范约束磁盘 I/O 侧的 GPU 加速能力,包括:

- **批量区块加载**:服务器启动加载、流式加载的吞吐优化;
- **序列化管线**:区块保存路径的 GPU 化。

不在本规范范围内(但列为前置条件,见 §9.6):

- 当前三大 GPU 加速器(noise / batch / light)的计算逻辑——它们操作内存中的 `ChunkData`,与磁盘格式无关;
- 内存侧 palette 展开(`HeterogeneousPaletteData` 索引 → 全局 `BlockStateId`)与光照定长数组的 SoA 上传,属上传前置条件,不属格式选型。

### 9.2 候选格式

| 格式 | 实现文件 | 扩展名 | 定位 |
|------|---------|--------|------|
| Anvil | `crates/pumpkin-world/src/chunk/format/anvil.rs` | `.mca` | 原版兼容 |
| Linear v2 | `crates/pumpkin-world/src/chunk/format/linear.rs` | `.linear` | 连续紧凑布局 |
| Pump | `crates/pumpkin-world/src/chunk/format/pump.rs` | `.pump` | 自研 NBT 包裹 |

### 9.3 结构对比

| 维度 | Anvil | Linear v2 | Pump |
|------|-------|-----------|------|
| 头部/索引 | 固定 8 KiB(1024×u32 扇区偏移 + 1024×u32 时间戳) | 26B 超级块 + 128B 位图 + 变长特性字典(恒空) + N×13B 桶表 | 无索引,整文件为 NBT 复合标签 |
| 数据布局 | 每 chunk 独立压缩,4 KiB 扇区对齐,散布 | 桶数据紧密连续,尾部 8B 签名 | chunk 以字节数组内嵌于 NBT 树 |
| 压缩 | 每 chunk 可选 zlib / gzip / lz4 / 不压缩(异构) | 每桶独立 zstd 流,桶数由 grid_size 决定(默认 4,最大 1024) | 每 chunk 独立 zstd |
| 校验 | 无 | xxhash64(每桶) | 无 |
| 版本/时间戳 | 时间戳表 | 版本字节 + 每 chunk 时间戳 | 无 |
| 随机访问 | O(1) 扇区偏移 | 桶偏移 = 桶表前缀和 | 必须先串行解析整个 NBT |

### 9.4 评估维度(选型必须全部满足)

| # | 维度 | 要求 |
|---|------|------|
| 1 | 可并行性 | 至少 1 个 GPU work-item 对应 1 个独立压缩流 |
| 2 | 访存模式 | 整块连续上传 + 合并访问(coalesced);禁止 scatter/gather |
| 3 | 索引可计算性 | 数据偏移可由定长头部 + 并行前缀和(scan)计算;禁止串行走树 |
| 4 | 编解码器 GPU 亲和度 | 块化压缩格式(zstd);禁止以强串行熵解码(zlib/gzip)为主路径 |
| 5 | 校验并行性 | 校验和可 SIMD/GPU 并行计算(xxhash64) |

### 9.5 逐格式结论

#### Anvil(`.mca`)—— 定级:兼容保留,不入选

- ✅ 每 chunk 独立压缩,并行度天然 1024;8 KiB 头部支持 O(1) 随机访问;
- ❌ 4 KiB 扇区对齐导致 chunk 散布,无法整块连续上传,只能 scatter-gather;
- ❌ 压缩算法每 chunk 异构(zlib/gzip/lz4/不压缩),zlib/gzip 的 Huffman + LZ77 反向引用为强串行算法,GPU 效率极差;仅 LZ4 chunk 具备 GPU 解压价值;
- ❌ 4 KiB 填充浪费 PCIe 传输带宽。
- 处理:**保留只读兼容**,GPU 管线仅对 LZ4 chunk 生效。

#### Linear v2(`.linear`)—— 定级:入选(GPU 管线目标格式)

- ✅ 定长头部(26B + 128B + 13B×N) + 连续桶数据,整文件一次连续读入/上传,零填充浪费;
- ✅ 桶偏移 = 桶表并行前缀和,GPU 原生数据并行模式;
- ✅ 每桶独立 zstd 流 → 桶级并行;`grid_size` 上调至 32×32(1024 桶)即每 chunk 一桶,并行度与 Anvil 相同且保持数据连续;
- ✅ zstd 块化结构适合 GPU 解码(如 nvCOMP 路线),现有 `ruzstd` 纯 Rust 实现替换接口干净;
- ✅ xxhash64 校验可整块搬上 GPU 并行验证;
- ✅ 现有 `write()`/`read()` 的 `spawn_blocking` 桶循环(`linear.rs` 写路径 L410、读路径 L523)即天然内核边界,替换成本最低;
- ⚠️ 默认 `grid_size = 2`(4 桶)并行度不足,须按 §9.7 调整。

#### Pump(`.pump`)—— 定级:弃用

- ❌ 整文件 NBT 复合标签,须先串行走树解析(tag 类型字节 + 变长字符串键 + 递归结构),GPU 无法数据并行;
- ❌ 无索引/固定头部,无 O(1) 偏移,无 prefix-scan 可做;chunk 键为十进制字符串(`"0"`..`"1023"`),需逐条 string→int;
- ❌ 无校验和、无版本字节、无时间戳,GPU 管线无法完整性验证,格式演进困难;
- ✅ 仅每 chunk 独立 zstd 可并行解压,但解析瓶颈已堵死 GPU 路径。
- 处理:**弃用或仅用于小型存档**,不进入 GPU 管线。

### 9.6 选型决策

1. **GPU 管线目标格式 = Linear v2(`.linear`)**;
2. Anvil 保留只读兼容(原版存档互通),GPU 路径仅覆盖 LZ4 chunk;
3. Pump 格式退出 GPU 规划,保留现有 CPU 读写代码直至弃用;
4. 内存侧前置条件(不属本规范,但 GPU 上传前必须满足):
   - palette 索引在 CPU 侧展开为全局 `BlockStateId`;
   - 光照定长数组沿用 SoA 布局上传(与 `light_accel` 现有路径一致)。

### 9.7 落地要求与验收标准

| # | 要求 | 验收 |
|---|------|------|
| 1 | `linear.rs` 默认 `grid_size` 提升至 32(每 chunk 一桶) | 新世界默认 1024 桶,写放大可接受 |
| 2 | 新增 `pumpkin-gpu::chunk_io` 模块:zstd 解压 + xxhash64 校验内核 | `--features gpu` 构建通过,内核字符串嵌入二进制 |
| 3 | 替换 `linear.rs` 写路径(L410)与读路径(L523)的 `spawn_blocking` 桶循环为 GPU 管线,CPU 回退路径保留 | CPU/GPU 双路径均通过现有 roundtrip 测试 |
| 4 | 一致性:GPU 解压结果与 CPU(`ruzstd`)逐位一致;xxhash 校验结果一致 | 指纹/逐位断言套件通过 |
| 5 | 回退策略:GPU 不可用、桶损坏、grid_size 非法时回退 CPU 路径 | 与 `noise_accel` / `light_accel` 回退语义一致 |
| 6 | 基准报告新增「区块格式」章节,记录批量加载吞吐(CPU vs GPU 加速比) | 报告含 §2.4 格式的基准表 |
