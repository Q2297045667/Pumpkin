// == Pumpkin GPU Kernels ==
// Target: NVIDIA CUDA (PTX)
// Precision: f64 (double)
//
// 编译命令 (使用 nvcc):
//   nvcc -ptx -arch=sm_70 -o pumpkin_kernels.ptx pumpkin_kernels.cu
//
// 注意: 此文件仅作为参考，实际编译会在 build.rs 中通过 NVRTC 完成。
// 所有 Kernel 使用 double (f64) 以保证与 Vanilla Minecraft 的一致性。

extern "C" {

// === Perlin 噪声批量采样 ===
// 对 N 个 (x, y, z) 坐标并行计算 Perlin 噪声值。
// grid: ceil(N / 256), block: 256
// positions: [N * 3]f64 — 交错存储的 (x, y, z) 坐标
// permutations: [512]u8 — 排列表（通常 2×256）
// origins: [3]f64 — (x_origin, y_origin, z_origin)
// results: [N]f64 — 输出噪声值
__global__ void perlin_sample_f64(
    const double* positions,
    const unsigned char* permutations,
    const double* origins,
    double* results,
    int N
);

// === 三线性插值批量处理 ===
// 对 M 个 cell 并行执行三线性插值。
// grid: ceil(M / 256), block: 256
// cell_corners: [M * 8]f64 — 每个 cell 的 8 个角点密度值
// deltas: [M * 3]f64 — 每个 cell 的 (dx, dy, dz) 归一化坐标 (0..1)
// results: [M]f64 — 输出插值结果
__global__ void trilinear_interpolate_f64(
    const double* cell_corners,
    const double* deltas,
    double* results,
    int M
);

// === 光照距离场迭代 ===
// 一次迭代：并行更新所有节点的光等级。
// grid: ceil(N / 256), block: 256
// light: [N]u8 — 输入/输出光等级
// opacity: [N]u8 — 对应节点的透明度
// neighbors: [N * 6]i32 — 6 个邻居索引 (-1 表示边界外)
// changed: [1]u8 — 输出：是否有任何节点发生变化
// N: 节点总数
__global__ void light_propagate_u8(
    unsigned char* light,
    const unsigned char* opacity,
    const int* neighbors,
    unsigned char* changed,
    int N
);

} // extern "C"
