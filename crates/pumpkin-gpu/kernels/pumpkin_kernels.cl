// == Pumpkin GPU Kernels (OpenCL) ==
// Target: OpenCL 3.0
// Precision: double (f64)
//
// 编译命令 (使用 clspv 或 clang):
//   clspv pumpkin_kernels.cl -o pumpkin_kernels.spv
// 或在线编译:
//   clBuildProgram(program, ...)
//
// 注意: OpenCL C 要求 double 支持需启 cl_khr_fp64 扩展。
// 运行时需检查设备是否支持 double 精度。

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// === Perlin 噪声辅助函数 ===
// 梯度计算
static inline double grad(int hash, double x, double y, double z) {
    int h = hash & 15;
    double u = h < 8 ? x : y;
    double v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// 平滑插值
static inline double fade(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// === Perlin 噪声批量采样 ===
// global_size: N (1D)
__kernel void perlin_sample_f64(
    __global const double* positions,   // N × 3
    __global const uchar* permutations, // 512
    __global const double* origins,     // 3
    __global double* results,           // N
    int N
) {
    int idx = get_global_id(0);
    if (idx >= N) return;

    double x = positions[idx * 3] + origins[0];
    double y = positions[idx * 3 + 1] + origins[1];
    double z = positions[idx * 3 + 2] + origins[2];

    // 整数部分和小数部分
    int xi = (int)floor(x);
    int yi = (int)floor(y);
    int zi = (int)floor(z);
    double xf = x - xi;
    double yf = y - yi;
    double zf = z - zi;

    // 排列表索引
    xi &= 255;
    yi &= 255;
    zi &= 255;

    // 8 个角点的排列值
    int aaa = permutations[permutations[permutations[xi] + yi] + zi];
    int aba = permutations[permutations[permutations[xi] + yi + 1] + zi];
    int aab = permutations[permutations[permutations[xi] + yi] + zi + 1];
    int abb = permutations[permutations[permutations[xi] + yi + 1] + zi + 1];
    int baa = permutations[permutations[permutations[xi + 1] + yi] + zi];
    int bba = permutations[permutations[permutations[xi + 1] + yi + 1] + zi];
    int bab = permutations[permutations[permutations[xi + 1] + yi] + zi + 1];
    int bbb = permutations[permutations[permutations[xi + 1] + yi + 1] + zi + 1];

    // 平滑权重
    double u = fade(xf);
    double v = fade(yf);
    double w = fade(zf);

    // 三线性插值
    double result = mix(
        mix(mix(grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf), u),
            mix(grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf), u), v),
        mix(mix(grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1), u),
            mix(grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1), u), v),
        w
    );

    results[idx] = result;
}

// === 三线性插值批量处理 ===
// global_size: M (1D)
__kernel void trilinear_interpolate_f64(
    __global const double* cell_corners, // M × 8
    __global const double* deltas,       // M × 3
    __global double* results,            // M
    int M
) {
    int idx = get_global_id(0);
    if (idx >= M) return;

    int base = idx * 8;
    double c000 = cell_corners[base];
    double c100 = cell_corners[base + 1];
    double c010 = cell_corners[base + 2];
    double c110 = cell_corners[base + 3];
    double c001 = cell_corners[base + 4];
    double c101 = cell_corners[base + 5];
    double c011 = cell_corners[base + 6];
    double c111 = cell_corners[base + 7];

    double dx = deltas[idx * 3];
    double dy = deltas[idx * 3 + 1];
    double dz = deltas[idx * 3 + 2];

    // 展开的三线性插值
    double result = c000 * (1 - dx) * (1 - dy) * (1 - dz)
                  + c100 * dx * (1 - dy) * (1 - dz)
                  + c010 * (1 - dx) * dy * (1 - dz)
                  + c110 * dx * dy * (1 - dz)
                  + c001 * (1 - dx) * (1 - dy) * dz
                  + c101 * dx * (1 - dy) * dz
                  + c011 * (1 - dx) * dy * dz
                  + c111 * dx * dy * dz;

    results[idx] = result;
}

// === 光照传播单步迭代 ===
// global_size: N (1D)
__kernel void light_propagate_u8(
    __global uchar* light,        // N
    __global const uchar* opacity, // N
    __global const int* neighbors, // N × 6
    __global uchar* changed,       // 1
    int N
) {
    int idx = get_global_id(0);
    if (idx >= N) return;

    uchar current = light[idx];
    uchar op = opacity[idx];

    // 从 6 个邻居中取传播后的最大值
    uchar best = current;
    for (int d = 0; d < 6; d++) {
        int n = neighbors[idx * 6 + d];
        if (n < 0 || n >= N) continue;  // 边界检查
        uchar n_light = light[n];
        // 邻居传播：距离衰减 1 + 透明度衰减
        uchar propagated = (n_light > (uchar)(1 + op))
            ? (uchar)(n_light - (uchar)(1 + op))
            : 0;
        best = max(best, propagated);
    }

    if (best > current) {
        light[idx] = best;
        *changed = 1;  // 标记有变化
    }
}
