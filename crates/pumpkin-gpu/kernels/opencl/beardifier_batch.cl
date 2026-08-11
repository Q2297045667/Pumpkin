// Beardifier batch: 累加来自结构与连接点的地形适应量。
// structures: [num_structures * 9] →
//   (center_x, center_y, center_z,
//    radius_x, radius_y, radius_z,
//    min_y, ground_delta_y, max_y)
// junctions: [num_junctions * 3] → (x, y, z)
// beard_kernel: 预计算 24³ 核表 (f64)
__kernel void beardifier_batch_f64(
    __global const double* pos,              // [N*3] 位置
    __global const double* beard_kernel,     // [K*K*K] 预计算核 (K=24)
    __global const double* structures,       // [num_structures * 9] 结构包围盒
    __global const double* junctions,        // [num_junctions * 3] 连接点
    __global const int* structure_to_junction, // [num_structures] 每个结构对应的连接点索引
    __global double* beard_values,           // [N] beard 贡献输出
    int N,
    int num_structures,
    int num_junctions,
    int kernel_size,                         // 24
    double kernel_scale                      // 1.0 / kernel_size, 用于坐标映射
) {
    int i = get_global_id(0);
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    double beard = 0.0;
    int ks  = kernel_size;
    int ks2 = ks * ks;
    double ksh = (double)ks * 0.5;           // kernel half-size

    // ---- 结构贡献 ----
    for (int s = 0; s < num_structures; s++) {
        int sb = s * 9;
        double cx = structures[sb];
        double cy = structures[sb + 1];
        double cz = structures[sb + 2];
        double rx = structures[sb + 3];
        double ry = structures[sb + 4];
        double rz = structures[sb + 5];
        double min_y = structures[sb + 6];
        double ground_dy = structures[sb + 7];
        double max_y = structures[sb + 8];

        if (rx <= 0.0 || ry <= 0.0 || rz <= 0.0) continue;

        // 将世界坐标映射到核坐标
        double kx = (x - cx) * kernel_scale / rx + ksh;
        double ky = (y - cy) * kernel_scale / ry + ksh;
        double kz = (z - cz) * kernel_scale / rz + ksh;

        if (kx < 0.0 || kx >= (double)(ks - 1) ||
            ky < 0.0 || ky >= (double)(ks - 1) ||
            kz < 0.0 || kz >= (double)(ks - 1)) {
            continue;
        }

        int ix = (int)floor(kx);
        int iy = (int)floor(ky);
        int iz = (int)floor(kz);

        // 边界钳制
        if (ix < 0) ix = 0; if (ix > ks - 2) ix = ks - 2;
        if (iy < 0) iy = 0; if (iy > ks - 2) iy = ks - 2;
        if (iz < 0) iz = 0; if (iz > ks - 2) iz = ks - 2;

        double fx = kx - (double)ix;
        double fy = ky - (double)iy;
        double fz = kz - (double)iz;

        // 三线性采样
        double c000 = beard_kernel[ ix      * ks2 +  iy      * ks +  iz     ];
        double c100 = beard_kernel[(ix + 1) * ks2 +  iy      * ks +  iz     ];
        double c010 = beard_kernel[ ix      * ks2 + (iy + 1) * ks +  iz     ];
        double c110 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks +  iz     ];
        double c001 = beard_kernel[ ix      * ks2 +  iy      * ks + (iz + 1)];
        double c101 = beard_kernel[(ix + 1) * ks2 +  iy      * ks + (iz + 1)];
        double c011 = beard_kernel[ ix      * ks2 + (iy + 1) * ks + (iz + 1)];
        double c111 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks + (iz + 1)];

        double v00 = c000 + fx * (c100 - c000);
        double v10 = c010 + fx * (c110 - c010);
        double v01 = c001 + fx * (c101 - c001);
        double v11 = c011 + fx * (c111 - c011);
        double v0  = v00  + fy * (v10  - v00 );
        double v1  = v01  + fy * (v11  - v01 );
        double val = v0   + fz * (v1   - v0  );

        // 含高度约束的贡献权重
        double y_contrib = 1.0;
        if (y < min_y) {
            y_contrib = 0.0;
        } else if (y < min_y + ground_dy && ground_dy > 0.0) {
            y_contrib = (y - min_y) / ground_dy;
        }
        beard += val * y_contrib * kernel_scale;
    }

    // ---- 连接点贡献 ----
    for (int j = 0; j < num_junctions; j++) {
        int jb = j * 3;
        double jx = junctions[jb];
        double jy = junctions[jb + 1];
        double jz = junctions[jb + 2];

        double dx = x - jx;
        double dy = y - jy;
        double dz = z - jz;

        // 连接点半径 = 12 格 (Minecraft 默认)
        double jr = 12.0;
        double kx = dx * kernel_scale / jr + ksh;
        double ky = dy * kernel_scale / jr + ksh;
        double kz = dz * kernel_scale / jr + ksh;

        if (kx < 0.0 || kx >= (double)(ks - 1) ||
            ky < 0.0 || ky >= (double)(ks - 1) ||
            kz < 0.0 || kz >= (double)(ks - 1)) {
            continue;
        }

        int ix = (int)floor(kx);
        int iy = (int)floor(ky);
        int iz = (int)floor(kz);

        if (ix < 0) ix = 0; if (ix > ks - 2) ix = ks - 2;
        if (iy < 0) iy = 0; if (iy > ks - 2) iy = ks - 2;
        if (iz < 0) iz = 0; if (iz > ks - 2) iz = ks - 2;

        double fx = kx - (double)ix;
        double fy = ky - (double)iy;
        double fz = kz - (double)iz;

        double c000 = beard_kernel[ ix      * ks2 +  iy      * ks +  iz     ];
        double c100 = beard_kernel[(ix + 1) * ks2 +  iy      * ks +  iz     ];
        double c010 = beard_kernel[ ix      * ks2 + (iy + 1) * ks +  iz     ];
        double c110 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks +  iz     ];
        double c001 = beard_kernel[ ix      * ks2 +  iy      * ks + (iz + 1)];
        double c101 = beard_kernel[(ix + 1) * ks2 +  iy      * ks + (iz + 1)];
        double c011 = beard_kernel[ ix      * ks2 + (iy + 1) * ks + (iz + 1)];
        double c111 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks + (iz + 1)];

        double v00 = c000 + fx * (c100 - c000);
        double v10 = c010 + fx * (c110 - c010);
        double v01 = c001 + fx * (c101 - c001);
        double v11 = c011 + fx * (c111 - c011);
        double v0  = v00  + fy * (v10  - v00 );
        double v1  = v01  + fy * (v11  - v01 );
        double val = v0   + fz * (v1   - v0  );

        beard += val * kernel_scale;
    }

    beard_values[i] = beard;
}
