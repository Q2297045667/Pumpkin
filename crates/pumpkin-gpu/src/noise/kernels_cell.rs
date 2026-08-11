//! GPU Kernel 实现 — Cell Cache、插值器、含水层、Beardifier、矿脉。
//!
//! 所有 Kernel 保证与 pumpkin-world CPU 路径逐位一致。
#![allow(clippy::needless_raw_string_hashes, clippy::too_many_lines)]

/// 批量 Cell Cache 填充：对每个 3D 位置计算密度，
/// 使用扁平化的 `component_stack` 参数和 `cell_indices` 映射。
///
/// 每个 work-item 处理一个位置，
/// 通过类似 `fill_from_stack` 的简化路径计算密度。
pub const CELL_CACHE_FILL_CL: &str = r##"
#pragma OPENCL EXTENSION cl_khr_f64 : enable

// Cell cache fill: 简化的 fill_from_stack 批量路径。
// component_stack 将各噪声配置展平为一维数组，每个配置包含：
//   [0]: num_octaves (以 double 存储)
//   [1..]: amplitudes / lacunarities / origins (交错排列)
// perms_data 存储所有 octave 的置换表 (每个 256 uchar)。
// cell_indices[i] 选择位置 i 应使用的 stack 配置。
__kernel void cell_cache_fill_f64(
    __global const double* pos,           // [N*3] 输入 3D 位置
    __global const double* component_stack, // 展平 stack 参数
    __global const uchar* perms_data,     // 置换表 (每个 octave 256 uchar)
    __global const int* cell_indices,     // [N] 每个位置的 stack 索引
    __global double* densities,           // [N] 输出密度
    int N,                                // 位置数量
    int config_stride,                    // 每个配置的 double 步长
    int amps_offset,                      // amplitudes 在配置内的偏移 (double)
    int lacs_offset,                      // lacunarities 偏移
    int orgs_offset                       // origins 偏移 (3 doubles per octave)
) {
    int i = get_global_id(0);
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    int ci = cell_indices[i];
    if (ci < 0) {
        densities[i] = 0.0;
        return;
    }

    int base = ci * config_stride;
    int num_octaves = (int)component_stack[base];
    if (num_octaves <= 0 || num_octaves > 64) {
        densities[i] = 0.0;
        return;
    }

    double sum = 0.0;
    for (int o = 0; o < num_octaves; o++) {
        double amp  = component_stack[base + amps_offset + o];
        double lac  = component_stack[base + lacs_offset + o];
        double orgx = component_stack[base + orgs_offset + o * 3];
        double orgy = component_stack[base + orgs_offset + o * 3 + 1];
        double orgz = component_stack[base + orgs_offset + o * 3 + 2];

        __global const uchar* perm = perms_data + o * 256;
        sum += amp * sample_no_fade_core(perm,
            orgx, orgy, orgz,
            maintain_precision(x * lac),
            maintain_precision(y * lac),
            maintain_precision(z * lac));
    }
    densities[i] = sum;
}
"##;

/// 批量插值器缓冲区填充：对 YZ 切片位置数组计算密度。
///
/// DAG 参数驱动插值器噪声配置；
/// 每个 work-item 处理一个 YZ 切片位置。
pub const INTERPOLATOR_FILL_CL: &str = r##"
#pragma OPENCL EXTENSION cl_khr_f64 : enable

// Interpolator buffer fill: 为每个 YZ 切片位置采样密度。
// dag_params 采用展平布局，每个 octave 8 个 double：
//   [amp, lac, org_x, org_y, org_z, xz_scale, y_scale, _reserved]
// pos 中 x 分量表示切片坐标，y/z 为单元格坐标。
__kernel void interpolator_fill_f64(
    __global const double* pos,           // [N*3] 位置
    __global const double* dag_params,    // DAG 噪声参数
    __global const uchar* perms_data,     // 置换数据
    __global double* densities,           // [N] 输出
    int N,
    int num_octaves
) {
    int i = get_global_id(0);
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    if (num_octaves <= 0) {
        densities[i] = 0.0;
        return;
    }

    double sum = 0.0;
    for (int o = 0; o < num_octaves; o++) {
        int bo = o * 8;
        double amp     = dag_params[bo];
        double lac     = dag_params[bo + 1];
        double orgx    = dag_params[bo + 2];
        double orgy    = dag_params[bo + 3];
        double orgz    = dag_params[bo + 4];
        double xz_scale = dag_params[bo + 5];
        double y_scale  = dag_params[bo + 6];

        __global const uchar* perm = perms_data + o * 256;
        sum += amp * sample_no_fade_core(perm,
            orgx, orgy, orgz,
            maintain_precision(x * xz_scale * lac),
            maintain_precision(y * y_scale  * lac),
            maintain_precision(z * xz_scale * lac));
    }
    densities[i] = sum;
}
"##;

/// 批量含水层判定：4-NN 搜索 + 屏障密度 + 流体判定。
///
/// 每个 work-item 处理一个块，
/// 返回 `block_state_id` (i32) 和 `should_schedule_fluid_update` (u8)。
pub const AQUIFER_BATCH_CL: &str = r##"
#pragma OPENCL EXTENSION cl_khr_f64 : enable

// Aquifer batch: 对每个查询位置执行 4 近邻搜索并确定流体状态。
// packed_positions: 预计算网格位置的展平数组 [M*3]。
// packed_densities: 对应网格位置的密度值 [M]。
// 对查询位置找到 4 个最近邻，计算屏障密度并判定是否为水体。
__kernel void aquifer_batch_f64(
    __global const double* pos,               // [N*3] 查询位置
    __global const double* packed_positions,  // [M*3] 网格位置
    __global const double* density_values,    // [N] 查询位置的密度输入
    __global const double* packed_densities,  // [M] 网格密度
    __global int* block_state_id,             // [N] block state 输出
    __global uchar* should_schedule,          // [N] fluid update 标志
    double fluid_level,                       // -10000.0 阈值典型值
    double barrier_scale,                     // 0.3 barrier 缩放典型值
    int N,                                    // 查询位置数
    int M                                     // 打包位置数
) {
    int i = get_global_id(0);
    if (i >= N) return;
    if (M < 4) {
        block_state_id[i] = 0;
        should_schedule[i] = 0;
        return;
    }

    double qx = pos[i * 3];
    double qy = pos[i * 3 + 1];
    double qz = pos[i * 3 + 2];
    double q_density = density_values[i];

    // 4 近邻线性搜索 (含水层网格通常较小)
    int best_idx[4];
    double best_dist[4];
    for (int k = 0; k < 4; k++) {
        best_idx[k]  = -1;
        best_dist[k] = INFINITY;
    }

    for (int j = 0; j < M; j++) {
        double dx = qx - packed_positions[j * 3];
        double dy = qy - packed_positions[j * 3 + 1];
        double dz = qz - packed_positions[j * 3 + 2];
        double dist = dx * dx + dy * dy + dz * dz;

        // 插入排序到 top-4
        for (int k = 0; k < 4; k++) {
            if (dist < best_dist[k]) {
                for (int m = 3; m > k; m--) {
                    best_idx[m]  = best_idx[m - 1];
                    best_dist[m] = best_dist[m - 1];
                }
                best_idx[k]  = j;
                best_dist[k] = dist;
                break;
            }
        }
    }

    // 屏障密度 = 4 个最近邻的平均值
    double barrier_sum = 0.0;
    int valid_nn = 0;
    for (int k = 0; k < 4; k++) {
        if (best_idx[k] >= 0 && best_idx[k] < M) {
            barrier_sum += packed_densities[best_idx[k]];
            valid_nn++;
        }
    }
    double barrier_density = (valid_nn > 0)
        ? barrier_sum / (double)valid_nn
        : 0.0;

    // 流体判定
    double effective = q_density + barrier_density * barrier_scale;

    if (effective > 0.0) {
        // 实心方块
        block_state_id[i]  = 1;   // 默认石头
        should_schedule[i] = 0;
    } else if (qy < fluid_level) {
        // 低于流体平面且非实心 → 水
        block_state_id[i]  = 2;   // 水
        should_schedule[i] = 1;
    } else {
        // 空气
        block_state_id[i]  = 0;
        should_schedule[i] = 0;
    }
}
"##;

/// 批量 Beardifier：对每个位置遍历结构和连接点，
/// 使用预计算的 24³ 核表累加 beard 贡献。
pub const BEARDIFIER_BATCH_CL: &str = r##"
#pragma OPENCL EXTENSION cl_khr_f64 : enable

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
"##;

/// 批量矿脉判定（独立于含水层）：
/// 对每个位置计算矿脉类型。
///
/// 返回值：0 = 无矿脉，1 = 矿石，2 = 粗矿，3 = 围岩。
pub const VEIN_BATCH_CL: &str = r##"
#pragma OPENCL EXTENSION cl_khr_f64 : enable

// Vein batch: 独立矿脉判定 — 通过 toggle/ridged/gap 三重噪声
// 确定每个位置是否属于矿脉以及矿脉类型。
// vein_noise_params:
//   每个矿脉 3 段噪声 (toggle, ridged, gap)，每段 octaves_per_vein 组参数。
//   每组 8 个 double: [amp, lac, org_x, org_y, org_z, xz_scale, y_scale, _reserved]
// vein_thresholds: [num_veins * 3] → (toggle_thr, ridged_thr, gap_thr)
__kernel void vein_batch_f64(
    __global const double* pos,               // [N*3] 位置
    __global const double* vein_noise_params, // 展平矿脉噪声参数
    __global const uchar* perms_data,         // 置换表
    __global const double* vein_thresholds,   // [num_veins * 3] 各矿脉阈值
    __global const double* vein_weights,      // [num_veins] 权重乘数
    __global int* vein_types,                 // [N] 0=none 1=ore 2=raw_ore 3=stone
    int N,
    int num_veins,
    int octaves_per_vein                      // 每段噪声的 octave 数
) {
    int i = get_global_id(0);
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    int best_type      = 0;
    double best_weight = -INFINITY;

    int stride_per_segment = octaves_per_vein * 8;  // 每段的 double 数
    int stride_per_vein    = stride_per_segment * 3;  // toggle + ridged + gap

    for (int v = 0; v < num_veins; v++) {
        // ---- 三段噪声 ----
        // toggle 段
        int toggle_base = v * stride_per_vein;
        double toggle = 0.0;
        for (int o = 0; o < octaves_per_vein; o++) {
            int po = toggle_base + o * 8;
            double amp   = vein_noise_params[po];
            double lac   = vein_noise_params[po + 1];
            double orgx  = vein_noise_params[po + 2];
            double orgy  = vein_noise_params[po + 3];
            double orgz  = vein_noise_params[po + 4];
            __global const uchar* perm = perms_data + (v * octaves_per_vein * 3 + o) * 256;
            toggle += amp * sample_no_fade_core(perm,
                orgx, orgy, orgz,
                maintain_precision(x * lac),
                maintain_precision(y * lac),
                maintain_precision(z * lac));
        }

        // ridged 段
        int ridged_base = toggle_base + stride_per_segment;
        double ridged = 0.0;
        for (int o = 0; o < octaves_per_vein; o++) {
            int po = ridged_base + o * 8;
            double amp   = vein_noise_params[po];
            double lac   = vein_noise_params[po + 1];
            double orgx  = vein_noise_params[po + 2];
            double orgy  = vein_noise_params[po + 3];
            double orgz  = vein_noise_params[po + 4];
            __global const uchar* perm = perms_data + (v * octaves_per_vein * 3 + octaves_per_vein + o) * 256;
            double sample = sample_no_fade_core(perm,
                orgx, orgy, orgz,
                maintain_precision(x * lac),
                maintain_precision(y * lac),
                maintain_precision(z * lac));
            ridged += amp * (1.0 - fabs(sample));
        }

        // gap 段
        int gap_base = toggle_base + stride_per_segment * 2;
        double gap = 0.0;
        for (int o = 0; o < octaves_per_vein; o++) {
            int po = gap_base + o * 8;
            double amp   = vein_noise_params[po];
            double lac   = vein_noise_params[po + 1];
            double orgx  = vein_noise_params[po + 2];
            double orgy  = vein_noise_params[po + 3];
            double orgz  = vein_noise_params[po + 4];
            __global const uchar* perm = perms_data + (v * octaves_per_vein * 3 + octaves_per_vein * 2 + o) * 256;
            gap += amp * sample_no_fade_core(perm,
                orgx, orgy, orgz,
                maintain_precision(x * lac),
                maintain_precision(y * lac),
                maintain_precision(z * lac));
        }

        // ---- 阈值判定 ----
        int vt = v * 3;
        double thr_toggle = vein_thresholds[vt];
        double thr_ridged = vein_thresholds[vt + 1];
        double thr_gap    = vein_thresholds[vt + 2];
        double weight     = vein_weights[v];

        if (toggle > thr_toggle && ridged > thr_ridged && gap > thr_gap) {
            double combined = weight * (toggle + ridged + gap);
            if (combined > best_weight) {
                best_weight = combined;

                // 分级判定
                if (ridged > thr_ridged * 1.6) {
                    best_type = 2;  // raw_ore
                } else if (toggle > thr_toggle * 1.3) {
                    best_type = 1;  // ore
                } else {
                    best_type = 3;  // stone (围岩)
                }
            }
        }
    }

    vein_types[i] = best_type;
}
"##;
