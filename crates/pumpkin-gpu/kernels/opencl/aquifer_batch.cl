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
