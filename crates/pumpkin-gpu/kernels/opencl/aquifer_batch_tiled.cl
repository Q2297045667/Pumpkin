// aquifer_batch_tiled.cl - Tiled variant with local memory
// 如果 M <= 2048 使用此 kernel，否则回退到原始 aquifer_batch.cl
//
// 语义与 aquifer_batch_f64 逐位一致：
// - 4-NN 选择与插入顺序（严格小于、先到先得）相同；
// - 屏障密度 = sum(4 NN 密度) / 4（求和顺序 k=0..3 与标准 kernel 一致）；
// - 判定：effective > 0 → 实心(1)；否则 qy < fluid_level → 水(2)；否则空气(0)；
// - M < 4 → 输出全 0。
// 参数顺序与主机端一致（与 aquifer_batch_f64 完全相同），尾部追加两个
// __local 缓冲区（主机通过 set_arg_local_buffer 设置大小）。

__kernel void aquifer_batch_tiled_f64(
    __global const double* pos,               // [N*3] 查询位置
    __global const double* packed_positions,  // [M*3] 网格位置
    __global const double* density_values,    // [N] 查询位置的密度输入
    __global const double* packed_densities,  // [M] 网格密度
    __global int* block_state_id,             // [N] block state 输出
    __global uchar* should_schedule,          // [N] fluid update 标志
    double fluid_level,                       // -10000.0 阈值典型值
    double barrier_scale,                     // 0.3 barrier 缩放典型值
    int N,                                    // 查询位置数
    int M,                                    // 打包位置数
    // 动态 local memory 参数（由 host 端设置大小）
    __local double* tile_positions,
    __local double* tile_densities
) {
    int i = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // Phase 1: 协作加载 packed 数据到 local memory
    for (int j = lid * 3; j < M * 3; j += group_size * 3) {
        if (j < M * 3) tile_positions[j] = packed_positions[j];
        if (j + 1 < M * 3) tile_positions[j + 1] = packed_positions[j + 1];
        if (j + 2 < M * 3) tile_positions[j + 2] = packed_positions[j + 2];
    }
    for (int j = lid; j < M; j += group_size) {
        tile_densities[j] = packed_densities[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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

    // Phase 2: 4-NN 搜索（从 local memory 读取，与标准 kernel 相同的插入逻辑）
    int best_idx[4];
    double best_dist[4];
    for (int k = 0; k < 4; k++) {
        best_idx[k]  = -1;
        best_dist[k] = INFINITY;
    }

    for (int j = 0; j < M; j++) {
        double dx = qx - tile_positions[j * 3];
        double dy = qy - tile_positions[j * 3 + 1];
        double dz = qz - tile_positions[j * 3 + 2];
        double dist = dx * dx + dy * dy + dz * dz;

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

    // Phase 3: 屏障密度 = 4 个最近邻的平均值（与标准 kernel 相同的顺序与除法）
    double barrier_sum = 0.0;
    int valid_nn = 0;
    for (int k = 0; k < 4; k++) {
        if (best_idx[k] >= 0 && best_idx[k] < M) {
            barrier_sum += tile_densities[best_idx[k]];
            valid_nn++;
        }
    }
    double barrier_density = (valid_nn > 0)
        ? barrier_sum / (double)valid_nn
        : 0.0;

    // 流体判定（与标准 kernel 逐位一致）
    double effective = q_density + barrier_density * barrier_scale;

    if (effective > 0.0) {
        block_state_id[i]  = 1;
        should_schedule[i] = 0;
    } else if (qy < fluid_level) {
        block_state_id[i]  = 2;
        should_schedule[i] = 1;
    } else {
        block_state_id[i]  = 0;
        should_schedule[i] = 0;
    }
}
