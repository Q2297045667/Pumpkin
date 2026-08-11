// aquifer_batch_tiled.cl - Tiled variant with local memory
// 如果 M <= 2048 使用此 kernel，否则回退到原始 aquifer_batch.cl

__kernel void aquifer_batch_tiled_f64(
    __global const double* pos,
    __global const double* densities,
    __global const double* packed_positions,  // [M*3]
    __global const double* packed_densities,  // [M]
    __global int* block_ids,
    __global uchar* fluid_updates,
    double fluid_level,
    double barrier_scale,
    int N,
    int M,
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

    double qx = pos[i * 3], qy = pos[i * 3 + 1], qz = pos[i * 3 + 2];
    double q_density = densities[i];

    // Phase 2: 4-NN 搜索（从 local memory 读取）
    int best_idx[4] = {0, 0, 0, 0};
    double best_dist[4] = {INFINITY, INFINITY, INFINITY, INFINITY};

    for (int j = 0; j < M; j++) {
        double dx = qx - tile_positions[j * 3];
        double dy = qy - tile_positions[j * 3 + 1];
        double dz = qz - tile_positions[j * 3 + 2];
        double dist = dx * dx + dy * dy + dz * dz;

        // 插入排序到 top-4
        for (int k = 0; k < 4; k++) {
            if (dist < best_dist[k]) {
                for (int kk = 3; kk > k; kk--) {
                    best_idx[kk] = best_idx[kk - 1];
                    best_dist[kk] = best_dist[kk - 1];
                }
                best_idx[k] = j;
                best_dist[k] = dist;
                break;
            }
        }
    }

    // Phase 3: 屏障计算（与原始相同）
    double barrier = 0.0;
    for (int k = 0; k < 4; k++) {
        if (best_dist[k] < 1e12) {
            barrier += tile_densities[best_idx[k]];
        }
    }
    barrier = barrier * barrier_scale / 4.0;

    if (q_density + barrier > 0.0) {
        block_ids[i] = 0;
    } else {
        block_ids[i] = (int)(fluid_level);
    }
    fluid_updates[i] = (uchar)((q_density + barrier <= 0.0) ? 1 : 0);
}
