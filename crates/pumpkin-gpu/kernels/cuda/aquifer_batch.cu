// Aquifer batch: 4-nearest-neighbor search for each query position and fluid state determination.
// packed_positions: flattened precomputed grid positions [M*3].
// packed_densities: corresponding grid density values [M].
// For each query position find the 4 nearest neighbors, compute barrier density and determine water status.
extern "C" __global__ void aquifer_batch_f64(
    const double* pos,                      // [N*3] query positions
    const double* packed_positions,         // [M*3] grid positions
    const double* density_values,           // [N] density input for query positions
    const double* packed_densities,         // [M] grid densities
    int* block_state_id,                    // [N] block state output
    unsigned char* should_schedule,         // [N] fluid update flag
    double fluid_level,                     // -10000.0 typical threshold
    double barrier_scale,                   // 0.3 typical barrier scale
    int N,                                  // number of query positions
    int M                                   // number of packed positions
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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

    // 4-NN linear search (aquifer grids are typically small)
    int best_idx[4];
    // NVRTC 不定义 INFINITY 宏；用远大于任何现实坐标距离平方的有限值代替。
    // （网格坐标为小整数，dist = dx²+dy²+dz² 远小于 1e300）
    double best_dist[4];
    for (int k = 0; k < 4; k++) {
        best_dist[k] = 1.0e300;
    }

    for (int j = 0; j < M; j++) {
        double dx = qx - packed_positions[j * 3];
        double dy = qy - packed_positions[j * 3 + 1];
        double dz = qz - packed_positions[j * 3 + 2];
        double dist = dx * dx + dy * dy + dz * dz;

        // Insertion sort into top-4
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

    // Barrier density = average of 4 nearest neighbors
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

    // Fluid determination
    double effective = q_density + barrier_density * barrier_scale;

    if (effective > 0.0) {
        // Solid block
        block_state_id[i]  = 1;   // default stone
        should_schedule[i] = 0;
    } else if (qy < fluid_level) {
        // Below fluid plane and non-solid → water
        block_state_id[i]  = 2;   // water
        should_schedule[i] = 1;
    } else {
        // Air
        block_state_id[i]  = 0;
        should_schedule[i] = 0;
    }
}
