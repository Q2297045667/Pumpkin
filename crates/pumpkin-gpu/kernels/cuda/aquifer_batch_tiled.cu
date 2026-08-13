// aquifer_batch_tiled.cu - Tiled variant with shared memory
// Use this kernel if M <= 2048, otherwise fall back to original aquifer_batch.cu
//
// Semantics are bitwise-identical to aquifer_batch_f64:
// - 4-NN selection / insertion order (strictly-less, first-come) identical;
// - barrier density = sum(4 NN densities) / 4 (summation order k=0..3 identical);
// - decision: effective > 0 → solid(1); else qy < fluid_level → water(2); else air(0);
// - M < 4 → all-zero output.
// Parameter order identical to the host side / aquifer_batch_f64, followed by
// dynamic shared memory (extern __shared__, size set via LaunchConfig.shared_mem_bytes).

extern "C" __global__ void aquifer_batch_tiled_f64(
    const double* pos,                       // [N*3] query positions
    const double* packed_positions,          // [M*3] grid positions
    const double* density_values,            // [N] query density inputs
    const double* packed_densities,          // [M] grid densities
    int* block_state_id,                     // [N] block state output
    unsigned char* should_schedule,          // [N] fluid update flag
    double fluid_level,                      // -10000.0 typical threshold
    double barrier_scale,                    // 0.3 typical barrier scale
    int N,                                   // query count
    int M                                    // packed position count
) {
    // Dynamic shared memory allocation
    extern __shared__ double shared_data[];
    double* tile_positions = shared_data;
    double* tile_densities = shared_data + M * 3;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int group_size = blockDim.x;

    // Phase 1: Cooperative load of packed data into shared memory
    for (int j = lid * 3; j < M * 3; j += group_size * 3) {
        if (j < M * 3) tile_positions[j] = packed_positions[j];
        if (j + 1 < M * 3) tile_positions[j + 1] = packed_positions[j + 1];
        if (j + 2 < M * 3) tile_positions[j + 2] = packed_positions[j + 2];
    }
    for (int j = lid; j < M; j += group_size) {
        tile_densities[j] = packed_densities[j];
    }
    __syncthreads();

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

    // Phase 2: 4-NN search (from shared memory; insertion logic identical to standard kernel)
    // NVRTC 不定义 INFINITY 宏；用远大于任何现实坐标距离平方的有限值代替。
    int best_idx[4];
    double best_dist[4];
    for (int k = 0; k < 4; k++) {
        best_idx[k]  = -1;
        best_dist[k] = 1.0e300;
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

    // Phase 3: Barrier density = mean of 4 nearest neighbors (same order & division)
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

    // Fluid decision (bitwise identical to standard kernel)
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
