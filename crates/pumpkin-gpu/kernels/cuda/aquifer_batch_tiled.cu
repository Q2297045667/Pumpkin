// aquifer_batch_tiled.cu - Tiled variant with shared memory
// Use this kernel if M <= 2048, otherwise fall back to original aquifer_batch.cu

extern "C" __global__ void aquifer_batch_tiled_f64(
    const double* pos,
    const double* densities,
    const double* packed_positions,         // [M*3]
    const double* packed_densities,         // [M]
    int* block_ids,
    unsigned char* fluid_updates,
    double fluid_level,
    double barrier_scale,
    int N,
    int M
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

    double qx = pos[i * 3], qy = pos[i * 3 + 1], qz = pos[i * 3 + 2];
    double q_density = densities[i];

    // Phase 2: 4-NN search (read from shared memory)
    int best_idx[4] = {0, 0, 0, 0};
    double best_dist[4] = {INFINITY, INFINITY, INFINITY, INFINITY};

    for (int j = 0; j < M; j++) {
        double dx = qx - tile_positions[j * 3];
        double dy = qy - tile_positions[j * 3 + 1];
        double dz = qz - tile_positions[j * 3 + 2];
        double dist = dx * dx + dy * dy + dz * dz;

        // Insertion sort into top-4
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

    // Phase 3: Barrier calculation (same as original)
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
    fluid_updates[i] = (unsigned char)((q_density + barrier <= 0.0) ? 1 : 0);
}
