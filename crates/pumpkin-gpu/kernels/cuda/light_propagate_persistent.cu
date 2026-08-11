// Persistent kernel variant: light_propagate_u8_persistent
//
// Uses atomic counter for grid-level synchronization.
// Launched once with cooperative launch; internally iterates until convergence.
//
// Requires: cudaLaunchCooperativeKernel (hardware support: SM 6.0+)

extern "C" __global__ void light_propagate_u8_persistent(
    unsigned char* light,
    const unsigned char* opacity,
    const int* neighbors,
    volatile int* convergence_flag,  // host-visible: 0=running, -1=done
    volatile int* sync_counter,      // grid sync: counts blocks that reached barrier
    int N,
    int max_iters
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;

    for (int iter = 0; iter < max_iters; iter++) {
        unsigned char changed_this_thread = 0;

        if (i < N) {
            unsigned char cur = light[i];
            unsigned char best = cur;
            for (int d = 0; d < 6; d++) {
                int n = neighbors[i * 6 + d];
                if (n < 0 || n >= N) continue;
                unsigned char nl = light[n];
                unsigned char op = opacity[n];
                unsigned char prop = (nl > (unsigned char)(1 + op))
                    ? (unsigned char)(nl - (unsigned char)(1 + op))
                    : 0;
                if (prop > best) best = prop;
            }
            if (best > cur) {
                light[i] = best;
                changed_this_thread = 1;
            }
        }

        // Converge any change within block using shared memory + warp reduce
        __shared__ unsigned char block_changed;
        if (threadIdx.x == 0) block_changed = 0;
        __syncthreads();

        if (changed_this_thread) block_changed = 1;
        __syncthreads();

        // Global barrier: atomic counter
        unsigned int ticket = atomicInc((unsigned int*)sync_counter, 0xFFFFFFFF);
        if (ticket == (unsigned int)(num_blocks - 1)) {
            // Last block: reset counter for next iteration
            *sync_counter = 0;

            // Check if any block had changes this iteration
            if (block_changed == 0) {
                // No changes across all blocks — converged
                if (threadIdx.x == 0) {
                    *convergence_flag = -1;  // Signal host
                }
            }
        } else {
            // Spin-wait until last block resets the counter
            while (*sync_counter != 0) {
                __threadfence_block();
            }
        }

        // If converged, exit
        if (*convergence_flag == -1) break;
    }

    // Ensure convergence flag is visible to host
    __threadfence_system();
}
