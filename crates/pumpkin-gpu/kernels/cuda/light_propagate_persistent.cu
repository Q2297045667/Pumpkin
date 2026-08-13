// Persistent kernel variant: light_propagate_u8_persistent
//
// Launched once with cooperative launch; internally iterates until convergence.
// Requires: cudaLaunchCooperativeKernel (hardware support: SM 6.0+).
//
// Grid barrier design (corrected):
// - Each block reduces its changes into `block_changed` (shared memory) and
//   publishes it to `changed_flags[blockIdx.x]` with a `__threadfence()`.
// - A monotonic atomic counter implements the grid barrier (no reset — a reset
//   would race with the next iteration's atomicInc and deadlock spinning blocks).
//   After iteration `iter` the counter equals (iter+1) * num_blocks.
// - After the barrier, EVERY block independently scans `changed_flags` to decide
//   convergence. This avoids a cross-block "convergence flag" whose write could
//   race with blocks exiting their spin (the earlier design deadlocked exactly
//   there: the last-arriving block alone decided convergence and exited while
//   the others, not yet seeing the flag, entered the next iteration and spun
//   forever on a counter that nobody would ever increment again).
// - `__syncthreads()` after the barrier keeps each block in lockstep so no
//   thread starts the next iteration (and writes light[]) while other blocks
//   are still reading it.

extern "C" __global__ void light_propagate_u8_persistent(
    unsigned char* light,
    const unsigned char* opacity,
    const int* neighbors,
    volatile int* sync_counter,           // grid sync: counts blocks that reached barrier
    unsigned char* changed_flags,         // [num_blocks] per-block change flags
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

        // Block-level reduction of changes
        __shared__ unsigned char block_changed;
        if (threadIdx.x == 0) block_changed = 0;
        __syncthreads();
        if (changed_this_thread) block_changed = 1;
        __syncthreads();

        // Publish block change flag (device-scope visible to all blocks)
        if (threadIdx.x == 0) {
            changed_flags[blockIdx.x] = block_changed;
        }
        __threadfence();
        __syncthreads();

        // Grid barrier: only thread 0 of each block participates.
        // Monotonic counter design (no reset, see header comment).
        if (threadIdx.x == 0) {
            unsigned int target = (unsigned int)(iter + 1) * (unsigned int)num_blocks;
            unsigned int ticket = atomicInc((unsigned int*)sync_counter, 0xFFFFFFFF);
            if (ticket != target - 1) {
                // Spin until every block of this iteration has arrived
                while (*sync_counter < target) {
                    // volatile read via the sync_counter pointer
                }
            }
        }
        __syncthreads();

        // Every block decides convergence independently: the barrier guarantees
        // all blocks' flags were written (flag write + fence precedes each
        // block's atomicInc, and the spin exits only after the last arrival).
        __shared__ unsigned char converged_shared;
        if (threadIdx.x == 0) {
            __threadfence();
            int any_changed = 0;
            for (int b = 0; b < num_blocks; b++) {
                any_changed |= changed_flags[b];
            }
            converged_shared = (any_changed == 0) ? 1 : 0;
        }
        __syncthreads();

        if (converged_shared) break;
    }

    // Ensure final light[] writes are visible to the host before kernel exit
    __threadfence_system();
}
