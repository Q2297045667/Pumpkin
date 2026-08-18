// Sky light horizontal/cascade propagation kernel (CUDA).
// See OpenCL version for detailed comments.
extern "C" __global__ void sky_light_horizontal_propagate_u8(
    unsigned char* sky_light,            // [width * height * depth] 3D sky light values
    const unsigned char* opacity,        // [width * height * depth] opacity values
    int* changed,                        // [1] convergence flag (atomicOr)
    int width,                            // X dimension
    int depth,                            // Z dimension
    int height                            // Y dimension
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int x = linear % width;
    int z = linear / width;
    if (x >= width || z >= depth) return;

    int base = z * (width * height) + x * height;
    int stride_z = width * height;
    int stride_x = height;

    for (int y = height - 1; y >= 0; y--) {
        int idx = base + y;
        unsigned char cur = sky_light[idx];
        unsigned char best = cur;

        // 1. Horizontal propagation: 4 cardinal neighbors
        int neighbor_idxs[4];
        int has_neighbor[4];

        has_neighbor[0] = (x > 0);
        neighbor_idxs[0] = base - stride_x + y;

        has_neighbor[1] = (x < width - 1);
        neighbor_idxs[1] = base + stride_x + y;

        has_neighbor[2] = (z > 0);
        neighbor_idxs[2] = base - stride_z + y;

        has_neighbor[3] = (z < depth - 1);
        neighbor_idxs[3] = base + stride_z + y;

        for (int d = 0; d < 4; d++) {
            if (!has_neighbor[d]) continue;
            unsigned char nl = sky_light[neighbor_idxs[d]];
            if (nl > 1) {
                unsigned char prop = nl - 1;
                if (prop > best) best = prop;
            }
        }

        // 2. Downward cascade
        if (y < height - 1) {
            unsigned char light_above = sky_light[idx + 1];
            if (light_above == 15 && opacity[idx] == 0) {
                if (15 > best) best = 15;
            }
        }

        if (best > cur) {
            sky_light[idx] = best;
            atomicOr(changed, 1);
        }
    }
}
