// Sky light horizontal/cascade propagation kernel.
// Operates on an already vertically-filled 3D sky_light array [width * height * depth].
// Each thread processes one (x, z) column:
//   1. Checks 4 horizontal neighbors at each Y level, propagating light with attenuation=1.
//   2. Cascades light downward: if position above has light 15 and current block is air, keep 15.
// This kernel is called iteratively until convergence (changed flag).
__kernel void sky_light_horizontal_propagate_u8(
    __global uchar* sky_light,       // [width * height * depth] 3D sky light values
    __global const uchar* opacity,   // [width * height * depth] opacity values
    __global int* changed,           // [1] convergence flag (atomic_or)
    int width,                        // X dimension (typically 18 for 16+2 borders)
    int depth,                        // Z dimension (typically 18)
    int height                        // Y dimension
) {
    int linear = get_global_id(0);
    int x = linear % width;
    int z = linear / width;
    if (x >= width || z >= depth) return;

    int base = z * (width * height) + x * height;
    int stride_z = width * height;   // north/south step
    int stride_x = height;           // west/east step

    // Process top-down (Y descending) so downward cascade propagates correctly within a single pass
    for (int y = height - 1; y >= 0; y--) {
        int idx = base + y;
        uchar cur = sky_light[idx];
        uchar best = cur;

        // 1. Horizontal propagation: check 4 cardinal neighbors at same Y
        int neighbor_idxs[4];
        int has_neighbor[4];

        // West (x - 1)
        has_neighbor[0] = (x > 0);
        neighbor_idxs[0] = base - stride_x + y;

        // East (x + 1)
        has_neighbor[1] = (x < width - 1);
        neighbor_idxs[1] = base + stride_x + y;

        // North (z - 1)
        has_neighbor[2] = (z > 0);
        neighbor_idxs[2] = base - stride_z + y;

        // South (z + 1)
        has_neighbor[3] = (z < depth - 1);
        neighbor_idxs[3] = base + stride_z + y;

        for (int d = 0; d < 4; d++) {
            if (!has_neighbor[d]) continue;
            uchar nl = sky_light[neighbor_idxs[d]];
            if (nl > 1) {
                uchar prop = nl - 1;
                if (prop > best) best = prop;
            }
        }

        // 2. Downward cascade: if position above has light 15 and current is air, keep 15
        //    (position above = y+1 = idx + 1)
        if (y < height - 1) {
            uchar light_above = sky_light[idx + 1];
            if (light_above == 15 && opacity[idx] == 0) {
                if (15 > best) best = 15;
            }
        }

        if (best > cur) {
            sky_light[idx] = best;
            atomic_or(changed, 1);
        }
    }
}
