// Beardifier batch: accumulate terrain adaptation contributions from structures and junctions.
// structures: [num_structures * 9] →
//   (center_x, center_y, center_z,
//    radius_x, radius_y, radius_z,
//    min_y, ground_delta_y, max_y)
// junctions: [num_junctions * 3] → (x, y, z)
// beard_kernel: precomputed 24³ kernel table (f64)
extern "C" __global__ void beardifier_batch_f64(
    const double* pos,                      // [N*3] positions
    const double* beard_kernel,             // [K*K*K] precomputed kernel (K=24)
    const double* structures,               // [num_structures * 9] structure bounding boxes
    const double* junctions,                // [num_junctions * 3] junction points
    const int* structure_to_junction,       // [num_structures] junction index per structure
    double* beard_values,                   // [N] beard contribution output
    int N,
    int num_structures,
    int num_junctions,
    int kernel_size,                        // 24
    double kernel_scale                     // 1.0 / kernel_size, for coordinate mapping
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    double beard = 0.0;
    int ks  = kernel_size;
    int ks2 = ks * ks;
    double ksh = (double)ks * 0.5;           // kernel half-size

    // ---- Structure contributions ----
    for (int s = 0; s < num_structures; s++) {
        int sb = s * 9;
        double cx = structures[sb];
        double cy = structures[sb + 1];
        double cz = structures[sb + 2];
        double rx = structures[sb + 3];
        double ry = structures[sb + 4];
        double rz = structures[sb + 5];
        double min_y = structures[sb + 6];
        double ground_dy = structures[sb + 7];
        double max_y = structures[sb + 8];

        if (rx <= 0.0 || ry <= 0.0 || rz <= 0.0) continue;

        // Map world coordinates to kernel coordinates
        double kx = (x - cx) * kernel_scale / rx + ksh;
        double ky = (y - cy) * kernel_scale / ry + ksh;
        double kz = (z - cz) * kernel_scale / rz + ksh;

        if (kx < 0.0 || kx >= (double)(ks - 1) ||
            ky < 0.0 || ky >= (double)(ks - 1) ||
            kz < 0.0 || kz >= (double)(ks - 1)) {
            continue;
        }

        int ix = (int)floor(kx);
        int iy = (int)floor(ky);
        int iz = (int)floor(kz);

        // Boundary clamping
        if (ix < 0) ix = 0; if (ix > ks - 2) ix = ks - 2;
        if (iy < 0) iy = 0; if (iy > ks - 2) iy = ks - 2;
        if (iz < 0) iz = 0; if (iz > ks - 2) iz = ks - 2;

        double fx = kx - (double)ix;
        double fy = ky - (double)iy;
        double fz = kz - (double)iz;

        // Trilinear sampling
        double c000 = beard_kernel[ ix      * ks2 +  iy      * ks +  iz     ];
        double c100 = beard_kernel[(ix + 1) * ks2 +  iy      * ks +  iz     ];
        double c010 = beard_kernel[ ix      * ks2 + (iy + 1) * ks +  iz     ];
        double c110 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks +  iz     ];
        double c001 = beard_kernel[ ix      * ks2 +  iy      * ks + (iz + 1)];
        double c101 = beard_kernel[(ix + 1) * ks2 +  iy      * ks + (iz + 1)];
        double c011 = beard_kernel[ ix      * ks2 + (iy + 1) * ks + (iz + 1)];
        double c111 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks + (iz + 1)];

        double v00 = c000 + fx * (c100 - c000);
        double v10 = c010 + fx * (c110 - c010);
        double v01 = c001 + fx * (c101 - c001);
        double v11 = c011 + fx * (c111 - c011);
        double v0  = v00  + fy * (v10  - v00 );
        double v1  = v01  + fy * (v11  - v01 );
        double val = v0   + fz * (v1   - v0  );

        // Height-constrained contribution weight
        double y_contrib = 1.0;
        if (y < min_y) {
            y_contrib = 0.0;
        } else if (y < min_y + ground_dy && ground_dy > 0.0) {
            y_contrib = (y - min_y) / ground_dy;
        }
        beard += val * y_contrib * kernel_scale;
    }

    // ---- Junction contributions ----
    for (int j = 0; j < num_junctions; j++) {
        int jb = j * 3;
        double jx = junctions[jb];
        double jy = junctions[jb + 1];
        double jz = junctions[jb + 2];

        double dx = x - jx;
        double dy = y - jy;
        double dz = z - jz;

        // Junction radius = 12 blocks (Minecraft default)
        double jr = 12.0;
        double kx = dx * kernel_scale / jr + ksh;
        double ky = dy * kernel_scale / jr + ksh;
        double kz = dz * kernel_scale / jr + ksh;

        if (kx < 0.0 || kx >= (double)(ks - 1) ||
            ky < 0.0 || ky >= (double)(ks - 1) ||
            kz < 0.0 || kz >= (double)(ks - 1)) {
            continue;
        }

        int ix = (int)floor(kx);
        int iy = (int)floor(ky);
        int iz = (int)floor(kz);

        if (ix < 0) ix = 0; if (ix > ks - 2) ix = ks - 2;
        if (iy < 0) iy = 0; if (iy > ks - 2) iy = ks - 2;
        if (iz < 0) iz = 0; if (iz > ks - 2) iz = ks - 2;

        double fx = kx - (double)ix;
        double fy = ky - (double)iy;
        double fz = kz - (double)iz;

        double c000 = beard_kernel[ ix      * ks2 +  iy      * ks +  iz     ];
        double c100 = beard_kernel[(ix + 1) * ks2 +  iy      * ks +  iz     ];
        double c010 = beard_kernel[ ix      * ks2 + (iy + 1) * ks +  iz     ];
        double c110 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks +  iz     ];
        double c001 = beard_kernel[ ix      * ks2 +  iy      * ks + (iz + 1)];
        double c101 = beard_kernel[(ix + 1) * ks2 +  iy      * ks + (iz + 1)];
        double c011 = beard_kernel[ ix      * ks2 + (iy + 1) * ks + (iz + 1)];
        double c111 = beard_kernel[(ix + 1) * ks2 + (iy + 1) * ks + (iz + 1)];

        double v00 = c000 + fx * (c100 - c000);
        double v10 = c010 + fx * (c110 - c010);
        double v01 = c001 + fx * (c101 - c001);
        double v11 = c011 + fx * (c111 - c011);
        double v0  = v00  + fy * (v10  - v00 );
        double v1  = v01  + fy * (v11  - v01 );
        double val = v0   + fz * (v1   - v0  );

        beard += val * kernel_scale;
    }

    beard_values[i] = beard;
}
