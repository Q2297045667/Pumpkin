extern "C" __global__ void trilinear_interpolate_f64(
    const double* corners,
    const double* deltas,
    double* results, int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= M) return;
    int b = i * 8;
    double c000=corners[b], c100=corners[b+1], c010=corners[b+2], c110=corners[b+3];
    double c001=corners[b+4], c101=corners[b+5], c011=corners[b+6], c111=corners[b+7];
    double dx=deltas[i*3], dy=deltas[i*3+1], dz=deltas[i*3+2];
    results[i] = c000*(1-dx)*(1-dy)*(1-dz) + c100*dx*(1-dy)*(1-dz)
               + c010*(1-dx)*dy*(1-dz) + c110*dx*dy*(1-dz)
               + c001*(1-dx)*(1-dy)*dz + c101*dx*(1-dy)*dz
               + c011*(1-dx)*dy*dz + c111*dx*dy*dz;
}
