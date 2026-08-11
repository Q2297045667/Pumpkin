extern "C" __global__ void octave_perlin_sample_soa_f64(
    const double* pos_x,
    const double* pos_y,
    const double* pos_z,
    const unsigned char* perms,
    const double* amps,
    const double* lacs,
    const double* orgs,
    double* res,
    int N,
    int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double x = pos_x[i], y = pos_y[i], z = pos_z[i], sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        sum += amps[o] * sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), maintain_precision(y*lac), maintain_precision(z*lac));
    }
    res[i] = sum;
}
