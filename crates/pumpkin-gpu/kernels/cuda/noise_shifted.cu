extern "C" __global__ void shifted_noise_sample_f64(
    const double* pos,
    const unsigned char* perms, const double* amps, const double* lacs,
    const double* orgs, const double* shifts,
    double xz_scale, double y_scale, int use_per_sample_shift,
    double* res, int N, int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    double sx, sy, sz;
    if (use_per_sample_shift) {
        sx = shifts[i*3]; sy = shifts[i*3+1]; sz = shifts[i*3+2];
    } else {
        sx = shifts[0]; sy = shifts[1]; sz = shifts[2];
    }
    double sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        sum += amps[o] * sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision((x*xz_scale + sx)*lac),
            maintain_precision((y*y_scale + sy)*lac),
            maintain_precision((z*xz_scale + sz)*lac));
    }
    res[i] = sum;
}
