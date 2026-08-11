extern "C" __global__ void flatcache_precompute_f64(
    const double* pos,
    const unsigned char* perms,
    const double* amps, const double* lacs,
    const double* orgs,
    double* res, int N, int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
    double x = pos[i*2], z = pos[i*2+1], sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        sum += amps[o] * sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), 0.0, maintain_precision(z*lac));
    }
    res[i] = sum;
}
