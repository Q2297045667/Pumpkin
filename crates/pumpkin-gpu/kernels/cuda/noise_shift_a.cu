extern "C" __global__ void shift_a_sample_f64(
    const double* pos, const unsigned char* perms,
    const double* amps, const double* pers,
    const double* lacs, const double* orgs, double* res, int N, int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
    double x = pos[i*2] * 0.25, z = pos[i*2+1] * 0.25;
    double sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        double s = sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), 0.0, maintain_precision(z*lac));
        // 与 CPU 路径 (amplitude * sample) * persistence 的结合顺序逐位一致
        sum += (amps[o] * s) * pers[o];
    }
    res[i] = sum * 4.0;
}
