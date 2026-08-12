__kernel void shift_b_sample_f64(
    __global const double* pos, __global const uchar* perms,
    __global const double* amps, __global const double* pers,
    __global const double* lacs, __global const double* orgs, __global double* res,
    int N, int M
) {
    int i = get_global_id(0); if (i >= N) return;
    double z = pos[i*2] * 0.25, x = pos[i*2+1] * 0.25;
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
