__kernel void octave_perlin_sample_f64(
    __global const double* pos, __global const uchar* perms,
    __global const double* amps, __global const double* lacs,
    __global const double* orgs, __global double* res, int N, int M
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2], sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        sum += amps[o] * sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), maintain_precision(y*lac), maintain_precision(z*lac));
    }
    res[i] = sum;
}
