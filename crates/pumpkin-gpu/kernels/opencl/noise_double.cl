__kernel void double_perlin_sample_f64(
    __global const double* pos, __global const uchar* perms1,
    __global const double* amps1, __global const double* pers1, __global const double* lacs1,
    __global const double* orgs1, __global const uchar* perms2,
    __global const double* amps2, __global const double* pers2, __global const double* lacs2,
    __global const double* orgs2, double amp,
    __global double* res, int N, int M1, int M2
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    double s1 = 0.0, s2 = 0.0;
    for (int o = 0; o < M1; o++) {
        double lac = lacs1[o];
        double s = sample_no_fade_core(perms1 + o*256,
            orgs1[o*3], orgs1[o*3+1], orgs1[o*3+2],
            maintain_precision(x*lac), maintain_precision(y*lac), maintain_precision(z*lac));
        // 与 CPU 路径 (amplitude * sample) * persistence 的结合顺序逐位一致
        s1 += (amps1[o] * s) * pers1[o];
    }
    double c = 1.0181268882175227;
    for (int o = 0; o < M2; o++) {
        double lac = lacs2[o];
        double s = sample_no_fade_core(perms2 + o*256,
            orgs2[o*3], orgs2[o*3+1], orgs2[o*3+2],
            maintain_precision(x*c*lac), maintain_precision(y*c*lac), maintain_precision(z*c*lac));
        s2 += (amps2[o] * s) * pers2[o];
    }
    res[i] = (s1 + s2) * amp;
}
