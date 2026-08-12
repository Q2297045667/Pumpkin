// SoA variant: octave_perlin — 使用独立 x/y/z 数组
__kernel void octave_perlin_sample_soa_f64(
    __global const double* pos_x,    // [N]
    __global const double* pos_y,    // [N]
    __global const double* pos_z,    // [N]
    __global const uchar* perms,     // [M*256]
    __global const double* amps,     // [M]
    __global const double* pers,     // [M]
    __global const double* lacs,     // [M]
    __global const double* orgs,     // [M*3]
    __global double* res,            // [N]
    int N,
    int M
) {
    int i = get_global_id(0);
    if (i >= N) return;
    double x = pos_x[i], y = pos_y[i], z = pos_z[i], sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        double s = sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), maintain_precision(y*lac), maintain_precision(z*lac));
        // 与 CPU 路径 (amplitude * sample) * persistence 的结合顺序逐位一致
        sum += (amps[o] * s) * pers[o];
    }
    res[i] = sum;
}
