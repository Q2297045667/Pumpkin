//! 补充 GPU Kernel — 三线性插值、FlatCache 预计算。
#![allow(clippy::needless_raw_string_hashes)]

/// 三线性插值批量处理。
pub const TRILINEAR_INTERPOLATE_CL: &str = r##"
__kernel void trilinear_interpolate_f64(
    __global const double* corners,
    __global const double* deltas,
    __global double* results, int M
) {
    int i = get_global_id(0); if (i >= M) return;
    int b = i * 8;
    double c000=corners[b], c100=corners[b+1], c010=corners[b+2], c110=corners[b+3];
    double c001=corners[b+4], c101=corners[b+5], c011=corners[b+6], c111=corners[b+7];
    double dx=deltas[i*3], dy=deltas[i*3+1], dz=deltas[i*3+2];
    results[i] = c000*(1-dx)*(1-dy)*(1-dz) + c100*dx*(1-dy)*(1-dz)
               + c010*(1-dx)*dy*(1-dz) + c110*dx*dy*(1-dz)
               + c001*(1-dx)*(1-dy)*dz + c101*dx*(1-dy)*dz
               + c011*(1-dx)*dy*dz + c111*dx*dy*dz;
}
"##;

/// FlatCache 预计算：对 2D 网格批量采样噪声。
pub const FLATCACHE_PRECOMPUTE_CL: &str = r##"
__kernel void flatcache_precompute_f64(
    __global const double* pos,
    __global const uchar* perms,
    __global const double* amps, __global const double* lacs,
    __global const double* orgs,
    __global double* res, int N, int M
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*2], z = pos[i*2+1], sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        sum += amps[o] * sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), 0.0, maintain_precision(z*lac));
    }
    res[i] = sum;
}
"##;
