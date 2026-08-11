//! GPU Kernel 实现 — 完整噪声类型。
//!
//! 所有 Kernel 保证与 pumpkin-world CPU 路径逐位一致。
#![allow(clippy::needless_raw_string_hashes, clippy::too_many_lines)]

/// 基础 Perlin 噪声采样器 (sample_no_fade)。
/// 由 octave_perlin 和 double_perlin 共享使用。
pub const PERLIN_CORE_CL: &str = r##"
#pragma OPENCL EXTENSION cl_khr_f64 : enable

__constant double GRADIENTS[16][3] = {
    {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}, {1.0, -1.0, 0.0}, {-1.0, -1.0, 0.0},
    {1.0, 0.0, 1.0}, {-1.0, 0.0, 1.0}, {1.0, 0.0, -1.0}, {-1.0, 0.0, -1.0},
    {0.0, 1.0, 1.0}, {0.0, -1.0, 1.0}, {0.0, 1.0, -1.0}, {0.0, -1.0, -1.0},
    {1.0, 1.0, 0.0}, {0.0, -1.0, 1.0}, {-1.0, 1.0, 0.0}, {0.0, -1.0, -1.0}
};

static inline double perlin_fade(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
static inline double grad(int hash, double x, double y, double z) {
    int h = hash & 15;
    return GRADIENTS[h][0] * x + GRADIENTS[h][1] * y + GRADIENTS[h][2] * z;
}
static inline double maintain_precision(double value) {
    return value - floor(value / 33554432.0 + 0.5) * 33554432.0;
}
static inline double sample_no_fade_core(
    __global const uchar* perm, double ox, double oy, double oz,
    double x, double y, double z
) {
    double tx = x + ox, ty = y + oy, tz = z + oz;
    int xi = (int)floor(tx), yi = (int)floor(ty), zi = (int)floor(tz);
    double lx = tx - xi, ly = ty - yi, lz = tz - zi;
    int ix = xi & 255, iy = yi & 255, iz = zi & 255;
    int i = perm[ix], j = perm[(ix+1)&255];
    int k = perm[(i+iy)&255], l = perm[(i+iy+1)&255];
    int m = perm[(j+iy)&255], n = perm[(j+iy+1)&255];
    double d = grad(perm[(k+iz)&255], lx, ly, lz);
    double e = grad(perm[(m+iz)&255], lx-1.0, ly, lz);
    double f = grad(perm[(l+iz)&255], lx, ly-1.0, lz);
    double g = grad(perm[(n+iz)&255], lx-1.0, ly-1.0, lz);
    double h = grad(perm[(k+iz+1)&255], lx, ly, lz-1.0);
    double o = grad(perm[(m+iz+1)&255], lx-1.0, ly, lz-1.0);
    double p = grad(perm[(l+iz+1)&255], lx, ly-1.0, lz-1.0);
    double q = grad(perm[(n+iz+1)&255], lx-1.0, ly-1.0, lz-1.0);
    double u = perlin_fade(lx), v = perlin_fade(ly), w = perlin_fade(lz);
    double du0 = d + u*(e-d), du1 = f + u*(g-f), du2 = h + u*(o-h), du3 = p + u*(q-p);
    double dv0 = du0 + v*(du1-du0), dv1 = du2 + v*(du3-du2);
    return dv0 + w*(dv1-dv0);
}
"##;

/// 八度 Perlin 噪声批量采样。
pub const OCTAVE_PERLIN_SAMPLE_CL: &str = r##"
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
"##;

/// 双 Perlin 噪声批量采样。
pub const DOUBLE_PERLIN_SAMPLE_CL: &str = r##"
__kernel void double_perlin_sample_f64(
    __global const double* pos, __global const uchar* perms1,
    __global const double* amps1, __global const double* lacs1,
    __global const double* orgs1, __global const uchar* perms2,
    __global const double* amps2, __global const double* lacs2,
    __global const double* orgs2, double amp,
    __global double* res, int N, int M1, int M2
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    double s1 = 0.0, s2 = 0.0;
    for (int o = 0; o < M1; o++) {
        double lac = lacs1[o];
        s1 += amps1[o] * sample_no_fade_core(perms1 + o*256,
            orgs1[o*3], orgs1[o*3+1], orgs1[o*3+2],
            maintain_precision(x*lac), maintain_precision(y*lac), maintain_precision(z*lac));
    }
    double c = 1.0181268882175227;
    for (int o = 0; o < M2; o++) {
        double lac = lacs2[o];
        s2 += amps2[o] * sample_no_fade_core(perms2 + o*256,
            orgs2[o*3], orgs2[o*3+1], orgs2[o*3+2],
            maintain_precision(x*c*lac), maintain_precision(y*c*lac), maintain_precision(z*c*lac));
    }
    res[i] = (s1 + s2) * amp;
}
"##;

/// 偏移噪声批量采样 (ShiftA / ShiftB / ShiftedNoise)。
pub const SHIFTED_NOISE_SAMPLE_CL: &str = r##"
__kernel void shifted_noise_sample_f64(
    __global const double* pos,
    __global const uchar* perms, __global const double* amps, __global const double* lacs,
    __global const double* orgs, __global const double* shifts,
    double xz_scale, double y_scale, int use_per_sample_shift,
    __global double* res, int N, int M
) {
    int i = get_global_id(0); if (i >= N) return;
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
"##;

/// ShiftA 专用。
pub const SHIFT_A_SAMPLE_CL: &str = r##"
__kernel void shift_a_sample_f64(
    __global const double* pos, __global const uchar* perms,
    __global const double* amps, __global const double* lacs,
    __global const double* orgs, __global double* res, int N, int M
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*2] * 0.25, z = pos[i*2+1] * 0.25;
    double sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        sum += amps[o] * sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), 0.0, maintain_precision(z*lac));
    }
    res[i] = sum * 4.0;
}
"##;

/// ShiftB 专用。
pub const SHIFT_B_SAMPLE_CL: &str = r##"
__kernel void shift_b_sample_f64(
    __global const double* pos, __global const uchar* perms,
    __global const double* amps, __global const double* lacs,
    __global const double* orgs, __global double* res, int N, int M
) {
    int i = get_global_id(0); if (i >= N) return;
    double z = pos[i*2] * 0.25, x = pos[i*2+1] * 0.25;
    double sum = 0.0;
    for (int o = 0; o < M; o++) {
        double lac = lacs[o];
        sum += amps[o] * sample_no_fade_core(perms + o*256,
            orgs[o*3], orgs[o*3+1], orgs[o*3+2],
            maintain_precision(x*lac), 0.0, maintain_precision(z*lac));
    }
    res[i] = sum * 4.0;
}
"##;

/// 插值噪声采样。
pub const INTERPOLATED_NOISE_SAMPLE_CL: &str = r##"
__kernel void interpolated_noise_sample_f64(
    __global const double* pos,
    __global const double* noise_perms_data, __global const double* lower_perms_data,
    __global const double* upper_perms_data,
    __global const double* noise_amps, __global const double* noise_lacs,
    __global const double* noise_orgs,
    __global const double* lower_amps, __global const double* lower_lacs,
    __global const double* lower_orgs,
    __global const double* upper_amps, __global const double* upper_lacs,
    __global const double* upper_orgs,
    double xz_factor, double y_factor, double smear_scale_multiplier,
    double scaled_xz_scale, double scaled_y_scale, double y_multiplier,
    __global const double* fractions,
    __global double* res, int N, int noise_M, int lower_M, int upper_M
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    double xzm = scaled_xz_scale * 684.412;
    double ym = y_multiplier;
    double d = x * xzm, e = y * ym, f = z * xzm;
    double g = d / xz_factor, h = e / y_factor, i2 = f / xz_factor;
    double j = ym * smear_scale_multiplier, k = j / y_factor;
    double n_sum = 0.0;
    __global const uchar* np = (__global const uchar*)noise_perms_data;
    for (int o = 0; o < noise_M; o++) {
        int ri = noise_M - 1 - o;
        double frac = fractions[ri];
        double mx = maintain_precision(g * frac), my = maintain_precision(h * frac), mz = maintain_precision(i2 * frac);
        n_sum += sample_no_fade_core(np + ri*256, noise_orgs[ri*3], noise_orgs[ri*3+1], noise_orgs[ri*3+2],
            maintain_precision(mx*noise_lacs[ri]), maintain_precision(my*noise_lacs[ri]), maintain_precision(mz*noise_lacs[ri]))
            * noise_amps[ri] / frac;
    }
    double q = (n_sum / 10.0 + 1.0) * 0.5;
    double l_sum = 0.0, u_sum = 0.0;
    if (q < 1.0) {
        __global const uchar* lp = (__global const uchar*)lower_perms_data;
        for (int o = 0; o < lower_M; o++) {
            int ri = lower_M - 1 - o;
            double frac = fractions[ri];
            l_sum += sample_no_fade_core(lp + ri*256, lower_orgs[ri*3], lower_orgs[ri*3+1], lower_orgs[ri*3+2],
                maintain_precision(d*frac*lower_lacs[ri]), maintain_precision(e*frac*lower_lacs[ri]), maintain_precision(f*frac*lower_lacs[ri]))
                * lower_amps[ri] / frac;
        }
    }
    if (q > 0.0) {
        __global const uchar* up = (__global const uchar*)upper_perms_data;
        for (int o = 0; o < upper_M; o++) {
            int ri = upper_M - 1 - o;
            double frac = fractions[ri];
            u_sum += sample_no_fade_core(up + ri*256, upper_orgs[ri*3], upper_orgs[ri*3+1], upper_orgs[ri*3+2],
                maintain_precision(d*frac*upper_lacs[ri]), maintain_precision(e*frac*upper_lacs[ri]), maintain_precision(f*frac*upper_lacs[ri]))
                * upper_amps[ri] / frac;
        }
    }
    double clamped_q = clamp(q, 0.0, 1.0);
    res[i] = (l_sum / 512.0 * (1.0 - clamped_q) + u_sum / 512.0 * clamped_q) / 128.0;
}
"##;

/// 矿脉噪声批量采样。
pub const VEIN_NOISE_SAMPLE_CL: &str = r##"
__kernel void vein_noise_sample_f64(
    __global const double* pos,
    __global const double* toggle_config, __global const double* ridged_config,
    __global const double* gap_config,
    __global double* toggle_out, __global double* ridged_out,
    __global double* gap_out, int N
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    __global const uchar* tp = (__global const uchar*)toggle_config;
    double tsum = 0.0; int tM = (int)toggle_config[0];
    for (int o = 0; o < tM; o++) { int off = 1 + o * 64; tsum += sample_no_fade_core(tp + off, 0,0,0, x,y,z); }
    toggle_out[i] = tsum;
    __global const uchar* rp = (__global const uchar*)ridged_config;
    double rsum = 0.0; int rM = (int)ridged_config[0];
    for (int o = 0; o < rM; o++) { int off = 1 + o * 64; rsum += sample_no_fade_core(rp + off, 0,0,0, x,y,z); }
    ridged_out[i] = rsum;
    __global const uchar* gp = (__global const uchar*)gap_config;
    double gsum = 0.0; int gM = (int)gap_config[0];
    for (int o = 0; o < gM; o++) { int off = 1 + o * 64; gsum += sample_no_fade_core(gp + off, 0,0,0, x,y,z); }
    gap_out[i] = gsum;
}
"##;

/// 密度采样。
pub const DENSITY_SAMPLE_CL: &str = r##"
__kernel void batch_density_sample_f64(
    __global const double* pos, __global double* res,
    __global const double* perlin_configs, int N, int num_samplers
) {
    int i = get_global_id(0); if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    double sum = 0.0;
    for (int s = 0; s < num_samplers; s++) {
        int base = s * 64;
        __global const uchar* p = (__global const uchar*)(perlin_configs + base);
        sum += sample_no_fade_core(p, 0,0,0, x,y,z);
    }
    res[i] = sum;
}
"##;
