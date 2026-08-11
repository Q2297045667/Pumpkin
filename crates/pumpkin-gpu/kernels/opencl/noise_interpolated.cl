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
