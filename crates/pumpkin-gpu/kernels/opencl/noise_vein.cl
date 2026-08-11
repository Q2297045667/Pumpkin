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
