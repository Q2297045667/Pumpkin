extern "C" __global__ void vein_noise_sample_f64(
    const double* pos,
    const double* toggle_config, const double* ridged_config,
    const double* gap_config,
    double* toggle_out, double* ridged_out,
    double* gap_out, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    const unsigned char* tp = (const unsigned char*)toggle_config;
    double tsum = 0.0; int tM = (int)toggle_config[0];
    for (int o = 0; o < tM; o++) { int off = 1 + o * 64; tsum += sample_no_fade_core(tp + off, 0,0,0, x,y,z); }
    toggle_out[i] = tsum;
    const unsigned char* rp = (const unsigned char*)ridged_config;
    double rsum = 0.0; int rM = (int)ridged_config[0];
    for (int o = 0; o < rM; o++) { int off = 1 + o * 64; rsum += sample_no_fade_core(rp + off, 0,0,0, x,y,z); }
    ridged_out[i] = rsum;
    const unsigned char* gp = (const unsigned char*)gap_config;
    double gsum = 0.0; int gM = (int)gap_config[0];
    for (int o = 0; o < gM; o++) { int off = 1 + o * 64; gsum += sample_no_fade_core(gp + off, 0,0,0, x,y,z); }
    gap_out[i] = gsum;
}
