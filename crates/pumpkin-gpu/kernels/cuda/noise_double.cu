extern "C" __global__ void double_perlin_sample_f64(
    const double* pos, const unsigned char* perms1,
    const double* amps1, const double* lacs1,
    const double* orgs1, const unsigned char* perms2,
    const double* amps2, const double* lacs2,
    const double* orgs2, double amp,
    double* res, int N, int M1, int M2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
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
