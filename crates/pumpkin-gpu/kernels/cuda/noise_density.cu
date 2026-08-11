extern "C" __global__ void batch_density_sample_f64(
    const double* pos, double* res,
    const double* perlin_configs, int N, int num_samplers
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
    double x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    double sum = 0.0;
    for (int s = 0; s < num_samplers; s++) {
        int base = s * 64;
        const unsigned char* p = (const unsigned char*)(perlin_configs + base);
        sum += sample_no_fade_core(p, 0,0,0, x,y,z);
    }
    res[i] = sum;
}
