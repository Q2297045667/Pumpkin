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
