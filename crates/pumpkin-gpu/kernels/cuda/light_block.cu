extern "C" __global__ void block_light_scan_u8(
    const unsigned char* luminances,
    unsigned char* block_light,
    int* sources_out,
    int* source_count,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned char lum = luminances[i];
    block_light[i] = lum;
    if (lum > 0) {
        int idx = atomicAdd(source_count, 1);
        sources_out[idx] = i;
    }
}
