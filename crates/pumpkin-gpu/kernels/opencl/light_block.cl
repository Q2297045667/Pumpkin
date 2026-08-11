__kernel void block_light_scan_u8(
    __global const uchar* luminances,
    __global uchar* block_light,
    __global int* sources_out,
    __global int* source_count,
    int N
) {
    int i = get_global_id(0);
    if (i >= N) return;
    uchar lum = luminances[i];
    block_light[i] = lum;
    if (lum > 0) {
        int idx = atomic_add(source_count, 1);
        sources_out[idx] = i;
    }
}
