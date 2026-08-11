extern "C" __global__ void light_propagate_u8(
    unsigned char* light,
    const unsigned char* opacity,
    const int* neighbors,
    unsigned char* changed,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned char cur = light[i];
    unsigned char best = cur;
    for (int d = 0; d < 6; d++) {
        int n = neighbors[i * 6 + d];
        if (n < 0 || n >= N) continue;
        unsigned char nl = light[n];
        unsigned char op = opacity[n];
        unsigned char prop = (nl > (unsigned char)(1 + op)) ? (unsigned char)(nl - (unsigned char)(1 + op)) : 0;
        if (prop > best) best = prop;
    }
    if (best > cur) {
        light[i] = best;
        *changed = 1;
    }
}
