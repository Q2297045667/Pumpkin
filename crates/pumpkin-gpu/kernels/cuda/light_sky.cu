extern "C" __global__ void sky_light_fill_u8(
    const int* heightmap,
    const unsigned char* opacity,
    unsigned char* sky_light,
    int N, int max_height, int bottom_y
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    int top = heightmap[col];
    for (int y = top + 1; y < max_height; y++) {
        sky_light[col * max_height + y] = 15;
    }
    unsigned char light = 15;
    for (int y = top; y >= 0; y--) {
        int idx = col * max_height + y;
        unsigned char op = opacity[idx];
        light = (light > op) ? (unsigned char)(light - op) : 0;
        sky_light[idx] = light;
    }
}
