__kernel void sky_light_fill_u8(
    __global const int* heightmap,
    __global const uchar* opacity,
    __global uchar* sky_light,
    int N, int max_height
) {
    int col = get_global_id(0);
    if (col >= N) return;
    int top = clamp(heightmap[col], -1, max_height - 1);
    for (int y = top + 1; y < max_height; y++) {
        sky_light[col * max_height + y] = 15;
    }
    uchar light = 15;
    for (int y = top; y >= 0; y--) {
        int idx = col * max_height + y;
        uchar op = opacity[idx];
        light = (light > op) ? (uchar)(light - op) : 0;
        sky_light[idx] = light;
    }
}
