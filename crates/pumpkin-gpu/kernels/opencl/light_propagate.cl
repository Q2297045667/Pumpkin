__kernel void light_propagate_u8(
    __global uchar* light,
    __global const uchar* opacity,
    __global const int* neighbors,
    __global uchar* changed,
    int N
) {
    int i = get_global_id(0);
    if (i >= N) return;
    uchar cur = light[i];
    uchar best = cur;
    for (int d = 0; d < 6; d++) {
        int n = neighbors[i * 6 + d];
        if (n < 0 || n >= N) continue;
        uchar nl = light[n];
        uchar op = opacity[n];
        uchar prop = (nl > (uchar)(1 + op)) ? (uchar)(nl - (uchar)(1 + op)) : 0;
        if (prop > best) best = prop;
    }
    if (best > cur) {
        light[i] = best;
        *changed = 1;
    }
}
