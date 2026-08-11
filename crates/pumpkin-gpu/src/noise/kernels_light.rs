//! Lighting GPU Kernel 实现。
#![allow(clippy::needless_raw_string_hashes)]

/// 天空光垂直填充：对每个 (x,z) 列计算 light = 15 - cumulative_opacity。
pub const SKY_LIGHT_FILL_CL: &str = r##"
__kernel void sky_light_fill_u8(
    __global const int* heightmap,
    __global const uchar* opacity,
    __global uchar* sky_light,
    int N, int max_height, int bottom_y
) {
    int col = get_global_id(0);
    if (col >= N) return;
    int top = heightmap[col];
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
"##;

/// 方块光扫描：找出所有发光方块并设置初始光等级。
pub const BLOCK_LIGHT_SCAN_CL: &str = r##"
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
"##;

/// 光照传播单步迭代（迭代式距离场替代 BFS）。
pub const LIGHT_PROPAGATE_CL: &str = r##"
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
"##;
