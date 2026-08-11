// Cell cache fill: 简化的 fill_from_stack 批量路径。
// component_stack 将各噪声配置展平为一维数组，每个配置包含：
//   [0]: num_octaves (以 double 存储)
//   [1..]: amplitudes / lacunarities / origins (交错排列)
// perms_data 存储所有 octave 的置换表 (每个 256 uchar)。
// cell_indices[i] 选择位置 i 应使用的 stack 配置。
__kernel void cell_cache_fill_f64(
    __global const double* pos,           // [N*3] 输入 3D 位置
    __global const double* component_stack, // 展平 stack 参数
    __global const uchar* perms_data,     // 置换表 (每个 octave 256 uchar)
    __global const int* cell_indices,     // [N] 每个位置的 stack 索引
    __global double* densities,           // [N] 输出密度
    int N,                                // 位置数量
    int config_stride,                    // 每个配置的 double 步长
    int amps_offset,                      // amplitudes 在配置内的偏移 (double)
    int lacs_offset,                      // lacunarities 偏移
    int orgs_offset                       // origins 偏移 (3 doubles per octave)
) {
    int i = get_global_id(0);
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    int ci = cell_indices[i];
    if (ci < 0) {
        densities[i] = 0.0;
        return;
    }

    int base = ci * config_stride;
    int num_octaves = (int)component_stack[base];
    if (num_octaves <= 0 || num_octaves > 64) {
        densities[i] = 0.0;
        return;
    }

    double sum = 0.0;
    for (int o = 0; o < num_octaves; o++) {
        double amp  = component_stack[base + amps_offset + o];
        double lac  = component_stack[base + lacs_offset + o];
        double orgx = component_stack[base + orgs_offset + o * 3];
        double orgy = component_stack[base + orgs_offset + o * 3 + 1];
        double orgz = component_stack[base + orgs_offset + o * 3 + 2];

        __global const uchar* perm = perms_data + o * 256;
        sum += amp * sample_no_fade_core(perm,
            orgx, orgy, orgz,
            maintain_precision(x * lac),
            maintain_precision(y * lac),
            maintain_precision(z * lac));
    }
    densities[i] = sum;
}
