// Vein batch: 独立矿脉判定 — 通过 toggle/ridged/gap 三重噪声
// 确定每个位置是否属于矿脉以及矿脉类型。
// vein_noise_params:
//   每个矿脉 3 段噪声 (toggle, ridged, gap)，每段 octaves_per_vein 组参数。
//   每组 8 个 double: [amp, lac, org_x, org_y, org_z, xz_scale, y_scale, _reserved]
// vein_thresholds: [num_veins * 3] → (toggle_thr, ridged_thr, gap_thr)
__kernel void vein_batch_f64(
    __global const double* pos,               // [N*3] 位置
    __global const double* vein_noise_params, // 展平矿脉噪声参数
    __global const uchar* perms_data,         // 置换表
    __global const double* vein_thresholds,   // [num_veins * 3] 各矿脉阈值
    __global const double* vein_weights,      // [num_veins] 权重乘数
    __global int* vein_types,                 // [N] 0=none 1=ore 2=raw_ore 3=stone
    int N,
    int num_veins,
    int octaves_per_vein                      // 每段噪声的 octave 数
) {
    int i = get_global_id(0);
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    int best_type      = 0;
    double best_weight = -INFINITY;

    int stride_per_segment = octaves_per_vein * 8;  // 每段的 double 数
    int stride_per_vein    = stride_per_segment * 3;  // toggle + ridged + gap

    for (int v = 0; v < num_veins; v++) {
        // ---- 三段噪声 ----
        // toggle 段
        int toggle_base = v * stride_per_vein;
        double toggle = 0.0;
        for (int o = 0; o < octaves_per_vein; o++) {
            int po = toggle_base + o * 8;
            double amp   = vein_noise_params[po];
            double lac   = vein_noise_params[po + 1];
            double orgx  = vein_noise_params[po + 2];
            double orgy  = vein_noise_params[po + 3];
            double orgz  = vein_noise_params[po + 4];
            __global const uchar* perm = perms_data + (v * octaves_per_vein * 3 + o) * 256;
            toggle += amp * sample_no_fade_core(perm,
                orgx, orgy, orgz,
                maintain_precision(x * lac),
                maintain_precision(y * lac),
                maintain_precision(z * lac));
        }

        // ridged 段
        int ridged_base = toggle_base + stride_per_segment;
        double ridged = 0.0;
        for (int o = 0; o < octaves_per_vein; o++) {
            int po = ridged_base + o * 8;
            double amp   = vein_noise_params[po];
            double lac   = vein_noise_params[po + 1];
            double orgx  = vein_noise_params[po + 2];
            double orgy  = vein_noise_params[po + 3];
            double orgz  = vein_noise_params[po + 4];
            __global const uchar* perm = perms_data + (v * octaves_per_vein * 3 + octaves_per_vein + o) * 256;
            double sample = sample_no_fade_core(perm,
                orgx, orgy, orgz,
                maintain_precision(x * lac),
                maintain_precision(y * lac),
                maintain_precision(z * lac));
            ridged += amp * (1.0 - fabs(sample));
        }

        // gap 段
        int gap_base = toggle_base + stride_per_segment * 2;
        double gap = 0.0;
        for (int o = 0; o < octaves_per_vein; o++) {
            int po = gap_base + o * 8;
            double amp   = vein_noise_params[po];
            double lac   = vein_noise_params[po + 1];
            double orgx  = vein_noise_params[po + 2];
            double orgy  = vein_noise_params[po + 3];
            double orgz  = vein_noise_params[po + 4];
            __global const uchar* perm = perms_data + (v * octaves_per_vein * 3 + octaves_per_vein * 2 + o) * 256;
            gap += amp * sample_no_fade_core(perm,
                orgx, orgy, orgz,
                maintain_precision(x * lac),
                maintain_precision(y * lac),
                maintain_precision(z * lac));
        }

        // ---- 阈值判定 ----
        int vt = v * 3;
        double thr_toggle = vein_thresholds[vt];
        double thr_ridged = vein_thresholds[vt + 1];
        double thr_gap    = vein_thresholds[vt + 2];
        double weight     = vein_weights[v];

        if (toggle > thr_toggle && ridged > thr_ridged && gap > thr_gap) {
            double combined = weight * (toggle + ridged + gap);
            if (combined > best_weight) {
                best_weight = combined;

                // 分级判定
                if (ridged > thr_ridged * 1.6) {
                    best_type = 2;  // raw_ore
                } else if (toggle > thr_toggle * 1.3) {
                    best_type = 1;  // ore
                } else {
                    best_type = 3;  // stone (围岩)
                }
            }
        }
    }

    vein_types[i] = best_type;
}
