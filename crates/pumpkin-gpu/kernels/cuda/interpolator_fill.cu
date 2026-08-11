// Interpolator buffer fill: sample density for each YZ slice position.
// dag_params uses flattened layout, each octave 8 doubles:
//   [amp, lac, org_x, org_y, org_z, xz_scale, y_scale, _reserved]
// x in pos represents slice coordinate, y/z are cell coordinates.
extern "C" __global__ void interpolator_fill_f64(
    const double* pos,                      // [N*3] positions
    const double* dag_params,               // DAG noise params
    const unsigned char* perms_data,        // permutation data
    double* densities,                      // [N] output
    int N,
    int num_octaves
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];

    if (num_octaves <= 0) {
        densities[i] = 0.0;
        return;
    }

    double sum = 0.0;
    for (int o = 0; o < num_octaves; o++) {
        int bo = o * 8;
        double amp     = dag_params[bo];
        double lac     = dag_params[bo + 1];
        double orgx    = dag_params[bo + 2];
        double orgy    = dag_params[bo + 3];
        double orgz    = dag_params[bo + 4];
        double xz_scale = dag_params[bo + 5];
        double y_scale  = dag_params[bo + 6];

        const unsigned char* perm = perms_data + o * 256;
        sum += amp * sample_no_fade_core(perm,
            orgx, orgy, orgz,
            maintain_precision(x * xz_scale * lac),
            maintain_precision(y * y_scale  * lac),
            maintain_precision(z * xz_scale * lac));
    }
    densities[i] = sum;
}
