// Cell cache fill: simplified fill_from_stack batch path.
// component_stack stores each noise config flattened into 1D, each config has:
//   [0]: num_octaves (stored as double)
//   [1..]: amplitudes / lacunarities / origins (interleaved)
// perms_data stores all octave permutation tables (256 uchar each).
// cell_indices[i] selects which stack config position i should use.
extern "C" __global__ void cell_cache_fill_f64(
    const double* pos,                      // [N*3] input 3D positions
    const double* component_stack,          // flattened stack params
    const unsigned char* perms_data,        // permutation tables (256 uchar per octave)
    const int* cell_indices,                // [N] stack index per position
    double* densities,                      // [N] output densities
    int N,                                  // number of positions
    int config_stride,                      // double stride per config
    int amps_offset,                        // amplitudes offset in config (doubles)
    int lacs_offset,                        // lacunarities offset
    int orgs_offset                         // origins offset (3 doubles per octave)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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

        const unsigned char* perm = perms_data + o * 256;
        sum += amp * sample_no_fade_core(perm,
            orgx, orgy, orgz,
            maintain_precision(x * lac),
            maintain_precision(y * lac),
            maintain_precision(z * lac));
    }
    densities[i] = sum;
}
