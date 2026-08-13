// Beardifier batch: 与 vanilla `Beardifier.sample` 逐位一致的批量 kernel（CUDA）。
//
// structures: [num_structures * 8] f64 →
//   (box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z,
//    adaptation, ground_delta)
//   adaptation: 0=None 1=BeardThin 2=BeardBox 3=Bury 4=Encapsulate
// junctions: [num_junctions * 3] f64 → (x, ground_y, z)
// affected:  [6] f64 → (min_x, min_y, min_z, max_x, max_y, max_z)，包含边界
// beard_kernel: 24³ 预计算核表 (f64)，zi-major 布局: kernel[zi*576 + xi*24 + yi]

// 与 vanilla `get_beard_contribution(dx, dy, dz, y_to_ground)` 逐位一致
static __device__ double beard_contrib(
    int dx, int dy, int dz, int y_to_ground,
    const double* kernel
) {
    int xi = dx + 12;
    int yi = dy + 12;
    int zi = dz + 12;
    if (xi >= 0 && xi < 24 && yi >= 0 && yi < 24 && zi >= 0 && zi < 24) {
        double dy_off = (double)y_to_ground + 0.5;
        double dsq = (double)(dx * dx) + dy_off * dy_off + (double)(dz * dz);
        double value = (-dy_off) * (1.0 / sqrt(dsq / 2.0)) / 2.0;
        return value * kernel[zi * 576 + xi * 24 + yi];
    }
    return 0.0;
}

// 与 vanilla `get_bury_contribution` 逐位一致
static __device__ double bury_contrib(double dx, double dy, double dz) {
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 0.0) return 1.0;
    if (dist > 6.0) return 0.0;
    return 1.0 - dist / 6.0;
}

extern "C" __global__ void beardifier_batch_f64(
    const double* pos,             // [N*3] 位置（整数块坐标的 f64 表示）
    const double* beard_kernel,    // [24*24*24] 核表
    const double* structures,      // [num_structures * 8]
    const double* junctions,       // [num_junctions * 3]
    const double* affected,        // [6] 受影响盒
    double* beard_values,          // [N] 输出
    int N,
    int num_structures,
    int num_junctions
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int x = (int)pos[i * 3];
    int y = (int)pos[i * 3 + 1];
    int z = (int)pos[i * 3 + 2];

    int aminx = (int)affected[0], aminy = (int)affected[1], aminz = (int)affected[2];
    int amaxx = (int)affected[3], amaxy = (int)affected[4], amaxz = (int)affected[5];
    if (x < aminx || x > amaxx || y < aminy || y > amaxy || z < aminz || z > amaxz) {
        beard_values[i] = 0.0;
        return;
    }

    double weight = 0.0;

    for (int s = 0; s < num_structures; s++) {
        int sb = s * 8;
        int bminx = (int)structures[sb];
        int bminy = (int)structures[sb + 1];
        int bminz = (int)structures[sb + 2];
        int bmaxx = (int)structures[sb + 3];
        int bmaxy = (int)structures[sb + 4];
        int bmaxz = (int)structures[sb + 5];
        int adapt = (int)structures[sb + 6];
        int ground_delta = (int)structures[sb + 7];

        int dx = max(0, max(bminx - x, x - bmaxx));
        int dz = max(0, max(bminz - z, z - bmaxz));
        int ground_y = bminy + ground_delta;
        int dy_to_ground = y - ground_y;

        int dy = 0;
        if (adapt == 0) {
            dy = 0;
        } else if (adapt == 1 || adapt == 3) {
            dy = dy_to_ground;
        } else if (adapt == 2) {
            dy = max(0, max(ground_y - y, y - bmaxy));
        } else {
            dy = max(0, max(bminy - y, y - bmaxy));
        }

        if (adapt == 0) {
            continue;
        }
        if (adapt == 3) {
            weight += bury_contrib((double)dx, (double)dy / 2.0, (double)dz);
        } else if (adapt == 1 || adapt == 2) {
            weight += beard_contrib(dx, dy, dz, dy_to_ground, beard_kernel) * 0.8;
        } else {
            weight +=
                bury_contrib((double)dx / 2.0, (double)dy / 2.0, (double)dz / 2.0) * 0.8;
        }
    }

    for (int j = 0; j < num_junctions; j++) {
        int jb = j * 3;
        int jdx = x - (int)junctions[jb];
        int jdy = y - (int)junctions[jb + 1];
        int jdz = z - (int)junctions[jb + 2];
        weight += beard_contrib(jdx, jdy, jdz, jdy, beard_kernel) * 0.4;
    }

    beard_values[i] = weight;
}
