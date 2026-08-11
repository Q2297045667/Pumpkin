#pragma OPENCL EXTENSION cl_khr_f64 : enable

__constant double GRADIENTS[16][3] = {
    {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}, {1.0, -1.0, 0.0}, {-1.0, -1.0, 0.0},
    {1.0, 0.0, 1.0}, {-1.0, 0.0, 1.0}, {1.0, 0.0, -1.0}, {-1.0, 0.0, -1.0},
    {0.0, 1.0, 1.0}, {0.0, -1.0, 1.0}, {0.0, 1.0, -1.0}, {0.0, -1.0, -1.0},
    {1.0, 1.0, 0.0}, {0.0, -1.0, 1.0}, {-1.0, 1.0, 0.0}, {0.0, -1.0, -1.0}
};

static inline double perlin_fade(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
static inline double grad(int hash, double x, double y, double z) {
    int h = hash & 15;
    return GRADIENTS[h][0] * x + GRADIENTS[h][1] * y + GRADIENTS[h][2] * z;
}
static inline double maintain_precision(double value) {
    return value - floor(value / 33554432.0 + 0.5) * 33554432.0;
}
static inline double sample_no_fade_core(
    __global const uchar* perm, double ox, double oy, double oz,
    double x, double y, double z
) {
    double tx = x + ox, ty = y + oy, tz = z + oz;
    int xi = (int)floor(tx), yi = (int)floor(ty), zi = (int)floor(tz);
    double lx = tx - xi, ly = ty - yi, lz = tz - zi;
    int ix = xi & 255, iy = yi & 255, iz = zi & 255;
    int i = perm[ix], j = perm[(ix+1)&255];
    int k = perm[(i+iy)&255], l = perm[(i+iy+1)&255];
    int m = perm[(j+iy)&255], n = perm[(j+iy+1)&255];
    double d = grad(perm[(k+iz)&255], lx, ly, lz);
    double e = grad(perm[(m+iz)&255], lx-1.0, ly, lz);
    double f = grad(perm[(l+iz)&255], lx, ly-1.0, lz);
    double g = grad(perm[(n+iz)&255], lx-1.0, ly-1.0, lz);
    double h = grad(perm[(k+iz+1)&255], lx, ly, lz-1.0);
    double o = grad(perm[(m+iz+1)&255], lx-1.0, ly, lz-1.0);
    double p = grad(perm[(l+iz+1)&255], lx, ly-1.0, lz-1.0);
    double q = grad(perm[(n+iz+1)&255], lx-1.0, ly-1.0, lz-1.0);
    double u = perlin_fade(lx), v = perlin_fade(ly), w = perlin_fade(lz);
    double du0 = d + u*(e-d), du1 = f + u*(g-f), du2 = h + u*(o-h), du3 = p + u*(q-p);
    double dv0 = du0 + v*(du1-du0), dv1 = du2 + v*(du3-du2);
    return dv0 + w*(dv1-dv0);
}
