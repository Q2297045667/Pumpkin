//! 数据布局转换辅助（AoS ↔ SoA）。
//!
//! 当 `soa_layout` 配置启用时，位置数据以独立 X/Y/Z 数组格式上传，
//! 改善 GPU 内存合并访问效率。
//!
//! AoS (Array of Structures): [x0,y0,z0, x1,y1,z1, ...]  — 交错格式
//! SoA (Structure of Arrays): x[0..N], y[0..N], z[0..N]   — 独立数组

/// 将 3D AoS 交错格式转换为 SoA 独立数组。
#[must_use]
pub fn aos3d_to_soa(interleaved: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = interleaved.len() / 3;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        x.push(interleaved[i * 3]);
        y.push(interleaved[i * 3 + 1]);
        z.push(interleaved[i * 3 + 2]);
    }
    (x, y, z)
}

/// 将 2D AoS 交错格式转换为 SoA 独立数组。
#[must_use]
pub fn aos2d_to_soa(interleaved: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = interleaved.len() / 2;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push(interleaved[i * 2]);
        b.push(interleaved[i * 2 + 1]);
    }
    (a, b)
}

/// 将 SoA 独立数组转换回 3D AoS 交错格式。
#[must_use]
pub fn soa_to_aos3d(x: &[f64], y: &[f64], z: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len()).min(z.len());
    let mut interleaved = Vec::with_capacity(n * 3);
    for i in 0..n {
        interleaved.push(x[i]);
        interleaved.push(y[i]);
        interleaved.push(z[i]);
    }
    interleaved
}
