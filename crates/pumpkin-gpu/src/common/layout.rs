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
