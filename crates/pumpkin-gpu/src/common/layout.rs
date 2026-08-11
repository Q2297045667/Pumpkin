//! 数据布局转换辅助（AoS ↔ SoA）。

/// 将交错格式 [x0,y0,z0, x1,y1,z1, ...] 转换为独立数组。
#[must_use]
pub fn aos_to_soa_f64(interleaved: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

/// 将 AoS 格式 [x0,z0, x1,z1, ...] (2D) 转换为独立数组。
#[must_use]
pub fn aos2d_to_soa_f64(interleaved: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = interleaved.len() / 2;
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        x.push(interleaved[i * 2]);
        z.push(interleaved[i * 2 + 1]);
    }
    (x, z)
}
