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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aos3d_to_soa_roundtrip() {
        let aos = [
            1.0, 2.0, 3.0, // p0
            4.0, 5.0, 6.0, // p1
            7.0, 8.0, 9.0, // p2
        ];
        let (x, y, z) = aos3d_to_soa(&aos);
        assert_eq!(x, vec![1.0, 4.0, 7.0]);
        assert_eq!(y, vec![2.0, 5.0, 8.0]);
        assert_eq!(z, vec![3.0, 6.0, 9.0]);

        // 逐位保持（含负数与特殊值）
        let special = [0.0, -1.5, f64::INFINITY];
        let (sx, sy, sz) = aos3d_to_soa(&special);
        assert_eq!(sx, vec![0.0]);
        assert_eq!(sy, vec![-1.5]);
        assert_eq!(sz, vec![f64::INFINITY]);
    }

    #[test]
    fn aos3d_to_soa_empty_and_partial() {
        assert_eq!(aos3d_to_soa(&[]), (vec![], vec![], vec![]));
        // 长度不是 3 的倍数时按完整三元组截断（调用方保证长度正确）
        assert_eq!(aos3d_to_soa(&[1.0, 2.0]), (vec![], vec![], vec![]));
        let (x, y, z) = aos3d_to_soa(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(x, vec![1.0]);
        assert_eq!(y, vec![2.0]);
        assert_eq!(z, vec![3.0]);
    }
}
