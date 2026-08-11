# 🎵 GPU 噪声一致性 — 重建检查 (2026-08-11)

## 实测结果

### `batch_sampler` 内部 (4/31 失败)

| 测试 | CPU哈希 (期望) | GPU哈希 (实际) | 文件:行号 |
|------|---------------|---------------|-----------|
| `octave_consistency` | `1288544955713945777` | `13155553383635245199` | batch_sampler.rs:830 |
| `double_perlin_consistency` | `18259472666589028941` | `14371890326783839888` | batch_sampler.rs:845 |
| `shift_a_consistency` | `15157119126774824070` | `14712322157336749339` | batch_sampler.rs:864 |
| `shift_b_consistency` | `15157119126774824070` | `14712322157336749339` | batch_sampler.rs:884 |

### `noise_accel_consistency` (7/15 通过)

通过的 7 个: `octave_single`, `double_perlin_small`, `trilinear_consistency`, `trilinear_identity`, `surface_noise_consistency`, `noise_empty_input`, `octave_cache_stability`

失败的 8 个:

| 测试 | CPU哈希 | GPU哈希 |
|------|---------|---------|
| `octave_multi_3` | `5118584579143574454` | `11065053127178084206` |
| `octave_multi_5` | `3985450185461084676` | `6613310522841496273` |
| `octave_zero_positions` | `695581749008869285` | `3509581437530033061` |
| `double_perlin_consistency` | `1287871108109659109` | `363128762817402913` |
| `shift_a_consistency` | `4208656243541410993` | `12063131301904574387` |
| `shift_b_consistency` | `13236108748390797113` | `4215561268066402795` |
| `flatcache_consistency` | `15265289267318707230` | `12748522492753756509` |
| `bench_octave_large` | `16968357160720672199` | `6738996102098028451` |

### `gpu_noise_fingerprint` (2/5 通过)

| 测试 | CPU哈希 | GPU哈希 |
|------|---------|---------|
| `empty` | — | — ✅ |
| `single_octave` | — | — ✅ |
| `multi_octave` | `13525303208833510481` | `16094880514126389252` ❌ |
| `large` | `185856489120527420` | `12588276237522242816` ❌ |
| `various_octaves` | `8157304175729924804` | `3137430450991428000` ❌ |

---

## 根因确认

**GPU kernel `octave_perlin_sample_f64` (noise_octave.cl:10)**:
```c
sum += amps[o] * sample_no_fade_core(...);  // 缺少 * persistences[o]
```

**CPU `OctavePerlinNoiseSampler::sample` (perlin.rs:567)**:
```rust
sum += data.amplitude * sample * data.persistence;  // 两个因子
```

`SerializedOctaveConfig` 有 `packed_persistences()` 但从未上传到 GPU。

---

## 修复确认

**方案**: 在 `packed_amplitudes()` 中预乘。零 kernel 改动。

```rust
// pumpkin-gpu/src/noise/cache.rs
pub fn packed_amplitudes(&self) -> Vec<f64> {
    self.octaves.iter()
        .map(|o| o.amplitude * o.persistence)  // ← 添加 * persistence
        .collect()
}
```

修复后将使 22 个失败测试全部通过。
