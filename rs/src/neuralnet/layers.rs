/// CPU layer computations – WASM-compatible translation of eigenbackend.cpp.
///
/// All tensors are stored flat in **NHWC row-major** order:
///   4-D: index (n, y, x, c) → n*H*W*C + y*W*C + x*C + c
///   3-D mask: index (n, y, x) → n*H*W + y*W + x
///   2-D: index (c, n) → c*N + n   (channel × batch)
///
/// This mirrors the C++ Eigen layout which is column-major [C,X,Y,N] — the
/// element ordering is identical; only the conceptual axis labels differ.

use crate::model::{
  Activation, BatchNormLayerDesc, ConvLayerDesc, GlobalPoolingResidualBlockDesc,
  MatBiasLayerDesc, MatMulLayerDesc, NestedBottleneckResidualBlockDesc,
  ResidualBlockDesc, SgfMetadataEncoderDesc, BlockDesc,
};

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn apply_activation(x: f32, act: Activation) -> f32 {
  match act {
    Activation::Identity => x,
    Activation::Relu => x.max(0.0),
    Activation::Mish => {
      // mish(x) = x * tanh(softplus(x))
      // softplus(x) = ln(1 + e^x) — clamped to avoid overflow
      let sp = if x >= 20.0 {
        x
      } else {
        (1.0_f32 + x.exp()).ln()
      };
      x * sp.tanh()
    }
    Activation::MishScale8 => {
      // Not used in CPU/WASM path (only in fp16 GPU path)
      panic!("MishScale8 not supported on CPU backend")
    }
  }
}

// ---------------------------------------------------------------------------
// Mask sum
// ---------------------------------------------------------------------------

/// Compute the sum of each batch element's mask entries.
/// `mask`: flat [N*H*W], each element is 0 or 1.
pub fn compute_mask_sum(mask: &[f32], nn_x: usize, nn_y: usize, n: usize) -> Vec<f32> {
  let hw = nn_x * nn_y;
  let mut result = vec![0.0f32; n];
  for ni in 0..n {
    let mut s = 0.0f32;
    for i in 0..hw {
      s += mask[ni * hw + i];
    }
    result[ni] = s;
  }
  result
}

// ---------------------------------------------------------------------------
// Winograd helper transforms (for 3×3 and 5×5 convolutions)
// ---------------------------------------------------------------------------

#[inline(always)]
fn transform3x3_6(a: &mut [f32; 6]) {
  let z0 = a[0];
  let z1 = a[1];
  let z2 = a[2];
  a[0] = 0.25 * z0;
  a[1] = (-z0 - z1 - z2) * (1.0 / 6.0);
  a[2] = (-z0 + z1 - z2) * (1.0 / 6.0);
  a[3] = (z0 + 2.0 * z1 + 4.0 * z2) * (1.0 / 24.0);
  a[4] = (z0 - 2.0 * z1 + 4.0 * z2) * (1.0 / 24.0);
  a[5] = z2;
}

#[inline(always)]
fn transform5x5_6(a: &mut [f32; 6]) {
  let z0 = a[0];
  let z1 = a[1];
  let z2 = a[2];
  let z3 = a[3];
  let z4 = a[4];
  a[0] = 0.25 * z0;
  a[1] = (-z0 - z1 - z2 - z3 - z4) * (1.0 / 6.0);
  a[2] = (-z0 + z1 - z2 + z3 - z4) * (1.0 / 6.0);
  a[3] = (z0 + 2.0 * z1 + 4.0 * z2 + 8.0 * z3 + 16.0 * z4) * (1.0 / 24.0);
  a[4] = (z0 - 2.0 * z1 + 4.0 * z2 - 8.0 * z3 + 16.0 * z4) * (1.0 / 24.0);
  a[5] = z4;
}

// ---------------------------------------------------------------------------
// ConvLayer
// ---------------------------------------------------------------------------

/// Convolution layer (zero-padded, dilation=1 only).
///
/// Weights stored as `[oc * in_c * conv_y * conv_x]` (C++ weight order:
/// `oc, ic, y, x`).  For 3×3 and 5×5 filters a Winograd transform is used.
/// For other sizes (e.g. 1×1) a direct im2col-style loop is used.
pub struct ConvLayer {
  pub name: String,
  pub conv_y: usize,
  pub conv_x: usize,
  pub in_c: usize,
  pub out_c: usize,
  pub nn_x: usize,
  pub nn_y: usize,

  /// For 3×3 / 5×5: winograd-transformed kernel [inTileXY * in_c * out_c].
  /// For other sizes: direct kernel [out_c * in_c * conv_y * conv_x].
  kernel: Vec<f32>,

  // Winograd metadata (zero if not using winograd)
  num_tiles_x: usize,
  num_tiles_y: usize,
  in_tile_xy: usize,  // inTileXSize * inTileYSize
  out_tile_x: usize,
  out_tile_y: usize,
}

impl ConvLayer {
  pub fn new(desc: &ConvLayerDesc, nn_x: usize, nn_y: usize) -> Self {
    assert!(
      desc.dilation_x == 1 && desc.dilation_y == 1,
      "CPU backend: dilated convolutions not supported"
    );
    let cy = desc.conv_y_size as usize;
    let cx = desc.conv_x_size as usize;
    let ic = desc.in_channels as usize;
    let oc = desc.out_channels as usize;

    assert!(cx % 2 == 1 && cy % 2 == 1, "filter sizes must be odd");

    if (cx == 3 && cy == 3) || (cx == 5 && cy == 5) {
      // Winograd F(4×4, 3×3) or F(2×2, 5×5), both mapped to 6×6 tiles
      let in_tile_x = 6usize;
      let in_tile_y = 6usize;
      let out_tile_x = if cx == 5 { 2 } else { 4 };
      let out_tile_y = if cy == 5 { 2 } else { 4 };
      let num_tiles_x = (nn_x + out_tile_x - 1) / out_tile_x;
      let num_tiles_y = (nn_y + out_tile_y - 1) / out_tile_y;
      let in_tile_xy = in_tile_x * in_tile_y;

      // Transform weights: layout [inTileXY * ic * oc]
      let mut trans_w = vec![0.0f32; in_tile_xy * ic * oc];

      for oc_i in 0..oc {
        for ic_i in 0..ic {
          // Gather raw kernel for this (oc, ic) pair
          let mut tmp = [[0.0f32; 6]; 6];
          for sy in 0..cy {
            for sx in 0..cx {
              let w = desc.weights[((oc_i * ic + ic_i) * cy + sy) * cx + sx];
              tmp[sy][sx] = w;
            }
          }

          // Apply row transforms
          if cx == 3 {
            for row in tmp.iter_mut() {
              let mut a = [row[0], row[1], row[2], 0.0, 0.0, 0.0];
              transform3x3_6(&mut a);
              row.copy_from_slice(&a);
            }
          } else {
            for row in tmp.iter_mut() {
              let mut a = [row[0], row[1], row[2], row[3], row[4], 0.0];
              transform5x5_6(&mut a);
              row.copy_from_slice(&a);
            }
          }

          // Apply column transforms
          for sx in 0..in_tile_x {
            let mut a = [
              tmp[0][sx], tmp[1][sx], tmp[2][sx],
              tmp[3][sx], tmp[4][sx], tmp[5][sx],
            ];
            if cy == 3 {
              transform3x3_6(&mut a);
            } else {
              transform5x5_6(&mut a);
            }
            for (sy, &v) in a.iter().enumerate() {
              tmp[sy][sx] = v;
            }
          }

          // Store in trans_w: index (subY*inTileX + subX) * ic * oc
          for sy in 0..in_tile_y {
            for sx in 0..in_tile_x {
              let tile_idx = sy * in_tile_x + sx;
              trans_w[(tile_idx * ic + ic_i) * oc + oc_i] = tmp[sy][sx];
            }
          }
        }
      }

      ConvLayer {
        name: desc.name.clone(),
        conv_y: cy,
        conv_x: cx,
        in_c: ic,
        out_c: oc,
        nn_x,
        nn_y,
        kernel: trans_w,
        num_tiles_x,
        num_tiles_y,
        in_tile_xy,
        out_tile_x,
        out_tile_y,
      }
    } else {
      // Direct convolution — store kernel as-is (oc, ic, y, x)
      ConvLayer {
        name: desc.name.clone(),
        conv_y: cy,
        conv_x: cx,
        in_c: ic,
        out_c: oc,
        nn_x,
        nn_y,
        kernel: desc.weights.clone(),
        num_tiles_x: 0,
        num_tiles_y: 0,
        in_tile_xy: 0,
        out_tile_x: 0,
        out_tile_y: 0,
      }
    }
  }

  /// Apply convolution.
  ///
  /// `input`/`output` are NHWC flat: index (n,y,x,c) = n*H*W*C + y*W*C + x*C + c.
  /// `accumulate`: if true, add to `output`; otherwise overwrite.
  pub fn apply(
    &self,
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    accumulate: bool,
  ) {
    if (self.conv_x == 3 && self.conv_y == 3)
      || (self.conv_x == 5 && self.conv_y == 5)
    {
      self.apply_winograd(input, output, batch, accumulate);
    } else {
      self.apply_direct(input, output, batch, accumulate);
    }
  }

  // -----------------------------------------------------------------------
  // Winograd path
  // -----------------------------------------------------------------------

  fn apply_winograd(
    &self,
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    accumulate: bool,
  ) {
    let in_tile_x = 6usize;
    let in_tile_y = 6usize;
    let out_tile_x = self.out_tile_x;
    let out_tile_y = self.out_tile_y;
    let nt_x = self.num_tiles_x;
    let nt_y = self.num_tiles_y;
    let ic = self.in_c;
    let oc = self.out_c;
    let w = self.nn_x;
    let h = self.nn_y;
    let in_offset_x: isize = if self.conv_x == 5 { -2 } else { -1 };
    let in_offset_y: isize = if self.conv_y == 5 { -2 } else { -1 };
    let in_tile_xy = in_tile_x * in_tile_y;

    // Temporary buffers: transformed input/output [ic/oc, N*ntY*ntX, inTileXY]
    // Stored as [batch * nt_y * nt_x * inTileXY * ic/oc] – we use a simpler
    // flat layout: [inTileXY][batch*nt_y*nt_x][ic/oc]
    let batch_tiles = batch * nt_y * nt_x;
    let mut t_in = vec![0.0f32; in_tile_xy * batch_tiles * ic];
    let mut t_out = vec![0.0f32; in_tile_xy * batch_tiles * oc];

    // Tile workspace
    let mut tile = vec![0.0f32; in_tile_xy * ic.max(oc)];

    // ---- Transform input ----
    for n in 0..batch {
      for yt in 0..nt_y {
        for xt in 0..nt_x {
          let bt = n * nt_y * nt_x + yt * nt_x + xt;
          // Gather tile from input
          for dy in 0..in_tile_y {
            for dx in 0..in_tile_x {
              let xi = xt as isize * out_tile_x as isize + dx as isize + in_offset_x;
              let yi = yt as isize * out_tile_y as isize + dy as isize + in_offset_y;
              let sub = dy * in_tile_x + dx;
              if xi < 0 || yi < 0 || xi >= w as isize || yi >= h as isize {
                for c in 0..ic {
                  tile[sub * ic + c] = 0.0;
                }
              } else {
                for c in 0..ic {
                  tile[sub * ic + c] =
                    input[(n * h * w + yi as usize * w + xi as usize) * ic + c];
                }
              }
            }
          }

          // Apply row transforms in-place (across x dimension of each row)
          for sy in 0..in_tile_y {
            for ic_i in 0..ic {
              let mut a = [
                tile[(sy * in_tile_x + 0) * ic + ic_i],
                tile[(sy * in_tile_x + 1) * ic + ic_i],
                tile[(sy * in_tile_x + 2) * ic + ic_i],
                tile[(sy * in_tile_x + 3) * ic + ic_i],
                tile[(sy * in_tile_x + 4) * ic + ic_i],
                tile[(sy * in_tile_x + 5) * ic + ic_i],
              ];
              Self::winograd_input_transform_row(&mut a);
              for (dx, &v) in a.iter().enumerate() {
                tile[(sy * in_tile_x + dx) * ic + ic_i] = v;
              }
            }
          }
          // Apply column transforms
          for sx in 0..in_tile_x {
            for ic_i in 0..ic {
              let mut a = [
                tile[(0 * in_tile_x + sx) * ic + ic_i],
                tile[(1 * in_tile_x + sx) * ic + ic_i],
                tile[(2 * in_tile_x + sx) * ic + ic_i],
                tile[(3 * in_tile_x + sx) * ic + ic_i],
                tile[(4 * in_tile_x + sx) * ic + ic_i],
                tile[(5 * in_tile_x + sx) * ic + ic_i],
              ];
              Self::winograd_input_transform_row(&mut a);
              for (dy, &v) in a.iter().enumerate() {
                tile[(dy * in_tile_x + sx) * ic + ic_i] = v;
              }
            }
          }
          // Store in t_in: layout [in_tile_xy * batch_tiles * ic]
          // index (sub, bt, ic_i) = sub * batch_tiles * ic + bt * ic + ic_i
          for sub in 0..in_tile_xy {
            for ic_i in 0..ic {
              t_in[sub * batch_tiles * ic + bt * ic + ic_i] =
                tile[sub * ic + ic_i];
            }
          }
        }
      }
    }

    // ---- Batched matrix multiply: t_out[sub] = kernel[sub] * t_in[sub] ----
    // kernel stored as [inTileXY * ic * oc], i.e. for each sub: ic×oc matrix
    // t_in  slice: [batch_tiles * ic]  → matrix (ic × batch_tiles)
    // t_out slice: [batch_tiles * oc]  → matrix (oc × batch_tiles)
    for sub in 0..in_tile_xy {
      let k_base = sub * ic * oc;
      let ti_base = sub * batch_tiles * ic;
      let to_base = sub * batch_tiles * oc;
      // t_out[oc_i, bt] = sum_ic_i kernel[ic_i, oc_i] * t_in[bt, ic_i]
      for oc_i in 0..oc {
        for bt in 0..batch_tiles {
          let mut acc = 0.0f32;
          for ic_i in 0..ic {
            acc += self.kernel[k_base + ic_i * oc + oc_i]
              * t_in[ti_base + bt * ic + ic_i];
          }
          t_out[to_base + bt * oc + oc_i] = acc;
        }
      }
    }

    // ---- Inverse-transform t_out and scatter to output ----
    for n in 0..batch {
      for yt in 0..nt_y {
        for xt in 0..nt_x {
          let bt = n * nt_y * nt_x + yt * nt_x + xt;
          // Gather from t_out into tile
          for sub in 0..in_tile_xy {
            for oc_i in 0..oc {
              tile[sub * oc + oc_i] = t_out[sub * batch_tiles * oc + bt * oc + oc_i];
            }
          }

          // Apply inverse row transforms
          if self.conv_x == 5 {
            for sy in 0..in_tile_y {
              for oc_i in 0..oc {
                let mut a = [
                  tile[(sy * in_tile_x + 0) * oc + oc_i],
                  tile[(sy * in_tile_x + 1) * oc + oc_i],
                  tile[(sy * in_tile_x + 2) * oc + oc_i],
                  tile[(sy * in_tile_x + 3) * oc + oc_i],
                  tile[(sy * in_tile_x + 4) * oc + oc_i],
                  tile[(sy * in_tile_x + 5) * oc + oc_i],
                ];
                Self::winograd_output_transform_5x5_row(&mut a);
                for (dx, &v) in a.iter().enumerate() {
                  tile[(sy * in_tile_x + dx) * oc + oc_i] = v;
                }
              }
            }
            for sx in 0..out_tile_x {
              for oc_i in 0..oc {
                let mut a = [
                  tile[(0 * in_tile_x + sx) * oc + oc_i],
                  tile[(1 * in_tile_x + sx) * oc + oc_i],
                  tile[(2 * in_tile_x + sx) * oc + oc_i],
                  tile[(3 * in_tile_x + sx) * oc + oc_i],
                  tile[(4 * in_tile_x + sx) * oc + oc_i],
                  tile[(5 * in_tile_x + sx) * oc + oc_i],
                ];
                Self::winograd_output_transform_5x5_row(&mut a);
                for (dy, &v) in a.iter().enumerate() {
                  tile[(dy * in_tile_x + sx) * oc + oc_i] = v;
                }
              }
            }
          } else {
            // 3×3 → 4×4 output
            for sy in 0..in_tile_y {
              for oc_i in 0..oc {
                let mut a = [
                  tile[(sy * in_tile_x + 0) * oc + oc_i],
                  tile[(sy * in_tile_x + 1) * oc + oc_i],
                  tile[(sy * in_tile_x + 2) * oc + oc_i],
                  tile[(sy * in_tile_x + 3) * oc + oc_i],
                  tile[(sy * in_tile_x + 4) * oc + oc_i],
                  tile[(sy * in_tile_x + 5) * oc + oc_i],
                ];
                Self::winograd_output_transform_3x3_row(&mut a);
                for (dx, &v) in a.iter().enumerate() {
                  tile[(sy * in_tile_x + dx) * oc + oc_i] = v;
                }
              }
            }
            for sx in 0..out_tile_x {
              for oc_i in 0..oc {
                let mut a = [
                  tile[(0 * in_tile_x + sx) * oc + oc_i],
                  tile[(1 * in_tile_x + sx) * oc + oc_i],
                  tile[(2 * in_tile_x + sx) * oc + oc_i],
                  tile[(3 * in_tile_x + sx) * oc + oc_i],
                  tile[(4 * in_tile_x + sx) * oc + oc_i],
                  tile[(5 * in_tile_x + sx) * oc + oc_i],
                ];
                Self::winograd_output_transform_3x3_row(&mut a);
                for (dy, &v) in a.iter().enumerate() {
                  tile[(dy * in_tile_x + sx) * oc + oc_i] = v;
                }
              }
            }
          }

          // Scatter to output
          for dy in 0..out_tile_y {
            for dx in 0..out_tile_x {
              let xi = xt * out_tile_x + dx;
              let yi = yt * out_tile_y + dy;
              if xi < w && yi < h {
                let sub = dy * in_tile_x + dx;
                let out_base = (n * h * w + yi * w + xi) * oc;
                for oc_i in 0..oc {
                  let v = tile[sub * oc + oc_i];
                  if accumulate {
                    output[out_base + oc_i] += v;
                  } else {
                    output[out_base + oc_i] = v;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /// Winograd input transform for 3×3 (B^T d B), row-wise: maps 6 values.
  /// `4z0 - 5z2 + z4` pattern.
  #[inline(always)]
  fn winograd_input_transform_row(a: &mut [f32; 6]) {
    let z0 = a[0];
    let z1 = a[1];
    let z2 = a[2];
    let z3 = a[3];
    let z4 = a[4];
    let z5 = a[5];
    a[0] = 4.0 * z0 - 5.0 * z2 + z4;
    a[1] = -4.0 * z1 - 4.0 * z2 + z3 + z4;
    a[2] =  4.0 * z1 - 4.0 * z2 - z3 + z4;
    a[3] = -2.0 * z1 - z2 + 2.0 * z3 + z4;
    a[4] =  2.0 * z1 - z2 - 2.0 * z3 + z4;
    a[5] = 4.0 * z1 - 5.0 * z3 + z5;
  }

  /// Winograd output inverse transform row for 3×3 → 4 outputs.
  #[inline(always)]
  fn winograd_output_transform_3x3_row(a: &mut [f32; 6]) {
    let z0 = a[0];
    let z1 = a[1];
    let z2 = a[2];
    let z3 = a[3];
    let z4 = a[4];
    let z5 = a[5];
    a[0] = z0 + z1 + z2 + z3 + z4;
    a[1] = (z1 - z2) + 2.0 * (z3 - z4);
    a[2] = (z1 + z2) + 4.0 * (z3 + z4);
    a[3] = (z1 - z2) + 8.0 * (z3 - z4) + z5;
    // a[4], a[5] unused
  }

  /// Winograd output inverse transform row for 5×5 → 2 outputs.
  #[inline(always)]
  fn winograd_output_transform_5x5_row(a: &mut [f32; 6]) {
    let z0 = a[0];
    let z1 = a[1];
    let z2 = a[2];
    let z3 = a[3];
    let z4 = a[4];
    let z5 = a[5];
    a[0] = z0 + z1 + z2 + z3 + z4;
    a[1] = (z1 - z2) + 2.0 * (z3 - z4) + z5;
    // a[2..5] unused
  }

  // -----------------------------------------------------------------------
  // Direct (im2col) path for non-3×3 / non-5×5 filters
  // -----------------------------------------------------------------------

  fn apply_direct(
    &self,
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    accumulate: bool,
  ) {
    let cy = self.conv_y;
    let cx = self.conv_x;
    let ic = self.in_c;
    let oc = self.out_c;
    let w = self.nn_x;
    let h = self.nn_y;
    let pad_y = (cy / 2) as isize;
    let pad_x = (cx / 2) as isize;

    if !accumulate {
      for v in output.iter_mut() {
        *v = 0.0;
      }
    }

    for n in 0..batch {
      for yi in 0..h {
        for xi in 0..w {
          let out_base = (n * h * w + yi * w + xi) * oc;
          for oc_i in 0..oc {
            let mut acc = 0.0f32;
            for sy in 0..cy {
              let iy = yi as isize + sy as isize - pad_y;
              if iy < 0 || iy >= h as isize {
                continue;
              }
              for sx in 0..cx {
                let ix = xi as isize + sx as isize - pad_x;
                if ix < 0 || ix >= w as isize {
                  continue;
                }
                let in_base =
                  (n * h * w + iy as usize * w + ix as usize) * ic;
                let k_base = ((oc_i * ic) * cy + sy) * cx + sx;
                // kernel layout: [oc, ic, cy, cx]
                // k_base above needs to account for ic stride properly:
                // index = oc_i * (ic * cy * cx) + ic_i * (cy * cx) + sy * cx + sx
                for ic_i in 0..ic {
                  let k_idx = oc_i * ic * cy * cx + ic_i * cy * cx + sy * cx + sx;
                  acc += self.kernel[k_idx] * input[in_base + ic_i];
                }
              }
            }
            if accumulate {
              output[out_base + oc_i] += acc;
            } else {
              output[out_base + oc_i] = acc;
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// BatchNormLayer (with fused activation)
// ---------------------------------------------------------------------------

/// Batch-normalisation layer with pre-fused scale/bias.
/// Optionally applies an activation in the same pass.
pub struct BatchNormLayer {
  pub name: String,
  pub activation: Activation,
  pub merged_scale: Vec<f32>,
  pub merged_bias: Vec<f32>,
}

impl BatchNormLayer {
  pub fn new(desc: &BatchNormLayerDesc, act: Activation) -> Self {
    BatchNormLayer {
      name: desc.name.clone(),
      activation: act,
      merged_scale: desc.merged_scale.clone(),
      merged_bias: desc.merged_bias.clone(),
    }
  }

  /// Apply BN + activation.
  ///
  /// `input` / `output`: NHWC [N*H*W*C].
  /// `mask`: NHW flat [N*H*W], 0.0 or 1.0; masked-off cells are zeroed.
  pub fn apply(
    &self,
    input: &[f32],
    output: &mut [f32],
    mask: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) {
    let nc = self.merged_scale.len();
    let hw = nn_x * nn_y;
    for n in 0..batch {
      for yi in 0..nn_y {
        for xi in 0..nn_x {
          let pos = n * hw + yi * nn_x + xi;
          let m = mask[pos];
          let base = pos * nc;
          for c in 0..nc {
            let x = input[base + c] * self.merged_scale[c] + self.merged_bias[c];
            let v = if m == 1.0 { apply_activation(x, self.activation) } else { 0.0 };
            output[base + c] = v;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// MatMulLayer
// ---------------------------------------------------------------------------

/// Dense matrix multiply: out[oc, n] = sum_ic weights[oc, ic] * in[ic, n].
///
/// Weights layout in desc: [ic * oc] (file order ic,oc).
/// We store transposed to [oc * ic] for efficient row-access.
pub struct MatMulLayer {
  pub name: String,
  pub in_c: usize,
  pub out_c: usize,
  /// Weights [out_c * in_c]: w[oc * in_c + ic]
  weights: Vec<f32>,
}

impl MatMulLayer {
  pub fn new(desc: &MatMulLayerDesc) -> Self {
    let ic = desc.in_channels as usize;
    let oc = desc.out_channels as usize;
    // desc.weights is in [ic, oc] order; transpose to [oc, ic]
    let mut w = vec![0.0f32; oc * ic];
    for ic_i in 0..ic {
      for oc_i in 0..oc {
        w[oc_i * ic + ic_i] = desc.weights[ic_i * oc + oc_i];
      }
    }
    MatMulLayer { name: desc.name.clone(), in_c: ic, out_c: oc, weights: w }
  }

  /// `input`: [in_c * batch], `output`: [out_c * batch].
  pub fn apply(&self, input: &[f32], output: &mut [f32], batch: usize) {
    for oc_i in 0..self.out_c {
      for n in 0..batch {
        let mut acc = 0.0f32;
        for ic_i in 0..self.in_c {
          acc +=
            self.weights[oc_i * self.in_c + ic_i] * input[ic_i * batch + n];
        }
        output[oc_i * batch + n] = acc;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// MatBiasLayer
// ---------------------------------------------------------------------------

pub struct MatBiasLayer {
  pub name: String,
  pub weights: Vec<f32>,
}

impl MatBiasLayer {
  pub fn new(desc: &MatBiasLayerDesc) -> Self {
    MatBiasLayer { name: desc.name.clone(), weights: desc.weights.clone() }
  }

  /// Add bias to `mat` in-place. `mat`: [num_channels * batch].
  pub fn apply(&self, mat: &mut [f32], batch: usize) {
    let nc = self.weights.len();
    for c in 0..nc {
      for n in 0..batch {
        mat[c * batch + n] += self.weights[c];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// ActivationLayer (standalone, for non-BN paths)
// ---------------------------------------------------------------------------

pub struct ActivationLayer {
  pub name: String,
  pub activation: Activation,
}

impl ActivationLayer {
  pub fn new(
    name: impl Into<String>,
    activation: Activation,
  ) -> Self {
    ActivationLayer { name: name.into(), activation }
  }

  pub fn apply_inplace(&self, data: &mut [f32]) {
    for v in data.iter_mut() {
      *v = apply_activation(*v, self.activation);
    }
  }
}

// ---------------------------------------------------------------------------
// Global-pooling helper
// ---------------------------------------------------------------------------

/// Pool a 4-D NHWC tensor into a 2-D [3C × N] tensor using mean/sqrt-mean/max
/// (gpool variant, used in residual blocks and policy head).
///
/// `in4d`: NHWC [N*H*W*C_in].
/// `out2d`: [3*C_in * N] — written in-place (not accumulated).
/// `mask`: NHW flat [N*H*W].
/// `mask_sum`: per-batch element sum [N].
pub fn pool_rows_gpool(
  in4d: &[f32],
  out2d: &mut [f32],
  mask: &[f32],
  mask_sum: &[f32],
  batch: usize,
  nn_x: usize,
  nn_y: usize,
  c_in: usize,
) {
  let hw = nn_x * nn_y;
  for n in 0..batch {
    let div = mask_sum[n];
    let sqrtdiv = div.sqrt();
    for c in 0..c_in {
      let mut s = 0.0f32;
      let mut m = -1.0f32;
      for yi in 0..nn_y {
        for xi in 0..nn_x {
          let pos = n * hw + yi * nn_x + xi;
          let x = in4d[pos * c_in + c];
          s += x;
          let mask_val = mask[pos];
          let candidate = x + (mask_val - 1.0);
          if candidate > m {
            m = candidate;
          }
        }
      }
      let mean = s / div;
      out2d[c * batch + n] = mean;
      out2d[(c + c_in) * batch + n] = mean * (sqrtdiv - 14.0) * 0.1;
      out2d[(c + 2 * c_in) * batch + n] = m;
    }
  }
}

/// Pool variant for the value head (mean + sqrt-weighted mean + quadratic term).
pub fn pool_rows_value_head(
  in4d: &[f32],
  out2d: &mut [f32],
  mask_sum: &[f32],
  batch: usize,
  nn_x: usize,
  nn_y: usize,
  c_in: usize,
) {
  let hw = nn_x * nn_y;
  for n in 0..batch {
    let div = mask_sum[n];
    let sqrtdiv = div.sqrt();
    for c in 0..c_in {
      let mut s = 0.0f32;
      for yi in 0..nn_y {
        for xi in 0..nn_x {
          let pos = n * hw + yi * nn_x + xi;
          s += in4d[pos * c_in + c];
        }
      }
      let mean = s / div;
      let sd14 = sqrtdiv - 14.0;
      out2d[c * batch + n] = mean;
      out2d[(c + c_in) * batch + n] = mean * sd14 * 0.1;
      out2d[(c + 2 * c_in) * batch + n] = mean * (sd14 * sd14 * 0.01 - 0.1);
    }
  }
}

// ---------------------------------------------------------------------------
// Add bias NC in-place
// ---------------------------------------------------------------------------

/// Add a [C × N] bias tensor into a NHWC 4-D [N*H*W*C] tensor.
/// bias[c, n] is added to all spatial positions.
pub fn add_nc_bias_inplace(
  tensor: &mut [f32],
  bias: &[f32],
  batch: usize,
  nn_x: usize,
  nn_y: usize,
  channels: usize,
) {
  let hw = nn_x * nn_y;
  for n in 0..batch {
    for yi in 0..nn_y {
      for xi in 0..nn_x {
        let base = (n * hw + yi * nn_x + xi) * channels;
        for c in 0..channels {
          tensor[base + c] += bias[c * batch + n];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// NormActConv (BN + activation + Conv)
// ---------------------------------------------------------------------------

pub struct NormActConv {
  pub norm: BatchNormLayer,
  pub conv: ConvLayer,
  pub in_c: usize,
  pub out_c: usize,
}

impl NormActConv {
  pub fn new(
    bn_desc: &BatchNormLayerDesc,
    act: Activation,
    conv_desc: &ConvLayerDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    NormActConv {
      norm: BatchNormLayer::new(bn_desc, act),
      conv: ConvLayer::new(conv_desc, nn_x, nn_y),
      in_c: conv_desc.in_channels as usize,
      out_c: conv_desc.out_channels as usize,
    }
  }

  /// Apply BN+act on `input` into `scratch`, then convolve into `output`.
  pub fn apply(
    &self,
    input: &[f32],
    scratch: &mut [f32],
    output: &mut [f32],
    mask: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
    accumulate: bool,
  ) {
    self.norm.apply(input, scratch, mask, batch, nn_x, nn_y);
    self.conv.apply(scratch, output, batch, accumulate);
  }
}

// ---------------------------------------------------------------------------
// ResidualBlock
// ---------------------------------------------------------------------------

pub struct ResidualBlock {
  pub name: String,
  pub nac1: NormActConv,
  pub nac2: NormActConv,
}

impl ResidualBlock {
  pub fn new(desc: &ResidualBlockDesc, nn_x: usize, nn_y: usize) -> Self {
    ResidualBlock {
      name: desc.name.clone(),
      nac1: NormActConv::new(
        &desc.pre_bn,
        desc.pre_activation.activation,
        &desc.regular_conv,
        nn_x,
        nn_y,
      ),
      nac2: NormActConv::new(
        &desc.mid_bn,
        desc.mid_activation.activation,
        &desc.final_conv,
        nn_x,
        nn_y,
      ),
    }
  }

  pub fn apply(
    &self,
    trunk: &mut [f32],
    scratch: &mut [f32],
    mask: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) {
    let mid_elts = batch * nn_y * nn_x * self.nac1.out_c;
    let mut mid = vec![0.0f32; mid_elts];
    let mut mid_scratch = vec![0.0f32; mid_elts];

    self.nac1.apply(trunk, scratch, &mut mid, mask, batch, nn_x, nn_y, false);
    self.nac2.apply(&mid, &mut mid_scratch, trunk, mask, batch, nn_x, nn_y, true);
  }
}

// ---------------------------------------------------------------------------
// GlobalPoolingResidualBlock
// ---------------------------------------------------------------------------

pub struct GlobalPoolingResidualBlock {
  pub name: String,
  pub pre_bn: BatchNormLayer,
  pub regular_conv: ConvLayer,
  pub gpool_conv: ConvLayer,
  pub gpool_bn: BatchNormLayer,
  pub gpool_to_bias_mul: MatMulLayer,
  pub nac2: NormActConv,
}

impl GlobalPoolingResidualBlock {
  pub fn new(
    desc: &GlobalPoolingResidualBlockDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    GlobalPoolingResidualBlock {
      name: desc.name.clone(),
      pre_bn: BatchNormLayer::new(&desc.pre_bn, desc.pre_activation.activation),
      regular_conv: ConvLayer::new(&desc.regular_conv, nn_x, nn_y),
      gpool_conv: ConvLayer::new(&desc.gpool_conv, nn_x, nn_y),
      gpool_bn: BatchNormLayer::new(
        &desc.gpool_bn,
        desc.gpool_activation.activation,
      ),
      gpool_to_bias_mul: MatMulLayer::new(&desc.gpool_to_bias_mul),
      nac2: NormActConv::new(
        &desc.mid_bn,
        desc.mid_activation.activation,
        &desc.final_conv,
        nn_x,
        nn_y,
      ),
    }
  }

  pub fn apply(
    &self,
    trunk: &mut [f32],
    trunk_scratch: &mut [f32],
    mask: &[f32],
    mask_sum: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) {
    let reg_c = self.regular_conv.out_c;
    let gpc = self.gpool_conv.out_c;
    let hw = nn_x * nn_y;

    let mut regular_out = vec![0.0f32; batch * hw * reg_c];
    let mut regular_scratch = vec![0.0f32; batch * hw * reg_c];
    let mut gpool_out = vec![0.0f32; batch * hw * gpc];
    let mut gpool_out2 = vec![0.0f32; batch * hw * gpc];
    let mut gpool_concat = vec![0.0f32; gpc * 3 * batch];
    let mut gpool_bias = vec![0.0f32; reg_c * batch];

    self.pre_bn.apply(trunk, trunk_scratch, mask, batch, nn_x, nn_y);
    self.regular_conv.apply(trunk_scratch, &mut regular_out, batch, false);
    self.gpool_conv.apply(trunk_scratch, &mut gpool_out, batch, false);
    self.gpool_bn.apply(&gpool_out.clone(), &mut gpool_out2, mask, batch, nn_x, nn_y);
    pool_rows_gpool(&gpool_out2, &mut gpool_concat, mask, mask_sum, batch, nn_x, nn_y, gpc);
    self.gpool_to_bias_mul.apply(&gpool_concat, &mut gpool_bias, batch);
    add_nc_bias_inplace(&mut regular_out, &gpool_bias, batch, nn_x, nn_y, reg_c);
    self.nac2.apply(&regular_out, &mut regular_scratch, trunk, mask, batch, nn_x, nn_y, true);
  }
}

// ---------------------------------------------------------------------------
// BlockStack (dynamic dispatch via enum)
// ---------------------------------------------------------------------------

pub enum ResidualBlockKind {
  Ordinary(ResidualBlock),
  GlobalPooling(GlobalPoolingResidualBlock),
  NestedBottleneck(NestedBottleneckResidualBlockLayer),
}

impl ResidualBlockKind {
  pub fn apply(
    &self,
    trunk: &mut [f32],
    trunk_scratch: &mut [f32],
    mask: &[f32],
    mask_sum: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) {
    match self {
      ResidualBlockKind::Ordinary(b) => {
        b.apply(trunk, trunk_scratch, mask, batch, nn_x, nn_y)
      }
      ResidualBlockKind::GlobalPooling(b) => {
        b.apply(trunk, trunk_scratch, mask, mask_sum, batch, nn_x, nn_y)
      }
      ResidualBlockKind::NestedBottleneck(b) => {
        b.apply(trunk, trunk_scratch, mask, mask_sum, batch, nn_x, nn_y)
      }
    }
  }
}

pub struct BlockStack {
  pub blocks: Vec<ResidualBlockKind>,
}

impl BlockStack {
  pub fn new(
    desc_blocks: &[BlockDesc],
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    let mut blocks = Vec::with_capacity(desc_blocks.len());
    for bd in desc_blocks {
      let b = match bd {
        BlockDesc::Ordinary(d) => {
          ResidualBlockKind::Ordinary(ResidualBlock::new(d, nn_x, nn_y))
        }
        BlockDesc::GlobalPooling(d) => {
          ResidualBlockKind::GlobalPooling(
            GlobalPoolingResidualBlock::new(d, nn_x, nn_y),
          )
        }
        BlockDesc::NestedBottleneck(d) => {
          ResidualBlockKind::NestedBottleneck(
            NestedBottleneckResidualBlockLayer::new(d, nn_x, nn_y),
          )
        }
      };
      blocks.push(b);
    }
    BlockStack { blocks }
  }

  pub fn apply(
    &self,
    trunk: &mut [f32],
    trunk_scratch: &mut [f32],
    mask: &[f32],
    mask_sum: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) {
    for b in &self.blocks {
      b.apply(trunk, trunk_scratch, mask, mask_sum, batch, nn_x, nn_y);
    }
  }
}

// ---------------------------------------------------------------------------
// NestedBottleneckResidualBlock
// ---------------------------------------------------------------------------

pub struct NestedBottleneckResidualBlockLayer {
  pub name: String,
  pub nac1: NormActConv,
  pub inner: BlockStack,
  pub nac2: NormActConv,
}

impl NestedBottleneckResidualBlockLayer {
  pub fn new(
    desc: &NestedBottleneckResidualBlockDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    NestedBottleneckResidualBlockLayer {
      name: desc.name.clone(),
      nac1: NormActConv::new(
        &desc.pre_bn,
        desc.pre_activation.activation,
        &desc.pre_conv,
        nn_x,
        nn_y,
      ),
      inner: BlockStack::new(&desc.blocks, nn_x, nn_y),
      nac2: NormActConv::new(
        &desc.post_bn,
        desc.post_activation.activation,
        &desc.post_conv,
        nn_x,
        nn_y,
      ),
    }
  }

  pub fn apply(
    &self,
    trunk: &mut [f32],
    trunk_scratch: &mut [f32],
    mask: &[f32],
    mask_sum: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) {
    let mid_c = self.nac1.out_c;
    let hw = nn_x * nn_y;
    let mut mid = vec![0.0f32; batch * hw * mid_c];
    let mut mid_scratch = vec![0.0f32; batch * hw * mid_c];

    self.nac1.apply(trunk, trunk_scratch, &mut mid, mask, batch, nn_x, nn_y, false);
    self.inner.apply(&mut mid, &mut mid_scratch, mask, mask_sum, batch, nn_x, nn_y);
    self.nac2.apply(&mid, &mut mid_scratch, trunk, mask, batch, nn_x, nn_y, true);
  }
}

// ---------------------------------------------------------------------------
// SGFMetadataEncoder
// ---------------------------------------------------------------------------

pub struct SgfMetadataEncoder {
  pub name: String,
  pub mul1: MatMulLayer,
  pub bias1: MatBiasLayer,
  pub act1: ActivationLayer,
  pub mul2: MatMulLayer,
  pub bias2: MatBiasLayer,
  pub act2: ActivationLayer,
  pub mul3: MatMulLayer,
}

impl SgfMetadataEncoder {
  pub fn new(desc: &SgfMetadataEncoderDesc) -> Self {
    SgfMetadataEncoder {
      name: desc.name.clone(),
      mul1: MatMulLayer::new(&desc.mul1),
      bias1: MatBiasLayer::new(&desc.bias1),
      act1: ActivationLayer::new(&desc.act1.name, desc.act1.activation),
      mul2: MatMulLayer::new(&desc.mul2),
      bias2: MatBiasLayer::new(&desc.bias2),
      act2: ActivationLayer::new(&desc.act2.name, desc.act2.activation),
      mul3: MatMulLayer::new(&desc.mul3),
    }
  }

  /// `input`: [meta_c * batch], `output`: [out_c * batch].
  pub fn apply(&self, input: &[f32], output: &mut [f32], batch: usize) {
    let c1 = self.mul1.out_c;
    let c2 = self.mul2.out_c;
    let max_c = c1.max(c2);
    let mut buf1 = vec![0.0f32; max_c * batch];
    let mut buf2 = vec![0.0f32; max_c * batch];

    self.mul1.apply(input, &mut buf1[..c1 * batch], batch);
    self.bias1.apply(&mut buf1[..c1 * batch], batch);
    self.act1.apply_inplace(&mut buf1[..c1 * batch]);
    self.mul2.apply(&buf1[..c1 * batch], &mut buf2[..c2 * batch], batch);
    self.bias2.apply(&mut buf2[..c2 * batch], batch);
    self.act2.apply_inplace(&mut buf2[..c2 * batch]);
    self.mul3.apply(&buf2[..c2 * batch], output, batch);
  }
}

// ---------------------------------------------------------------------------
// Trunk
// ---------------------------------------------------------------------------

pub struct Trunk {
  pub name: String,
  pub initial_conv: ConvLayer,
  pub initial_mat_mul: MatMulLayer,
  pub sgf_meta_encoder: Option<SgfMetadataEncoder>,
  pub blocks: BlockStack,
  pub trunk_tip_bn: BatchNormLayer,
  pub trunk_c: usize,
}

impl Trunk {
  pub fn new(
    desc: &crate::model::TrunkDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    let meta_enc = desc.sgf_metadata_encoder.as_ref().map(SgfMetadataEncoder::new);
    Trunk {
      name: desc.name.clone(),
      initial_conv: ConvLayer::new(&desc.initial_conv, nn_x, nn_y),
      initial_mat_mul: MatMulLayer::new(&desc.initial_mat_mul),
      sgf_meta_encoder: meta_enc,
      blocks: BlockStack::new(&desc.blocks, nn_x, nn_y),
      trunk_tip_bn: BatchNormLayer::new(
        &desc.trunk_tip_bn,
        desc.trunk_tip_activation.activation,
      ),
      trunk_c: desc.trunk_num_channels as usize,
    }
  }

  /// Run the trunk.
  ///
  /// Returns the post-BN trunk tensor in `trunk_out` (NHWC [N*H*W*C]).
  #[allow(clippy::too_many_arguments)]
  pub fn apply(
    &self,
    input: &[f32],
    input_global: &[f32],
    input_meta: Option<&[f32]>,
    trunk_out: &mut [f32],
    mask: &[f32],
    mask_sum: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) {
    let tc = self.trunk_c;
    let hw = nn_x * nn_y;

    // trunk_scratch will hold the post-conv result before BN/tip
    let mut trunk_scratch = vec![0.0f32; batch * hw * tc];
    let mut mat_out = vec![0.0f32; tc * batch];

    // initial conv: input → trunk_scratch
    self.initial_conv.apply(input, &mut trunk_scratch, batch, false);

    // initial mat mul: global → mat_out, then broadcast-add to trunk_scratch
    self.initial_mat_mul.apply(input_global, &mut mat_out, batch);
    add_nc_bias_inplace(&mut trunk_scratch, &mat_out, batch, nn_x, nn_y, tc);

    // optional SGF metadata encoder
    if let (Some(enc), Some(meta)) = (&self.sgf_meta_encoder, input_meta) {
      enc.apply(meta, &mut mat_out, batch);
      add_nc_bias_inplace(&mut trunk_scratch, &mat_out, batch, nn_x, nn_y, tc);
    }

    // residual block stack: flip trunk_scratch ↔ trunk_out as double-buffer
    self.blocks.apply(&mut trunk_scratch, trunk_out, mask, mask_sum, batch, nn_x, nn_y);

    // final BN tip: trunk_scratch → trunk_out
    let ts_clone = trunk_scratch.clone();
    self.trunk_tip_bn.apply(&ts_clone, trunk_out, mask, batch, nn_x, nn_y);
  }
}

// ---------------------------------------------------------------------------
// PolicyHead
// ---------------------------------------------------------------------------

pub struct PolicyHead {
  pub name: String,
  pub model_version: i32,
  pub p1_conv: ConvLayer,
  pub g1_conv: ConvLayer,
  pub g1_bn: BatchNormLayer,
  pub gpool_to_bias_mul: MatMulLayer,
  pub p1_bn: BatchNormLayer,
  pub p2_conv: ConvLayer,
  pub gpool_to_pass_mul: MatMulLayer,
  /// v15+
  pub gpool_to_pass_bias: Option<MatBiasLayer>,
  pub pass_activation: Option<ActivationLayer>,
  pub gpool_to_pass_mul2: Option<MatMulLayer>,
}

impl PolicyHead {
  pub fn new(
    desc: &crate::model::PolicyHeadDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    PolicyHead {
      name: desc.name.clone(),
      model_version: 0, // will be set from model
      p1_conv: ConvLayer::new(&desc.p1_conv, nn_x, nn_y),
      g1_conv: ConvLayer::new(&desc.g1_conv, nn_x, nn_y),
      g1_bn: BatchNormLayer::new(&desc.g1_bn, desc.g1_activation.activation),
      gpool_to_bias_mul: MatMulLayer::new(&desc.gpool_to_bias_mul),
      p1_bn: BatchNormLayer::new(&desc.p1_bn, desc.p1_activation.activation),
      p2_conv: ConvLayer::new(&desc.p2_conv, nn_x, nn_y),
      gpool_to_pass_mul: MatMulLayer::new(&desc.gpool_to_pass_mul),
      gpool_to_pass_bias: desc
        .gpool_to_pass_bias
        .as_ref()
        .map(MatBiasLayer::new),
      pass_activation: desc.pass_activation.as_ref().map(|a| {
        ActivationLayer::new(&a.name, a.activation)
      }),
      gpool_to_pass_mul2: desc
        .gpool_to_pass_mul2
        .as_ref()
        .map(MatMulLayer::new),
    }
  }

  /// Returns `(policy_pass, policy_spatial)`.
  /// `policy_pass`: [policy_ch * batch]
  /// `policy_spatial`: NHWC [N*H*W*policy_ch]
  #[allow(clippy::too_many_arguments)]
  pub fn apply(
    &self,
    trunk: &[f32],
    mask: &[f32],
    mask_sum: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
    model_version: i32,
  ) -> (Vec<f32>, Vec<f32>) {
    let hw = nn_x * nn_y;
    let p1c = self.p1_conv.out_c;
    let g1c = self.g1_conv.out_c;
    let p2c = self.p2_conv.out_c;

    let mut p1_out = vec![0.0f32; batch * hw * p1c];
    let mut p1_out2 = vec![0.0f32; batch * hw * p1c];
    let mut g1_out = vec![0.0f32; batch * hw * g1c];
    let mut g1_out2 = vec![0.0f32; batch * hw * g1c];
    let mut g1_concat = vec![0.0f32; g1c * 3 * batch];
    let mut g1_bias = vec![0.0f32; p1c * batch];
    let mut policy_pass = vec![0.0f32; p2c * batch];
    let mut policy_spatial = vec![0.0f32; batch * hw * p2c];

    self.p1_conv.apply(trunk, &mut p1_out, batch, false);
    self.g1_conv.apply(trunk, &mut g1_out, batch, false);
    self.g1_bn.apply(&g1_out.clone(), &mut g1_out2, mask, batch, nn_x, nn_y);
    pool_rows_gpool(&g1_out2, &mut g1_concat, mask, mask_sum, batch, nn_x, nn_y, g1c);
    self.gpool_to_bias_mul.apply(&g1_concat, &mut g1_bias, batch);
    add_nc_bias_inplace(&mut p1_out, &g1_bias, batch, nn_x, nn_y, p1c);
    self.p1_bn.apply(&p1_out.clone(), &mut p1_out2, mask, batch, nn_x, nn_y);
    self.p2_conv.apply(&p1_out2, &mut policy_spatial, batch, false);

    if model_version >= 15 {
      // v15+: pass = mul(g1_concat) + bias → act → mul2
      let mut p1_pass = vec![0.0f32; p1c * batch];
      self.gpool_to_pass_mul.apply(&g1_concat, &mut p1_pass, batch);
      if let Some(b) = &self.gpool_to_pass_bias {
        b.apply(&mut p1_pass, batch);
      }
      if let Some(a) = &self.pass_activation {
        a.apply_inplace(&mut p1_pass);
      }
      if let Some(m) = &self.gpool_to_pass_mul2 {
        m.apply(&p1_pass, &mut policy_pass, batch);
      }
    } else {
      self.gpool_to_pass_mul.apply(&g1_concat, &mut policy_pass, batch);
    }

    (policy_pass, policy_spatial)
  }
}

// ---------------------------------------------------------------------------
// ValueHead
// ---------------------------------------------------------------------------

pub struct ValueHead {
  pub name: String,
  pub v1_conv: ConvLayer,
  pub v1_bn: BatchNormLayer,
  pub v2_mul: MatMulLayer,
  pub v2_bias: MatBiasLayer,
  pub v2_activation: ActivationLayer,
  pub v3_mul: MatMulLayer,
  pub v3_bias: MatBiasLayer,
  pub sv3_mul: MatMulLayer,
  pub sv3_bias: MatBiasLayer,
  pub v_ownership_conv: ConvLayer,
}

impl ValueHead {
  pub fn new(
    desc: &crate::model::ValueHeadDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    ValueHead {
      name: desc.name.clone(),
      v1_conv: ConvLayer::new(&desc.v1_conv, nn_x, nn_y),
      v1_bn: BatchNormLayer::new(&desc.v1_bn, desc.v1_activation.activation),
      v2_mul: MatMulLayer::new(&desc.v2_mul),
      v2_bias: MatBiasLayer::new(&desc.v2_bias),
      v2_activation: ActivationLayer::new(
        &desc.v2_activation.name,
        desc.v2_activation.activation,
      ),
      v3_mul: MatMulLayer::new(&desc.v3_mul),
      v3_bias: MatBiasLayer::new(&desc.v3_bias),
      sv3_mul: MatMulLayer::new(&desc.sv3_mul),
      sv3_bias: MatBiasLayer::new(&desc.sv3_bias),
      v_ownership_conv: ConvLayer::new(&desc.v_ownership_conv, nn_x, nn_y),
    }
  }

  /// Returns `(value, score_value, ownership)`.
  /// `value`: [value_ch * batch]
  /// `score_value`: [score_ch * batch]
  /// `ownership`: NHWC [N*H*W*ownership_ch]
  pub fn apply(
    &self,
    trunk: &[f32],
    mask: &[f32],
    mask_sum: &[f32],
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hw = nn_x * nn_y;
    let v1c = self.v1_conv.out_c;
    let v2c = self.v2_mul.out_c;
    let v3c = self.v3_mul.out_c;
    let sv3c = self.sv3_mul.out_c;
    let owc = self.v_ownership_conv.out_c;

    let mut v1_out = vec![0.0f32; batch * hw * v1c];
    let mut v1_out2 = vec![0.0f32; batch * hw * v1c];
    let mut v1_mean = vec![0.0f32; v1c * 3 * batch];
    let mut v2_out = vec![0.0f32; v2c * batch];
    let mut value = vec![0.0f32; v3c * batch];
    let mut score_value = vec![0.0f32; sv3c * batch];
    let mut ownership = vec![0.0f32; batch * hw * owc];

    self.v1_conv.apply(trunk, &mut v1_out, batch, false);
    self.v1_bn.apply(&v1_out.clone(), &mut v1_out2, mask, batch, nn_x, nn_y);
    pool_rows_value_head(&v1_out2, &mut v1_mean, mask_sum, batch, nn_x, nn_y, v1c);
    self.v2_mul.apply(&v1_mean, &mut v2_out, batch);
    self.v2_bias.apply(&mut v2_out, batch);
    self.v2_activation.apply_inplace(&mut v2_out);
    self.v3_mul.apply(&v2_out, &mut value, batch);
    self.v3_bias.apply(&mut value, batch);
    self.sv3_mul.apply(&v2_out, &mut score_value, batch);
    self.sv3_bias.apply(&mut score_value, batch);
    self.v_ownership_conv.apply(&v1_out2, &mut ownership, batch, false);

    (value, score_value, ownership)
  }
}

// ---------------------------------------------------------------------------
// Model (assembled network)
// ---------------------------------------------------------------------------

pub struct Model {
  pub name: String,
  pub model_version: i32,
  pub num_input_channels: usize,
  pub num_input_global_channels: usize,
  pub num_input_meta_channels: usize,
  pub num_policy_channels: usize,
  pub num_value_channels: usize,
  pub num_score_value_channels: usize,
  pub num_ownership_channels: usize,
  pub trunk: Trunk,
  pub policy_head: PolicyHead,
  pub value_head: ValueHead,
}

impl Model {
  pub fn new(
    desc: &crate::model::ModelDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Self {
    Model {
      name: desc.name.clone(),
      model_version: desc.model_version,
      num_input_channels: desc.num_input_channels as usize,
      num_input_global_channels: desc.num_input_global_channels as usize,
      num_input_meta_channels: desc.num_input_meta_channels as usize,
      num_policy_channels: desc.num_policy_channels as usize,
      num_value_channels: desc.num_value_channels as usize,
      num_score_value_channels: desc.num_score_value_channels as usize,
      num_ownership_channels: desc.num_ownership_channels as usize,
      trunk: Trunk::new(&desc.trunk, nn_x, nn_y),
      policy_head: PolicyHead::new(&desc.policy_head, nn_x, nn_y),
      value_head: ValueHead::new(&desc.value_head, nn_x, nn_y),
    }
  }

  /// Run inference for a batch.
  ///
  /// Input tensors:
  ///   `input`:        NHWC [N * H * W * num_input_channels]
  ///   `input_global`: [num_input_global_channels * N]
  ///   `input_meta`:   [num_input_meta_channels * N] (or None)
  ///
  /// Returns `(policy_pass, policy_spatial, value, score_value, ownership)`.
  pub fn apply(
    &self,
    input: &[f32],
    input_global: &[f32],
    input_meta: Option<&[f32]>,
    batch: usize,
    nn_x: usize,
    nn_y: usize,
  ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let tc = self.trunk.trunk_c;
    let hw = nn_x * nn_y;

    // Compute mask from channel 0 of input (NHW binary mask)
    let ic = self.num_input_channels;
    let mut mask = vec![0.0f32; batch * hw];
    for n in 0..batch {
      for pos in 0..hw {
        mask[n * hw + pos] = input[(n * hw + pos) * ic];
      }
    }
    let mask_sum = compute_mask_sum(&mask, nn_x, nn_y, batch);

    let mut trunk_out = vec![0.0f32; batch * hw * tc];
    self.trunk.apply(
      input,
      input_global,
      input_meta,
      &mut trunk_out,
      &mask,
      &mask_sum,
      batch,
      nn_x,
      nn_y,
    );

    let (policy_pass, policy_spatial) = self.policy_head.apply(
      &trunk_out,
      &mask,
      &mask_sum,
      batch,
      nn_x,
      nn_y,
      self.model_version,
    );

    let (value, score_value, ownership) = self.value_head.apply(
      &trunk_out,
      &mask,
      &mask_sum,
      batch,
      nn_x,
      nn_y,
    );

    (policy_pass, policy_spatial, value, score_value, ownership)
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;
  use crate::model::{
    Activation, ActivationLayerDesc, BatchNormLayerDesc, ConvLayerDesc,
    MatBiasLayerDesc, MatMulLayerDesc,
  };

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  fn assert_close(a: f32, b: f32, tol: f32, label: &str) {
    assert!(
      (a - b).abs() <= tol,
      "{label}: expected {b}, got {a} (diff {})",
      (a - b).abs()
    );
  }

  fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
      assert!(
        (av - bv).abs() <= tol,
        "{label}[{i}]: expected {bv}, got {av} (diff {})",
        (av - bv).abs()
      );
    }
  }

  /// Build a minimal `ConvLayerDesc` from raw parts (no file I/O needed).
  fn make_conv_desc(
    ky: i32,
    kx: i32,
    ic: i32,
    oc: i32,
    weights: Vec<f32>,
  ) -> ConvLayerDesc {
    ConvLayerDesc {
      name: "test_conv".into(),
      conv_y_size: ky,
      conv_x_size: kx,
      in_channels: ic,
      out_channels: oc,
      dilation_y: 1,
      dilation_x: 1,
      weights,
    }
  }

  fn make_bn_desc(nc: i32, scale: Vec<f32>, bias: Vec<f32>) -> BatchNormLayerDesc {
    // merged_scale = scale / sqrt(var + eps) with var=0, eps=1e-5
    // → scale / sqrt(1e-5)  but here we let caller pass pre-merged values
    // directly via merged_scale/merged_bias for clarity.
    let eps = 1e-5f32;
    let merged_scale: Vec<f32> = scale
      .iter()
      .map(|&s| s / (0.0f32 + eps).sqrt())
      .collect();
    let merged_bias: Vec<f32> = bias
      .iter()
      .zip(merged_scale.iter())
      .map(|(&b, &ms)| b - ms * 0.0f32)
      .collect();
    BatchNormLayerDesc {
      name: "test_bn".into(),
      num_channels: nc,
      epsilon: eps,
      has_scale: true,
      has_bias: true,
      mean: vec![0.0f32; nc as usize],
      variance: vec![0.0f32; nc as usize],
      scale: scale.clone(),
      bias: bias.clone(),
      merged_scale,
      merged_bias,
    }
  }

  /// Build a thin `BatchNormLayerDesc` whose merged_scale and merged_bias are
  /// already exactly what we want (used by `BatchNormLayer::new` directly).
  fn make_bn_desc_merged(merged_scale: Vec<f32>, merged_bias: Vec<f32>) -> BatchNormLayerDesc {
    let nc = merged_scale.len() as i32;
    BatchNormLayerDesc {
      name: "test_bn".into(),
      num_channels: nc,
      epsilon: 1e-5,
      has_scale: true,
      has_bias: true,
      mean: vec![0.0f32; nc as usize],
      variance: vec![0.0f32; nc as usize],
      scale: merged_scale.clone(),
      bias: merged_bias.clone(),
      merged_scale,
      merged_bias,
    }
  }

  fn make_matmul_desc(ic: i32, oc: i32, weights_ic_oc: Vec<f32>) -> MatMulLayerDesc {
    MatMulLayerDesc {
      name: "test_mm".into(),
      in_channels: ic,
      out_channels: oc,
      weights: weights_ic_oc,
    }
  }

  fn make_matbias_desc(nc: i32, weights: Vec<f32>) -> MatBiasLayerDesc {
    MatBiasLayerDesc { name: "test_bias".into(), num_channels: nc, weights }
  }

  // -------------------------------------------------------------------------
  // compute_mask_sum
  // -------------------------------------------------------------------------

  #[test]
  fn mask_sum_simple() {
    // 2 batch, 2×2 spatial.  batch 0 = all ones; batch 1 = two ones.
    let mask: Vec<f32> = vec![
      1.0, 1.0, 1.0, 1.0, // n=0
      1.0, 1.0, 0.0, 0.0, // n=1
    ];
    let sums = compute_mask_sum(&mask, 2, 2, 2);
    assert_close(sums[0], 4.0, 1e-6, "mask_sum[0]");
    assert_close(sums[1], 2.0, 1e-6, "mask_sum[1]");
  }

  // -------------------------------------------------------------------------
  // apply_activation
  // -------------------------------------------------------------------------

  #[test]
  fn activation_identity() {
    assert_close(apply_activation(-3.0, Activation::Identity), -3.0, 1e-6, "id");
    assert_close(apply_activation(5.0, Activation::Identity), 5.0, 1e-6, "id");
  }

  #[test]
  fn activation_relu() {
    assert_close(apply_activation(-1.0, Activation::Relu), 0.0, 1e-6, "relu<0");
    assert_close(apply_activation(0.0, Activation::Relu), 0.0, 1e-6, "relu=0");
    assert_close(apply_activation(2.5, Activation::Relu), 2.5, 1e-6, "relu>0");
  }

  #[test]
  fn activation_mish() {
    // mish(0) = 0 * tanh(ln2) ≈ 0
    assert_close(apply_activation(0.0, Activation::Mish), 0.0, 1e-5, "mish(0)");
    // mish(1) = tanh(ln(1+e)) ≈ 0.86509
    let expected = 1.0_f32 * (1.0_f32 + 1.0_f32.exp()).ln().tanh();
    assert_close(apply_activation(1.0, Activation::Mish), expected, 1e-5, "mish(1)");
    // mish(-5) should be close to 0 (negative saturation)
    let v = apply_activation(-5.0, Activation::Mish);
    assert!(v < 0.0 && v > -1.0, "mish(-5)={v} not in (-1,0)");
    // For large x, mish(20) ≈ 20 (tanh(softplus(20))≈1)
    let v20 = apply_activation(20.0, Activation::Mish);
    assert_close(v20, 20.0, 0.01, "mish(20)");
  }

  // -------------------------------------------------------------------------
  // MatBiasLayer
  // -------------------------------------------------------------------------

  #[test]
  fn mat_bias_adds_correctly() {
    // 3 channels, batch 2. Layout [c * batch + n]:
    // channel 0 bias=1, channel 1 bias=2, channel 2 bias=3.
    let desc = make_matbias_desc(3, vec![1.0, 2.0, 3.0]);
    let layer = MatBiasLayer::new(&desc);
    // mat: [c=0,n=0]=10, [c=0,n=1]=20, [c=1,n=0]=30, [c=1,n=1]=40, ...
    let mut mat = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    layer.apply(&mut mat, 2);
    // After: [c=0,n=0]=11, [c=0,n=1]=21, [c=1,n=0]=32, [c=1,n=1]=42, [c=2]=53,63
    assert_slice_close(&mat, &[11.0, 21.0, 32.0, 42.0, 53.0, 63.0], 1e-6, "mat_bias");
  }

  // -------------------------------------------------------------------------
  // MatMulLayer
  // -------------------------------------------------------------------------

  #[test]
  fn matmul_identity_2x2() {
    // Identity: weights[ic,oc] = delta(ic,oc). File order: ic=0 oc=0,  ic=0
    // oc=1,  ic=1 oc=0,  ic=1 oc=1  →  [1,0,0,1].
    let desc = make_matmul_desc(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let layer = MatMulLayer::new(&desc);
    // input [ic * batch]: ic=0 → [3,4], ic=1 → [5,6]
    let input = vec![3.0f32, 4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; 4];
    layer.apply(&input, &mut output, 2);
    // Should be identity: output == input
    assert_slice_close(&output, &input, 1e-6, "matmul_id");
  }

  #[test]
  fn matmul_2x2_known() {
    // W = [[2,0],[1,3]] in [oc,ic] convention.
    // File (ic,oc) order:  ic=0 oc=0 → 2, ic=0 oc=1 → 0, ic=1 oc=0 → 1,
    // ic=1 oc=1 → 3.   File vec: [2,0,1,3].
    let desc = make_matmul_desc(2, 2, vec![2.0, 0.0, 1.0, 3.0]);
    let layer = MatMulLayer::new(&desc);
    // input for batch=1: ic=0 → 1.0, ic=1 → 2.0
    let input = vec![1.0f32, 2.0]; // [ic * batch] with batch=1
    let mut output = vec![0.0f32; 2];
    layer.apply(&input, &mut output, 1);
    // oc=0: 2*1 + 1*2 = 4   (W_T[oc=0] = (W[ic,oc=0]) = [2,1])
    // Wait – file order [ic,oc]: w[ic=0,oc=0]=2, w[ic=0,oc=1]=0, w[ic=1,oc=0]=1, w[ic=1,oc=1]=3
    // After transposing in MatMulLayer::new to [oc,ic]:
    //   w_t[oc=0,ic=0]=2, w_t[oc=0,ic=1]=1, w_t[oc=1,ic=0]=0, w_t[oc=1,ic=1]=3
    // out[oc=0] = 2*1 + 1*2 = 4
    // out[oc=1] = 0*1 + 3*2 = 6
    assert_slice_close(&output, &[4.0, 6.0], 1e-6, "matmul_2x2");
  }

  // -------------------------------------------------------------------------
  // BatchNormLayer
  // -------------------------------------------------------------------------

  #[test]
  fn batch_norm_scales_and_masks() {
    // 1 channel, 2×2 spatial, batch=1.
    // merged_scale=2, merged_bias=1. mask: top-left ON, rest OFF.
    let desc = make_bn_desc_merged(vec![2.0], vec![1.0]);
    let layer = BatchNormLayer::new(&desc, Activation::Identity);
    // input: [N*H*W*C] = 4 values, channel dim=1 so just [4]
    let input = vec![3.0f32, 5.0, 7.0, 9.0]; // positions 0..3
    let mask = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut output = vec![0.0f32; 4];
    layer.apply(&input, &mut output, &mask, 1, 2, 2);
    // pos 0 (mask=1): 3*2+1=7; others (mask=0): 0
    assert_slice_close(&output, &[7.0, 0.0, 0.0, 0.0], 1e-6, "bn_mask");
  }

  #[test]
  fn batch_norm_relu_activation() {
    let desc = make_bn_desc_merged(vec![1.0], vec![-4.0]); // shift by -4
    let layer = BatchNormLayer::new(&desc, Activation::Relu);
    // input = 3,5  → after BN: 3-4=-1 (relu→0), 5-4=1 (relu→1)
    let input = vec![3.0f32, 5.0]; // 1 channel, 2 positions, batch=1, nn_x=2,nn_y=1
    let mask = vec![1.0f32, 1.0];
    let mut output = vec![0.0f32; 2];
    layer.apply(&input, &mut output, &mask, 1, 2, 1);
    assert_slice_close(&output, &[0.0, 1.0], 1e-6, "bn_relu");
  }

  // -------------------------------------------------------------------------
  // add_nc_bias_inplace
  // -------------------------------------------------------------------------

  #[test]
  fn add_nc_bias_broadcasts_over_spatial() {
    // 2 channels, 2×2 spatial, batch=1.
    // bias [c=0,n=0]=10, [c=1,n=0]=20.
    // tensor: pos 0 → (c0=1, c1=2), pos 1 → (c0=3, c1=4), …
    let mut tensor = vec![
      1.0f32, 2.0, // pos 0 (y=0,x=0): c0, c1
      3.0, 4.0,   // pos 1 (y=0,x=1)
      5.0, 6.0,   // pos 2 (y=1,x=0)
      7.0, 8.0,   // pos 3 (y=1,x=1)
    ];
    let bias = vec![10.0f32, 20.0]; // [c * batch] with batch=1: c0=10, c1=20
    add_nc_bias_inplace(&mut tensor, &bias, 1, 2, 2, 2);
    let expected = [11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0];
    assert_slice_close(&tensor, &expected, 1e-6, "add_bias");
  }

  // -------------------------------------------------------------------------
  // pool_rows_gpool
  // -------------------------------------------------------------------------

  #[test]
  fn pool_gpool_single_channel_known_values() {
    // 1 channel, 3×1 spatial (H=3, W=1), batch=1.
    // mask all ones, mask_sum=3. input values: 2, 4, 6 → mean=4.
    // sqrt(3)≈1.732, out[0]=mean=4, out[1]=mean*(sqrt-14)*0.1 ≈ 4*(1.732-14)*0.1=-4.907
    // out[2]=max = max(2+(1-1), 4+(1-1), 6+(1-1)) = 6
    let input = vec![2.0f32, 4.0, 6.0]; // NHWC [1*3*1*1]
    let mask = vec![1.0f32, 1.0, 1.0];
    let mask_sum = vec![3.0f32];
    let mut out = vec![0.0f32; 3]; // 3 channels (mean, sqrt-mean, max)
    pool_rows_gpool(&input, &mut out, &mask, &mask_sum, 1, 1, 3, 1);
    let sqrtdiv = 3.0f32.sqrt();
    assert_close(out[0], 4.0, 1e-5, "gpool mean");
    assert_close(out[1], 4.0 * (sqrtdiv - 14.0) * 0.1, 1e-5, "gpool sqrt-mean");
    assert_close(out[2], 6.0, 1e-5, "gpool max");
  }

  #[test]
  fn pool_gpool_mask_affects_max() {
    // 1 channel, 2×1 spatial, batch=1. mask: first ON, second OFF.
    // input: 5, 100. max should be 5 (100 is padded → 100+(0-1)=99 < 5+(1-1)=5).
    // Actually the max init is -1.0, then: cand(0)=5+(1-1)=5, cand(1)=100+(0-1)=99.
    // Wait — 99 > 5, so max=99. That is the C++ design: padded cells are
    // 100 + (0-1)=99 which would beat 5. But all *valid* cells give 5+(1-1)=5.
    // The C++ comment says padding MUST be zero (after BN+mask). Here we test
    // with input=100 at a masked-off position to confirm the formula.
    let input = vec![5.0f32, 100.0]; // NHWC
    let mask = vec![1.0f32, 0.0];
    let mask_sum = vec![1.0f32];
    let mut out = vec![0.0f32; 3];
    pool_rows_gpool(&input, &mut out, &mask, &mask_sum, 1, 1, 2, 1);
    // mean over ALL cells (not masked): 5+100=105, / 1 = 105  (matches C++)
    assert_close(out[0], 105.0, 1e-4, "gpool mean unmasked");
    // max: max(-1, 5+(1-1), 100+(0-1)) = max(-1,5,99) = 99
    assert_close(out[2], 99.0, 1e-4, "gpool max with pad value");
  }

  // -------------------------------------------------------------------------
  // pool_rows_value_head
  // -------------------------------------------------------------------------

  #[test]
  fn pool_value_head_single_channel() {
    // 1 channel, 2×1, batch=1. values=3,7 → mean=5. mask_sum=2, sqrt=sqrt(2).
    let input = vec![3.0f32, 7.0];
    let mask_sum = vec![2.0f32];
    let mut out = vec![0.0f32; 3];
    pool_rows_value_head(&input, &mut out, &mask_sum, 1, 1, 2, 1);
    let sqrtdiv = 2.0f32.sqrt();
    let mean = 5.0f32;
    let sd14 = sqrtdiv - 14.0;
    assert_close(out[0], mean, 1e-5, "vhead mean");
    assert_close(out[1], mean * sd14 * 0.1, 1e-5, "vhead sqrt-term");
    assert_close(out[2], mean * (sd14 * sd14 * 0.01 - 0.1), 1e-5, "vhead quad-term");
  }

  // -------------------------------------------------------------------------
  // ConvLayer — 1×1 (direct path)
  // -------------------------------------------------------------------------

  #[test]
  fn conv1x1_identity_single_channel() {
    // 1×1 conv, 1 in_c, 1 out_c, weight=1.0.  Should copy input unchanged.
    let desc = make_conv_desc(1, 1, 1, 1, vec![1.0]);
    let layer = ConvLayer::new(&desc, 3, 3);
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect(); // 3×3×1 NHWC
    let mut output = vec![0.0f32; 9];
    layer.apply(&input, &mut output, 1, false);
    assert_slice_close(&output, &input, 1e-5, "1x1 identity");
  }

  #[test]
  fn conv1x1_scale_two_channels() {
    // 1×1 conv, 2 in_c, 2 out_c.
    // Weight in [oc, ic, y, x] order: oc=0→(2,0), oc=1→(0,3).
    // File order (y,x,ic,oc): [2,0,0,3]  (for 1×1: one (y,x) tile, ic first then oc)
    // y=0,x=0: ic=0→(oc=0=2, oc=1=0), ic=1→(oc=0=0, oc=1=3)  → file: [2,0,0,3]
    let desc = make_conv_desc(1, 1, 2, 2, vec![2.0, 0.0, 0.0, 3.0]);
    let layer = ConvLayer::new(&desc, 2, 2);
    // 2×2 spatial, batch=1, 2 channels.
    // input NHWC: [pos=0: c0=1,c1=2, pos=1: c0=3,c1=4, pos=2: c0=5,c1=6, pos=3: c0=7,c1=8]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 8];
    layer.apply(&input, &mut output, 1, false);
    // pos=0: out_c0 = 2*1+0*2=2, out_c1 = 0*1+3*2=6
    // pos=1: out_c0 = 2*3+0*4=6, out_c1 = 0*3+3*4=12
    // pos=2: out_c0 = 2*5=10, out_c1=3*6=18
    // pos=3: out_c0 = 2*7=14, out_c1=3*8=24
    let expected = [2.0, 6.0, 6.0, 12.0, 10.0, 18.0, 14.0, 24.0];
    assert_slice_close(&output, &expected, 1e-4, "1x1 2ch");
  }

  #[test]
  fn conv1x1_accumulate_adds_to_existing() {
    let desc = make_conv_desc(1, 1, 1, 1, vec![1.0]);
    let layer = ConvLayer::new(&desc, 2, 1);
    let input = vec![3.0f32, 5.0];
    let mut output = vec![10.0f32, 20.0];
    layer.apply(&input, &mut output, 1, true); // accumulate=true
    assert_slice_close(&output, &[13.0, 25.0], 1e-5, "1x1 accumulate");
  }

  // -------------------------------------------------------------------------
  // ConvLayer — 3×3 Winograd vs direct reference
  // -------------------------------------------------------------------------

  /// Reference direct 3×3 convolution (no Winograd), used for cross-checking.
  fn conv3x3_reference(
    input: &[f32],
    kernel_oc_ic_y_x: &[f32],
    batch: usize,
    h: usize,
    w: usize,
    ic: usize,
    oc: usize,
  ) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * h * w * oc];
    for n in 0..batch {
      for yi in 0..h {
        for xi in 0..w {
          for oc_i in 0..oc {
            let mut acc = 0.0f32;
            for sy in 0..3usize {
              let iy = yi as isize + sy as isize - 1;
              if iy < 0 || iy >= h as isize { continue; }
              for sx in 0..3usize {
                let ix = xi as isize + sx as isize - 1;
                if ix < 0 || ix >= w as isize { continue; }
                for ic_i in 0..ic {
                  let k = kernel_oc_ic_y_x[oc_i * ic * 9 + ic_i * 9 + sy * 3 + sx];
                  let inp = input[(n * h * w + iy as usize * w + ix as usize) * ic + ic_i];
                  acc += k * inp;
                }
              }
            }
            out[(n * h * w + yi * w + xi) * oc + oc_i] = acc;
          }
        }
      }
    }
    out
  }

  #[test]
  fn conv3x3_winograd_matches_direct_1ch() {
    // 1 in_c, 1 out_c, 4×4 spatial.
    // Kernel: 3×3 edge-detector (Laplacian-ish): center=4, neighbours=-1, corners=0.
    // File order [y,x,ic=0,oc=0] (ic,oc vary in inner loop, but ic=oc=1 here):
    //   y,x iterates col-major within the filter.
    let kernel_oc_ic_y_x = [
      0.0f32, -1.0,  0.0,
      -1.0,   4.0,  -1.0,
       0.0,  -1.0,   0.0,
    ]; // [oc=0, ic=0, y, x]
    // File order for parse (y outer, x inner, ic, oc all 1):
    let file_weights: Vec<f32> = kernel_oc_ic_y_x.to_vec();
    let desc = make_conv_desc(3, 3, 1, 1, file_weights.clone());
    let layer = ConvLayer::new(&desc, 4, 4);

    let input: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect(); // 4×4 NHWC, 1ch

    let mut winograd_out = vec![0.0f32; 16];
    layer.apply(&input, &mut winograd_out, 1, false);

    let ref_out = conv3x3_reference(&input, &kernel_oc_ic_y_x, 1, 4, 4, 1, 1);
    assert_slice_close(&winograd_out, &ref_out, 0.01, "3x3_winograd_1ch");
  }

  #[test]
  fn conv3x3_winograd_matches_direct_multichannel() {
    // 2 in_c, 2 out_c, 5×5 spatial, batch=2.
    // Use random-ish but deterministic weights.
    let ic = 2usize;
    let oc = 2usize;
    let h = 5usize;
    let w = 5usize;
    let batch = 2usize;

    // kernel [oc, ic, y, x] — 2*2*9 = 36 values
    let kernel_oc_ic_y_x: Vec<f32> = (0..36)
      .map(|i| ((i as f32 * 0.1 - 1.8) * 0.5))
      .collect();

    // ConvLayerDesc.weights uses [oc, ic, y, x] order (same as kernel_oc_ic_y_x).
    // The C++ desc.cpp re-orders file bytes from [y,x,ic,oc] to [oc,ic,y,x] before
    // storing in desc.weights, so we pass kernel_oc_ic_y_x directly here.
    let desc = make_conv_desc(3, 3, ic as i32, oc as i32, kernel_oc_ic_y_x.clone());
    let layer = ConvLayer::new(&desc, w, h);

    let input: Vec<f32> = (0..(batch * h * w * ic))
      .map(|i| (i as f32 + 1.0) * 0.3)
      .collect();

    let mut winograd_out = vec![0.0f32; batch * h * w * oc];
    layer.apply(&input, &mut winograd_out, batch, false);

    let ref_out = conv3x3_reference(&input, &kernel_oc_ic_y_x, batch, h, w, ic, oc);
    assert_slice_close(&winograd_out, &ref_out, 0.05, "3x3_winograd_multichan");
  }

  #[test]
  fn conv3x3_winograd_accumulate() {
    // Verify that accumulate=true sums into pre-filled output.
    let kernel_oc_ic_y_x = [1.0f32; 9]; // all-ones 3×3, 1ch
    let desc = make_conv_desc(3, 3, 1, 1, vec![1.0f32; 9]);
    let layer = ConvLayer::new(&desc, 3, 3);
    let input = vec![1.0f32; 9]; // 3×3, all ones
    let mut out = vec![5.0f32; 9]; // pre-filled with 5
    layer.apply(&input, &mut out, 1, true);

    // Direct convolution of all-ones 3×3 input with all-ones 3×3 kernel:
    // corners→4, edges→6, center→9 (neighbour counts).
    let ref_out = conv3x3_reference(&vec![1.0f32; 9], &kernel_oc_ic_y_x, 1, 3, 3, 1, 1);
    let expected: Vec<f32> = ref_out.iter().map(|&v| v + 5.0).collect();
    assert_slice_close(&out, &expected, 0.01, "3x3_winograd_accum");
  }

  // -------------------------------------------------------------------------
  // ConvLayer — edge: 1×1 identity is no-op for non-winograd path
  // -------------------------------------------------------------------------

  #[test]
  fn conv_zero_weights_produces_zeros() {
    let desc = make_conv_desc(1, 1, 2, 3, vec![0.0f32; 6]);
    let layer = ConvLayer::new(&desc, 3, 3);
    let input: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let mut output = vec![99.0f32; 27]; // pre-filled
    layer.apply(&input, &mut output, 1, false);
    assert!(output.iter().all(|&v| v == 0.0), "zero weights: {output:?}");
  }

  // -------------------------------------------------------------------------
  // BatchNormLayer multi-channel
  // -------------------------------------------------------------------------

  #[test]
  fn batch_norm_multi_channel_mish() {
    // 2 channels: scale=[1,2], bias=[0,1]. merged_scale/bias set directly.
    // Input pos 0 ch 0 = 1.0 → mish(1*1+0) = mish(1) ≈ 0.865
    // Input pos 0 ch 1 = 2.0 → mish(2*2+1) = mish(5) ≈ 4.988 (≈5 for large x)
    let desc = make_bn_desc_merged(vec![1.0, 2.0], vec![0.0, 1.0]);
    let layer = BatchNormLayer::new(&desc, Activation::Mish);
    let input = vec![1.0f32, 2.0]; // 1 pos, 2 channels, batch=1, nn_x=1,nn_y=1
    let mask = vec![1.0f32];
    let mut output = vec![0.0f32; 2];
    layer.apply(&input, &mut output, &mask, 1, 1, 1);

    let expected0 = apply_activation(1.0 * 1.0 + 0.0, Activation::Mish);
    let expected1 = apply_activation(2.0 * 2.0 + 1.0, Activation::Mish);
    assert_close(output[0], expected0, 1e-5, "bn_mish ch0");
    assert_close(output[1], expected1, 1e-5, "bn_mish ch1");
  }

  // -------------------------------------------------------------------------
  // ActivationLayer (standalone)
  // -------------------------------------------------------------------------

  #[test]
  fn activation_layer_relu_inplace() {
    let layer = ActivationLayer::new("test", Activation::Relu);
    let mut data = vec![-3.0f32, 0.0, 2.5, -0.1, 4.0];
    layer.apply_inplace(&mut data);
    assert_slice_close(&data, &[0.0, 0.0, 2.5, 0.0, 4.0], 1e-6, "act_relu");
  }

  // -------------------------------------------------------------------------
  // Winograd transform helper round-trips
  // -------------------------------------------------------------------------

  /// Verify that `B^T d B` and `A^T (GgG^T) A` compose correctly for 3×3.
  /// For a unit impulse at the centre of a 4×4 output tile, the input
  /// transform and output transform should be inverses.
  #[test]
  fn winograd_3x3_transform_round_trip() {
    // Use identity kernel (only center=1, rest=0).
    let mut kernel = [0.0f32; 9]; // [y,x], 1ch
    kernel[4] = 1.0; // centre of 3×3 filter
    let desc = make_conv_desc(3, 3, 1, 1, kernel.to_vec());
    let layer = ConvLayer::new(&desc, 4, 4);

    // Input: only cell (y=1, x=1) = 1.0
    let mut input = vec![0.0f32; 16];
    input[1 * 4 + 1] = 1.0;

    let mut output = vec![0.0f32; 16];
    layer.apply(&input, &mut output, 1, false);

    // With identity kernel and a single 1.0: output should equal input.
    assert_slice_close(&output, &input, 0.01, "winograd_identity_round_trip");
  }
}
