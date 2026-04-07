/// KataGo neural network model descriptor and loader.
///
/// Translates the C++ model loading code from `cpp/neuralnet/desc.cpp` and
/// `cpp/neuralnet/desc.h` into Rust.  Supports `.bin.gz` (binary floats,
/// gzip-compressed) and `.txt.gz` (text floats, gzip-compressed) as well as
/// their uncompressed variants.
use std::io::{self, BufRead, Read};

// ---------------------------------------------------------------------------
// Activation kinds (from activations.h)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
  Identity,
  Relu,
  Mish,
  /// Mish with 1/8 scale applied to reduce activation magnitude.
  MishScale8,
}

// ---------------------------------------------------------------------------
// Low-level token / float reader
// ---------------------------------------------------------------------------

/// A minimal whitespace-delimited token reader that can also extract raw
/// binary float blocks in KataGo's `@BIN@` format.
pub struct TokenReader<R: Read> {
  inner: io::BufReader<R>,
  /// If true the float blocks use `@BIN@` + little-endian binary encoding.
  pub binary_floats: bool,
}

impl<R: Read> TokenReader<R> {
  pub fn new(reader: R, binary_floats: bool) -> Self {
    Self {
      inner: io::BufReader::new(reader),
      binary_floats,
    }
  }

  /// Read the next whitespace-delimited token (like `in >> str` in C++).
  pub fn read_token(&mut self) -> Result<String, String> {
    // Skip leading whitespace
    loop {
      let buf = self.inner.fill_buf().map_err(|e| e.to_string())?;
      if buf.is_empty() {
        return Err("unexpected end of stream while reading token".into());
      }
      if !buf[0].is_ascii_whitespace() {
        break;
      }
      self.inner.consume(1);
    }
    // Read until whitespace
    let mut token = Vec::new();
    loop {
      let buf = self.inner.fill_buf().map_err(|e| e.to_string())?;
      if buf.is_empty() || buf[0].is_ascii_whitespace() {
        break;
      }
      token.push(buf[0]);
      self.inner.consume(1);
    }
    String::from_utf8(token).map_err(|e| e.to_string())
  }

  /// Read a single byte from the underlying stream (no buffering flush).
  fn read_byte(&mut self) -> Result<u8, String> {
    let buf = self.inner.fill_buf().map_err(|e| e.to_string())?;
    if buf.is_empty() {
      return Err("unexpected end of stream reading byte".into());
    }
    let b = buf[0];
    self.inner.consume(1);
    Ok(b)
  }

  pub fn read_int(&mut self) -> Result<i32, String> {
    let tok = self.read_token()?;
    tok
      .parse::<i32>()
      .map_err(|_| format!("expected integer, got {:?}", tok))
  }

  pub fn read_f64(&mut self) -> Result<f64, String> {
    let tok = self.read_token()?;
    tok
      .parse::<f64>()
      .map_err(|_| format!("expected float, got {:?}", tok))
  }

  pub fn read_bool(&mut self) -> Result<bool, String> {
    let tok = self.read_token()?;
    match tok.as_str() {
      "0" => Ok(false),
      "1" => Ok(true),
      _ => Err(format!("expected 0 or 1 (bool), got {:?}", tok)),
    }
  }

  /// Read `count` floats, respecting the binary/text mode.
  ///
  /// Binary mode: skip whitespace until `@` then read `BIN@` marker,
  /// then read `count * 4` raw little-endian bytes.
  /// Text mode: read `count` whitespace-delimited ASCII floats.
  pub fn read_floats(
    &mut self,
    count: usize,
    ctx: &str,
  ) -> Result<Vec<f32>, String> {
    let mut buf = vec![0f32; count];
    if self.binary_floats {
      // Consume whitespace up to and including the leading '@'
      let mut chars_before = 0usize;
      loop {
        let b = self.read_byte().map_err(|e| format!("{ctx}: {e}"))?;
        if b == b'@' {
          break;
        }
        if !b.is_ascii_whitespace() || chars_before > 100 {
          return Err(format!(
            "{ctx}: could not find @BIN@ header (got non-whitespace before '@')"
          ));
        }
        chars_before += 1;
      }
      // Read 'B', 'I', 'N', '@'
      let mut marker = [0u8; 4];
      for m in &mut marker {
        *m = self.read_byte().map_err(|e| format!("{ctx}: {e}"))?;
      }
      if &marker != b"BIN@" {
        return Err(format!(
          "{ctx}: expected BIN@ marker, got {:?}",
          std::str::from_utf8(&marker).unwrap_or("???")
        ));
      }
      // Read raw little-endian f32s
      let byte_count = count * 4;
      let mut bytes = vec![0u8; byte_count];
      // BufReader::read_exact goes through the internal buffer correctly
      use io::Read as _;
      self
        .inner
        .read_exact(&mut bytes)
        .map_err(|e| format!("{ctx}: failed reading binary floats: {e}"))?;
      for (i, f) in buf.iter_mut().enumerate() {
        let le = u32::from_le_bytes([
          bytes[i * 4],
          bytes[i * 4 + 1],
          bytes[i * 4 + 2],
          bytes[i * 4 + 3],
        ]);
        *f = f32::from_bits(le);
        if !f.is_finite() {
          return Err(format!("{ctx}: NaN or infinite weight at index {i}"));
        }
      }
    } else {
      for i in 0..count {
        let tok = self.read_token().map_err(|e| format!("{ctx}: {e}"))?;
        let v = tok.parse::<f32>().map_err(|_| {
          format!("{ctx}: expected float at index {i}, got {:?}", tok)
        })?;
        if !v.is_finite() {
          return Err(format!("{ctx}: NaN or infinite weight at index {i}"));
        }
        buf[i] = v;
      }
    }
    Ok(buf)
  }
}

// ---------------------------------------------------------------------------
// Layer descriptors
// ---------------------------------------------------------------------------

/// Convolutional layer.
///
/// File format: name convYSize convXSize inChannels outChannels dilationY
/// dilationX  [floats in y,x,ic,oc order]
///
/// Memory layout (matching CUDA col-major): oc, ic, y, x.
#[derive(Debug, Clone)]
pub struct ConvLayerDesc {
  pub name: String,
  pub conv_y_size: i32,
  pub conv_x_size: i32,
  pub in_channels: i32,
  pub out_channels: i32,
  pub dilation_y: i32,
  pub dilation_x: i32,
  /// Weights in `[oc][ic][y][x]` order (CUDA col-major).
  pub weights: Vec<f32>,
}

impl ConvLayerDesc {
  pub fn parse<R: Read>(r: &mut TokenReader<R>) -> Result<Self, String> {
    let name = r.read_token()?;
    let conv_y_size = r.read_int()?;
    let conv_x_size = r.read_int()?;
    let in_channels = r.read_int()?;
    let out_channels = r.read_int()?;
    let dilation_y = r.read_int()?;
    let dilation_x = r.read_int()?;

    if conv_x_size <= 0 || conv_y_size <= 0 {
      return Err(format!("{name}: conv filter sizes must be positive"));
    }
    if in_channels <= 0 || out_channels <= 0 {
      return Err(format!("{name}: in/out channels must be positive"));
    }
    if dilation_x <= 0 || dilation_y <= 0 {
      return Err(format!("{name}: dilation factors must be positive"));
    }
    if conv_x_size % 2 == 0 || conv_y_size % 2 == 0 {
      return Err(format!("{name}: conv filter sizes must be odd"));
    }

    let num_weights =
      (conv_y_size * conv_x_size * in_channels * out_channels) as usize;
    let floats = r.read_floats(num_weights, &name)?;

    // Reorder from file order y,x,ic,oc  →  memory order oc,ic,y,x
    let oc_stride = (conv_y_size * conv_x_size * in_channels) as usize;
    let ic_stride = (conv_y_size * conv_x_size) as usize;
    let y_stride = conv_x_size as usize;
    let x_stride = 1usize;

    let mut weights = vec![0f32; num_weights];
    let mut idx = 0usize;
    for y in 0..conv_y_size as usize {
      for x in 0..conv_x_size as usize {
        for ic in 0..in_channels as usize {
          for oc in 0..out_channels as usize {
            weights
              [oc * oc_stride + ic * ic_stride + y * y_stride + x * x_stride] =
              floats[idx];
            idx += 1;
          }
        }
      }
    }

    Ok(Self {
      name,
      conv_y_size,
      conv_x_size,
      in_channels,
      out_channels,
      dilation_y,
      dilation_x,
      weights,
    })
  }
}

// ---------------------------------------------------------------------------

/// Batch normalisation layer.
///
/// File format: name numChannels epsilon hasScale hasBias  [mean] [variance]
/// [scale if hasScale] [bias if hasBias]
///
/// Also pre-computes `merged_scale` and `merged_bias`:
///   merged_scale[c] = scale[c] / sqrt(variance[c] + epsilon)
///   merged_bias[c]  = bias[c]  - merged_scale[c] * mean[c]
#[derive(Debug, Clone)]
pub struct BatchNormLayerDesc {
  pub name: String,
  pub num_channels: i32,
  pub epsilon: f32,
  pub has_scale: bool,
  pub has_bias: bool,
  pub mean: Vec<f32>,
  pub variance: Vec<f32>,
  pub scale: Vec<f32>,
  pub bias: Vec<f32>,
  /// Pre-fused: scale / sqrt(variance + epsilon)
  pub merged_scale: Vec<f32>,
  /// Pre-fused: bias - merged_scale * mean
  pub merged_bias: Vec<f32>,
}

impl BatchNormLayerDesc {
  pub fn parse<R: Read>(r: &mut TokenReader<R>) -> Result<Self, String> {
    let name = r.read_token()?;
    let num_channels = r.read_int()?;
    let epsilon = {
      let tok = r.read_token()?;
      tok
        .parse::<f32>()
        .map_err(|_| format!("{name}: expected epsilon float, got {tok:?}"))?
    };
    let has_scale = r.read_bool()?;
    let has_bias = r.read_bool()?;

    if num_channels < 1 {
      return Err(format!("{name}: numChannels ({num_channels}) < 1"));
    }
    if epsilon <= 0.0 {
      return Err(format!("{name}: epsilon ({epsilon}) <= 0"));
    }

    let nc = num_channels as usize;
    let mean = r.read_floats(nc, &name)?;
    let variance = r.read_floats(nc, &name)?;

    let scale = if has_scale {
      r.read_floats(nc, &name)?
    } else {
      vec![1.0f32; nc]
    };

    let bias = if has_bias {
      r.read_floats(nc, &name)?
    } else {
      vec![0.0f32; nc]
    };

    let mut merged_scale = vec![0f32; nc];
    let mut merged_bias = vec![0f32; nc];
    for c in 0..nc {
      merged_scale[c] = scale[c] / (variance[c] + epsilon).sqrt();
      merged_bias[c] = bias[c] - merged_scale[c] * mean[c];
    }

    Ok(Self {
      name,
      num_channels,
      epsilon,
      has_scale,
      has_bias,
      mean,
      variance,
      scale,
      bias,
      merged_scale,
      merged_bias,
    })
  }
}

// ---------------------------------------------------------------------------

/// Activation layer.
///
/// File format: name [activation_kind_string if modelVersion >= 11]
#[derive(Debug, Clone)]
pub struct ActivationLayerDesc {
  pub name: String,
  pub activation: Activation,
}

impl ActivationLayerDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;
    let activation = if model_version >= 11 {
      let kind = r.read_token()?;
      match kind.as_str() {
        "ACTIVATION_IDENTITY" => Activation::Identity,
        "ACTIVATION_RELU" => Activation::Relu,
        "ACTIVATION_MISH" => Activation::Mish,
        _ => return Err(format!("{name}: unknown activation {kind:?}")),
      }
    } else {
      Activation::Relu
    };
    Ok(Self { name, activation })
  }
}

// ---------------------------------------------------------------------------

/// Dense matrix multiply layer.
///
/// File format: name inChannels outChannels  [weights in ic,oc order]
/// Memory layout: ic, oc (same as file — matches cuBLAS transpose convention).
#[derive(Debug, Clone)]
pub struct MatMulLayerDesc {
  pub name: String,
  pub in_channels: i32,
  pub out_channels: i32,
  /// Weights in `[ic][oc]` order.
  pub weights: Vec<f32>,
}

impl MatMulLayerDesc {
  pub fn parse<R: Read>(r: &mut TokenReader<R>) -> Result<Self, String> {
    let name = r.read_token()?;
    let in_channels = r.read_int()?;
    let out_channels = r.read_int()?;

    if in_channels <= 0 || out_channels <= 0 {
      return Err(format!("{name}: in/out channels must be positive"));
    }

    let num_weights = (in_channels * out_channels) as usize;
    let floats = r.read_floats(num_weights, &name)?;

    // File order is ic,oc – same as memory order, just copy directly.
    Ok(Self {
      name,
      in_channels,
      out_channels,
      weights: floats,
    })
  }
}

// ---------------------------------------------------------------------------

/// Bias layer (dense).
///
/// File format: name numChannels  [bias weights]
#[derive(Debug, Clone)]
pub struct MatBiasLayerDesc {
  pub name: String,
  pub num_channels: i32,
  pub weights: Vec<f32>,
}

impl MatBiasLayerDesc {
  pub fn parse<R: Read>(r: &mut TokenReader<R>) -> Result<Self, String> {
    let name = r.read_token()?;
    let num_channels = r.read_int()?;

    if num_channels <= 0 {
      return Err(format!("{name}: numChannels must be positive"));
    }

    let weights = r.read_floats(num_channels as usize, &name)?;
    Ok(Self {
      name,
      num_channels,
      weights,
    })
  }
}

// ---------------------------------------------------------------------------
// Block types
// ---------------------------------------------------------------------------

/// Standard residual block: preBN → preAct → regularConv → midBN → midAct → finalConv
#[derive(Debug, Clone)]
pub struct ResidualBlockDesc {
  pub name: String,
  pub pre_bn: BatchNormLayerDesc,
  pub pre_activation: ActivationLayerDesc,
  pub regular_conv: ConvLayerDesc,
  pub mid_bn: BatchNormLayerDesc,
  pub mid_activation: ActivationLayerDesc,
  pub final_conv: ConvLayerDesc,
}

impl ResidualBlockDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;
    let pre_bn = BatchNormLayerDesc::parse(r)?;
    let pre_activation = ActivationLayerDesc::parse(r, model_version)?;
    let regular_conv = ConvLayerDesc::parse(r)?;
    let mid_bn = BatchNormLayerDesc::parse(r)?;
    let mid_activation = ActivationLayerDesc::parse(r, model_version)?;
    let final_conv = ConvLayerDesc::parse(r)?;

    if pre_bn.num_channels != regular_conv.in_channels {
      return Err(format!(
        "{name}: preBN.numChannels ({}) != regularConv.inChannels ({})",
        pre_bn.num_channels, regular_conv.in_channels
      ));
    }
    if mid_bn.num_channels != regular_conv.out_channels {
      return Err(format!(
        "{name}: midBN.numChannels ({}) != regularConv.outChannels ({})",
        mid_bn.num_channels, regular_conv.out_channels
      ));
    }
    if mid_bn.num_channels != final_conv.in_channels {
      return Err(format!(
        "{name}: midBN.numChannels ({}) != finalConv.inChannels ({})",
        mid_bn.num_channels, final_conv.in_channels
      ));
    }

    Ok(Self {
      name,
      pre_bn,
      pre_activation,
      regular_conv,
      mid_bn,
      mid_activation,
      final_conv,
    })
  }
}

// ---------------------------------------------------------------------------

/// Global-pooling residual block.
#[derive(Debug, Clone)]
pub struct GlobalPoolingResidualBlockDesc {
  pub name: String,
  pub pre_bn: BatchNormLayerDesc,
  pub pre_activation: ActivationLayerDesc,
  pub regular_conv: ConvLayerDesc,
  pub gpool_conv: ConvLayerDesc,
  pub gpool_bn: BatchNormLayerDesc,
  pub gpool_activation: ActivationLayerDesc,
  pub gpool_to_bias_mul: MatMulLayerDesc,
  pub mid_bn: BatchNormLayerDesc,
  pub mid_activation: ActivationLayerDesc,
  pub final_conv: ConvLayerDesc,
}

impl GlobalPoolingResidualBlockDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;
    let pre_bn = BatchNormLayerDesc::parse(r)?;
    let pre_activation = ActivationLayerDesc::parse(r, model_version)?;
    let regular_conv = ConvLayerDesc::parse(r)?;
    let gpool_conv = ConvLayerDesc::parse(r)?;
    let gpool_bn = BatchNormLayerDesc::parse(r)?;
    let gpool_activation = ActivationLayerDesc::parse(r, model_version)?;
    let gpool_to_bias_mul = MatMulLayerDesc::parse(r)?;
    let mid_bn = BatchNormLayerDesc::parse(r)?;
    let mid_activation = ActivationLayerDesc::parse(r, model_version)?;
    let final_conv = ConvLayerDesc::parse(r)?;

    if pre_bn.num_channels != regular_conv.in_channels {
      return Err(format!(
        "{name}: preBN.numChannels ({}) != regularConv.inChannels ({})",
        pre_bn.num_channels, regular_conv.in_channels
      ));
    }
    if pre_bn.num_channels != gpool_conv.in_channels {
      return Err(format!(
        "{name}: preBN.numChannels ({}) != gpoolConv.inChannels ({})",
        pre_bn.num_channels, gpool_conv.in_channels
      ));
    }
    if gpool_bn.num_channels != gpool_conv.out_channels {
      return Err(format!(
        "{name}: gpoolBN.numChannels ({}) != gpoolConv.outChannels ({})",
        gpool_bn.num_channels, gpool_conv.out_channels
      ));
    }
    // gpoolToBiasMul.inChannels == gpoolBN.numChannels * 3 (mean, max, mean-of-sq)
    if gpool_to_bias_mul.in_channels != gpool_bn.num_channels * 3 {
      return Err(format!(
        "{name}: gpoolToBiasMul.inChannels ({}) != gpoolBN.numChannels*3 ({})",
        gpool_to_bias_mul.in_channels,
        gpool_bn.num_channels * 3
      ));
    }
    if mid_bn.num_channels != regular_conv.out_channels {
      return Err(format!(
        "{name}: midBN.numChannels ({}) != regularConv.outChannels ({})",
        mid_bn.num_channels, regular_conv.out_channels
      ));
    }
    if mid_bn.num_channels != final_conv.in_channels {
      return Err(format!(
        "{name}: midBN.numChannels ({}) != finalConv.inChannels ({})",
        mid_bn.num_channels, final_conv.in_channels
      ));
    }

    Ok(Self {
      name,
      pre_bn,
      pre_activation,
      regular_conv,
      gpool_conv,
      gpool_bn,
      gpool_activation,
      gpool_to_bias_mul,
      mid_bn,
      mid_activation,
      final_conv,
    })
  }
}

// ---------------------------------------------------------------------------

/// Nested bottleneck residual block (recursive – contains inner blocks).
#[derive(Debug, Clone)]
pub struct NestedBottleneckResidualBlockDesc {
  pub name: String,
  pub pre_bn: BatchNormLayerDesc,
  pub pre_activation: ActivationLayerDesc,
  pub pre_conv: ConvLayerDesc,
  pub blocks: Vec<BlockDesc>,
  pub post_bn: BatchNormLayerDesc,
  pub post_activation: ActivationLayerDesc,
  pub post_conv: ConvLayerDesc,
}

impl NestedBottleneckResidualBlockDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;
    let num_blocks = r.read_int()?;

    let pre_bn = BatchNormLayerDesc::parse(r)?;
    let pre_activation = ActivationLayerDesc::parse(r, model_version)?;
    let pre_conv = ConvLayerDesc::parse(r)?;

    let mid_channels = pre_conv.out_channels;
    let blocks = parse_residual_block_stack(
      r,
      model_version,
      num_blocks as usize,
      mid_channels,
    )?;

    let post_bn = BatchNormLayerDesc::parse(r)?;
    let post_activation = ActivationLayerDesc::parse(r, model_version)?;
    let post_conv = ConvLayerDesc::parse(r)?;

    Ok(Self {
      name,
      pre_bn,
      pre_activation,
      pre_conv,
      blocks,
      post_bn,
      post_activation,
      post_conv,
    })
  }
}

// ---------------------------------------------------------------------------

/// A discriminated union of the three block kinds, mirroring the C++
/// `pair<int, unique_ptr_void>` blocks vector.
#[derive(Debug, Clone)]
pub enum BlockDesc {
  Ordinary(ResidualBlockDesc),
  GlobalPooling(GlobalPoolingResidualBlockDesc),
  NestedBottleneck(NestedBottleneckResidualBlockDesc),
}

/// Parse `num_blocks` residual blocks from the stream, reading the kind tag
/// before each block (matches `parseResidualBlockStack` in desc.cpp).
fn parse_residual_block_stack<R: Read>(
  r: &mut TokenReader<R>,
  model_version: i32,
  num_blocks: usize,
  trunk_num_channels: i32,
) -> Result<Vec<BlockDesc>, String> {
  let mut blocks = Vec::with_capacity(num_blocks);
  for _ in 0..num_blocks {
    let kind = r.read_token()?;
    let block = match kind.as_str() {
      "ordinary_block" => {
        let desc = ResidualBlockDesc::parse(r, model_version)?;
        if desc.pre_bn.num_channels != trunk_num_channels {
          return Err(format!(
            "{}: preBN.numChannels ({}) != trunkNumChannels ({})",
            desc.name, desc.pre_bn.num_channels, trunk_num_channels
          ));
        }
        if desc.final_conv.out_channels != trunk_num_channels {
          return Err(format!(
            "{}: finalConv.outChannels ({}) != trunkNumChannels ({})",
            desc.name, desc.final_conv.out_channels, trunk_num_channels
          ));
        }
        BlockDesc::Ordinary(desc)
      }
      "gpool_block" => {
        let desc = GlobalPoolingResidualBlockDesc::parse(r, model_version)?;
        if desc.pre_bn.num_channels != trunk_num_channels {
          return Err(format!(
            "{}: preBN.numChannels ({}) != trunkNumChannels ({})",
            desc.name, desc.pre_bn.num_channels, trunk_num_channels
          ));
        }
        if desc.final_conv.out_channels != trunk_num_channels {
          return Err(format!(
            "{}: finalConv.outChannels ({}) != trunkNumChannels ({})",
            desc.name, desc.final_conv.out_channels, trunk_num_channels
          ));
        }
        BlockDesc::GlobalPooling(desc)
      }
      "nested_bottleneck_block" => {
        let desc = NestedBottleneckResidualBlockDesc::parse(r, model_version)?;
        if desc.pre_bn.num_channels != trunk_num_channels {
          return Err(format!(
            "{}: preBN.numChannels ({}) != trunkNumChannels ({})",
            desc.name, desc.pre_bn.num_channels, trunk_num_channels
          ));
        }
        if desc.post_conv.out_channels != trunk_num_channels {
          return Err(format!(
            "{}: postConv.outChannels ({}) != trunkNumChannels ({})",
            desc.name, desc.post_conv.out_channels, trunk_num_channels
          ));
        }
        BlockDesc::NestedBottleneck(desc)
      }
      _ => return Err(format!("unknown block kind: {kind:?}")),
    };
    blocks.push(block);
  }
  Ok(blocks)
}

// ---------------------------------------------------------------------------
// SGF metadata encoder (version >= 15 with metaEncoderVersion > 0)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SgfMetadataEncoderDesc {
  pub name: String,
  pub meta_encoder_version: i32,
  pub num_input_meta_channels: i32,
  pub mul1: MatMulLayerDesc,
  pub bias1: MatBiasLayerDesc,
  pub act1: ActivationLayerDesc,
  pub mul2: MatMulLayerDesc,
  pub bias2: MatBiasLayerDesc,
  pub act2: ActivationLayerDesc,
  pub mul3: MatMulLayerDesc,
}

impl SgfMetadataEncoderDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
    meta_encoder_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;
    let num_input_meta_channels = r.read_int()?;
    let mul1 = MatMulLayerDesc::parse(r)?;
    let bias1 = MatBiasLayerDesc::parse(r)?;
    let act1 = ActivationLayerDesc::parse(r, model_version)?;
    let mul2 = MatMulLayerDesc::parse(r)?;
    let bias2 = MatBiasLayerDesc::parse(r)?;
    let act2 = ActivationLayerDesc::parse(r, model_version)?;
    let mul3 = MatMulLayerDesc::parse(r)?;
    Ok(Self {
      name,
      meta_encoder_version,
      num_input_meta_channels,
      mul1,
      bias1,
      act1,
      mul2,
      bias2,
      act2,
      mul3,
    })
  }
}

// ---------------------------------------------------------------------------
// Top-level network components
// ---------------------------------------------------------------------------

/// The main network trunk.
#[derive(Debug, Clone)]
pub struct TrunkDesc {
  pub name: String,
  pub num_blocks: i32,
  pub trunk_num_channels: i32,
  pub mid_num_channels: i32,
  pub regular_num_channels: i32,
  pub gpool_num_channels: i32,
  pub meta_encoder_version: i32,
  pub initial_conv: ConvLayerDesc,
  pub initial_mat_mul: MatMulLayerDesc,
  pub sgf_metadata_encoder: Option<SgfMetadataEncoderDesc>,
  pub blocks: Vec<BlockDesc>,
  pub trunk_tip_bn: BatchNormLayerDesc,
  pub trunk_tip_activation: ActivationLayerDesc,
}

impl TrunkDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
    meta_encoder_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;
    let num_blocks = r.read_int()?;
    let trunk_num_channels = r.read_int()?;
    let mid_num_channels = r.read_int()?;
    let regular_num_channels = r.read_int()?;
    let _dilated_num_channels = r.read_int()?; // present but unused
    let gpool_num_channels = r.read_int()?;

    // modelVersion >= 15: 6 unused ints
    if model_version >= 15 {
      for _ in 0..6 {
        r.read_int()?;
      }
    }

    if num_blocks < 1 {
      return Err(format!("{name}: num blocks must be positive"));
    }
    if trunk_num_channels <= 0
      || mid_num_channels <= 0
      || regular_num_channels <= 0
      || gpool_num_channels <= 0
    {
      return Err(format!("{name}: all channel counts must be positive"));
    }

    let initial_conv = ConvLayerDesc::parse(r)?;
    if initial_conv.out_channels != trunk_num_channels {
      return Err(format!(
        "{name}: initialConv.outChannels ({}) != trunkNumChannels ({})",
        initial_conv.out_channels, trunk_num_channels
      ));
    }

    let initial_mat_mul = MatMulLayerDesc::parse(r)?;
    if initial_mat_mul.out_channels != trunk_num_channels {
      return Err(format!(
        "{name}: initialMatMul.outChannels ({}) != trunkNumChannels ({})",
        initial_mat_mul.out_channels, trunk_num_channels
      ));
    }

    let sgf_metadata_encoder = if meta_encoder_version > 0 {
      Some(SgfMetadataEncoderDesc::parse(
        r,
        model_version,
        meta_encoder_version,
      )?)
    } else {
      None
    };

    let blocks = parse_residual_block_stack(
      r,
      model_version,
      num_blocks as usize,
      trunk_num_channels,
    )?;

    let trunk_tip_bn = BatchNormLayerDesc::parse(r)?;
    let trunk_tip_activation = ActivationLayerDesc::parse(r, model_version)?;

    if trunk_tip_bn.num_channels != trunk_num_channels {
      return Err(format!(
        "{name}: trunkTipBN.numChannels ({}) != trunkNumChannels ({})",
        trunk_tip_bn.num_channels, trunk_num_channels
      ));
    }

    Ok(Self {
      name,
      num_blocks,
      trunk_num_channels,
      mid_num_channels,
      regular_num_channels,
      gpool_num_channels,
      meta_encoder_version,
      initial_conv,
      initial_mat_mul,
      sgf_metadata_encoder,
      blocks,
      trunk_tip_bn,
      trunk_tip_activation,
    })
  }
}

// ---------------------------------------------------------------------------

/// Policy output head.
#[derive(Debug, Clone)]
pub struct PolicyHeadDesc {
  pub name: String,
  pub policy_out_channels: i32,
  pub p1_conv: ConvLayerDesc,
  pub g1_conv: ConvLayerDesc,
  pub g1_bn: BatchNormLayerDesc,
  pub g1_activation: ActivationLayerDesc,
  pub gpool_to_bias_mul: MatMulLayerDesc,
  pub p1_bn: BatchNormLayerDesc,
  pub p1_activation: ActivationLayerDesc,
  pub p2_conv: ConvLayerDesc,
  pub gpool_to_pass_mul: MatMulLayerDesc,
  /// Only present for modelVersion >= 15.
  pub gpool_to_pass_bias: Option<MatBiasLayerDesc>,
  pub pass_activation: Option<ActivationLayerDesc>,
  pub gpool_to_pass_mul2: Option<MatMulLayerDesc>,
}

impl PolicyHeadDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;

    let policy_out_channels = if model_version >= 16 {
      4
    } else if model_version >= 12 {
      2
    } else {
      1
    };

    let p1_conv = ConvLayerDesc::parse(r)?;
    let g1_conv = ConvLayerDesc::parse(r)?;
    let g1_bn = BatchNormLayerDesc::parse(r)?;
    let g1_activation = ActivationLayerDesc::parse(r, model_version)?;
    let gpool_to_bias_mul = MatMulLayerDesc::parse(r)?;
    let p1_bn = BatchNormLayerDesc::parse(r)?;
    let p1_activation = ActivationLayerDesc::parse(r, model_version)?;
    let p2_conv = ConvLayerDesc::parse(r)?;
    let gpool_to_pass_mul = MatMulLayerDesc::parse(r)?;

    let (gpool_to_pass_bias, pass_activation, gpool_to_pass_mul2) =
      if model_version >= 15 {
        let b = MatBiasLayerDesc::parse(r)?;
        let a = ActivationLayerDesc::parse(r, model_version)?;
        let m = MatMulLayerDesc::parse(r)?;
        (Some(b), Some(a), Some(m))
      } else {
        (None, None, None)
      };

    // Structural validation
    if p1_conv.out_channels != p1_bn.num_channels {
      return Err(format!(
        "{name}: p1Conv.outChannels ({}) != p1BN.numChannels ({})",
        p1_conv.out_channels, p1_bn.num_channels
      ));
    }
    if g1_conv.out_channels != g1_bn.num_channels {
      return Err(format!(
        "{name}: g1Conv.outChannels ({}) != g1BN.numChannels ({})",
        g1_conv.out_channels, g1_bn.num_channels
      ));
    }
    if gpool_to_bias_mul.in_channels != g1_bn.num_channels * 3 {
      return Err(format!(
        "{name}: gpoolToBiasMul.inChannels ({}) != g1BN.numChannels*3 ({})",
        gpool_to_bias_mul.in_channels,
        g1_bn.num_channels * 3
      ));
    }
    if gpool_to_bias_mul.out_channels != p1_bn.num_channels {
      return Err(format!(
        "{name}: gpoolToBiasMul.outChannels ({}) != p1BN.numChannels ({})",
        gpool_to_bias_mul.out_channels, p1_bn.num_channels
      ));
    }
    if p2_conv.in_channels != p1_bn.num_channels {
      return Err(format!(
        "{name}: p2Conv.inChannels ({}) != p1BN.numChannels ({})",
        p2_conv.in_channels, p1_bn.num_channels
      ));
    }
    if gpool_to_pass_mul.in_channels != g1_bn.num_channels * 3 {
      return Err(format!(
        "{name}: gpoolToPassMul.inChannels ({}) != g1BN.numChannels*3 ({})",
        gpool_to_pass_mul.in_channels,
        g1_bn.num_channels * 3
      ));
    }
    if p2_conv.out_channels != policy_out_channels {
      return Err(format!(
        "{name}: p2Conv.outChannels ({}) != policyOutChannels ({})",
        p2_conv.out_channels, policy_out_channels
      ));
    }

    Ok(Self {
      name,
      policy_out_channels,
      p1_conv,
      g1_conv,
      g1_bn,
      g1_activation,
      gpool_to_bias_mul,
      p1_bn,
      p1_activation,
      p2_conv,
      gpool_to_pass_mul,
      gpool_to_pass_bias,
      pass_activation,
      gpool_to_pass_mul2,
    })
  }
}

// ---------------------------------------------------------------------------

/// Value output head.
#[derive(Debug, Clone)]
pub struct ValueHeadDesc {
  pub name: String,
  pub v1_conv: ConvLayerDesc,
  pub v1_bn: BatchNormLayerDesc,
  pub v1_activation: ActivationLayerDesc,
  pub v2_mul: MatMulLayerDesc,
  pub v2_bias: MatBiasLayerDesc,
  pub v2_activation: ActivationLayerDesc,
  pub v3_mul: MatMulLayerDesc,
  pub v3_bias: MatBiasLayerDesc,
  pub sv3_mul: MatMulLayerDesc,
  pub sv3_bias: MatBiasLayerDesc,
  pub v_ownership_conv: ConvLayerDesc,
}

impl ValueHeadDesc {
  pub fn parse<R: Read>(
    r: &mut TokenReader<R>,
    model_version: i32,
  ) -> Result<Self, String> {
    let name = r.read_token()?;
    let v1_conv = ConvLayerDesc::parse(r)?;
    let v1_bn = BatchNormLayerDesc::parse(r)?;
    let v1_activation = ActivationLayerDesc::parse(r, model_version)?;
    let v2_mul = MatMulLayerDesc::parse(r)?;
    let v2_bias = MatBiasLayerDesc::parse(r)?;
    let v2_activation = ActivationLayerDesc::parse(r, model_version)?;
    let v3_mul = MatMulLayerDesc::parse(r)?;
    let v3_bias = MatBiasLayerDesc::parse(r)?;
    let sv3_mul = MatMulLayerDesc::parse(r)?;
    let sv3_bias = MatBiasLayerDesc::parse(r)?;
    let v_ownership_conv = ConvLayerDesc::parse(r)?;

    if v1_conv.out_channels != v1_bn.num_channels {
      return Err(format!(
        "{name}: v1Conv.outChannels ({}) != v1BN.numChannels ({})",
        v1_conv.out_channels, v1_bn.num_channels
      ));
    }
    if v2_mul.in_channels != v1_bn.num_channels * 3 {
      return Err(format!(
        "{name}: v2Mul.inChannels ({}) != v1BN.numChannels*3 ({})",
        v2_mul.in_channels,
        v1_bn.num_channels * 3
      ));
    }
    if v2_mul.out_channels != v2_bias.num_channels {
      return Err(format!(
        "{name}: v2Mul.outChannels ({}) != v2Bias.numChannels ({})",
        v2_mul.out_channels, v2_bias.num_channels
      ));
    }
    if v2_mul.out_channels != v3_mul.in_channels {
      return Err(format!(
        "{name}: v2Mul.outChannels ({}) != v3Mul.inChannels ({})",
        v2_mul.out_channels, v3_mul.in_channels
      ));
    }

    Ok(Self {
      name,
      v1_conv,
      v1_bn,
      v1_activation,
      v2_mul,
      v2_bias,
      v2_activation,
      v3_mul,
      v3_bias,
      sv3_mul,
      sv3_bias,
      v_ownership_conv,
    })
  }
}

// ---------------------------------------------------------------------------

/// Post-processing scaling parameters (modelVersion >= 13).
#[derive(Debug, Clone)]
pub struct ModelPostProcessParams {
  pub td_score_multiplier: f64,
  pub score_mean_multiplier: f64,
  pub score_stdev_multiplier: f64,
  pub lead_multiplier: f64,
  pub variance_time_multiplier: f64,
  pub shortterm_value_error_multiplier: f64,
  pub shortterm_score_error_multiplier: f64,
}

impl Default for ModelPostProcessParams {
  fn default() -> Self {
    // KataGo defaults (from the C++ default constructor)
    Self {
      td_score_multiplier: 1.0,
      score_mean_multiplier: 1.0,
      score_stdev_multiplier: 1.0,
      lead_multiplier: 1.0,
      variance_time_multiplier: 1.0,
      shortterm_value_error_multiplier: 1.0,
      shortterm_score_error_multiplier: 1.0,
    }
  }
}

// ---------------------------------------------------------------------------

/// The complete model descriptor, analogous to `ModelDesc` in desc.h.
#[derive(Debug, Clone)]
pub struct ModelDesc {
  pub name: String,
  pub model_version: i32,
  pub num_input_channels: i32,
  pub num_input_global_channels: i32,
  pub num_input_meta_channels: i32,
  pub num_policy_channels: i32,
  pub num_value_channels: i32,
  pub num_score_value_channels: i32,
  pub num_ownership_channels: i32,
  pub meta_encoder_version: i32,
  pub post_process_params: ModelPostProcessParams,
  pub trunk: TrunkDesc,
  pub policy_head: PolicyHeadDesc,
  pub value_head: ValueHeadDesc,
}

impl ModelDesc {
  /// Parse a model from an open (already-decompressed) stream.
  pub fn parse<R: Read>(r: &mut TokenReader<R>) -> Result<Self, String> {
    let name = r.read_token()?;
    let model_version = r.read_int()?;

    if model_version < 0 {
      return Err(format!("invalid model version {model_version}"));
    }
    if model_version < 3 {
      return Err(format!(
        "model version {model_version} is too old (minimum supported: 3)"
      ));
    }
    if model_version > 16 {
      return Err(format!(
        "model version {model_version} requires a newer KataGo"
      ));
    }

    let num_input_channels = r.read_int()?;
    if num_input_channels <= 0 {
      return Err(format!("{name}: numInputChannels must be positive"));
    }

    let num_input_global_channels = r.read_int()?;
    if num_input_global_channels <= 0 {
      return Err(format!("{name}: numInputGlobalChannels must be positive"));
    }

    let post_process_params = if model_version >= 13 {
      let td = r.read_f64()?;
      let sm = r.read_f64()?;
      let ss = r.read_f64()?;
      let lm = r.read_f64()?;
      let vt = r.read_f64()?;
      let sve = r.read_f64()?;
      let sse = r.read_f64()?;
      ModelPostProcessParams {
        td_score_multiplier: td,
        score_mean_multiplier: sm,
        score_stdev_multiplier: ss,
        lead_multiplier: lm,
        variance_time_multiplier: vt,
        shortterm_value_error_multiplier: sve,
        shortterm_score_error_multiplier: sse,
      }
    } else {
      ModelPostProcessParams::default()
    };

    let (meta_encoder_version, num_input_meta_channels) = if model_version >= 15
    {
      let mev = r.read_int()?;
      if mev < 0 || mev > 1 {
        return Err(format!("{name}: metaEncoderVersion {mev} not supported"));
      }
      // 7 unused ints follow
      for _ in 0..7 {
        r.read_int()?;
      }
      let nmc = if mev == 0 {
        0
      } else {
        // SGFMetadata::METADATA_INPUT_NUM_CHANNELS — hard-coded to 76 in KataGo source
        76
      };
      (mev, nmc)
    } else {
      (0, 0)
    };

    let trunk = TrunkDesc::parse(r, model_version, meta_encoder_version)?;
    let policy_head = PolicyHeadDesc::parse(r, model_version)?;
    let value_head = ValueHeadDesc::parse(r, model_version)?;

    let num_policy_channels = policy_head.policy_out_channels;
    let num_value_channels = value_head.v3_mul.out_channels;
    let num_score_value_channels = value_head.sv3_mul.out_channels;
    let num_ownership_channels = value_head.v_ownership_conv.out_channels;

    // Cross-component channel consistency
    if num_input_channels != trunk.initial_conv.in_channels {
      return Err(format!(
        "{name}: numInputChannels ({num_input_channels}) != trunk.initialConv.inChannels ({})",
        trunk.initial_conv.in_channels
      ));
    }
    if num_input_global_channels != trunk.initial_mat_mul.in_channels {
      return Err(format!(
        "{name}: numInputGlobalChannels ({num_input_global_channels}) != trunk.initialMatMul.inChannels ({})",
        trunk.initial_mat_mul.in_channels
      ));
    }
    if trunk.trunk_num_channels != policy_head.p1_conv.in_channels {
      return Err(format!(
        "{name}: trunk.trunkNumChannels ({}) != policyHead.p1Conv.inChannels ({})",
        trunk.trunk_num_channels, policy_head.p1_conv.in_channels
      ));
    }
    if trunk.trunk_num_channels != policy_head.g1_conv.in_channels {
      return Err(format!(
        "{name}: trunk.trunkNumChannels ({}) != policyHead.g1Conv.inChannels ({})",
        trunk.trunk_num_channels, policy_head.g1_conv.in_channels
      ));
    }
    if trunk.trunk_num_channels != value_head.v1_conv.in_channels {
      return Err(format!(
        "{name}: trunk.trunkNumChannels ({}) != valueHead.v1Conv.inChannels ({})",
        trunk.trunk_num_channels, value_head.v1_conv.in_channels
      ));
    }

    Ok(Self {
      name,
      model_version,
      num_input_channels,
      num_input_global_channels,
      num_input_meta_channels,
      num_policy_channels,
      num_value_channels,
      num_score_value_channels,
      num_ownership_channels,
      meta_encoder_version,
      post_process_params,
      trunk,
      policy_head,
      value_head,
    })
  }

  /// Parse a model from already-decompressed bytes.
  ///
  /// `binary_floats` selects between binary (`@BIN@`) and text float blocks.
  pub fn load_from_bytes(
    data: &[u8],
    binary_floats: bool,
  ) -> Result<Self, String> {
    let mut reader = TokenReader::new(io::Cursor::new(data), binary_floats);
    Self::parse(&mut reader)
  }

  /// Parse a model from gzip-compressed bytes.
  ///
  /// `binary_floats` selects between binary (`@BIN@`) and text float blocks
  /// (i.e. whether the original file was `.bin.gz` or `.txt.gz`).
  pub fn load_from_gz_bytes(
    data: &[u8],
    binary_floats: bool,
  ) -> Result<Self, String> {
    let decompressed = decompress_gzip(data)
      .map_err(|e| format!("gzip decompression failed: {e}"))?;
    Self::load_from_bytes(&decompressed, binary_floats)
  }

  /// Load a model from a `.bin.gz`, `.txt.gz`, `.bin`, or `.txt` file.
  ///
  /// Not available on `wasm32` targets — use [`Self::load_from_bytes`] or
  /// [`Self::load_from_gz_bytes`] instead.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn load_from_file(
    path: impl AsRef<std::path::Path>,
  ) -> Result<Self, String> {
    let path = path.as_ref();
    let lower = path.to_string_lossy().to_lowercase();

    let raw = std::fs::read(path)
      .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    if lower.ends_with(".txt.gz")
      || lower.ends_with(".bin.gz")
      || lower.ends_with(".gz")
    {
      let binary = !lower.ends_with(".txt.gz");
      Self::load_from_gz_bytes(&raw, binary)
    } else if lower.ends_with(".bin") {
      Self::load_from_bytes(&raw, true)
    } else if lower.ends_with(".txt") {
      Self::load_from_bytes(&raw, false)
    } else {
      Err(format!(
        "unrecognised model file extension for {}",
        path.display()
      ))
    }
  }
}

// ---------------------------------------------------------------------------
// gzip decompression helper
// ---------------------------------------------------------------------------

fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>, String> {
  use flate2::read::GzDecoder;
  let mut decoder = GzDecoder::new(data);
  let mut out = Vec::new();
  decoder.read_to_end(&mut out).map_err(|e| e.to_string())?;
  Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::Cursor;

  // -----------------------------------------------------------------------
  // TokenReader unit tests
  // -----------------------------------------------------------------------

  #[test]
  fn token_reader_reads_text_tokens() {
    let data = b"hello 42 3.14";
    let mut r = TokenReader::new(Cursor::new(data.as_ref()), false);
    assert_eq!(r.read_token().unwrap(), "hello");
    assert_eq!(r.read_int().unwrap(), 42);
    let f = r.read_token().unwrap().parse::<f32>().unwrap();
    assert!((f - 3.14f32).abs() < 1e-4);
  }

  #[test]
  fn token_reader_reads_binary_floats() {
    // Build a binary float block: " @BIN@" + 2 LE f32s (1.0, -2.5)
    let mut data: Vec<u8> = b" @BIN@".to_vec();
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&(-2.5f32).to_le_bytes());
    let mut r = TokenReader::new(Cursor::new(data), true);
    let floats = r.read_floats(2, "test").unwrap();
    assert!((floats[0] - 1.0).abs() < 1e-7);
    assert!((floats[1] - (-2.5)).abs() < 1e-7);
  }

  // -----------------------------------------------------------------------
  // File-based tests — not available on wasm32 (no filesystem)
  // -----------------------------------------------------------------------

  #[cfg(not(target_arch = "wasm32"))]
  mod file_tests {
    use super::super::ModelDesc;

    fn workspace_root() -> std::path::PathBuf {
      std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
    }

    fn g170_bin_gz_path() -> std::path::PathBuf {
      workspace_root()
        .join("cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz")
    }

    fn g170_txt_gz_path() -> std::path::PathBuf {
      workspace_root()
        .join("cpp/tests/models/g170-b6c96-s175395328-d26788732.txt.gz")
    }

    // .network.bin.gz  (the user's model)

    #[test]
    fn load_network_bin_gz_succeeds() {
      let path = workspace_root().join(".network.bin.gz");
      let m = ModelDesc::load_from_file(&path)
        .expect("loading .network.bin.gz should succeed");

      assert!(
        m.model_version >= 3 && m.model_version <= 16,
        "model_version {} out of expected range",
        m.model_version
      );
      assert!(m.num_input_channels > 0);
      assert!(m.num_input_global_channels > 0);
      assert!(!m.name.is_empty());
      assert!(!m.trunk.blocks.is_empty());
      assert!(m.num_policy_channels > 0);
      assert!(m.num_value_channels > 0);
      assert!(m.num_score_value_channels > 0);
      assert!(m.num_ownership_channels > 0);
    }

    #[test]
    fn network_trunk_channel_counts_are_consistent() {
      let m =
        ModelDesc::load_from_file(workspace_root().join(".network.bin.gz"))
          .unwrap();

      assert_eq!(
        m.trunk.initial_conv.out_channels,
        m.trunk.trunk_num_channels
      );
      assert_eq!(
        m.trunk.initial_mat_mul.out_channels,
        m.trunk.trunk_num_channels
      );
      assert_eq!(
        m.trunk.trunk_tip_bn.num_channels,
        m.trunk.trunk_num_channels
      );
      assert_eq!(
        m.trunk.trunk_num_channels,
        m.policy_head.p1_conv.in_channels
      );
      assert_eq!(
        m.trunk.trunk_num_channels,
        m.policy_head.g1_conv.in_channels
      );
      assert_eq!(m.trunk.trunk_num_channels, m.value_head.v1_conv.in_channels);
    }

    #[test]
    fn network_policy_head_shapes_are_valid() {
      let m =
        ModelDesc::load_from_file(workspace_root().join(".network.bin.gz"))
          .unwrap();
      let ph = &m.policy_head;

      assert_eq!(ph.p2_conv.out_channels, ph.policy_out_channels);
      assert_eq!(ph.p1_conv.out_channels, ph.p1_bn.num_channels);
      assert_eq!(ph.g1_conv.out_channels, ph.g1_bn.num_channels);
      assert_eq!(ph.gpool_to_bias_mul.in_channels, ph.g1_bn.num_channels * 3);
    }

    #[test]
    fn network_value_head_shapes_are_valid() {
      let m =
        ModelDesc::load_from_file(workspace_root().join(".network.bin.gz"))
          .unwrap();
      let vh = &m.value_head;

      assert_eq!(vh.v1_conv.out_channels, vh.v1_bn.num_channels);
      assert_eq!(vh.v2_mul.in_channels, vh.v1_bn.num_channels * 3);
      assert_eq!(vh.v2_mul.out_channels, vh.v2_bias.num_channels);
      assert_eq!(vh.v2_mul.out_channels, vh.v3_mul.in_channels);
      assert_eq!(vh.v3_mul.out_channels, vh.v3_bias.num_channels);
      assert_eq!(vh.sv3_mul.out_channels, vh.sv3_bias.num_channels);
    }

    #[test]
    fn network_conv_weights_have_correct_size() {
      let m =
        ModelDesc::load_from_file(workspace_root().join(".network.bin.gz"))
          .unwrap();
      let ic = &m.trunk.initial_conv;
      let expected =
        (ic.conv_y_size * ic.conv_x_size * ic.in_channels * ic.out_channels)
          as usize;
      assert_eq!(
        ic.weights.len(),
        expected,
        "initial_conv weight vector has wrong size"
      );
    }

    #[test]
    fn network_bn_merged_params_are_finite() {
      let m =
        ModelDesc::load_from_file(workspace_root().join(".network.bin.gz"))
          .unwrap();
      let bn = &m.trunk.trunk_tip_bn;
      for (i, (&s, &b)) in bn
        .merged_scale
        .iter()
        .zip(bn.merged_bias.iter())
        .enumerate()
      {
        assert!(s.is_finite(), "merged_scale[{i}] is not finite: {s}");
        assert!(b.is_finite(), "merged_bias[{i}] is not finite: {b}");
      }
    }

    // g170-b6c96 model (both .bin.gz and .txt.gz)

    #[test]
    fn load_g170_bin_gz_succeeds() {
      let m = ModelDesc::load_from_file(g170_bin_gz_path())
        .expect("loading g170 bin.gz should succeed");
      assert!(m.model_version >= 3);
      assert!(!m.trunk.blocks.is_empty());
    }

    #[test]
    fn load_g170_txt_gz_succeeds() {
      let m = ModelDesc::load_from_file(g170_txt_gz_path())
        .expect("loading g170 txt.gz should succeed");
      assert!(m.model_version >= 3);
      assert!(!m.trunk.blocks.is_empty());
    }

    /// Binary and text variants of the same model must produce identical
    /// metadata and first few weight values.
    #[test]
    fn g170_bin_and_txt_are_consistent() {
      let mb = ModelDesc::load_from_file(g170_bin_gz_path()).unwrap();
      let mt = ModelDesc::load_from_file(g170_txt_gz_path()).unwrap();

      // The two files have slightly different embedded names; compare structure.
      assert_eq!(mb.model_version, mt.model_version);
      assert_eq!(mb.num_input_channels, mt.num_input_channels);
      assert_eq!(mb.num_input_global_channels, mt.num_input_global_channels);
      assert_eq!(mb.trunk.trunk_num_channels, mt.trunk.trunk_num_channels);
      assert_eq!(mb.trunk.num_blocks, mt.trunk.num_blocks);

      let bw = &mb.trunk.initial_conv.weights;
      let tw = &mt.trunk.initial_conv.weights;
      assert_eq!(bw.len(), tw.len());
      for i in 0..bw.len().min(16) {
        assert!(
          (bw[i] - tw[i]).abs() < 1e-5,
          "initial_conv.weights[{i}] differs: bin={} txt={}",
          bw[i],
          tw[i]
        );
      }
    }

    // g170e-b10c128 model

    #[test]
    fn load_g170e_bin_gz_succeeds() {
      let path = workspace_root()
        .join("cpp/tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz");
      let m = ModelDesc::load_from_file(path)
        .expect("loading g170e bin.gz should succeed");
      assert!(m.model_version >= 3);
      assert_eq!(m.trunk.trunk_num_channels, 128);
      assert_eq!(m.trunk.num_blocks, 10);
    }
  }
}
