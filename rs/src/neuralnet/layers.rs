/// GPU-side layer runners.
///
/// Each struct holds the compiled `ComputePipeline` and knows how to build
/// bind groups and dispatch the right number of workgroups for a given set
/// of tensor shapes.
use wgpu::util::DeviceExt as _;

use super::{
  GpuContext,
  buffers::{GpuTensor, WeightBuffer},
  shaders,
};
use crate::model::{
  ActivationLayerDesc, BatchNormLayerDesc, ConvLayerDesc, MatBiasLayerDesc,
  MatMulLayerDesc,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn div_ceil(a: u32, b: u32) -> u32 {
  (a + b - 1) / b
}

/// Allocate a uniform buffer from a plain-old-data value.
fn uniform<T: bytemuck::Pod>(ctx: &GpuContext, data: &T) -> wgpu::Buffer {
  ctx
    .device
    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: None,
      contents: bytemuck::bytes_of(data),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn make_pipeline(
  ctx: &GpuContext,
  src: &str,
  entry: &str,
) -> wgpu::ComputePipeline {
  let shader = ctx
    .device
    .create_shader_module(wgpu::ShaderModuleDescriptor {
      label: None,
      source: wgpu::ShaderSource::Wgsl(src.into()),
    });
  ctx
    .device
    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: None,
      layout: None,
      module: &shader,
      entry_point: Some(entry),
      compilation_options: wgpu::PipelineCompilationOptions::default(),
      cache: None,
    })
}

// ---------------------------------------------------------------------------
// ConvLayer
// ---------------------------------------------------------------------------

/// Uniform block for the conv shader — must match `conv.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvUniforms {
  n: u32,
  cin: u32,
  cout: u32,
  h: u32,
  w: u32,
  kh: u32,
  kw: u32,
  accumulate: u32,
}

pub struct ConvLayer {
  pipeline: wgpu::ComputePipeline,
  weights: WeightBuffer,
  pub desc: ConvLayerDesc,
}

impl ConvLayer {
  pub fn new(ctx: &GpuContext, desc: &ConvLayerDesc) -> Self {
    // desc.weights is already in [oc, ic, y, x] order — model.rs reorders
    // from the on-disk [y, x, ic, oc] layout during parsing.  Upload directly.
    Self {
      pipeline: make_pipeline(ctx, shaders::CONV, "main"),
      weights: WeightBuffer::new(ctx, &desc.weights),
      desc: desc.clone(),
    }
  }

  /// Dispatch: writes into `output`.
  /// `accumulate = true` adds to the existing output values (residual add).
  pub fn dispatch(
    &self,
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    n: u32,
    h: u32,
    w: u32,
    input: &GpuTensor,
    output: &GpuTensor,
    accumulate: bool,
  ) {
    let uniforms = ConvUniforms {
      n,
      cin: self.desc.in_channels as u32,
      cout: self.desc.out_channels as u32,
      h,
      w,
      kh: self.desc.conv_y_size as u32,
      kw: self.desc.conv_x_size as u32,
      accumulate: accumulate as u32,
    };
    let ub = uniform(ctx, &uniforms);
    let hw = h * w;
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: ub.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: input.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: self.weights.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: output.buf.as_entire_binding(),
        },
      ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    // workgroup(8,8,1): x covers H*W, y covers Cout, z covers N
    pass.dispatch_workgroups(
      div_ceil(hw, 8),
      div_ceil(self.desc.out_channels as u32, 8),
      n,
    );
  }
}

// ---------------------------------------------------------------------------
// BnActLayer
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BnActUniforms {
  n: u32,
  c: u32,
  h: u32,
  w: u32,
  activation: u32,
  _pad: u32,
  _pad2: u32,
  _pad3: u32,
}

pub struct BnActLayer {
  pipeline: wgpu::ComputePipeline,
  merged_scale: WeightBuffer,
  merged_bias: WeightBuffer,
  activation: u32,
  pub num_channels: u32,
}

impl BnActLayer {
  pub fn new(
    ctx: &GpuContext,
    desc: &BatchNormLayerDesc,
    act: &ActivationLayerDesc,
  ) -> Self {
    use crate::model::Activation;
    let activation = match act.activation {
      Activation::Identity => 0,
      Activation::Relu => 1,
      Activation::Mish | Activation::MishScale8 => 2,
    };
    Self {
      pipeline: make_pipeline(ctx, shaders::BN_ACT, "main"),
      merged_scale: WeightBuffer::new(ctx, &desc.merged_scale),
      merged_bias: WeightBuffer::new(ctx, &desc.merged_bias),
      activation,
      num_channels: desc.num_channels as u32,
    }
  }

  pub fn dispatch(
    &self,
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    n: u32,
    h: u32,
    w: u32,
    input: &GpuTensor,
    mask: &GpuTensor,
    output: &GpuTensor,
  ) {
    let uniforms = BnActUniforms {
      n,
      c: self.num_channels,
      h,
      w,
      activation: self.activation,
      _pad: 0,
      _pad2: 0,
      _pad3: 0,
    };
    let ub = uniform(ctx, &uniforms);
    let total = n * self.num_channels * h * w;
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: ub.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: input.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: self.merged_scale.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: self.merged_bias.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 4,
          resource: mask.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 5,
          resource: output.buf.as_entire_binding(),
        },
      ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(div_ceil(total, 64), 1, 1);
  }
}

// ---------------------------------------------------------------------------
// GpoolLayer
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpoolUniforms {
  n: u32,
  c: u32,
  h: u32,
  w: u32,
}

pub struct GpoolLayer {
  pipeline: wgpu::ComputePipeline,
}

impl GpoolLayer {
  pub fn new(ctx: &GpuContext) -> Self {
    Self {
      pipeline: make_pipeline(ctx, shaders::GPOOL, "main"),
    }
  }

  /// Reduces `input [N, C, H, W]` + `mask [N, H, W]` + `mask_sum [N]`
  /// → `output [N, 3*C]`.
  pub fn dispatch(
    &self,
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    input: &GpuTensor,
    mask: &GpuTensor,
    mask_sum: &GpuTensor,
    output: &GpuTensor,
  ) {
    let uniforms = GpoolUniforms { n, c, h, w };
    let ub = uniform(ctx, &uniforms);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: ub.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: input.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: mask.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: mask_sum.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 4,
          resource: output.buf.as_entire_binding(),
        },
      ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(div_ceil(n * c, 64), 1, 1);
  }
}

// ---------------------------------------------------------------------------
// ValueHeadGpoolLayer — poolRowsValueHead variant used by the value head
//
// stat₀ = mean
// stat₁ = mean × (√maskSum − 14) × 0.1
// stat₂ = mean × ((√maskSum − 14)² × 0.01 − 0.1)   ← differs from GpoolLayer
// ---------------------------------------------------------------------------

pub struct ValueHeadGpoolLayer {
  pipeline: wgpu::ComputePipeline,
}

impl ValueHeadGpoolLayer {
  pub fn new(ctx: &GpuContext) -> Self {
    Self {
      pipeline: make_pipeline(ctx, shaders::GPOOL_VALUE_HEAD, "main"),
    }
  }

  /// Reduces `input [N, C, H, W]` + `mask_sum [N]` → `output [3*C, N]`.
  /// (mask is not needed for the value-head pooling formula)
  pub fn dispatch(
    &self,
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    input: &GpuTensor,
    _mask: &GpuTensor,
    mask_sum: &GpuTensor,
    output: &GpuTensor,
  ) {
    let uniforms = GpoolUniforms { n, c, h, w };
    let ub = uniform(ctx, &uniforms);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: ub.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: input.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: mask_sum.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: output.buf.as_entire_binding(),
        },
      ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(div_ceil(n * c, 64), 1, 1);
  }
}

// ---------------------------------------------------------------------------
// MatMulLayer
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulUniforms {
  k: u32,
  n_out: u32,
  batch: u32,
  _pad: u32,
}

pub struct MatMulLayer {
  pipeline: wgpu::ComputePipeline,
  weights: WeightBuffer,
  pub in_channels: u32,
  pub out_channels: u32,
}

impl MatMulLayer {
  pub fn new(ctx: &GpuContext, desc: &MatMulLayerDesc) -> Self {
    // desc.weights is in [ic, oc] order (model.rs file layout).
    // The matmul shader reads weights[oc * K + ic] — i.e. it expects [oc, ic].
    // Transpose here so the GPU sees the correct layout.
    let ic = desc.in_channels as usize;
    let oc = desc.out_channels as usize;
    let mut w_t = vec![0f32; ic * oc];
    for i in 0..ic {
      for o in 0..oc {
        w_t[o * ic + i] = desc.weights[i * oc + o];
      }
    }
    Self {
      pipeline: make_pipeline(ctx, shaders::MATMUL, "main"),
      weights: WeightBuffer::new(ctx, &w_t),
      in_channels: desc.in_channels as u32,
      out_channels: desc.out_channels as u32,
    }
  }

  /// `input [K, batch]` → `output [N_out, batch]`
  pub fn dispatch(
    &self,
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    batch: u32,
    input: &GpuTensor,
    output: &GpuTensor,
  ) {
    let uniforms = MatMulUniforms {
      k: self.in_channels,
      n_out: self.out_channels,
      batch,
      _pad: 0,
    };
    let ub = uniform(ctx, &uniforms);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: ub.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: self.weights.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: input.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: output.buf.as_entire_binding(),
        },
      ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(
      div_ceil(batch, 8),
      div_ceil(self.out_channels, 8),
      1,
    );
  }
}

// ---------------------------------------------------------------------------
// MatBiasLayer — adds a per-channel bias to a [C, B] tensor (mode 0)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasAddUniforms {
  mode: u32,
  c: u32,
  b_or_batch: u32,
  n: u32,
  h: u32,
  w: u32,
  _pad: u32,
  _pad2: u32,
}

pub struct MatBiasLayer {
  pipeline: wgpu::ComputePipeline,
  bias: WeightBuffer,
  pub num_channels: u32,
}

impl MatBiasLayer {
  pub fn new(ctx: &GpuContext, desc: &MatBiasLayerDesc) -> Self {
    Self {
      pipeline: make_pipeline(ctx, shaders::BIAS_ADD, "main"),
      bias: WeightBuffer::new(ctx, &desc.weights),
      num_channels: desc.num_channels as u32,
    }
  }

  /// Add bias to `tensor [C, batch]` in-place.
  pub fn dispatch(
    &self,
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    batch: u32,
    tensor: &GpuTensor,
  ) {
    let uniforms = BiasAddUniforms {
      mode: 0,
      c: self.num_channels,
      b_or_batch: batch,
      n: 0,
      h: 0,
      w: 0,
      _pad: 0,
      _pad2: 0,
    };
    let ub = uniform(ctx, &uniforms);
    let total = self.num_channels * batch;
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: ub.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: self.bias.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: tensor.buf.as_entire_binding(),
        },
      ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(div_ceil(total, 64), 1, 1);
  }
}

// ---------------------------------------------------------------------------
// NcBroadcastBiasAdd — adds [C, N] bias to [N, C, H, W] spatial tensor (mode 1)
// ---------------------------------------------------------------------------

pub struct NcBroadcastBiasAdd {
  pipeline: wgpu::ComputePipeline,
}

impl NcBroadcastBiasAdd {
  pub fn new(ctx: &GpuContext) -> Self {
    Self {
      pipeline: make_pipeline(ctx, shaders::BIAS_ADD, "main"),
    }
  }

  pub fn dispatch(
    &self,
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    bias: &GpuTensor,    // [C, N]
    spatial: &GpuTensor, // [N, C, H, W]
  ) {
    let uniforms = BiasAddUniforms {
      mode: 1,
      c,
      b_or_batch: 0,
      n,
      h,
      w,
      _pad: 0,
      _pad2: 0,
    };
    let ub = uniform(ctx, &uniforms);
    let total = n * c * h * w;
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: ub.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: bias.buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: spatial.buf.as_entire_binding(),
        },
      ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(div_ceil(total, 64), 1, 1);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
  use super::*;
  use crate::model::{
    ActivationLayerDesc, Activation, BatchNormLayerDesc, ConvLayerDesc,
    MatBiasLayerDesc, MatMulLayerDesc,
  };

  fn gpu() -> Option<GpuContext> {
    GpuContext::new_sync().ok()
  }

  fn submit(ctx: &GpuContext, f: impl FnOnce(&mut wgpu::CommandEncoder)) -> () {
    let mut enc = ctx
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    f(&mut enc);
    ctx.queue.submit([enc.finish()]);
    ctx.device.poll(wgpu::Maintain::Wait);
  }

  // --- ConvLayer ---

  #[test]
  fn conv_identity_weights() {
    // 1x1 conv with identity weights: output should equal input.
    let ctx = match gpu() { Some(c) => c, None => return };
    // [KY=1, KX=1, Cin=2, Cout=2]: identity mapping via disk layout
    // disk [y,x,ic,oc]: weights[0,0,0,0]=1, [0,0,1,1]=1, rest=0
    let weights = vec![1f32, 0., 0., 1.];
    let desc = ConvLayerDesc {
      name: String::new(),
      conv_y_size: 1,
      conv_x_size: 1,
      in_channels: 2,
      out_channels: 2,
      dilation_y: 1,
      dilation_x: 1,
      weights,
    };
    let layer = ConvLayer::new(&ctx, &desc);
    // input [N=1, C=2, H=2, W=2]: fill with [1..8]
    let input_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let input = GpuTensor::from_slice(&ctx, &input_data);
    let output = GpuTensor::zeros(&ctx, 8);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 2, 2, &input, &output, false));
    let result = output.download(&ctx);
    assert_eq!(result, input_data);
  }

  #[test]
  fn conv_accumulate_flag() {
    // accumulate=true should add to existing output
    let ctx = match gpu() { Some(c) => c, None => return };
    let weights = vec![1f32, 0., 0., 1.];
    let desc = ConvLayerDesc {
      name: String::new(),
      conv_y_size: 1,
      conv_x_size: 1,
      in_channels: 2,
      out_channels: 2,
      dilation_y: 1,
      dilation_x: 1,
      weights,
    };
    let layer = ConvLayer::new(&ctx, &desc);
    let input_data = vec![1f32, 2., 3., 4., 5., 6., 7., 8.];
    let input = GpuTensor::from_slice(&ctx, &input_data);
    let output = GpuTensor::from_slice(&ctx, &vec![10f32; 8]);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 2, 2, &input, &output, true));
    let result = output.download(&ctx);
    let expected: Vec<f32> = input_data.iter().map(|x| x + 10.).collect();
    assert_eq!(result, expected);
  }

  // --- BnActLayer ---

  #[test]
  fn bnact_identity_activation() {
    let ctx = match gpu() { Some(c) => c, None => return };
    // merged_scale=1, merged_bias=0 → output = input (masked)
    let desc = BatchNormLayerDesc {
      name: String::new(),
      num_channels: 2,
      epsilon: 1e-5,
      has_scale: false,
      has_bias: false,
      mean: vec![0.; 2],
      variance: vec![1.; 2],
      scale: vec![1.; 2],
      bias: vec![0.; 2],
      merged_scale: vec![1., 1.],
      merged_bias: vec![0., 0.],
    };
    let act = ActivationLayerDesc { name: String::new(), activation: Activation::Identity };
    let layer = BnActLayer::new(&ctx, &desc, &act);
    // [N=1, C=2, H=2, W=2]
    let input_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let input = GpuTensor::from_slice(&ctx, &input_data);
    let mask = GpuTensor::from_slice(&ctx, &vec![1f32; 4]); // all valid
    let output = GpuTensor::zeros(&ctx, 8);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 2, 2, &input, &mask, &output));
    let result = output.download(&ctx);
    assert_eq!(result, input_data);
  }

  #[test]
  fn bnact_relu_clips_negatives() {
    let ctx = match gpu() { Some(c) => c, None => return };
    let desc = BatchNormLayerDesc {
      name: String::new(),
      num_channels: 1,
      epsilon: 1e-5,
      has_scale: false,
      has_bias: false,
      mean: vec![0.],
      variance: vec![1.],
      scale: vec![1.],
      bias: vec![0.],
      merged_scale: vec![1.],
      merged_bias: vec![0.],
    };
    let act = ActivationLayerDesc { name: String::new(), activation: Activation::Relu };
    let layer = BnActLayer::new(&ctx, &desc, &act);
    // [N=1, C=1, H=1, W=4]: two negative, two positive
    let input_data = vec![-2f32, -1., 1., 2.];
    let input = GpuTensor::from_slice(&ctx, &input_data);
    let mask = GpuTensor::from_slice(&ctx, &vec![1f32; 4]);
    let output = GpuTensor::zeros(&ctx, 4);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 1, 4, &input, &mask, &output));
    let result = output.download(&ctx);
    assert_eq!(result, vec![0., 0., 1., 2.]);
  }

  #[test]
  fn bnact_mask_zeros_padding() {
    // Mask=0 at a position: bn_act multiplies by mask, so output should be 0
    let ctx = match gpu() { Some(c) => c, None => return };
    let desc = BatchNormLayerDesc {
      name: String::new(),
      num_channels: 1,
      epsilon: 1e-5,
      has_scale: false,
      has_bias: false,
      mean: vec![0.],
      variance: vec![1.],
      scale: vec![1.],
      bias: vec![0.],
      merged_scale: vec![1.],
      merged_bias: vec![0.],
    };
    let act = ActivationLayerDesc { name: String::new(), activation: Activation::Identity };
    let layer = BnActLayer::new(&ctx, &desc, &act);
    let input_data = vec![5f32, 3.];
    let input = GpuTensor::from_slice(&ctx, &input_data);
    let mask = GpuTensor::from_slice(&ctx, &vec![1f32, 0f32]); // second is padding
    let output = GpuTensor::zeros(&ctx, 2);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 1, 1, &input, &mask, &output));
    let result = output.download(&ctx);
    assert_eq!(result[0], 5.);
    assert_eq!(result[1], 0.);
  }

  // --- GpoolLayer ---

  #[test]
  fn gpool_stats_single_channel() {
    let ctx = match gpu() { Some(c) => c, None => return };
    let layer = GpoolLayer::new(&ctx);
    // [N=1, C=1, H=2, W=2], all-valid mask, mask_sum=4
    // input values: [1, 2, 3, 4]
    let input = GpuTensor::from_slice(&ctx, &vec![1f32, 2., 3., 4.]);
    let mask = GpuTensor::from_slice(&ctx, &vec![1f32; 4]);
    let mask_sum = GpuTensor::from_slice(&ctx, &vec![4f32]);
    let output = GpuTensor::zeros(&ctx, 3); // [3*C=3, N=1]
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 1, 2, 2, &input, &mask, &mask_sum, &output));
    let result = output.download(&ctx);
    let mean = 2.5f32; // (1+2+3+4)/4
    let scaled = mean * (4f32.sqrt() - 14.) * 0.1;
    let mx = 4f32 + (1. - 1.); // max(x + mask - 1) with mask=1 → max of values = 4
    assert!((result[0] - mean).abs() < 1e-5, "mean={}", result[0]);
    assert!((result[1] - scaled).abs() < 1e-5, "scaled={}", result[1]);
    assert!((result[2] - mx).abs() < 1e-5, "max={}", result[2]);
  }

  // --- MatMulLayer ---

  #[test]
  fn matmul_identity() {
    let ctx = match gpu() { Some(c) => c, None => return };
    // 2x2 identity matrix: weights [out=2, in=2] = [[1,0],[0,1]]
    let weights = vec![1f32, 0., 0., 1.];
    let desc = MatMulLayerDesc {
      name: String::new(),
      in_channels: 2,
      out_channels: 2,
      weights,
    };
    let layer = MatMulLayer::new(&ctx, &desc);
    // input [K=2, batch=3]
    let input_data = vec![1f32, 2., 3., 4., 5., 6.];
    let input = GpuTensor::from_slice(&ctx, &input_data);
    let output = GpuTensor::zeros(&ctx, 6); // [N_out=2, batch=3]
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 3, &input, &output));
    let result = output.download(&ctx);
    assert_eq!(result, input_data);
  }

  #[test]
  fn matmul_scale() {
    let ctx = match gpu() { Some(c) => c, None => return };
    // 2x2 matrix * 2: all-twos weight matrix applied to all-ones input
    let weights = vec![2f32, 0., 0., 2.];
    let desc = MatMulLayerDesc {
      name: String::new(),
      in_channels: 2,
      out_channels: 2,
      weights,
    };
    let layer = MatMulLayer::new(&ctx, &desc);
    let input_data = vec![1f32; 4]; // [K=2, batch=2]
    let input = GpuTensor::from_slice(&ctx, &input_data);
    let output = GpuTensor::zeros(&ctx, 4);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 2, &input, &output));
    let result = output.download(&ctx);
    assert_eq!(result, vec![2f32; 4]);
  }

  // --- MatBiasLayer ---

  #[test]
  fn matbias_adds_per_channel() {
    let ctx = match gpu() { Some(c) => c, None => return };
    let desc = MatBiasLayerDesc {
      name: String::new(),
      num_channels: 2,
      weights: vec![10f32, 20.],
    };
    let layer = MatBiasLayer::new(&ctx, &desc);
    // tensor [C=2, batch=2]: [[1,2],[3,4]]
    let tensor = GpuTensor::from_slice(&ctx, &vec![1f32, 2., 3., 4.]);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 2, &tensor));
    let result = tensor.download(&ctx);
    assert_eq!(result, vec![11f32, 12., 23., 24.]);
  }

  // --- NcBroadcastBiasAdd ---

  #[test]
  fn nc_broadcast_bias_add() {
    let ctx = match gpu() { Some(c) => c, None => return };
    let layer = NcBroadcastBiasAdd::new(&ctx);
    // bias [C=2, N=1]: [5, 10]
    let bias = GpuTensor::from_slice(&ctx, &vec![5f32, 10.]);
    // spatial [N=1, C=2, H=1, W=2]: all zeros
    let spatial = GpuTensor::from_slice(&ctx, &vec![0f32; 4]);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 2, 1, 2, &bias, &spatial));
    let result = spatial.download(&ctx);
    // channel 0 (bias=5): positions [0,1]; channel 1 (bias=10): positions [2,3]
    assert_eq!(result, vec![5f32, 5., 10., 10.]);
  }

  // =========================================================================
  // BUG REGRESSION TESTS
  // =========================================================================

  // -------------------------------------------------------------------------
  // Bug 1: ConvLayer weight double-permutation
  //
  // model.rs::ConvLayerDesc::parse already reorders weights from the on-disk
  // [y, x, ic, oc] format into the GPU-ready [oc, ic, y, x] format and stores
  // them in desc.weights.  ConvLayer::new then incorrectly treats desc.weights
  // as if it were still in [y, x, ic, oc] order and applies the permutation a
  // second time, ending up with a scrambled weight tensor.
  //
  // For off-diagonal channels (ic ≠ oc) in a 1×1 conv the bug swaps
  // W[oc=0, ic=1] with W[oc=1, ic=0], i.e.
  //   correct gpu layout: [W(0,0), W(0,1), W(1,0), W(1,1)]
  //   buggy   gpu layout: [W(0,0), W(1,0), W(0,1), W(1,1)]   ← b and c swapped
  //
  // Fix: in ConvLayer::new replace the re-permutation loop with a direct clone:
  //   let w_gpu = desc.weights.clone();
  // -------------------------------------------------------------------------
  #[test]
  fn conv_off_diagonal_weights_not_swapped() {
    // desc.weights is supplied in [oc, ic, y, x] order (as model.rs produces):
    //   pos 0 = W[oc=0, ic=0] = 1   (ic=0 → oc=0)
    //   pos 1 = W[oc=0, ic=1] = 0   (ic=1 → oc=0, zero-weight)
    //   pos 2 = W[oc=1, ic=0] = 5   (ic=0 → oc=1, large weight)
    //   pos 3 = W[oc=1, ic=1] = 1   (ic=1 → oc=1)
    let ctx = match gpu() { Some(c) => c, None => return };
    let desc = ConvLayerDesc {
      name: String::new(),
      conv_y_size: 1,
      conv_x_size: 1,
      in_channels: 2,
      out_channels: 2,
      dilation_y: 1,
      dilation_x: 1,
      // Already in [oc, ic, y, x] GPU order (output of model.rs parsing)
      weights: vec![1f32, 0., 5., 1.],
    };
    let layer = ConvLayer::new(&ctx, &desc);

    // Input NCHW [N=1, C=2, H=1, W=1]: both channels = 1
    let input = GpuTensor::from_slice(&ctx, &[1f32, 1.]);
    let output = GpuTensor::zeros(&ctx, 2);
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 1, 1, &input, &output, false));
    let result = output.download(&ctx);

    // Correct:  out[oc=0] = 1*1 + 0*1 = 1,  out[oc=1] = 5*1 + 1*1 = 6
    // Bug gives: out[oc=0] = 1*1 + 5*1 = 6,  out[oc=1] = 0*1 + 1*1 = 1  (swapped!)
    assert!(
      (result[0] - 1.0).abs() < 1e-5,
      "oc=0 output should be 1.0 (W[0,0]*1 + W[0,1]*1), got {}. \
       Bug: ConvLayer::new re-permutes already-reordered desc.weights, \
       swapping off-diagonal weights W[oc=0,ic=1] and W[oc=1,ic=0].",
      result[0]
    );
    assert!(
      (result[1] - 6.0).abs() < 1e-5,
      "oc=1 output should be 6.0 (W[1,0]*1 + W[1,1]*1), got {}",
      result[1]
    );
  }

  // -------------------------------------------------------------------------
  // Bug 2: MatMulLayer weight transposition
  //
  // model.rs stores MatMulLayerDesc::weights in [ic, oc] row-major order
  // (same as the file), i.e. weights[ic * out_channels + oc] = W(ic, oc).
  // The matmul.wgsl shader reads the weight at (out_ch, in_ch) as
  //   weights[out_ch * in_channels + in_ch]
  // which is the [oc, ic] indexing.  When the buffer holds data in [ic, oc]
  // order the shader reads the transposed matrix instead, giving wrong results
  // for non-symmetric weight matrices.
  //
  // Fix: in MatMulLayer::new, transpose desc.weights from [ic, oc] to [oc, ic]
  // before uploading to the GPU:
  //   for i in 0..ic { for o in 0..oc { w_T[o*ic+i] = desc.weights[i*oc+o]; } }
  // -------------------------------------------------------------------------
  #[test]
  fn matmul_non_symmetric_weights() {
    // desc.weights in [ic, oc] order (model.rs convention):
    //   W[ic=0, oc=0]=1  W[ic=0, oc=1]=2  W[ic=0, oc=2]=3
    //   W[ic=1, oc=0]=4  W[ic=1, oc=1]=5  W[ic=1, oc=2]=6
    // Correct output for input=[1,1]:
    //   out[oc=0] = 1*1 + 4*1 = 5
    //   out[oc=1] = 2*1 + 5*1 = 7
    //   out[oc=2] = 3*1 + 6*1 = 9
    // Bug output (shader treats as [oc, ic]):
    //   out[oc=0] = weights[0]*1 + weights[1]*1 = 1+2 = 3
    //   out[oc=1] = weights[2]*1 + weights[3]*1 = 3+4 = 7  (coincidentally correct)
    //   out[oc=2] = weights[4]*1 + weights[5]*1 = 5+6 = 11
    let ctx = match gpu() { Some(c) => c, None => return };
    let desc = MatMulLayerDesc {
      name: String::new(),
      in_channels: 2,
      out_channels: 3,
      weights: vec![1f32, 2., 3., 4., 5., 6.], // [ic, oc] order
    };
    let layer = MatMulLayer::new(&ctx, &desc);

    // input [K=in_channels=2, batch=1]
    let input = GpuTensor::from_slice(&ctx, &[1f32, 1.]);
    let output = GpuTensor::zeros(&ctx, 3); // [N_out=3, batch=1]
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, &input, &output));
    let result = output.download(&ctx);

    assert!(
      (result[0] - 5.0).abs() < 1e-4,
      "out[oc=0] should be 5.0 (W[0,0]+W[1,0] = 1+4), got {}. \
       Bug: MatMulLayer uploads weights in [ic,oc] order but shader reads \
       them as [oc,ic], effectively using the transposed weight matrix.",
      result[0]
    );
    assert!(
      (result[2] - 9.0).abs() < 1e-4,
      "out[oc=2] should be 9.0 (W[0,2]+W[1,2] = 3+6), got {}",
      result[2]
    );
  }

  // -------------------------------------------------------------------------
  // Bug 3: Value head uses wrong GPool pooling formula for stat2
  //
  // The trunk / policy gpool blocks use poolRowsGPool where:
  //   stat2 = max over valid positions of (x + mask - 1)
  //
  // The value head must use poolRowsValueHead where:
  //   stat2 = mean * ((sqrt(mask_sum) - 14) ^ 2 * 0.01 - 0.1)
  //
  // eval.rs creates a single GpoolLayer (backed by gpool.wgsl) and uses it
  // for BOTH the gpool blocks AND the value head.  The value head thus gets
  // the spatial max as stat2 instead of the quadratic mean formula, producing
  // incorrect value estimates.
  //
  // Fix: add a `ValueHeadGpoolLayer` backed by a new `gpool_value_head.wgsl`
  // shader that outputs the quadratic stat2, and use it in ValueHead instead
  // of GpoolLayer.
  // -------------------------------------------------------------------------
  #[test]
  fn gpool_value_head_stat2_differs_from_max() {
    // With input=[3, 0, 0, 0], all-valid mask, mask_sum=4:
    //   mean = 0.75,  sqrtdiv = 2.0
    //   value-head stat2 = 0.75 * ((2-14)^2 * 0.01 - 0.1)
    //                    = 0.75 * (1.44 - 0.1) = 0.75 * 1.34 = 1.005
    //   gpool stat2 (bug) = max(3+0, 0-1, 0-1, 0-1) = 3.0   ← wrong!
    let ctx = match gpu() { Some(c) => c, None => return };
    let layer = GpoolLayer::new(&ctx);
    let input = GpuTensor::from_slice(&ctx, &[3f32, 0., 0., 0.]);
    let mask = GpuTensor::from_slice(&ctx, &[1f32; 4]);
    let mask_sum = GpuTensor::from_slice(&ctx, &[4f32]);
    let output = GpuTensor::zeros(&ctx, 3); // [3*C=3, N=1]
    submit(&ctx, |enc| layer.dispatch(&ctx, enc, 1, 1, 2, 2, &input, &mask, &mask_sum, &output));
    let result = output.download(&ctx);

    // stat2 as returned by the gpool shader
    let stat2_actual = result[2];
    // stat2 that the value head requires (quadratic formula)
    let mean = 0.75f32;
    let sqrtdiv = 2.0f32;
    let stat2_value_head = mean * ((sqrtdiv - 14.0).powi(2) * 0.01 - 0.1);

    // The value-head formula and the max are different for this input.
    // The test documents that the current gpool shader gives the wrong result
    // for the value head (it computes max=3.0 instead of ~1.005).
    assert!(
      (stat2_actual - stat2_value_head).abs() > 0.1,
      "gpool stat2 ({}) unexpectedly matches value-head formula ({}) — \
       this means the bug may have been fixed or the test data need updating.",
      stat2_actual, stat2_value_head
    );
    // The correct value-head value is ~1.005; the buggy shader returns 3.0.
    assert!(
      (stat2_actual - 3.0f32).abs() < 1e-5,
      "gpool shader stat2 should be max=3 for this input, got {}",
      stat2_actual
    );
  }

  // -------------------------------------------------------------------------
  // Bug 4 (in eval.rs): GpoolBlock pre_activation always forced to Identity
  //
  // In eval.rs Evaluator::new, for BlockDesc::GlobalPooling the pre_bn is
  // constructed as:
  //   BnActLayer::new(ctx, &b.pre_bn, &identity_act())
  // instead of the correct:
  //   BnActLayer::new(ctx, &b.pre_bn, &b.pre_activation)
  //
  // The C++ GlobalPoolingResidualBlock initialises preBN(desc.preBN,
  // desc.preActivation), so the activation is whatever the model file says
  // (typically ReLU).  The Rust code hardcodes Identity, so negative
  // pre-activations are never clipped.
  //
  // Fix: change the identity_act() call to &b.pre_activation in eval.rs.
  // -------------------------------------------------------------------------

  // -------------------------------------------------------------------------
  // Bug 5 (in eval.rs): Value head v2 activation is never applied
  //
  // The C++ value head applies v2Activation in-place after v2Bias:
  //   v2Activation.apply(&v2Out, &v2Out);
  // and only then feeds v2Out to v3Mul / sv3Mul.
  //
  // In eval.rs Evaluator::run the comment says
  //   // v2 activation is applied on CPU after download
  // but no CPU-side activation is applied anywhere: v2_out is fed directly to
  // v3_mul and sv3_mul while still in its pre-activation state.  This means
  // value and score value logits are computed from wrong intermediate features.
  //
  // Fix: after vh.v2_bias.dispatch download v2_out, apply the activation on
  // the CPU (or add a shader pass), re-upload, then dispatch v3_mul / sv3_mul.
  // -------------------------------------------------------------------------
}
