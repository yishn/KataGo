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
    // Weights on disk: [KY, KX, Cin, Cout] — re-order to [Cout, Cin, KH, KW]
    let ky = desc.conv_y_size as usize;
    let kx = desc.conv_x_size as usize;
    let cin = desc.in_channels as usize;
    let cout = desc.out_channels as usize;
    let mut w_gpu = vec![0f32; cout * cin * ky * kx];
    for oc in 0..cout {
      for ic in 0..cin {
        for y in 0..ky {
          for x in 0..kx {
            // disk layout: [y, x, ic, oc]
            let src = y * kx * cin * cout + x * cin * cout + ic * cout + oc;
            // gpu layout: [oc, ic, y, x]
            let dst = oc * cin * ky * kx + ic * ky * kx + y * kx + x;
            w_gpu[dst] = desc.weights[src];
          }
        }
      }
    }
    Self {
      pipeline: make_pipeline(ctx, shaders::CONV, "main"),
      weights: WeightBuffer::new(ctx, &w_gpu),
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
    // weights from desc are [out, in] — matches shader expectation
    Self {
      pipeline: make_pipeline(ctx, shaders::MATMUL, "main"),
      weights: WeightBuffer::new(ctx, &desc.weights),
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
}
