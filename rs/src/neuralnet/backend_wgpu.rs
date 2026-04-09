/// WebGPU backend using the `wgpu` crate.
///
/// # Architecture
///
/// Each arithmetic operation (convolution, batch-norm, matmul, pooling, …) is
/// implemented as one or more WGSL compute shaders.  On construction all
/// weight tensors are uploaded to `wgpu::Buffer`s labelled
/// `STORAGE | COPY_DST`.  The forward pass then:
///
/// 1. Uploads the per-call input tensors (spatial + global features).
/// 2. Chains a sequence of dispatch calls — one per logical layer — writing
///    intermediate activations into pre-allocated GPU buffers.
/// 3. Reads back the five output tensors (`policy_pass`, `policy_spatial`,
///    `value`, `score_value`, `ownership`) to CPU memory.
///
/// All tensors keep the same **NHWC row-major** layout as the CPU backend.
///
/// # Shader design
///
/// Every shader is a plain WGSL compute kernel with workgroup size 64.
/// Binding slots follow the convention:
/// ```text
///   @group(0) @binding(0)  input  / left operand   (read-only storage)
///   @group(0) @binding(1)  weight / right operand  (read-only storage)
///   @group(0) @binding(2)  output                  (read-write storage)
///   @group(0) @binding(3)  uniform / extra params  (when needed)
/// ```
///
/// # Synchronous execution
///
/// `wgpu` is async by nature.  On non-WASM targets this module uses the
/// `pollster` crate to block the calling thread until GPU work is complete,
/// matching the synchronous [`Backend`] contract.
///
/// # Feature coverage
///
/// The following operations are GPU-accelerated:
/// * Convolution (1×1, 3×3 direct) via `shader_conv_nhwc`
/// * BatchNorm + activation (ReLU / Mish / identity) via `shader_batchnorm_act`
/// * Matrix multiply `[OC, IC] × [IC, N]` via `shader_matmul`
/// * Matrix bias add (in-place, `[C, N]` layout) via `shader_matbias`
/// * NC-bias broadcast add into NHWC tensor via `shader_add_nc_bias`
/// * Global-pool gpool variant via `shader_gpool`
/// * Global-pool value-head variant via `shader_value_pool`
/// * Mask-sum reduction via `shader_mask_sum`
///
/// SGF metadata encoder, nested bottleneck residual blocks, and model
/// versions >= 15 (two-stage pass head) are all supported via the same
/// sequence as the CPU backend.
use futures_channel::oneshot;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::model::{
  Activation, BatchNormLayerDesc, BlockDesc, ConvLayerDesc, MatBiasLayerDesc,
  MatMulLayerDesc, ModelDesc,
};
use crate::neuralnet::backend::{Backend, EvalOutput, RunFuture};

// ============================================================================
// Helpers
// ============================================================================

/// Round `n` up to the nearest multiple of `align`.
#[inline(always)]
fn align_up(n: u64, align: u64) -> u64 {
  (n + align - 1) & !(align - 1)
}

/// Create a GPU buffer pre-filled with `data`.
fn upload_f32(
  device: &wgpu::Device,
  label: &str,
  data: &[f32],
) -> wgpu::Buffer {
  device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some(label),
    contents: bytemuck::cast_slice(data),
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
  })
}

/// Create an uninitialised STORAGE + COPY_SRC GPU buffer of `count` f32 elements.
fn alloc_f32(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
  device.create_buffer(&wgpu::BufferDescriptor {
    label: Some(label),
    size: (count * 4) as u64,
    usage: wgpu::BufferUsages::STORAGE
      | wgpu::BufferUsages::COPY_SRC
      | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  })
}

/// Write `data` into an existing buffer (offset 0).
fn write_f32(queue: &wgpu::Queue, buf: &wgpu::Buffer, data: &[f32]) {
  queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
}

/// Readback `count` f32 values from a GPU buffer (async; works on native and WASM).
async fn readback_f32(
  device: &wgpu::Device,
  queue: &wgpu::Queue,
  src: &wgpu::Buffer,
  count: usize,
) -> Vec<f32> {
  let size = (count * 4) as u64;
  let staging = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("staging_readback"),
    size,
    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });

  let mut enc =
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("readback_enc"),
    });
  enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
  queue.submit(std::iter::once(enc.finish()));

  let (tx, rx) = oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
  staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
    let _ = tx.send(r);
  });
  // On native we drive the event loop ourselves; on WASM the browser does it.
  device.poll(wgpu::Maintain::Wait);
  rx.await
    .expect("GPU map_async channel closed")
    .expect("GPU map error");

  let data = staging.slice(..).get_mapped_range();
  let v = bytemuck::cast_slice(&data).to_vec();
  drop(data);
  staging.unmap();
  v
}

/// Zero a GPU buffer via a fill pass.
fn zero_buffer(encoder: &mut wgpu::CommandEncoder, buf: &wgpu::Buffer) {
  encoder.clear_buffer(buf, 0, None);
}

// ============================================================================
// Uniform structs (must match WGSL struct layouts)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
  n: u32,
  h: u32,
  w: u32,
  ic: u32,
  oc: u32,
  ky: u32,
  kx: u32,
  accumulate: u32, // 0 = write, 1 = accumulate
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BnParams {
  nhw: u32,
  nc: u32,
  act: u32, // 0=identity, 1=relu, 2=mish
  _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
  ic: u32,
  oc: u32,
  batch: u32,
  _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatbiasParams {
  nc: u32,
  batch: u32,
  _pad0: u32,
  _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AddNcBiasParams {
  h: u32,
  w: u32,
  nc: u32,
  batch: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpoolParams {
  batch: u32,
  h: u32,
  w: u32,
  c_in: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaskSumParams {
  batch: u32,
  hw: u32,
  _pad0: u32,
  _pad1: u32,
}

// ============================================================================
// WGSL shader sources
// ============================================================================

/// 2-D direct convolution (any kernel size, no dilation).
/// Weights layout: [OC, IC, KY, KX] (same as CPU direct path).
const SHADER_CONV: &str = r#"
struct ConvParams {
  n: u32, h: u32, w: u32,
  ic: u32, oc: u32, ky: u32, kx: u32,
  accumulate: u32,
}
@group(0) @binding(0) var<storage, read>       inp    : array<f32>;
@group(0) @binding(1) var<storage, read>       weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> out    : array<f32>;
@group(0) @binding(3) var<uniform>             p      : ConvParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.n * p.h * p.w * p.oc;
  if idx >= total { return; }

  let oc_i = idx % p.oc;
  let rem  = idx / p.oc;
  let xi   = rem % p.w;
  let rem2 = rem / p.w;
  let yi   = rem2 % p.h;
  let ni   = rem2 / p.h;

  let pad_y = i32(p.ky) / 2;
  let pad_x = i32(p.kx) / 2;
  var acc: f32 = 0.0;

  for (var sy: u32 = 0u; sy < p.ky; sy += 1u) {
    let iy = i32(yi) + i32(sy) - pad_y;
    if iy < 0 || iy >= i32(p.h) { continue; }
    for (var sx: u32 = 0u; sx < p.kx; sx += 1u) {
      let ix = i32(xi) + i32(sx) - pad_x;
      if ix < 0 || ix >= i32(p.w) { continue; }
      for (var ic_i: u32 = 0u; ic_i < p.ic; ic_i += 1u) {
        let in_idx = (ni * p.h * p.w + u32(iy) * p.w + u32(ix)) * p.ic + ic_i;
        let k_idx  = (oc_i * p.ic * p.ky * p.kx)
                   + (ic_i * p.ky * p.kx)
                   + (sy   * p.kx)
                   + sx;
        acc += inp[in_idx] * weight[k_idx];
      }
    }
  }

  let out_idx = (ni * p.h * p.w + yi * p.w + xi) * p.oc + oc_i;
  if p.accumulate != 0u {
    out[out_idx] += acc;
  } else {
    out[out_idx] = acc;
  }
}
"#;

/// BatchNorm + optional activation.
/// Activation codes: 0=identity, 1=relu, 2=mish.
/// Masked cells (mask==0) are zeroed before activation.
const SHADER_BATCHNORM_ACT: &str = r#"
struct BnParams { nhw: u32, nc: u32, act: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read>       inp   : array<f32>;
@group(0) @binding(1) var<storage, read>       scale : array<f32>;
@group(0) @binding(2) var<storage, read>       bias  : array<f32>;
@group(0) @binding(3) var<storage, read>       mask  : array<f32>;
@group(0) @binding(4) var<storage, read_write> out   : array<f32>;
@group(0) @binding(5) var<uniform>             p     : BnParams;

fn mish(x: f32) -> f32 {
  let sp = select(log(1.0 + exp(x)), x, x >= 20.0);
  return x * tanh(sp);
}
fn activate(x: f32, act: u32) -> f32 {
  switch act {
    case 1u: { return max(x, 0.0); }
    case 2u: { return mish(x); }
    default: { return x; }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if idx >= p.nhw * p.nc { return; }
  let c   = idx % p.nc;
  let pos = idx / p.nc;  // flat position in NHW
  let m   = mask[pos];
  let v   = inp[idx] * scale[c] + bias[c];
  out[idx] = select(0.0, activate(v, p.act), m == 1.0);
}
"#;

/// Dense matrix multiply: out[OC, N] = W[OC, IC] * in[IC, N].
const SHADER_MATMUL: &str = r#"
struct MatmulParams { ic: u32, oc: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read>       inp    : array<f32>;
@group(0) @binding(1) var<storage, read>       weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> out    : array<f32>;
@group(0) @binding(3) var<uniform>             p      : MatmulParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if idx >= p.oc * p.batch { return; }
  let n    = idx % p.batch;
  let oc_i = idx / p.batch;
  var acc: f32 = 0.0;
  for (var ic_i: u32 = 0u; ic_i < p.ic; ic_i += 1u) {
    acc += weight[oc_i * p.ic + ic_i] * inp[ic_i * p.batch + n];
  }
  out[oc_i * p.batch + n] = acc;
}
"#;

/// Matrix bias add in-place: mat[c, n] += bias[c].
const SHADER_MATBIAS: &str = r#"
struct MatbiasParams { nc: u32, batch: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read>       bias : array<f32>;
@group(0) @binding(1) var<storage, read_write> mat  : array<f32>;
@group(0) @binding(2) var<uniform>             p    : MatbiasParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if idx >= p.nc * p.batch { return; }
  let n = idx % p.batch;
  let c = idx / p.batch;
  mat[c * p.batch + n] += bias[c];
}
"#;

/// Broadcast-add a `[C, N]` bias tensor into an NHWC `[N*H*W*C]` tensor.
const SHADER_ADD_NC_BIAS: &str = r#"
struct AddNcBiasParams { h: u32, w: u32, nc: u32, batch: u32 }
@group(0) @binding(0) var<storage, read>       bias   : array<f32>;
@group(0) @binding(1) var<storage, read_write> tensor : array<f32>;
@group(0) @binding(2) var<uniform>             p      : AddNcBiasParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let hw  = p.h * p.w;
  if idx >= p.batch * hw * p.nc { return; }
  let c  = idx % p.nc;
  let rem = idx / p.nc;
  let n  = rem / hw;
  tensor[idx] += bias[c * p.batch + n];
}
"#;

/// Global-pool (gpool variant) — mean / sqrt-weighted mean / max.
/// in4d: NHWC [N*H*W*C].  out: [3*C, N].  mask: [N*H*W].  mask_sum: [N].
const SHADER_GPOOL: &str = r#"
struct GpoolParams { batch: u32, h: u32, w: u32, c_in: u32 }
@group(0) @binding(0) var<storage, read>       in4d     : array<f32>;
@group(0) @binding(1) var<storage, read>       mask_buf : array<f32>;
@group(0) @binding(2) var<storage, read>       msum_buf : array<f32>;
@group(0) @binding(3) var<storage, read_write> out2d    : array<f32>;
@group(0) @binding(4) var<uniform>             p        : GpoolParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if idx >= p.batch * p.c_in { return; }
  let n = idx / p.c_in;
  let c = idx % p.c_in;
  let hw = p.h * p.w;
  let div      = msum_buf[n];
  let sqrtdiv  = sqrt(div);
  var s: f32 = 0.0;
  var m: f32 = -1.0;
  for (var yi: u32 = 0u; yi < p.h; yi += 1u) {
    for (var xi: u32 = 0u; xi < p.w; xi += 1u) {
      let pos  = n * hw + yi * p.w + xi;
      let x    = in4d[pos * p.c_in + c];
      s += x;
      let mv   = mask_buf[pos];
      let cand = x + (mv - 1.0);
      if cand > m { m = cand; }
    }
  }
  let mean = s / div;
  out2d[c * p.batch + n]               = mean;
  out2d[(c + p.c_in) * p.batch + n]    = mean * (sqrtdiv - 14.0) * 0.1;
  out2d[(c + 2u * p.c_in) * p.batch + n] = m;
}
"#;

/// Global-pool value-head variant — mean / sqrt-weighted mean / quadratic term.
const SHADER_VALUE_POOL: &str = r#"
struct GpoolParams { batch: u32, h: u32, w: u32, c_in: u32 }
@group(0) @binding(0) var<storage, read>       in4d     : array<f32>;
@group(0) @binding(1) var<storage, read>       msum_buf : array<f32>;
@group(0) @binding(2) var<storage, read_write> out2d    : array<f32>;
@group(0) @binding(3) var<uniform>             p        : GpoolParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if idx >= p.batch * p.c_in { return; }
  let n = idx / p.c_in;
  let c = idx % p.c_in;
  let hw  = p.h * p.w;
  let div     = msum_buf[n];
  let sqrtdiv = sqrt(div);
  var s: f32 = 0.0;
  for (var yi: u32 = 0u; yi < p.h; yi += 1u) {
    for (var xi: u32 = 0u; xi < p.w; xi += 1u) {
      let pos = n * hw + yi * p.w + xi;
      s += in4d[pos * p.c_in + c];
    }
  }
  let mean = s / div;
  let sd14 = sqrtdiv - 14.0;
  out2d[c * p.batch + n]               = mean;
  out2d[(c + p.c_in) * p.batch + n]    = mean * sd14 * 0.1;
  out2d[(c + 2u * p.c_in) * p.batch + n] = mean * (sd14 * sd14 * 0.01 - 0.1);
}
"#;

/// Mask-sum reduction: mask_sum[n] = sum of mask[n*hw .. (n+1)*hw].
const SHADER_MASK_SUM: &str = r#"
struct MaskSumParams { batch: u32, hw: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read>       mask : array<f32>;
@group(0) @binding(1) var<storage, read_write> msum : array<f32>;
@group(0) @binding(2) var<uniform>             p    : MaskSumParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if n >= p.batch { return; }
  var s: f32 = 0.0;
  for (var i: u32 = 0u; i < p.hw; i += 1u) {
    s += mask[n * p.hw + i];
  }
  msum[n] = s;
}
"#;

/// Extract the NHW mask from channel-0 of an NHWC tensor.
const SHADER_EXTRACT_MASK: &str = r#"
struct ConvParams {
  n: u32, h: u32, w: u32,
  ic: u32, oc: u32, ky: u32, kx: u32,
  accumulate: u32,
}
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read_write> mask : array<f32>;
@group(0) @binding(2) var<uniform>             p    : ConvParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pos = gid.x;
  if pos >= p.n * p.h * p.w { return; }
  mask[pos] = inp[pos * p.ic];
}
"#;

/// Activation-only in-place pass (for SGF encoder / v2 value path).
const SHADER_ACTIVATE_INPLACE: &str = r#"
struct ActParams { n: u32, act: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read_write> buf : array<f32>;
@group(0) @binding(1) var<uniform>             p   : ActParams;

fn mish(x: f32) -> f32 {
  let sp = select(log(1.0 + exp(x)), x, x >= 20.0);
  return x * tanh(sp);
}
fn activate(x: f32, act: u32) -> f32 {
  switch act {
    case 1u: { return max(x, 0.0); }
    case 2u: { return mish(x); }
    default: { return x; }
  }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= p.n { return; }
  buf[i] = activate(buf[i], p.act);
}
"#;

// ============================================================================
// GpuPipeline — holds one compiled pipeline and its bind-group layout
// ============================================================================

/// Describes how a single binding slot is declared in WGSL.
#[derive(Copy, Clone)]
enum BindSlot {
  /// `var<storage, read>` — read-only input / weight buffer.
  ReadOnly,
  /// `var<storage, read_write>` — read-write output buffer.
  ReadWrite,
  /// `var<uniform>` — small params / uniform buffer.
  Uniform,
}

struct GpuPipeline {
  pipeline: wgpu::ComputePipeline,
  bgl: wgpu::BindGroupLayout,
}

impl GpuPipeline {
  fn new(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    slots: &[BindSlot],
  ) -> Self {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some(label),
      source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    let entries: Vec<wgpu::BindGroupLayoutEntry> = slots
      .iter()
      .enumerate()
      .map(|(i, slot)| wgpu::BindGroupLayoutEntry {
        binding: i as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: match slot {
          BindSlot::ReadOnly => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          BindSlot::ReadWrite => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          BindSlot::Uniform => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
        },
        count: None,
      })
      .collect();

    let bgl =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bgl")),
        entries: &entries,
      });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some(&format!("{label}_pll")),
      bind_group_layouts: &[&bgl],
      push_constant_ranges: &[],
    });
    let pipeline =
      device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
      });
    GpuPipeline { pipeline, bgl }
  }

  fn bind_group(
    &self,
    device: &wgpu::Device,
    bufs: &[&wgpu::Buffer],
  ) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bufs
      .iter()
      .enumerate()
      .map(|(i, b)| wgpu::BindGroupEntry {
        binding: i as u32,
        resource: b.as_entire_binding(),
      })
      .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &self.bgl,
      entries: &entries,
    })
  }

  fn dispatch(
    &self,
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    bufs: &[&wgpu::Buffer],
    n_elements: u32,
  ) {
    let bg = self.bind_group(device, bufs);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups((n_elements + 63) / 64, 1, 1);
  }
}

// ============================================================================
// GPU-side weight / scratch buffer collections
// ============================================================================

/// Compiled + uploaded single convolution layer.
struct GpuConv {
  weight: wgpu::Buffer,
  ky: u32,
  kx: u32,
  ic: u32,
  oc: u32,
}

impl GpuConv {
  fn new(device: &wgpu::Device, desc: &ConvLayerDesc) -> Self {
    let ky = desc.conv_y_size as u32;
    let kx = desc.conv_x_size as u32;
    let ic = desc.in_channels as u32;
    let oc = desc.out_channels as u32;
    // Reorder from file layout [OC, IC, KY, KX] → same (already correct for direct).
    let weight =
      upload_f32(device, &format!("{}_weight", desc.name), &desc.weights);
    GpuConv {
      weight,
      ky,
      kx,
      ic,
      oc,
    }
  }
}

struct GpuBn {
  scale: wgpu::Buffer,
  bias: wgpu::Buffer,
  nc: u32,
  act: u32,
}

impl GpuBn {
  fn new(
    device: &wgpu::Device,
    desc: &BatchNormLayerDesc,
    act: Activation,
  ) -> Self {
    let act_code = match act {
      Activation::Identity => 0,
      Activation::Relu => 1,
      Activation::Mish | Activation::MishScale8 => 2,
    };
    GpuBn {
      scale: upload_f32(
        device,
        &format!("{}_scale", desc.name),
        &desc.merged_scale,
      ),
      bias: upload_f32(
        device,
        &format!("{}_bias", desc.name),
        &desc.merged_bias,
      ),
      nc: desc.num_channels as u32,
      act: act_code,
    }
  }
}

struct GpuMatMul {
  weight: wgpu::Buffer,
  ic: u32,
  oc: u32,
}

impl GpuMatMul {
  fn new(device: &wgpu::Device, desc: &MatMulLayerDesc) -> Self {
    let ic = desc.in_channels as u32;
    let oc = desc.out_channels as u32;
    // Transpose from [IC, OC] (file order) to [OC, IC] as needed by shader.
    let mut w = vec![0.0f32; (ic * oc) as usize];
    for ic_i in 0..ic as usize {
      for oc_i in 0..oc as usize {
        w[oc_i * ic as usize + ic_i] = desc.weights[ic_i * oc as usize + oc_i];
      }
    }
    GpuMatMul {
      weight: upload_f32(device, &format!("{}_weight", desc.name), &w),
      ic,
      oc,
    }
  }
}

struct GpuMatBias {
  weight: wgpu::Buffer,
  nc: u32,
}

impl GpuMatBias {
  fn new(device: &wgpu::Device, desc: &MatBiasLayerDesc) -> Self {
    GpuMatBias {
      weight: upload_f32(
        device,
        &format!("{}_weight", desc.name),
        &desc.weights,
      ),
      nc: desc.num_channels as u32,
    }
  }
}

// ============================================================================
// Compiled GPU model sub-graphs
// ============================================================================

struct GpuNormActConv {
  bn: GpuBn,
  conv: GpuConv,
}

struct GpuResBlock {
  nac1: GpuNormActConv,
  nac2: GpuNormActConv,
}

struct GpuGpoolResBlock {
  pre_bn: GpuBn,
  regular_conv: GpuConv,
  gpool_conv: GpuConv,
  gpool_bn: GpuBn,
  gpool_to_bias_mul: GpuMatMul,
  nac2: GpuNormActConv,
}

struct GpuNestedBottleneck {
  nac1: GpuNormActConv,
  inner: Vec<GpuBlock>,
  nac2: GpuNormActConv,
}

enum GpuBlock {
  Ordinary(GpuResBlock),
  GlobalPooling(GpuGpoolResBlock),
  NestedBottleneck(GpuNestedBottleneck),
}

struct GpuTrunk {
  initial_conv: GpuConv,
  initial_mat_mul: GpuMatMul,
  sgf_meta_encoder: Option<GpuSgfEncoder>,
  blocks: Vec<GpuBlock>,
  trunk_tip_bn: GpuBn,
  trunk_c: u32,
}

struct GpuSgfEncoder {
  mul1: GpuMatMul,
  bias1: GpuMatBias,
  act1: u32,
  mul2: GpuMatMul,
  bias2: GpuMatBias,
  act2: u32,
  mul3: GpuMatMul,
}

struct GpuPolicyHead {
  model_version: i32,
  p1_conv: GpuConv,
  g1_conv: GpuConv,
  g1_bn: GpuBn,
  gpool_to_bias_mul: GpuMatMul,
  p1_bn: GpuBn,
  p2_conv: GpuConv,
  gpool_to_pass_mul: GpuMatMul,
  gpool_to_pass_bias: Option<GpuMatBias>,
  pass_activation: u32,
  gpool_to_pass_mul2: Option<GpuMatMul>,
  p1c: u32,
  g1c: u32,
  p2c: u32,
}

struct GpuValueHead {
  v1_conv: GpuConv,
  v1_bn: GpuBn,
  v2_mul: GpuMatMul,
  v2_bias: GpuMatBias,
  v2_act: u32,
  v3_mul: GpuMatMul,
  v3_bias: GpuMatBias,
  sv3_mul: GpuMatMul,
  sv3_bias: GpuMatBias,
  v_ownership_conv: GpuConv,
  v1c: u32,
  v2c: u32,
  v3c: u32,
  sv3c: u32,
  owc: u32,
}

// ============================================================================
// GpuCompiled — encapsulates everything needed for repeated forward passes
// ============================================================================

struct Pipelines {
  conv: GpuPipeline,
  bn: GpuPipeline,
  matmul: GpuPipeline,
  matbias: GpuPipeline,
  add_nc_bias: GpuPipeline,
  gpool: GpuPipeline,
  value_pool: GpuPipeline,
  mask_sum: GpuPipeline,
  extract_mask: GpuPipeline,
  activate_inplace: GpuPipeline,
}

impl Pipelines {
  fn new(device: &wgpu::Device) -> Self {
    use BindSlot::{ReadOnly as R, ReadWrite as RW, Uniform as U};
    Pipelines {
      // SHADER_CONV:           inp(R), weight(R), out(RW), params(U)
      conv: GpuPipeline::new(device, "conv", SHADER_CONV, &[R, R, RW, U]),
      // SHADER_BATCHNORM_ACT:  inp(R), scale(R), bias(R), mask(R), out(RW), params(U)
      bn: GpuPipeline::new(
        device,
        "bn_act",
        SHADER_BATCHNORM_ACT,
        &[R, R, R, R, RW, U],
      ),
      // SHADER_MATMUL:         inp(R), weight(R), out(RW), params(U)
      matmul: GpuPipeline::new(device, "matmul", SHADER_MATMUL, &[R, R, RW, U]),
      // SHADER_MATBIAS:        bias(R), mat(RW), params(U)
      matbias: GpuPipeline::new(device, "matbias", SHADER_MATBIAS, &[R, RW, U]),
      // SHADER_ADD_NC_BIAS:    bias(R), tensor(RW), params(U)
      add_nc_bias: GpuPipeline::new(
        device,
        "add_nc_bias",
        SHADER_ADD_NC_BIAS,
        &[R, RW, U],
      ),
      // SHADER_GPOOL:          in4d(R), mask(R), msum(R), out(RW), params(U)
      gpool: GpuPipeline::new(device, "gpool", SHADER_GPOOL, &[R, R, R, RW, U]),
      // SHADER_VALUE_POOL:     in4d(R), msum(R), out(RW), params(U)
      value_pool: GpuPipeline::new(
        device,
        "value_pool",
        SHADER_VALUE_POOL,
        &[R, R, RW, U],
      ),
      // SHADER_MASK_SUM:       mask(R), msum(RW), params(U)
      mask_sum: GpuPipeline::new(
        device,
        "mask_sum",
        SHADER_MASK_SUM,
        &[R, RW, U],
      ),
      // SHADER_EXTRACT_MASK:   inp(R), mask(RW), params(U)
      extract_mask: GpuPipeline::new(
        device,
        "extract_mask",
        SHADER_EXTRACT_MASK,
        &[R, RW, U],
      ),
      // SHADER_ACTIVATE_INPLACE: buf(RW), params(U)
      activate_inplace: GpuPipeline::new(
        device,
        "activate_inplace",
        SHADER_ACTIVATE_INPLACE,
        &[RW, U],
      ),
    }
  }
}

// ============================================================================
// WgpuBackend
// ============================================================================

pub struct WgpuBackend {
  device: Arc<wgpu::Device>,
  queue: Arc<wgpu::Queue>,
  pipes: Pipelines,
  trunk: GpuTrunk,
  policy_head: GpuPolicyHead,
  value_head: GpuValueHead,
  meta: ModelMeta,
  nn_x: usize,
  nn_y: usize,
}

#[derive(Clone)]
struct ModelMeta {
  model_version: i32,
  num_input_channels: usize,
  num_input_global_channels: usize,
  num_input_meta_channels: usize,
  num_policy_channels: usize,
  num_value_channels: usize,
  num_score_value_channels: usize,
  num_ownership_channels: usize,
}

impl WgpuBackend {
  /// Try to create a [`WgpuBackend`].  Returns `Err` if no suitable GPU adapter
  /// is available.
  pub async fn new(
    desc: &ModelDesc,
    nn_x: usize,
    nn_y: usize,
  ) -> Result<Self, String> {
    // Obtain a wgpu device.
    let (device, queue) = Self::acquire_device().await?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let pipes = Pipelines::new(&device);

    let trunk = build_trunk(&device, &desc.trunk);
    let policy_head =
      build_policy_head(&device, &desc.policy_head, desc.model_version);
    let value_head = build_value_head(&device, &desc.value_head);

    let meta = ModelMeta {
      model_version: desc.model_version,
      num_input_channels: desc.num_input_channels as usize,
      num_input_global_channels: desc.num_input_global_channels as usize,
      num_input_meta_channels: desc.num_input_meta_channels as usize,
      num_policy_channels: desc.num_policy_channels as usize,
      num_value_channels: desc.num_value_channels as usize,
      num_score_value_channels: desc.num_score_value_channels as usize,
      num_ownership_channels: desc.num_ownership_channels as usize,
    };

    Ok(WgpuBackend {
      device,
      queue,
      pipes,
      trunk,
      policy_head,
      value_head,
      meta,
      nn_x,
      nn_y,
    })
  }

  async fn acquire_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
      backends: wgpu::Backends::all(),
      ..Default::default()
    });
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
      })
      .await
      .ok_or_else(|| "no wgpu adapter found".to_string())?;

    let limits = wgpu::Limits::default();

    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          label: Some("katago_wgpu"),
          required_features: wgpu::Features::empty(),
          required_limits: limits,
          memory_hints: Default::default(),
        },
        None, // no trace path
      )
      .await
      .map_err(|e| format!("wgpu device request failed: {e}"))?;
    Ok((device, queue))
  }

  /// Async forward pass — builds and submits the command buffer, then reads
  /// back all five output tensors.
  async fn run_internal(
    &self,
    spatial: &[f32],
    global: &[f32],
    meta: Option<&[f32]>,
    nn_x: usize,
    nn_y: usize,
  ) -> EvalOutput {
    let batch = 1u32;
    let hw = (nn_x * nn_y) as u32;
    let ic = self.meta.num_input_channels as u32;

    // Upload inputs
    let inp_buf = upload_f32(&self.device, "inp", spatial);
    let inp_global_buf = upload_f32(&self.device, "inp_global", global);
    let inp_meta_buf = meta.map(|m| upload_f32(&self.device, "inp_meta", m));

    // Allocate mask + mask_sum buffers
    let mask_buf = alloc_f32(&self.device, "mask", (batch * hw) as usize);
    let msum_buf = alloc_f32(&self.device, "msum", batch as usize);

    let ctx = ForwardCtx {
      device: &self.device,
      queue: &self.queue,
      pipes: &self.pipes,
      nn_x: nn_x as u32,
      nn_y: nn_y as u32,
      batch,
    };

    let mut enc =
      self
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
          label: Some("forward_pass"),
        });

    // Extract mask from channel 0, compute mask_sum
    ctx.extract_mask(&mut enc, &inp_buf, ic, &mask_buf);
    ctx.mask_sum(&mut enc, &mask_buf, &msum_buf);

    // Trunk
    let trunk_out = ctx.run_trunk(
      &mut enc,
      &inp_buf,
      &inp_global_buf,
      inp_meta_buf.as_ref(),
      &self.trunk,
      &mask_buf,
      &msum_buf,
    );

    // Policy head
    let (pp_buf, ps_buf) = ctx.run_policy_head(
      &mut enc,
      &trunk_out,
      &self.policy_head,
      &mask_buf,
      &msum_buf,
    );

    // Value head
    let (val_buf, sv_buf, own_buf) = ctx.run_value_head(
      &mut enc,
      &trunk_out,
      &self.value_head,
      &mask_buf,
      &msum_buf,
    );

    // Submit the entire forward pass as a single command buffer.
    self.queue.submit(std::iter::once(enc.finish()));

    // Async readbacks — each call submits a staging copy and awaits the map.
    let policy_pass = readback_f32(
      &self.device,
      &self.queue,
      &pp_buf,
      self.meta.num_policy_channels,
    )
    .await;
    let policy_spatial = readback_f32(
      &self.device,
      &self.queue,
      &ps_buf,
      nn_x * nn_y * self.meta.num_policy_channels,
    )
    .await;
    let value = readback_f32(
      &self.device,
      &self.queue,
      &val_buf,
      self.meta.num_value_channels,
    )
    .await;
    let score_value = readback_f32(
      &self.device,
      &self.queue,
      &sv_buf,
      self.meta.num_score_value_channels,
    )
    .await;
    let ownership = readback_f32(
      &self.device,
      &self.queue,
      &own_buf,
      nn_x * nn_y * self.meta.num_ownership_channels,
    )
    .await;

    EvalOutput {
      policy_pass,
      policy_spatial,
      value,
      score_value,
      ownership,
      nn_x,
      nn_y,
      policy_ch: self.meta.num_policy_channels,
      value_ch: self.meta.num_value_channels,
      score_ch: self.meta.num_score_value_channels,
      ownership_ch: self.meta.num_ownership_channels,
    }
  }
}

// ============================================================================
// Model builder helpers
// ============================================================================

fn act_code(act: Activation) -> u32 {
  match act {
    Activation::Identity => 0,
    Activation::Relu => 1,
    Activation::Mish | Activation::MishScale8 => 2,
  }
}

fn build_nac(
  device: &wgpu::Device,
  bn: &crate::model::BatchNormLayerDesc,
  act: Activation,
  conv: &crate::model::ConvLayerDesc,
) -> GpuNormActConv {
  GpuNormActConv {
    bn: GpuBn::new(device, bn, act),
    conv: GpuConv::new(device, conv),
  }
}

fn build_blocks(
  device: &wgpu::Device,
  desc_blocks: &[crate::model::BlockDesc],
) -> Vec<GpuBlock> {
  desc_blocks
    .iter()
    .map(|bd| match bd {
      BlockDesc::Ordinary(d) => GpuBlock::Ordinary(GpuResBlock {
        nac1: build_nac(
          device,
          &d.pre_bn,
          d.pre_activation.activation,
          &d.regular_conv,
        ),
        nac2: build_nac(
          device,
          &d.mid_bn,
          d.mid_activation.activation,
          &d.final_conv,
        ),
      }),
      BlockDesc::GlobalPooling(d) => {
        GpuBlock::GlobalPooling(GpuGpoolResBlock {
          pre_bn: GpuBn::new(device, &d.pre_bn, d.pre_activation.activation),
          regular_conv: GpuConv::new(device, &d.regular_conv),
          gpool_conv: GpuConv::new(device, &d.gpool_conv),
          gpool_bn: GpuBn::new(
            device,
            &d.gpool_bn,
            d.gpool_activation.activation,
          ),
          gpool_to_bias_mul: GpuMatMul::new(device, &d.gpool_to_bias_mul),
          nac2: build_nac(
            device,
            &d.mid_bn,
            d.mid_activation.activation,
            &d.final_conv,
          ),
        })
      }
      BlockDesc::NestedBottleneck(d) => {
        GpuBlock::NestedBottleneck(GpuNestedBottleneck {
          nac1: build_nac(
            device,
            &d.pre_bn,
            d.pre_activation.activation,
            &d.pre_conv,
          ),
          inner: build_blocks(device, &d.blocks),
          nac2: build_nac(
            device,
            &d.post_bn,
            d.post_activation.activation,
            &d.post_conv,
          ),
        })
      }
    })
    .collect()
}

fn build_trunk(
  device: &wgpu::Device,
  desc: &crate::model::TrunkDesc,
) -> GpuTrunk {
  let sgf_meta_encoder =
    desc.sgf_metadata_encoder.as_ref().map(|e| GpuSgfEncoder {
      mul1: GpuMatMul::new(device, &e.mul1),
      bias1: GpuMatBias::new(device, &e.bias1),
      act1: act_code(e.act1.activation),
      mul2: GpuMatMul::new(device, &e.mul2),
      bias2: GpuMatBias::new(device, &e.bias2),
      act2: act_code(e.act2.activation),
      mul3: GpuMatMul::new(device, &e.mul3),
    });
  GpuTrunk {
    initial_conv: GpuConv::new(device, &desc.initial_conv),
    initial_mat_mul: GpuMatMul::new(device, &desc.initial_mat_mul),
    sgf_meta_encoder,
    blocks: build_blocks(device, &desc.blocks),
    trunk_tip_bn: GpuBn::new(
      device,
      &desc.trunk_tip_bn,
      desc.trunk_tip_activation.activation,
    ),
    trunk_c: desc.trunk_num_channels as u32,
  }
}

fn build_policy_head(
  device: &wgpu::Device,
  desc: &crate::model::PolicyHeadDesc,
  model_version: i32,
) -> GpuPolicyHead {
  let p1c = desc.p1_conv.out_channels as u32;
  let g1c = desc.g1_conv.out_channels as u32;
  let p2c = desc.p2_conv.out_channels as u32;
  GpuPolicyHead {
    model_version,
    p1_conv: GpuConv::new(device, &desc.p1_conv),
    g1_conv: GpuConv::new(device, &desc.g1_conv),
    g1_bn: GpuBn::new(device, &desc.g1_bn, desc.g1_activation.activation),
    gpool_to_bias_mul: GpuMatMul::new(device, &desc.gpool_to_bias_mul),
    p1_bn: GpuBn::new(device, &desc.p1_bn, desc.p1_activation.activation),
    p2_conv: GpuConv::new(device, &desc.p2_conv),
    gpool_to_pass_mul: GpuMatMul::new(device, &desc.gpool_to_pass_mul),
    gpool_to_pass_bias: desc
      .gpool_to_pass_bias
      .as_ref()
      .map(|b| GpuMatBias::new(device, b)),
    pass_activation: desc
      .pass_activation
      .as_ref()
      .map(|a| act_code(a.activation))
      .unwrap_or(0),
    gpool_to_pass_mul2: desc
      .gpool_to_pass_mul2
      .as_ref()
      .map(|m| GpuMatMul::new(device, m)),
    p1c,
    g1c,
    p2c,
  }
}

fn build_value_head(
  device: &wgpu::Device,
  desc: &crate::model::ValueHeadDesc,
) -> GpuValueHead {
  GpuValueHead {
    v1c: desc.v1_conv.out_channels as u32,
    v2c: desc.v2_mul.out_channels as u32,
    v3c: desc.v3_mul.out_channels as u32,
    sv3c: desc.sv3_mul.out_channels as u32,
    owc: desc.v_ownership_conv.out_channels as u32,
    v1_conv: GpuConv::new(device, &desc.v1_conv),
    v1_bn: GpuBn::new(device, &desc.v1_bn, desc.v1_activation.activation),
    v2_mul: GpuMatMul::new(device, &desc.v2_mul),
    v2_bias: GpuMatBias::new(device, &desc.v2_bias),
    v2_act: act_code(desc.v2_activation.activation),
    v3_mul: GpuMatMul::new(device, &desc.v3_mul),
    v3_bias: GpuMatBias::new(device, &desc.v3_bias),
    sv3_mul: GpuMatMul::new(device, &desc.sv3_mul),
    sv3_bias: GpuMatBias::new(device, &desc.sv3_bias),
    v_ownership_conv: GpuConv::new(device, &desc.v_ownership_conv),
  }
}

// ============================================================================
// Forward-pass execution context
// ============================================================================

/// Per-call temporary GPU buffers.  Created fresh each `run()` call so that
/// we stay stateless and thread-safe.
struct ForwardCtx<'a> {
  device: &'a wgpu::Device,
  queue: &'a wgpu::Queue,
  pipes: &'a Pipelines,
  nn_x: u32,
  nn_y: u32,
  batch: u32,
}

impl<'a> ForwardCtx<'a> {
  fn hw(&self) -> u32 {
    self.nn_x * self.nn_y
  }

  // ------------------------------------------------------------------
  // Primitive operations
  // ------------------------------------------------------------------

  fn conv(
    &self,
    enc: &mut wgpu::CommandEncoder,
    inp: &wgpu::Buffer,
    gc: &GpuConv,
    out: &wgpu::Buffer,
    accumulate: bool,
  ) {
    let params = ConvParams {
      n: self.batch,
      h: self.nn_y,
      w: self.nn_x,
      ic: gc.ic,
      oc: gc.oc,
      ky: gc.ky,
      kx: gc.kx,
      accumulate: accumulate as u32,
    };
    let pu = upload_uniform(self.device, &params);
    let total = self.batch * self.hw() * gc.oc;
    self.pipes.conv.dispatch(
      enc,
      self.device,
      &[inp, &gc.weight, out, &pu],
      total,
    );
  }

  fn bn_act(
    &self,
    enc: &mut wgpu::CommandEncoder,
    inp: &wgpu::Buffer,
    gbn: &GpuBn,
    mask: &wgpu::Buffer,
    out: &wgpu::Buffer,
  ) {
    let params = BnParams {
      nhw: self.batch * self.hw(),
      nc: gbn.nc,
      act: gbn.act,
      _pad: 0,
    };
    let pu = upload_uniform(self.device, &params);
    let total = self.batch * self.hw() * gbn.nc;
    self.pipes.bn.dispatch(
      enc,
      self.device,
      &[inp, &gbn.scale, &gbn.bias, mask, out, &pu],
      total,
    );
  }

  fn matmul(
    &self,
    enc: &mut wgpu::CommandEncoder,
    inp: &wgpu::Buffer,
    gmm: &GpuMatMul,
    out: &wgpu::Buffer,
  ) {
    let params = MatmulParams {
      ic: gmm.ic,
      oc: gmm.oc,
      batch: self.batch,
      _pad: 0,
    };
    let pu = upload_uniform(self.device, &params);
    let total = gmm.oc * self.batch;
    self.pipes.matmul.dispatch(
      enc,
      self.device,
      &[inp, &gmm.weight, out, &pu],
      total,
    );
  }

  fn matbias(
    &self,
    enc: &mut wgpu::CommandEncoder,
    gmb: &GpuMatBias,
    mat: &wgpu::Buffer,
  ) {
    let params = MatbiasParams {
      nc: gmb.nc,
      batch: self.batch,
      _pad0: 0,
      _pad1: 0,
    };
    let pu = upload_uniform(self.device, &params);
    let total = gmb.nc * self.batch;
    self.pipes.matbias.dispatch(
      enc,
      self.device,
      &[&gmb.weight, mat, &pu],
      total,
    );
  }

  fn add_nc_bias(
    &self,
    enc: &mut wgpu::CommandEncoder,
    bias: &wgpu::Buffer,
    tensor: &wgpu::Buffer,
    nc: u32,
  ) {
    let params = AddNcBiasParams {
      h: self.nn_y,
      w: self.nn_x,
      nc,
      batch: self.batch,
    };
    let pu = upload_uniform(self.device, &params);
    let total = self.batch * self.hw() * nc;
    self.pipes.add_nc_bias.dispatch(
      enc,
      self.device,
      &[bias, tensor, &pu],
      total,
    );
  }

  fn gpool(
    &self,
    enc: &mut wgpu::CommandEncoder,
    in4d: &wgpu::Buffer,
    c_in: u32,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
    out: &wgpu::Buffer,
  ) {
    let params = GpoolParams {
      batch: self.batch,
      h: self.nn_y,
      w: self.nn_x,
      c_in,
    };
    let pu = upload_uniform(self.device, &params);
    let total = self.batch * c_in;
    self.pipes.gpool.dispatch(
      enc,
      self.device,
      &[in4d, mask, msum, out, &pu],
      total,
    );
  }

  fn value_pool(
    &self,
    enc: &mut wgpu::CommandEncoder,
    in4d: &wgpu::Buffer,
    c_in: u32,
    msum: &wgpu::Buffer,
    out: &wgpu::Buffer,
  ) {
    let params = GpoolParams {
      batch: self.batch,
      h: self.nn_y,
      w: self.nn_x,
      c_in,
    };
    let pu = upload_uniform(self.device, &params);
    let total = self.batch * c_in;
    self.pipes.value_pool.dispatch(
      enc,
      self.device,
      &[in4d, msum, out, &pu],
      total,
    );
  }

  fn mask_sum(
    &self,
    enc: &mut wgpu::CommandEncoder,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
  ) {
    let params = MaskSumParams {
      batch: self.batch,
      hw: self.hw(),
      _pad0: 0,
      _pad1: 0,
    };
    let pu = upload_uniform(self.device, &params);
    self.pipes.mask_sum.dispatch(
      enc,
      self.device,
      &[mask, msum, &pu],
      self.batch,
    );
  }

  fn extract_mask(
    &self,
    enc: &mut wgpu::CommandEncoder,
    inp: &wgpu::Buffer,
    ic: u32,
    mask: &wgpu::Buffer,
  ) {
    let params = ConvParams {
      n: self.batch,
      h: self.nn_y,
      w: self.nn_x,
      ic,
      oc: 0,
      ky: 0,
      kx: 0,
      accumulate: 0,
    };
    let pu = upload_uniform(self.device, &params);
    let total = self.batch * self.hw();
    self.pipes.extract_mask.dispatch(
      enc,
      self.device,
      &[inp, mask, &pu],
      total,
    );
  }

  fn activate_inplace(
    &self,
    enc: &mut wgpu::CommandEncoder,
    buf: &wgpu::Buffer,
    n: u32,
    act: u32,
  ) {
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ActParams {
      n: u32,
      act: u32,
      _p0: u32,
      _p1: u32,
    }
    let params = ActParams {
      n,
      act,
      _p0: 0,
      _p1: 0,
    };
    let pu = upload_uniform(self.device, &params);
    self
      .pipes
      .activate_inplace
      .dispatch(enc, self.device, &[buf, &pu], n);
  }

  // ------------------------------------------------------------------
  // Compound ops
  // ------------------------------------------------------------------

  fn nac(
    &self,
    enc: &mut wgpu::CommandEncoder,
    gnac: &GpuNormActConv,
    inp: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    out: &wgpu::Buffer,
    accumulate: bool,
  ) {
    let hw = self.batch * self.hw();
    let scratch =
      alloc_f32(self.device, "nac_scratch", (hw * gnac.bn.nc) as usize);
    self.bn_act(enc, inp, &gnac.bn, mask, &scratch);
    self.conv(enc, &scratch, &gnac.conv, out, accumulate);
  }

  fn apply_res_block(
    &self,
    enc: &mut wgpu::CommandEncoder,
    block: &GpuResBlock,
    trunk: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    tc: u32,
  ) {
    let hw = self.batch * self.hw();
    let mid_c = block.nac1.conv.oc;
    let mid = alloc_f32(self.device, "res_mid", (hw * mid_c) as usize);
    self.nac(enc, &block.nac1, trunk, mask, &mid, false);
    self.nac(enc, &block.nac2, &mid, mask, trunk, true);
    let _ = tc; // trunk channels — residual add is handled by accumulate=true
  }

  fn apply_gpool_block(
    &self,
    enc: &mut wgpu::CommandEncoder,
    block: &GpuGpoolResBlock,
    trunk: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
  ) {
    let hw = self.batch * self.hw();
    let reg_c = block.regular_conv.oc;
    let gpc = block.gpool_conv.oc;

    let trunk_scratch = alloc_f32(
      self.device,
      "gpool_trunk_scratch",
      (hw * block.pre_bn.nc) as usize,
    );
    let reg_out =
      alloc_f32(self.device, "gpool_reg_out", (hw * reg_c) as usize);
    let gp_out = alloc_f32(self.device, "gpool_gp_out", (hw * gpc) as usize);
    let gp_out2 = alloc_f32(self.device, "gpool_gp_out2", (hw * gpc) as usize);
    let gp_concat =
      alloc_f32(self.device, "gpool_concat", (gpc * 3 * self.batch) as usize);
    let gp_bias =
      alloc_f32(self.device, "gpool_bias", (reg_c * self.batch) as usize);

    self.bn_act(enc, trunk, &block.pre_bn, mask, &trunk_scratch);
    self.conv(enc, &trunk_scratch, &block.regular_conv, &reg_out, false);
    self.conv(enc, &trunk_scratch, &block.gpool_conv, &gp_out, false);
    self.bn_act(enc, &gp_out, &block.gpool_bn, mask, &gp_out2);
    self.gpool(enc, &gp_out2, gpc, mask, msum, &gp_concat);
    self.matmul(enc, &gp_concat, &block.gpool_to_bias_mul, &gp_bias);
    self.add_nc_bias(enc, &gp_bias, &reg_out, reg_c);
    self.nac(enc, &block.nac2, &reg_out, mask, trunk, true);
  }

  fn apply_nested_bottleneck(
    &self,
    enc: &mut wgpu::CommandEncoder,
    nb: &GpuNestedBottleneck,
    trunk: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
    tc: u32,
  ) {
    let hw = self.batch * self.hw();
    let mid_c = nb.nac1.conv.oc;
    let mid = alloc_f32(self.device, "nb_mid", (hw * mid_c) as usize);
    self.nac(enc, &nb.nac1, trunk, mask, &mid, false);
    self.apply_blocks(enc, &nb.inner, &mid, mask, msum, mid_c);
    self.nac(enc, &nb.nac2, &mid, mask, trunk, true);
    let _ = tc;
  }

  fn apply_blocks(
    &self,
    enc: &mut wgpu::CommandEncoder,
    blocks: &[GpuBlock],
    trunk: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
    tc: u32,
  ) {
    for blk in blocks {
      match blk {
        GpuBlock::Ordinary(b) => self.apply_res_block(enc, b, trunk, mask, tc),
        GpuBlock::GlobalPooling(b) => {
          self.apply_gpool_block(enc, b, trunk, mask, msum)
        }
        GpuBlock::NestedBottleneck(b) => {
          self.apply_nested_bottleneck(enc, b, trunk, mask, msum, tc)
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // Full trunk / head passes
  // ------------------------------------------------------------------

  fn run_trunk(
    &self,
    enc: &mut wgpu::CommandEncoder,
    inp: &wgpu::Buffer,
    inp_global: &wgpu::Buffer,
    inp_meta: Option<&wgpu::Buffer>,
    trunk: &GpuTrunk,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
  ) -> wgpu::Buffer {
    let tc = trunk.trunk_c;
    let hw = self.batch * self.hw();
    let trunk_buf = alloc_f32(self.device, "trunk_buf", (hw * tc) as usize);
    let mat_out = alloc_f32(self.device, "mat_out", (tc * self.batch) as usize);

    // initial conv + mat-mul
    self.conv(enc, inp, &trunk.initial_conv, &trunk_buf, false);
    self.matmul(enc, inp_global, &trunk.initial_mat_mul, &mat_out);
    self.add_nc_bias(enc, &mat_out, &trunk_buf, tc);

    // optional SGF metadata encoder
    if let (Some(enc_ref), Some(meta_buf)) = (&trunk.sgf_meta_encoder, inp_meta)
    {
      let c1 = enc_ref.mul1.oc;
      let c2 = enc_ref.mul2.oc;
      let buf1 = alloc_f32(self.device, "sgf_buf1", (c1 * self.batch) as usize);
      let buf2 = alloc_f32(self.device, "sgf_buf2", (c2 * self.batch) as usize);
      let meta_out = alloc_f32(
        self.device,
        "sgf_out",
        (enc_ref.mul3.oc * self.batch) as usize,
      );
      self.matmul(enc, meta_buf, &enc_ref.mul1, &buf1);
      self.matbias(enc, &enc_ref.bias1, &buf1);
      self.activate_inplace(enc, &buf1, c1 * self.batch, enc_ref.act1);
      self.matmul(enc, &buf1, &enc_ref.mul2, &buf2);
      self.matbias(enc, &enc_ref.bias2, &buf2);
      self.activate_inplace(enc, &buf2, c2 * self.batch, enc_ref.act2);
      self.matmul(enc, &buf2, &enc_ref.mul3, &meta_out);
      self.add_nc_bias(enc, &meta_out, &trunk_buf, tc);
    }

    // residual block stack
    self.apply_blocks(enc, &trunk.blocks, &trunk_buf, mask, msum, tc);

    // trunk tip BN
    let trunk_out = alloc_f32(self.device, "trunk_out", (hw * tc) as usize);
    self.bn_act(enc, &trunk_buf, &trunk.trunk_tip_bn, mask, &trunk_out);
    trunk_out
  }

  fn run_policy_head(
    &self,
    enc: &mut wgpu::CommandEncoder,
    trunk_out: &wgpu::Buffer,
    ph: &GpuPolicyHead,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
  ) -> (wgpu::Buffer, wgpu::Buffer) {
    let hw = self.batch * self.hw();
    let p1c = ph.p1c;
    let g1c = ph.g1c;
    let p2c = ph.p2c;

    let p1_out = alloc_f32(self.device, "p1_out", (hw * p1c) as usize);
    let g1_out = alloc_f32(self.device, "g1_out", (hw * g1c) as usize);
    let g1_out2 = alloc_f32(self.device, "g1_out2", (hw * g1c) as usize);
    let g1_concat =
      alloc_f32(self.device, "g1_concat", (g1c * 3 * self.batch) as usize);
    let g1_bias =
      alloc_f32(self.device, "g1_bias", (p1c * self.batch) as usize);
    let p1_out2 = alloc_f32(self.device, "p1_out2", (hw * p1c) as usize);
    let policy_spatial =
      alloc_f32(self.device, "policy_spatial", (hw * p2c) as usize);
    let policy_pass =
      alloc_f32(self.device, "policy_pass", (p2c * self.batch) as usize);

    self.conv(enc, trunk_out, &ph.p1_conv, &p1_out, false);
    self.conv(enc, trunk_out, &ph.g1_conv, &g1_out, false);
    self.bn_act(enc, &g1_out, &ph.g1_bn, mask, &g1_out2);
    self.gpool(enc, &g1_out2, g1c, mask, msum, &g1_concat);
    self.matmul(enc, &g1_concat, &ph.gpool_to_bias_mul, &g1_bias);
    self.add_nc_bias(enc, &g1_bias, &p1_out, p1c);
    self.bn_act(enc, &p1_out, &ph.p1_bn, mask, &p1_out2);
    self.conv(enc, &p1_out2, &ph.p2_conv, &policy_spatial, false);

    if ph.model_version >= 15 {
      let pass_tmp =
        alloc_f32(self.device, "pass_tmp", (p1c * self.batch) as usize);
      self.matmul(enc, &g1_concat, &ph.gpool_to_pass_mul, &pass_tmp);
      if let Some(b) = &ph.gpool_to_pass_bias {
        self.matbias(enc, b, &pass_tmp);
      }
      if ph.pass_activation != 0 {
        self.activate_inplace(
          enc,
          &pass_tmp,
          p1c * self.batch,
          ph.pass_activation,
        );
      }
      if let Some(m) = &ph.gpool_to_pass_mul2 {
        self.matmul(enc, &pass_tmp, m, &policy_pass);
      }
    } else {
      self.matmul(enc, &g1_concat, &ph.gpool_to_pass_mul, &policy_pass);
    }

    (policy_pass, policy_spatial)
  }

  fn run_value_head(
    &self,
    enc: &mut wgpu::CommandEncoder,
    trunk_out: &wgpu::Buffer,
    vh: &GpuValueHead,
    mask: &wgpu::Buffer,
    msum: &wgpu::Buffer,
  ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let hw = self.batch * self.hw();
    let v1c = vh.v1c;
    let v2c = vh.v2c;
    let v3c = vh.v3c;
    let sv3c = vh.sv3c;
    let owc = vh.owc;

    let v1_out = alloc_f32(self.device, "v1_out", (hw * v1c) as usize);
    let v1_out2 = alloc_f32(self.device, "v1_out2", (hw * v1c) as usize);
    let v1_mean =
      alloc_f32(self.device, "v1_mean", (v1c * 3 * self.batch) as usize);
    let v2_out = alloc_f32(self.device, "v2_out", (v2c * self.batch) as usize);
    let value = alloc_f32(self.device, "value", (v3c * self.batch) as usize);
    let score_value =
      alloc_f32(self.device, "score_value", (sv3c * self.batch) as usize);
    let ownership = alloc_f32(self.device, "ownership", (hw * owc) as usize);

    self.conv(enc, trunk_out, &vh.v1_conv, &v1_out, false);
    self.bn_act(enc, &v1_out, &vh.v1_bn, mask, &v1_out2);
    self.value_pool(enc, &v1_out2, v1c, msum, &v1_mean);
    self.matmul(enc, &v1_mean, &vh.v2_mul, &v2_out);
    self.matbias(enc, &vh.v2_bias, &v2_out);
    self.activate_inplace(enc, &v2_out, v2c * self.batch, vh.v2_act);
    self.matmul(enc, &v2_out, &vh.v3_mul, &value);
    self.matbias(enc, &vh.v3_bias, &value);
    self.matmul(enc, &v2_out, &vh.sv3_mul, &score_value);
    self.matbias(enc, &vh.sv3_bias, &score_value);
    self.conv(enc, &v1_out2, &vh.v_ownership_conv, &ownership, false);

    (value, score_value, ownership)
  }
}

// ============================================================================
// Uniform buffer helper (bypasses upload_f32 for typed structs)
// ============================================================================

fn upload_uniform<T: bytemuck::Pod>(
  device: &wgpu::Device,
  data: &T,
) -> wgpu::Buffer {
  device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("uniform"),
    contents: bytemuck::bytes_of(data),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
  })
}

// ============================================================================
// Backend impl
// ============================================================================

impl Backend for WgpuBackend {
  fn run<'a>(
    &'a self,
    spatial: &'a [f32],
    global: &'a [f32],
    meta: Option<&'a [f32]>,
    nn_x: usize,
    nn_y: usize,
  ) -> RunFuture<'a> {
    Box::pin(self.run_internal(spatial, global, meta, nn_x, nn_y))
  }

  fn model_version(&self) -> i32 {
    self.meta.model_version
  }
  fn num_input_channels(&self) -> usize {
    self.meta.num_input_channels
  }
  fn num_input_global_channels(&self) -> usize {
    self.meta.num_input_global_channels
  }
  fn num_input_meta_channels(&self) -> usize {
    self.meta.num_input_meta_channels
  }
  fn num_policy_channels(&self) -> usize {
    self.meta.num_policy_channels
  }
  fn num_value_channels(&self) -> usize {
    self.meta.num_value_channels
  }
  fn num_score_value_channels(&self) -> usize {
    self.meta.num_score_value_channels
  }
  fn num_ownership_channels(&self) -> usize {
    self.meta.num_ownership_channels
  }
}
