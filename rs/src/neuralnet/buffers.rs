use wgpu::util::DeviceExt;

use super::GpuContext;

// ---------------------------------------------------------------------------
// GpuTensor — a typed 1-D storage buffer on the GPU
// ---------------------------------------------------------------------------

/// A GPU-resident buffer of `f32` values that logically represents a
/// multi-dimensional tensor.  Dims are stored as metadata only; the GPU sees
/// a flat array.
///
/// Layout convention: **NCHW** (`[batch, channels, height, width]`).
/// For 2-D tensors the layout is `[channels, batch]` (columns = batch).
pub struct GpuTensor {
  pub buf: wgpu::Buffer,
  /// Total number of f32 elements.
  pub len: usize,
}

impl GpuTensor {
  /// Allocate an uninitialised read-write storage buffer.
  pub fn zeros(ctx: &GpuContext, len: usize) -> Self {
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: (len * 4) as u64,
      usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    Self { buf, len }
  }

  /// Upload a host slice to a new GPU buffer.
  pub fn from_slice(ctx: &GpuContext, data: &[f32]) -> Self {
    let buf =
      ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
          label: None,
          contents: bytemuck::cast_slice(data),
          usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        });
    Self {
      buf,
      len: data.len(),
    }
  }

  /// Upload new data into an existing buffer (must be same length).
  pub fn upload(&self, ctx: &GpuContext, data: &[f32]) {
    assert_eq!(data.len(), self.len);
    ctx
      .queue
      .write_buffer(&self.buf, 0, bytemuck::cast_slice(data));
  }

  /// Read back to host (blocking — submits + polls).
  ///
  /// Not available on WASM; there you must use the async version.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn download(&self, ctx: &GpuContext) -> Vec<f32> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("staging"),
      size: (self.len * 4) as u64,
      usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });

    let mut enc = ctx
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&self.buf, 0, &staging, 0, (self.len * 4) as u64);
    ctx.queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);

    let view = slice.get_mapped_range();
    bytemuck::cast_slice::<u8, f32>(&view).to_vec()
  }
}

// ---------------------------------------------------------------------------
// WeightBuffer — read-only storage buffer for layer weights
// ---------------------------------------------------------------------------

/// A read-only GPU buffer for layer weights / biases.
pub struct WeightBuffer {
  pub buf: wgpu::Buffer,
  pub len: usize,
}

impl WeightBuffer {
  pub fn new(ctx: &GpuContext, data: &[f32]) -> Self {
    let buf =
      ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
          label: None,
          contents: bytemuck::cast_slice(data),
          usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
    Self {
      buf,
      len: data.len(),
    }
  }
}

// ---------------------------------------------------------------------------
// Conversion helpers: NHWC (host) ↔ NCHW (GPU)
// ---------------------------------------------------------------------------

/// Reorder a batch of spatial tensors from NHWC to NCHW.
///
/// Input shape:  `[n, h, w, c]`  — standard KataGo host layout
/// Output shape: `[n, c, h, w]`  — GPU layout used by all shaders
pub fn nhwc_to_nchw(
  src: &[f32],
  n: usize,
  h: usize,
  w: usize,
  c: usize,
) -> Vec<f32> {
  let mut dst = vec![0f32; n * c * h * w];
  for ni in 0..n {
    for hi in 0..h {
      for wi in 0..w {
        for ci in 0..c {
          let src_idx = ni * (h * w * c) + hi * (w * c) + wi * c + ci;
          let dst_idx = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
  dst
}

/// Reorder a batch of spatial tensors from NCHW back to NHWC.
pub fn nchw_to_nhwc(
  src: &[f32],
  n: usize,
  c: usize,
  h: usize,
  w: usize,
) -> Vec<f32> {
  let mut dst = vec![0f32; n * h * w * c];
  for ni in 0..n {
    for ci in 0..c {
      for hi in 0..h {
        for wi in 0..w {
          let src_idx = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
          let dst_idx = ni * (h * w * c) + hi * (w * c) + wi * c + ci;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
  dst
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  // Pure conversion tests — no GPU required.

  #[test]
  fn nhwc_to_nchw_single_batch() {
    // [1, 2, 2, 2] NHWC:  pixel(h,w,c) arranged as rows
    //   h=0,w=0: c0=1, c1=2
    //   h=0,w=1: c0=3, c1=4
    //   h=1,w=0: c0=5, c1=6
    //   h=1,w=1: c0=7, c1=8
    let nhwc: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    let nchw = nhwc_to_nchw(&nhwc, 1, 2, 2, 2);
    // NCHW: channel-0 plane first, then channel-1
    //   c=0: [1, 3, 5, 7]  (scan over h,w)
    //   c=1: [2, 4, 6, 8]
    assert_eq!(nchw, vec![1., 3., 5., 7., 2., 4., 6., 8.]);
  }

  #[test]
  fn nchw_to_nhwc_single_batch() {
    let nchw: Vec<f32> = vec![1., 3., 5., 7., 2., 4., 6., 8.];
    let nhwc = nchw_to_nhwc(&nchw, 1, 2, 2, 2);
    assert_eq!(nhwc, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
  }

  #[test]
  fn nhwc_nchw_round_trip() {
    let n = 2usize;
    let (h, w, c) = (3, 4, 5);
    let src: Vec<f32> = (0..(n * h * w * c)).map(|i| i as f32).collect();
    let converted = nhwc_to_nchw(&src, n, h, w, c);
    let back = nchw_to_nhwc(&converted, n, c, h, w);
    assert_eq!(src, back);
  }

  // GPU round-trip tests — require a real GPU device.

  #[cfg(not(target_arch = "wasm32"))]
  #[test]
  fn gpu_upload_download_round_trip() {
    let ctx = match crate::neuralnet::GpuContext::new_sync() {
      Ok(c) => c,
      Err(_) => return, // skip if no GPU
    };
    let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    let tensor = GpuTensor::from_slice(&ctx, &data);
    let back = tensor.download(&ctx);
    assert_eq!(data, back);
  }

  #[cfg(not(target_arch = "wasm32"))]
  #[test]
  fn gpu_upload_then_overwrite() {
    let ctx = match crate::neuralnet::GpuContext::new_sync() {
      Ok(c) => c,
      Err(_) => return,
    };
    let original = vec![1f32, 2., 3., 4.];
    let tensor = GpuTensor::from_slice(&ctx, &original);
    let new_data = vec![9f32, 8., 7., 6.];
    tensor.upload(&ctx, &new_data);
    assert_eq!(tensor.download(&ctx), new_data);
  }
}
