/// KataGo neural network evaluation backed by wgpu compute shaders.
///
/// Data-flow:
///
/// ```text
/// [spatial f32 NHWC] + [global f32]
///        │
///   Trunk (initial conv + initial matmul + N residual blocks + tip BN)
///        │
///   ┌────┴────┐
/// Policy    Value
///  head      head
///   │          │
/// policy    (value, score_value, ownership)
/// ```
///
/// All intermediate tensors are kept in **NCHW** order on the GPU
/// (`[batch, channels, height, width]`), matching the Eigen backend's
/// internal layout.  The host-side helpers accept NHWC input and convert.
pub mod buffers;
pub mod eval;
pub mod layers;

mod shaders {
  pub const CONV: &str = include_str!("shaders/conv.wgsl");
  pub const BN_ACT: &str = include_str!("shaders/bn_act.wgsl");
  pub const GPOOL: &str = include_str!("shaders/gpool.wgsl");
  pub const GPOOL_VALUE_HEAD: &str = include_str!("shaders/gpool_value_head.wgsl");
  pub const MATMUL: &str = include_str!("shaders/matmul.wgsl");
  pub const BIAS_ADD: &str = include_str!("shaders/bias_add.wgsl");
}

use std::sync::Arc;

/// Shared GPU device + queue.  Cheaply clone-able (Arc inside).
#[derive(Clone)]
pub struct GpuContext {
  pub device: Arc<wgpu::Device>,
  pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
  /// Create a `GpuContext` by requesting the default high-performance
  /// adapter.  On WASM the caller must drive the future with
  /// `wasm_bindgen_futures::spawn_local`; on native you can use
  /// `pollster::block_on`.
  pub async fn new() -> Result<Self, String> {
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
      .ok_or_else(|| "no suitable GPU adapter found".to_string())?;

    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          label: Some("katago"),
          required_features: wgpu::Features::empty(),
          required_limits: wgpu::Limits::default(),
          memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
      )
      .await
      .map_err(|e| format!("device creation failed: {e}"))?;

    Ok(Self {
      device: Arc::new(device),
      queue: Arc::new(queue),
    })
  }

  /// Synchronous constructor — only available on non-WASM targets.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn new_sync() -> Result<Self, String> {
    pollster::block_on(Self::new())
  }
}
