// bn_act.wgsl — fused BatchNorm (merged-scale form) + activation + mask
//
// Applies the pre-merged parameters:
//   out = in * merged_scale[c] + merged_bias[c]
// then the requested activation, then masks padded positions to 0.
//
// Activation codes (must match model::Activation):
//   0 = IDENTITY,  1 = RELU,  2 = MISH
//
// Tensor layout: NCHW — flat index n*C*H*W + c*H*W + h*W + w
//
// Bindings:
//   0,0 : Params
//   0,1 : input         [N*C*H*W]   (may alias output for in-place)
//   0,2 : merged_scale  [C]
//   0,3 : merged_bias   [C]
//   0,4 : mask          [N*H*W]     (channel-0 of the input feature map)
//   0,5 : output        [N*C*H*W]

struct Params {
    N          : u32,
    C          : u32,
    H          : u32,
    W          : u32,
    activation : u32,   // 0=identity, 1=relu, 2=mish
}

@group(0) @binding(0) var<uniform>              params       : Params;
@group(0) @binding(1) var<storage, read>        input        : array<f32>;
@group(0) @binding(2) var<storage, read>        merged_scale : array<f32>;
@group(0) @binding(3) var<storage, read>        merged_bias  : array<f32>;
@group(0) @binding(4) var<storage, read>        mask         : array<f32>;
@group(0) @binding(5) var<storage, read_write>  output       : array<f32>;

fn mish(x: f32) -> f32 {
    // softplus(x) = log1p(exp(x)), clamped at 20 for stability
    let x_hi   = min(x, 20.0);
    let x_lo   = x - x_hi;          // = max(x-20, 0) carries the overflow
    let sp     = log(1.0 + exp(x_hi)) + x_lo;
    return x * tanh(sp);
}

fn activate(x: f32, mode: u32) -> f32 {
    switch mode {
        case 1u: { return max(x, 0.0); }          // RELU
        case 2u: { return mish(x); }               // MISH
        default: { return x; }                     // IDENTITY
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let hw   = params.H * params.W;
    let chw  = params.C * hw;
    let idx  = gid.x;             // flat index over N*C*H*W

    if idx >= params.N * chw { return; }

    let n    = idx / chw;
    let rem  = idx % chw;
    let c    = rem / hw;
    let flat = rem % hw;          // h*W + w

    let mask_idx = n * hw + flat;
    let m        = mask[mask_idx];

    var val = input[idx] * merged_scale[c] + merged_bias[c];
    val = activate(val, params.activation);
    val = val * m;                // zero-out padded positions

    output[idx] = val;
}
