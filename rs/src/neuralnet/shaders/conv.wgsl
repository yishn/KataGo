// conv.wgsl — 2-D convolution (NCHW, any kernel size, SAME zero-padding)
//
// Tensors are flat f32 arrays.  Indices use the standard NCHW formula:
//   idx(n, c, h, w) = n*C*H*W + c*H*W + h*W + w
//
// Uniforms (group 0, binding 0):
//   struct Params {
//       N, Cin, Cout : u32   — batch, input channels, output channels
//       H, W          : u32   — spatial dims (same for input and output)
//       KH, KW        : u32   — kernel height / width
//       accumulate    : u32   — if 1, add result to output instead of overwriting
//   }
//
// Bindings:
//   0,1 : Params
//   0,2 : input   [N * Cin * H * W]
//   0,3 : weights [Cout * Cin * KH * KW]  — stored as [oc, ic, ky, kx]
//   0,4 : output  [N * Cout * H * W]

struct Params {
    N          : u32,
    Cin        : u32,
    Cout       : u32,
    H          : u32,
    W          : u32,
    KH         : u32,
    KW         : u32,
    accumulate : u32,
}

@group(0) @binding(0) var<uniform>  params  : Params;
@group(0) @binding(1) var<storage, read>          input   : array<f32>;
@group(0) @binding(2) var<storage, read>          weights : array<f32>;
@group(0) @binding(3) var<storage, read_write>    output  : array<f32>;

// Each invocation computes one output element (n, oc, oh, ow).
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Map (gid.x -> ow*oh linearised, gid.y -> oc, gid.z -> n)
    let hw   = params.H * params.W;
    let flat = gid.x;               // flat spatial index into [H*W]
    let oc   = gid.y;
    let n    = gid.z;

    if flat >= hw || oc >= params.Cout || n >= params.N { return; }

    let oh = flat / params.W;
    let ow = flat % params.W;

    let pad_h = (params.KH - 1u) / 2u;
    let pad_w = (params.KW - 1u) / 2u;

    var sum : f32 = 0.0;
    for (var ic : u32 = 0u; ic < params.Cin; ic++) {
        for (var ky : u32 = 0u; ky < params.KH; ky++) {
            for (var kx : u32 = 0u; kx < params.KW; kx++) {
                // Input position with zero-padding
                let ih_i32 = i32(oh) + i32(ky) - i32(pad_h);
                let iw_i32 = i32(ow) + i32(kx) - i32(pad_w);
                if ih_i32 >= 0 && iw_i32 >= 0 {
                    let ih = u32(ih_i32);
                    let iw = u32(iw_i32);
                    if ih < params.H && iw < params.W {
                        let in_idx  = n  * params.Cin  * hw
                                    + ic * hw
                                    + ih * params.W + iw;
                        // weights layout: [Cout, Cin, KH, KW]
                        let w_idx   = oc * params.Cin * params.KH * params.KW
                                    + ic * params.KH * params.KW
                                    + ky * params.KW + kx;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
    }

    let out_idx = n * params.Cout * hw + oc * hw + oh * params.W + ow;
    if params.accumulate != 0u {
        output[out_idx] += sum;
    } else {
        output[out_idx] = sum;
    }
}
