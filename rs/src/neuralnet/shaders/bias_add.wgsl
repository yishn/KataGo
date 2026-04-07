// bias_add.wgsl — two operations in one shader, selected by `mode`:
//
//  mode 0 — MatBias (per-channel vector bias):
//              output[c, b] += bias[c]
//              Tensor layout [C, B] flat as output[c * B + b]
//
//  mode 1 — NC-broadcast bias add (add a [C, B] bias to a [C, H, W, B] tensor):
//              For each spatial position (h, w):
//                output[n, c, h, w] += bias[c, n]
//              Used to inject gpool-derived biases into the spatial feature map.
//              Spatial tensor: NCHW — flat [N*C*H*W]
//              Bias tensor:    [C, N] flat — bias[c * N + n]

struct Params {
    mode : u32,
    // mode 0: mat-bias
    C    : u32,
    B    : u32,
    // mode 1: spatial broadcast
    N    : u32,
    // C reused above
    H    : u32,
    W    : u32,
}

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read>       bias   : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;

    if params.mode == 0u {
        // MatBias: output has shape [C, B]
        let total = params.C * params.B;
        if idx >= total { return; }
        let c = idx / params.B;
        output[idx] += bias[c];

    } else {
        // NC-broadcast: output has shape [N, C, H, W]
        let hw    = params.H * params.W;
        let chw   = params.C * hw;
        let total = params.N * chw;
        if idx >= total { return; }
        let n     = idx / chw;
        let rem   = idx % chw;
        let c     = rem / hw;
        // bias shape [C, N], stored as bias[c * N + n]
        output[idx] += bias[c * params.N + n];
    }
}
