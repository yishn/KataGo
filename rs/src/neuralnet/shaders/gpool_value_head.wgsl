// gpool_value_head.wgsl — spatial global-pooling for the value head
//
// Replicates KataGo's `poolRowsValueHead` exactly:
//
//   mean     = sum(x) / maskSum
//   scaled   = mean * (sqrt(maskSum) - 14) * 0.1
//   quad     = mean * ((sqrt(maskSum) - 14)^2 * 0.01 - 0.1)
//
// This differs from the gpool/policy variant (gpool.wgsl) which uses
//   max over valid positions of (x + mask - 1)   for stat2.
//
// Input  shape: NCHW — [N, C, H, W]
// Mask   shape: [N, H, W]   (1.0 for valid cells, 0.0 for padding)
// maskSum shape: [N]
// Output shape: [3*C, N]  — layout (stat*C + c)*N + n
//                            stat 0 = mean, stat 1 = scaled, stat 2 = quad
//
// Each invocation handles one (n, c) pair.

struct Params {
    N : u32,
    C : u32,
    H : u32,
    W : u32,
}

@group(0) @binding(0) var<uniform>             params   : Params;
@group(0) @binding(1) var<storage, read>       input    : array<f32>;  // [N,C,H,W]
@group(0) @binding(2) var<storage, read>       mask_sum : array<f32>;  // [N]
@group(0) @binding(3) var<storage, read_write> output   : array<f32>;  // [3*C, N]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let nc = gid.x;
    if nc >= params.N * params.C { return; }

    let n = nc / params.C;
    let c = nc % params.C;

    let hw = params.H * params.W;
    var sum : f32 = 0.0;

    for (var h : u32 = 0u; h < params.H; h++) {
        for (var w : u32 = 0u; w < params.W; w++) {
            let in_idx = n * params.C * hw + c * hw + h * params.W + w;
            sum += input[in_idx];
        }
    }

    let div     = mask_sum[n];
    let sqrtdiv = sqrt(div);
    let mean    = sum / div;
    let t       = sqrtdiv - 14.0;

    // Output: [3*C, N] — layout (stat*C + c)*N + n
    output[(c + 0u * params.C) * params.N + n] = mean;
    output[(c + 1u * params.C) * params.N + n] = mean * t * 0.1;
    output[(c + 2u * params.C) * params.N + n] = mean * (t * t * 0.01 - 0.1);
}
