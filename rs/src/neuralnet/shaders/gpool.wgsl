// gpool.wgsl — spatial global-pooling producing 3 statistics per channel
//
// Replicates KataGo's `poolRowsGPool` exactly:
//
//   mean     = sum(x) / maskSum
//   scaled   = mean * (sqrt(maskSum) - 14) * 0.1
//   max_val  = max over valid positions of (x + mask - 1)
//
// Input  shape: NCHW — [N, C, H, W]
// Mask   shape: [N, H, W]   (1.0 for valid cells, 0.0 for padding)
// maskSum shape: [N]
// Output shape: [3*C, N]  — layout [stat*C + c, n] = (stat*C+c)*N + n
//                            stat 0 = mean, stat 1 = scaled, stat 2 = max
//
// This [K=3*C, N] layout matches what the matmul shader expects as input.
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
@group(0) @binding(2) var<storage, read>       mask     : array<f32>;  // [N,H,W]
@group(0) @binding(3) var<storage, read>       mask_sum : array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> output   : array<f32>;  // [N, 3*C]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let nc = gid.x;   // flat index over N*C
    if nc >= params.N * params.C { return; }

    let n  = nc / params.C;
    let c  = nc % params.C;

    let hw  = params.H * params.W;
    var sum : f32 = 0.0;
    var mx  : f32 = -1.0;

    for (var h : u32 = 0u; h < params.H; h++) {
        for (var w : u32 = 0u; w < params.W; w++) {
            let in_idx   = n * params.C * hw + c * hw + h * params.W + w;
            let mask_idx = n * hw + h * params.W + w;
            let x = input[in_idx];
            let m = mask[mask_idx];
            sum += x;
            mx   = max(mx, x + (m - 1.0));
        }
    }

    let div     = mask_sum[n];
    let sqrtdiv = sqrt(div);
    let mean    = sum / div;

    // Output: [3*C, N] — layout (stat*C + c)*N + n  →  matches [K, batch] for matmul
    output[(c + 0u * params.C) * params.N + n] = mean;
    output[(c + 1u * params.C) * params.N + n] = mean * (sqrtdiv - 14.0) * 0.1;
    output[(c + 2u * params.C) * params.N + n] = mx;
}
