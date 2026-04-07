// matmul.wgsl — batched matrix multiply  C = A * B
//
// A : [K, N]   (weights, read-only)  — K = in_channels, N = out_channels in
//                                       the transposed form used by KataGo:
//                                       weights are stored [out, in], so A is
//                                       weights^T from the caller's view.
// B : [K, B]   (input features, read-only)   — K rows, batch_size columns
// C : [N, B]   (output, read-write)
//
// We map gid.x → output column (batch index), gid.y → output row (out channel).
// Each thread computes one element of C by dotting a row of A with a column of B.
//
// Stored flat:
//   A[k, n]  = weights[n * K + k]   (row = out_ch, col = in_ch)
//   B[k, b]  = input[b * K + k]     (col-major: channel-first)
//   C[n, b]  = output[b * N + n]

struct Params {
    K : u32,   // in_channels
    N : u32,   // out_channels
    B : u32,   // batch_size
}

@group(0) @binding(0) var<uniform>             params  : Params;
@group(0) @binding(1) var<storage, read>       weights : array<f32>;  // [N, K]
@group(0) @binding(2) var<storage, read>       input   : array<f32>;  // [K, B]
@group(0) @binding(3) var<storage, read_write> output  : array<f32>;  // [N, B]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let b = gid.x;   // batch index
    let n = gid.y;   // out channel index

    if b >= params.B || n >= params.N { return; }

    var sum : f32 = 0.0;
    for (var k : u32 = 0u; k < params.K; k++) {
        // weights stored [out_ch, in_ch]: weights[n * K + k]
        // input  stored [in_ch, batch]:  input[k * B + b]
        sum += weights[n * params.K + k] * input[k * params.B + b];
    }

    output[n * params.B + b] = sum;
}
