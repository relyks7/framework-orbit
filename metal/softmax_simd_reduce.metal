#include <metal_stdlib>
using namespace metal;
#define T 128
#define LANES 32
#define WARPS T/(LANES)
kernel void softmax_simd_reduce(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device const float* global_max [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& b [[buffer(4)]],
    uint2 j [[thread_position_in_grid]],
    uint2 k [[threadgroup_position_in_grid]],
    uint si [[thread_index_in_simdgroup]],
    uint sj [[simdgroup_index_in_threadgroup]]
) {
    if (j.y>=b) return;
    float val=(j.x<n)?exp(A[j.y*n+j.x]-global_max[j.y]):0.0f;
    float local_sum=simd_sum(val);
    threadgroup float ps[WARPS];
    if (si==0){
        ps[sj]=local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint out_width = (n + T - 1) / T;
    if (sj==0){
        float xs = 0.0f;
        if (si < WARPS) {
            xs = ps[si]; 
        }
        float final_sum=simd_sum(xs);
        if (si == 0 && k.x < out_width) {
            B[j.y * out_width + k.x] = final_sum;
        }
    }
}