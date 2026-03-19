#include <metal_stdlib>
using namespace metal;
kernel void softmax(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    device const float* global_max[[buffer(2)]],
    device const float* denom[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    constant uint& b[[buffer(5)]],
    uint2 i[[thread_position_in_grid]]
) {
    if (i.x<n && i.y<b){
        B[i.y*n+i.x]=exp(A[i.y*n+i.x]-global_max[i.y])/denom[i.y];
    }
}