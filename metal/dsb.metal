#include <metal_stdlib>
using namespace metal;
kernel void dsb(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    device const float* G[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    constant uint& b[[buffer(5)]],
    constant float& l[[buffer(6)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        uint idx=i.y*n+i.x;
        float g = clamp(G[idx], 0.0f, 1.0f);
        C[idx] = max(0.0f, (A[idx]*(1-l) + B[idx])*g);
    }
}