#include <metal_stdlib>
using namespace metal;
kernel void bg_oja(
    device const float* g[[buffer(0)]],
    device float* W[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant uint& b[[buffer(3)]],
    constant float& eta[[buffer(4)]],
    constant float& w_max[[buffer(5)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.y<b && i.x<n){
        uint idx=i.y*n+i.x;
        W[idx]-=eta*g[i.y]*g[i.y]*W[idx];
        W[idx] = clamp(W[idx], -w_max, w_max);
    }
}