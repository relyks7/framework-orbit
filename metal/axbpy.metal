#include <metal_stdlib>
using namespace metal;
kernel void axbpy(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& b[[buffer(4)]],
    constant float& Y[[buffer(5)]],
    constant float& l[[buffer(6)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        uint idx=i.y*n+i.x;
        C[idx]=A[idx]*(1-Y-l)+B[idx]*Y;
    }
}