#include <metal_stdlib>
using namespace metal;
kernel void sigmoid(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant uint& b[[buffer(3)]],
    constant float& eta0[[buffer(4)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        B[i.y*n+i.x]=eta0/(1+exp(-A[i.y*n+i.x]));
    }
}