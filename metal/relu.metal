#include <metal_stdlib>
using namespace metal;
kernel void relu(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant uint& b[[buffer(3)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        B[i.y*n+i.x]=max(0.0f,A[i.y*n+i.x]);
    }
}