#include <metal_stdlib>
using namespace metal;
kernel void sub(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& b[[buffer(4)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        C[i.y*n+i.x]=A[i.y*n+i.x]-B[i.y*n+i.x];
    }
}