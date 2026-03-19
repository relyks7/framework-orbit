#include <metal_stdlib>
using namespace metal;
kernel void sqrt(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant uint& b[[buffer(3)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        B[i.y*n+i.x]=sqrt(A[i.y*n+i.x]+1e-6);
    }
}