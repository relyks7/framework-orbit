#include <metal_stdlib>
using namespace metal;
kernel void scale(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    constant float& alpha[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& b[[buffer(4)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        B[i.y*n+i.x]=A[i.y*n+i.x]*alpha;
    }
}