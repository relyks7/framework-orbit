#include <metal_stdlib>
using namespace metal;
kernel void zero(
    device float* A[[buffer(0)]],
    constant uint& n[[buffer(1)]],
    constant uint& b[[buffer(2)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        A[i.y*n+i.x]=0;
    }
}