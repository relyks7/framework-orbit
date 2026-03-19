#include <metal_stdlib>
using namespace metal;
kernel void fill(
    device float* A[[buffer(0)]],
    constant float& B[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant uint& b[[buffer(3)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        A[i.y*n+i.x]=B;
    }
}