#include <metal_stdlib>
using namespace metal;
kernel void add3(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device const float* C[[buffer(2)]],
    device float* D[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    constant uint& b[[buffer(5)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        uint idx=i.y*n+i.x;
        D[idx]=A[idx]+B[idx]+C[idx];
    }
}