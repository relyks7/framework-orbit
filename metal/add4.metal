#include <metal_stdlib>
using namespace metal;
kernel void add4(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device const float* C[[buffer(2)]],
    device const float* D[[buffer(3)]],
    device float* E[[buffer(4)]],
    constant uint& n[[buffer(5)]],
    constant uint& b[[buffer(6)]],
    constant float& alpha[[buffer(7)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        uint idx=i.y*n+i.x;
        E[idx]=(A[idx]+B[idx]+C[idx]+D[idx])*alpha;
    }
}