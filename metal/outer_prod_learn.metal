#include <metal_stdlib>
using namespace metal;
kernel void outer_prod_learn(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& b[[buffer(4)]],
    constant float& alpha[[buffer(5)]], 
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        C[i.y*n+i.x]=C[i.y*n+i.x]*0.9999-A[i.x]*B[i.y]*alpha;
    }
}