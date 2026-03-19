#include <metal_stdlib>
using namespace metal;
kernel void optra(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant uint& b[[buffer(3)]],
    constant float& alpha[[buffer(4)]],
    constant float& beta[[buffer(5)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        int idx=i.y*n+i.x;
        B[idx]=1+beta*tanh(alpha*A[idx]);
    }
}