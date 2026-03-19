#include <metal_stdlib>
using namespace metal;
kernel void final_oja_step(
    device const float* G1[[buffer(0)]],
    device const float* G2[[buffer(1)]],
    device const float* eta[[buffer(2)]],
    device const float* A[[buffer(3)]],
    device float* A_new[[buffer(4)]],
    constant uint& n[[buffer(5)]],
    constant uint& r[[buffer(6)]],
    constant float& lambda[[buffer(7)]],
    constant float& DA_c[[buffer(8)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<r && i.y<n){
        uint idx=i.y*r+i.x;
        float A_d=(eta[i.y]*(G1[idx] - G2[idx]))-lambda*A[idx];
        A_new[idx]=A[idx]+DA_c*A_d;
    }
}