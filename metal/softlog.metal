#include <metal_stdlib>
using namespace metal;
kernel void softlog(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant float& alpha[[buffer(3)]],
    constant uint& b[[buffer(4)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        float x=A[i.y*n+i.x];
        float ax=fabs(x);
        B[i.y*n+i.x]=x*log(1+alpha*ax)/(1e-20+ax);
    }
}