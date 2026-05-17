#include <metal_stdlib>
using namespace metal;
#define T 32
kernel void gemv(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& m[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    constant uint& b[[buffer(5)]],
    uint2 i[[threadgroup_position_in_grid]],
    uint j[[thread_position_in_threadgroup]]
){
    uint row=i.x*T+j;
    if (i.y<b && row<m){
        float acc=0;
        #pragma unroll 4
        for (uint k=0;k<n;k++){
            acc+=A[i.y*m*n+row*n+k]*B[i.y*n+k];
        }
        C[i.y*m+row]=acc;
    }
}