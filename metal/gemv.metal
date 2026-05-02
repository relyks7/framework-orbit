#include <metal_stdlib>
using namespace metal;
kernel void gemv(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& m[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    uint i[[thread_position_in_grid]]
){
    if (i<m){
        float acc=0;
        #pragma unroll 4
        for (uint k=0;k<n;k++){
            acc+=A[i*n+k]*B[k];
        }
        C[i]=acc;
    }
}