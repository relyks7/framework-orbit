#include <metal_stdlib>
using namespace metal;
kernel void layernorm(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    device const float* mu[[buffer(2)]],
    device const float* sigma2[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    constant float& eps[[buffer(5)]],
    constant uint& b[[buffer(6)]],
    uint2 i[[thread_position_in_grid]]
) {
    if (i.x<n && i.y<b){
        B[i.y*n+i.x]=(A[i.y*n+i.x]-mu[i.y])*rsqrt(sigma2[i.y]+eps);
    }
}