#include <metal_stdlib>
using namespace metal;
#define T 128
#define R 5
kernel void conv_r5(
    device const float* A[[buffer(0)]],
    constant float* W[[buffer(1)]],
    device float* B[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& b[[buffer(4)]],
    uint i[[threadgroup_position_in_grid]],
    uint j[[thread_position_in_threadgroup]]
){
    threadgroup float tile[T+2*R];
    uint b_row=i/((n+T-1)/T);
    uint b_col=i%((n+T-1)/T);
    int wl=b_col*T+j-R;
    if (b_row<b && 0<=wl && wl<n){
        tile[j]=A[b_row*n+wl];
    }
    else{
        tile[j]=0.0f;
    }
    if (b_row<b && j<R*2 && 0<=wl+T && wl+T<n){
        tile[j+T]=A[b_row*n+wl+T];
    }
    else{
        tile[j+T]=0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int nwl=b_col*T+j;
    if (b_row<b && 0<=nwl && nwl<n){
        float acc = 0.0f;
        #pragma unroll
        for (int k=-R;k<=R;k++){
            acc+=W[k+R]*tile[j+R+k];
        }
        B[b_row*n+nwl]=acc;
    }
}