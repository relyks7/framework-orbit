#include <metal_stdlib>
using namespace metal;
#define T 32
kernel void transpose(
    device const float* A[[buffer(0)]],
    device float* B[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant uint& m[[buffer(3)]],
    uint2 i[[threadgroup_position_in_grid]],
    uint2 j[[thread_position_in_threadgroup]]
){
    threadgroup float tile[T][T+1];
    if ((i.x*T+j.x)<n && (i.y*T+j.y)<m){
        tile[j.y][j.x]=A[(i.x*T+j.x)*m+(i.y*T+j.y)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((i.y*T+j.y)<m && (i.x*T+j.x)<n){
        B[(i.y*T+j.y)*n+(i.x*T+j.x)]=tile[j.x][j.y];
    }
}