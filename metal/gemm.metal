#include <metal_stdlib>
using namespace metal;
#define T 64
#define WIDTH 8
kernel void gemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    constant uint& b [[buffer(6)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]],
    uint si [[simdgroup_index_in_threadgroup]]
)
{
    threadgroup half tA[T][T];
    threadgroup half tB[T][T];
    simdgroup_float8x8 acc1; 
    simdgroup_float8x8 acc2; 
    simdgroup_float8x8 acc3; 
    simdgroup_float8x8 acc4; 
    simdgroup_half8x8 matA1;
    simdgroup_half8x8 matA2;
    simdgroup_half8x8 matA3;
    simdgroup_half8x8 matA4;
    simdgroup_half8x8 matB1;
    simdgroup_half8x8 matB2;
    simdgroup_half8x8 matB3;
    simdgroup_half8x8 matB4;
    acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    int diff=4;
    ushort2 offset = ushort2((si % diff) * WIDTH * 2, (si/ diff) * WIDTH * 2);
    uint blocks_per_batch = (m + T - 1) / T;
    uint layer = j.y / blocks_per_batch;
    uint block_row = j.y % blocks_per_batch;
    uint block_col = j.x;

    uint row = block_row * T + i.y;
    uint col = block_col * T + i.x*8;
    if (layer>=b) return;
    unsigned long offsetA = (unsigned long)layer * m * n;
    unsigned long offsetB = (unsigned long)layer * n * p;
    unsigned long offsetC = (unsigned long)layer * m * p;
    for (int curtile=0;curtile<(n+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x*8)+7 < n){
            *(threadgroup half4*)(&tA[i.y][i.x*8]) = half4(*(device const float4*)(&A[offsetA + row * n + (curtile * T + i.x * 8)]));
            *(threadgroup half4*)(&tA[i.y][i.x*8+4]) = half4(*(device const float4*)(&A[offsetA + row * n + (curtile * T + i.x * 8+4)]));
        }
        else{
            *(threadgroup half4*)(&tA[i.y][i.x*8]) = half4(0.0f);
            *(threadgroup half4*)(&tA[i.y][i.x*8+4]) = half4(0.0f);
        }
        if ((curtile*T + i.y) < n && col+7 < p){
            *(threadgroup half4*)(&tB[i.y][i.x*8]) = half4(*(device const float4*)(&B[offsetB+(curtile*T + i.y)*p + col]));
            *(threadgroup half4*)(&tB[i.y][i.x*8+4]) = half4(*(device const float4*)(&B[offsetB+(curtile*T + i.y)*p + col+4]));
        }
        else{
            *(threadgroup half4*)(&tB[i.y][i.x*8]) = half4(0.0f);
            *(threadgroup half4*)(&tB[i.y][i.x*8+4]) = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int k=0;k<T;k+=WIDTH*2){
            simdgroup_load(matA1, (threadgroup half*)&tA[0][0], T, ulong2(k, offset.y));
            simdgroup_load(matB1, (threadgroup half*)&tB[0][0], T, ulong2(offset.x, k));
            simdgroup_load(matA2, (threadgroup half*)&tA[0][0], T, ulong2(k+8, offset.y));
            simdgroup_load(matB2, (threadgroup half*)&tB[0][0], T, ulong2(offset.x+8, k));
            simdgroup_load(matA3, (threadgroup half*)&tA[0][0], T, ulong2(k, offset.y+8));
            simdgroup_load(matB3, (threadgroup half*)&tB[0][0], T, ulong2(offset.x, k+8));
            simdgroup_load(matA4, (threadgroup half*)&tA[0][0], T, ulong2(k+8, offset.y+8));
            simdgroup_load(matB4, (threadgroup half*)&tB[0][0], T, ulong2(offset.x+8, k+8));
            simdgroup_multiply_accumulate(acc1, matA1, matB1, acc1);
            simdgroup_multiply_accumulate(acc1, matA2, matB3, acc1);

            simdgroup_multiply_accumulate(acc2, matA1, matB2, acc2);
            simdgroup_multiply_accumulate(acc2, matA2, matB4, acc2);
            simdgroup_multiply_accumulate(acc3, matA3, matB1, acc3);
            simdgroup_multiply_accumulate(acc3, matA4, matB3, acc3);

            simdgroup_multiply_accumulate(acc4, matA3, matB2, acc4);
            simdgroup_multiply_accumulate(acc4, matA4, matB4, acc4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row0 = block_row * T + offset.y;
    uint col0 = block_col * T + offset.x;
    uint row1 = block_row * T + offset.y + 8;
    uint col1 = block_col * T + offset.x + 8;
    if (row0<m && col0<p){
        simdgroup_store(acc1, C+offsetC, p, ulong2(col0, row0));
    }
    if (row0<m && col1<p){
        simdgroup_store(acc2, C+offsetC, p, ulong2(col1, row0));
    }
    if (row1<m && col0<p){
        simdgroup_store(acc3, C+offsetC, p, ulong2(col0, row1));
    }
    if (row1<m && col1<p){
        simdgroup_store(acc4, C+offsetC, p, ulong2(col1, row1));
    }
}