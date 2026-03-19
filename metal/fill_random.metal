#include <metal_stdlib>
using namespace metal;
inline float randxy(uint x, float minv, float maxv){
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return minv+(maxv-minv)*float(x)*(1.0f / 4294967296.0f);
}
kernel void fill_random(
    device float* A[[buffer(0)]],
    constant uint& n[[buffer(1)]],
    constant uint& b[[buffer(2)]],
    constant float& x[[buffer(3)]],
    constant float& y[[buffer(4)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        uint seed=i.y*747796405 ^ i.x * 2891336453;
        A[i.y*n+i.x]=randxy(seed, x, y);
    }
}