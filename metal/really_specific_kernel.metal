#include <metal_stdlib>
using namespace metal;
kernel void really_specific_kernel(
    device float* G[[buffer(0)]],
    device const float* M[[buffer(1)]],
    device const float* ST[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant float& lambda[[buffer(4)]],
    constant float& DA_c[[buffer(5)]],
    constant float& k_low[[buffer(6)]],
    constant float& k_high[[buffer(7)]],
    uint i[[thread_position_in_grid]]
){
    if (i<n){
        float z=(G[i]-M[0])/(ST[0] + 1e-6f);
        z = clamp(z, -3.0f, 3.0f);
        float k = mix(k_low, k_high, clamp(DA_c, 0.0f, 1.0f));
        G[i] = exp(k * z);
    }
}