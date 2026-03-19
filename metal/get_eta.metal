#include <metal_stdlib>
using namespace metal;
kernel void get_eta(
    device const float* eligmean[[buffer(0)]],
    device float* eta[[buffer(1)]],
    constant float& eta_max[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant float& DA_c [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    uint i[[thread_position_in_grid]]
){
    if (i<n){
        eta[i]=eta_max*tanh(alpha*eligmean[i])*DA_c;
    }
}