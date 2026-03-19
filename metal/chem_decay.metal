#include <metal_stdlib>
using namespace metal;
kernel void chem_decay(
    device float* ACh[[buffer(0)]],
    device float* DA[[buffer(1)]],
    device float* NE[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant float& ACh_w[[buffer(4)]],
    constant float& DA_w[[buffer(5)]],
    constant float& NE_w[[buffer(6)]],
    uint i[[thread_position_in_grid]]
){
    if (i<n){
        ACh[i]*=ACh_w;
        DA[i]*=DA_w;
        NE[i]*=NE_w;
    }
}