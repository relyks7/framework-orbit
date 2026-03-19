#include <metal_stdlib>
using namespace metal;
kernel void write_point(
    device float* A [[buffer(0)]],
    constant float& x [[buffer(1)]],
    constant float& y [[buffer(2)]]
) {
    A[0]=x; 
    A[1]=y;
}