import Metal
@_cdecl("gemm_")
public func gemm_(
    streamptr: UnsafeRawPointer,
    Aptr: UnsafeRawPointer,
    Bptr: UnsafeRawPointer,
    Cptr: UnsafeRawPointer,
    m: UInt32,
    n: UInt32,
    p: UInt32,
    b: UInt32
) {
    let stream=Unmanaged<ComputeStream>.fromOpaque(streamptr).takeUnretainedValue();
    let A=Unmanaged<GPUBuffer<Float>>.fromOpaque(Aptr).takeUnretainedValue();
    let B=Unmanaged<GPUBuffer<Float>>.fromOpaque(Bptr).takeUnretainedValue();
    let C=Unmanaged<GPUBuffer<Float>>.fromOpaque(Cptr).takeUnretainedValue();
    gemm(stream: stream, A, B, C, m, n, p, b);
}
