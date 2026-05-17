import Metal
@_cdecl("create_context_")
public func create_context_(dirptr: UnsafePointer<CChar>) -> UnsafeMutableRawPointer {
    let path=String(cString: dirptr)
    let url=URL(fileURLWithPath: path)
    let ctx=MetalContext(kernelsDirectory: url)
    return Unmanaged.passRetained(ctx).toOpaque()
}
@_cdecl("gpu_buffer_f_")
public func gpu_buffer_f_(ctxeptr: UnsafeRawPointer, capacityptr: Int32) -> UnsafeMutablePointer{
    let ctx=Unmanaged<MetalContext>.fromOpaque(ctwptr).takeUnretainedValue()
    let buff=GPUBuffer<Float>(device: ctx.device, capacity: Int(capacity))
    return Unmanaged.passRetained(buff).toOpaque()
}
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
@_cdecl("gemv_")
public func gemv_(
    streamptr: UnsafeRawPointer,
    Aptr: UnsafeRawPointer,
    Bptr: UnsafeRawPointer,
    Cptr: UnsafeRawPointer,
    m: UInt32,
    n: UInt32,
    b: UInt32
) {
    let stream=Unmanaged<ComputeStream>.fromOpaque(streamptr).takeUnretainedValue();
    let A=Unmanaged<GPUBuffer<Float>>.fromOpaque(Aptr).takeUnretainedValue();
    let B=Unmanaged<GPUBuffer<Float>>.fromOpaque(Bptr).takeUnretainedValue();
    let C=Unmanaged<GPUBuffer<Float>>.fromOpaque(Cptr).takeUnretainedValue();
    gemv(stream: stream, A, B, C, m, n, b);
}