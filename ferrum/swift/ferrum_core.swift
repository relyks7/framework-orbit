import Metal
import Foundation
import Darwin

//<start AI_WRITTEN>

// ============================================================
// 1. SMALL HELPERS
// ============================================================

@inline(__always)
func bytes<T>(_ value: inout T) -> KernelArg {
    withUnsafeBytes(of: &value) {
        KernelArg.bytes($0.baseAddress!, $0.count)
    }
}

@inline(__always)
func load(_ src: [Float], into dst: GPUBuffer<Float>) {
    precondition(src.count <= dst.capacity)
    dst.ptr().update(from: src, count: src.count)
}

// ============================================================
// 2. METAL CONTEXT
// ============================================================

public final class MetalContext {

    public let device: MTLDevice
    public let libraries: [MTLLibrary]
    public let pipelines: [String: MTLComputePipelineState]

    public init(kernelsDirectory: URL) {

        device = MTLCreateSystemDefaultDevice()!

        let urls = (try? FileManager.default.contentsOfDirectory(
            at: kernelsDirectory,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension == "metallib" } ?? []

        precondition(!urls.isEmpty)

        var libs: [MTLLibrary] = []
        for u in urls {
            libs.append(try! device.makeLibrary(URL: u))
        }
        libraries = libs

        var pipes: [String: MTLComputePipelineState] = [:]

        for lib in libs {
            for name in lib.functionNames where pipes[name] == nil {
                let fn = lib.makeFunction(name: name)!
                pipes[name] = try! device.makeComputePipelineState(function: fn)
            }
        }

        precondition(!pipes.isEmpty)
        pipelines = pipes
    }
}

// ============================================================
// 3. GPU BUFFER
// ============================================================

public final class GPUBuffer<T> {

    public let buffer: MTLBuffer
    public let capacity: Int
    public var count: Int

    public init(device: MTLDevice, capacity: Int) {

        self.capacity = capacity
        self.count = capacity

        buffer = device.makeBuffer(
            length: capacity * MemoryLayout<T>.stride,
            options: .storageModeShared
        )!
    }

    @inline(__always)
    public func ptr() -> UnsafeMutablePointer<T> {
        buffer.contents().assumingMemoryBound(to: T.self)
    }
}

// ============================================================
// 4. KERNEL ARG
// ============================================================

public enum KernelArg {
    case buffer(MTLBuffer)
    case bytes(UnsafeRawPointer, Int)
}

// ============================================================
// 5. COMPUTE STREAM
// ============================================================

public final class ComputeStream {

    private let ctx: MetalContext
    private let queue: MTLCommandQueue

    private let inflight: Int

    private var cmdRing: [MTLCommandBuffer]
    private var encRing: [MTLComputeCommandEncoder]
    private var committed: [Bool]

    private var index: Int = 0

    // fast kernel cache
    private var kernelCache: [String: MTLComputePipelineState] = [:]

    // --------------------------------------------------------

    public init(context: MetalContext, inflightBuffers: Int = 3) {

        ctx = context
        queue = context.device.makeCommandQueue()!
        inflight = inflightBuffers

        cmdRing = []
        encRing = []
        committed = Array(repeating: false, count: inflight)

        for _ in 0..<inflight {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            cmdRing.append(cmd)
            encRing.append(enc)
        }
    }

    // --------------------------------------------------------
    // HOT PATH
    // --------------------------------------------------------

    @inline(__always)
    public func dispatch(
        kernel: String,
        args: [KernelArg],
        grid: MTLSize,
        threads: MTLSize
    ) {

        let enc = encRing[index]

        // cached lookup
        let pipe = kernelCache[kernel] ?? {
            let p = ctx.pipelines[kernel]!
            kernelCache[kernel] = p
            return p
        }()

        enc.setComputePipelineState(pipe)

        for (i, arg) in args.enumerated() {

            switch arg {

            case .buffer(let b):
                enc.setBuffer(b, offset: 0, index: i)

            case .bytes(let ptr, let size):
                enc.setBytes(ptr, length: size, index: i)
            }
        }

        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
    }

    // --------------------------------------------------------

    @inline(__always)
    public func advance() {

        let cmd = cmdRing[index]
        let enc = encRing[index]

        enc.endEncoding()
        cmd.commit()
        committed[index] = true

        index = (index + 1) % inflight

        if committed[index] {
            cmdRing[index].waitUntilCompleted()
            committed[index] = false
        }

        let nextCmd = queue.makeCommandBuffer()!
        let nextEnc = nextCmd.makeComputeCommandEncoder()!

        cmdRing[index] = nextCmd
        encRing[index] = nextEnc
    }

    // --------------------------------------------------------

    public func synchronize() {

        for i in 0..<inflight where committed[i] {
            cmdRing[i].waitUntilCompleted()
            committed[i] = false
        }
    }
}
//<end AI_WRITTEN>