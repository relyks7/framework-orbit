import Metal
import Foundation
import Darwin
public func gemm(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ m_: UInt32,
    _ n_: UInt32,
    _ p_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(m_*n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*p_*b_), "B has wrong size")
    precondition(C.count == Int(m_*p_*b_), "C has wrong size")
    var m=m_
    var n=n_
    var p=p_
    var b=b_
    stream.dispatch(
        kernel: "gemm",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&m),
            bytes(&n),
            bytes(&p),
            bytes(&b)
        ],
        grid: MTLSize(
            width: ((Int(p)+63)/64),
            height: ((Int(m)+63)/64)*Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 8,
            height: 64,
            depth: 1
        )
    )
}
public func gemv(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ m_: UInt32,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count==Int(b_*m_*n_))
    precondition(B.count==Int(b_*n_))
    precondition(C.count==Int(b_*m_))
    var m=m_
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "gemv",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&m),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: ((Int(m)+31)/32),
            height: b,
            depth: 1
        ),
        threads: MTLSize(
            width: 32,
            height: 1,
            depth: 1
        )
    )
}
