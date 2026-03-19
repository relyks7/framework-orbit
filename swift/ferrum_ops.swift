import Metal
import Foundation
import Darwin
public func optra(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ alpha_: Float,
    _ beta_: Float,
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    var alpha=alpha_
    var beta=beta_
    stream.dispatch(
        kernel: "optra",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&alpha),
            bytes(&beta)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func add(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "add",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func add_scaled(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ sa_: Float,
    _ sb_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    var sa=sa_
    var sb=sb_
    stream.dispatch(
        kernel: "add_scaled",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&sa),
            bytes(&sb)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func add3(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ D: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    precondition(D.count == Int(n_*b_), "D has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "add3",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .buffer(D.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func add4(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ D: GPUBuffer <Float>,
    _ E: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ alpha_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    precondition(D.count == Int(n_*b_), "D has wrong size")
    precondition(E.count == Int(n_*b_), "E has wrong size")
    var n=n_
    var b=b_
    var alpha=alpha_
    stream.dispatch(
        kernel: "add4",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .buffer(D.buffer),
            .buffer(E.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&alpha)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func axbpy(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ Y_: Float,
    _ l_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    var Y=Y_
    var l=l_
    stream.dispatch(
        kernel: "axbpy",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&Y),
            bytes(&l)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
//dynamic system base
public func dsb(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ G: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ l_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    precondition(G.count == Int(n_*b_), "G has wrong size")
    var n=n_
    var b=b_
    var l=l_
    stream.dispatch(
        kernel: "dsb",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .buffer(G.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&l)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func copy(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "copy",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func scale(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ alpha_: Float,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    var alpha=alpha_
    stream.dispatch(
        kernel: "scale",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&alpha),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func sqrt(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "sqrt",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func sigmoid(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ e0_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    var e0=e0_
    stream.dispatch(
        kernel: "sigmoid",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&e0)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func zero(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "zero",
        args: [
            .buffer(A.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func transpose(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ m_: UInt32
){
    precondition(A.count == Int(n_*m_), "A has wrong size")
    precondition(B.count == Int(n_*m_), "B has wrong size")
    var n=n_
    var m=m_
    stream.dispatch(
        kernel: "transpose",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&m)
        ],
        grid: MTLSize(
            width: (Int(n) + 31) / 32,
            height: (Int(m) + 31) / 32,
            depth: 1
        ),
        threads: MTLSize(
            width: 32,
            height: 32,
            depth: 1
        )
    )
}
public func div(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "div",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func mul(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "mul",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func sub(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "sub",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func embedding(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <UInt32>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ d_: UInt32,
    _ vocab_size_: UInt32
){
    precondition(A.count == Int(vocab_size_*d_), "A has wrong size")
    precondition(B.count == Int(n_), "B has wrong size")
    precondition(C.count == Int(n_*d_), "C has wrong size")
    var n=n_
    var d=d_
    var vocab_size=vocab_size_
    stream.dispatch(
        kernel: "embedding",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&d),
            bytes(&vocab_size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
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
            width: ((Int(p)+31)/32),
            height: ((Int(m)+31)/32)*Int(b),
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
){
    precondition(A.count==Int(m_*n_))
    precondition(B.count==Int(n_))
    precondition(C.count==Int(m_))
    var m=m_
    var n=n_
    stream.dispatch(
        kernel: "gemv",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&m),
            bytes(&n)
        ],
        grid: MTLSize(
            width: ((Int(m)+255)/256),
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func layernorm(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ mu: GPUBuffer <Float>,
    _ sigma2: GPUBuffer <Float>,
    _ n_: UInt32,
    _ eps_: Float,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(mu.count == Int(b_), "mu has wrong size")
    precondition(sigma2.count == Int(b_), "sigma2 has wrong size")
    var n=n_
    var eps=eps_
    var b=b_
    stream.dispatch(
        kernel: "layernorm",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(mu.buffer),
            .buffer(sigma2.buffer),
            bytes(&n),
            bytes(&eps),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func tanh(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "tanh",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func softlog(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ alpha_: Float,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var alpha=alpha_
    var b=b_
    stream.dispatch(
        kernel: "softlog",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&alpha),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func relu(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "relu",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func relu_s(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ theta_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    var theta=theta_
    stream.dispatch(
        kernel: "relu_s",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&theta)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func softmax_final(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ global_max: GPUBuffer<Float>,
    _ denom: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(global_max.count == Int(b_), "global_max has wrong size")
    precondition(denom.count == Int(b_), "denom has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "softmax",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(global_max.buffer),
            .buffer(denom.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func softmax_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ C: GPUBuffer<Float>,
    _ global_max: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    max_simd(stream: stream, A, scratch0, scratch1, global_max, n_, b_)
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    var isfirst=true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        if isfirst{
            stream.dispatch(
                kernel: "softmax_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    .buffer(global_max.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
            isfirst=false
        }
        else{
            stream.dispatch(
                kernel: "sum_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
        }
        if nextN == 1 { break }
        cur = out
        curN = nextN
    }
    softmax_final(stream: stream, A, C, global_max, B, n_, b_)
}
public func outer_prod(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ C: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ alpha_: Float
){
    precondition(A.count == Int(n_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    var alpha=alpha_
    stream.dispatch(
        kernel: "outer_prod",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&alpha)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func max_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "max_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                bytes(&n),
                bytes(&b)
            ],
            grid: MTLSize(
                width: nextN,
                height: batch,
                depth: 1
            ),
            threads: MTLSize(
                width: 128,
                height: 1,
                depth: 1
            )
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func sum_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "sum_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                bytes(&n),
                bytes(&b)
            ],
            grid: MTLSize(
                width: nextN,
                height: batch,
                depth: 1
            ),
            threads: MTLSize(
                width: 128,
                height: 1,
                depth: 1
            )
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func mean_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    var isfirst=true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        if isfirst{
            stream.dispatch(
                kernel: "mean_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
            isfirst=false
        }
        else{
            stream.dispatch(
                kernel: "sum_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
        }
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func sqmean_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    var isfirst=true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        if isfirst{
            stream.dispatch(
                kernel: "sqmean_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
            isfirst=false
        }
        else{
            stream.dispatch(
                kernel: "sum_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
        }
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func variance_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ mu: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    precondition(mu.count == Int(b_), "mu has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    var isfirst=true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        if isfirst{
            stream.dispatch(
                kernel: "variance_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    .buffer(mu.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
            isfirst=false
        }
        else{
            stream.dispatch(
                kernel: "sum_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
        }
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func abs_mean_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    var isfirst=true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        if isfirst{
            stream.dispatch(
                kernel: "abs_mean_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
            isfirst=false
        }
        else{
            stream.dispatch(
                kernel: "sum_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
                ],
                grid: MTLSize(
                    width: nextN,
                    height: batch,
                    depth: 1
                ),
                threads: MTLSize(
                    width: 128,
                    height: 1,
                    depth: 1
                )
            )
        }
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func conv_r3(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 7, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r3",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func conv_r5(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 11, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r5",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func conv_r7(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 15, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r7",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func conv_r11(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 23, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r11",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func inhib_sub_r3(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ scratch0: GPUBuffer <Float>,
    _ scratch1: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 7, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "inhib_sub_r3",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
    sum_simd(stream: stream, B, scratch0, scratch1, C, n_, b_);
}
public func inhib_div_r7(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ scratch0: GPUBuffer <Float>,
    _ scratch1: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 15, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "inhib_div_r7",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
    sum_simd(stream: stream, B, scratch0, scratch1, C, n_, b_);
}
public func final_oja_step(
    stream: ComputeStream,
    _ G1: GPUBuffer <Float>,
    _ G2: GPUBuffer <Float>,
    _ eta: GPUBuffer <Float>,
    _ A: GPUBuffer <Float>,
    _ A_new: GPUBuffer <Float>,
    _ n_: UInt32,
    _ r_: UInt32,
    _ lambda_: Float,
    _ DA_c_: Float
){
    precondition(G1.count == Int(n_*r_), "G1 has wrong size")
    precondition(G2.count == Int(n_*r_), "G2 has wrong size")
    precondition(eta.count == Int(n_), "eta has wrong size")
    precondition(A.count == Int(n_*r_), "A has wrong size")
    precondition(A_new.count == Int(n_*r_), "A_new has wrong size")
    var n=n_
    var r=r_
    var lambda=lambda_
    var DA_c=DA_c_
    stream.dispatch(
        kernel: "final_oja_step",
        args: [
            .buffer(G1.buffer),
            .buffer(G2.buffer),
            .buffer(eta.buffer),
            .buffer(A.buffer),
            .buffer(A_new.buffer),
            bytes(&n),
            bytes(&r),
            bytes(&lambda),
            bytes(&DA_c)
        ],
        grid: MTLSize(
            width: (Int(r) + 255) / 256,
            height: Int(n),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func get_eta(
    stream: ComputeStream,
    _ eta: GPUBuffer <Float>,
    _ elig: GPUBuffer <Float>,
    _ eligmean: GPUBuffer<Float>,
    _ scratch0: GPUBuffer <Float>,
    _ scratch1: GPUBuffer <Float>,
    _ n_: UInt32,
    _ k_:UInt32,
    _ eta_max_: Float,
    _ alpha_: Float,
    _ DA_c_: Float
){
    precondition(eta.count == Int(n_), "NE has wrong size")
    precondition(eligmean.count==Int(n_), "eligmean has wrong size")
    precondition(elig.count == Int(n_*k_), "elig has wrong size")
    var n=n_
    var eta_max=eta_max_
    var alpha=alpha_
    var DA_c=DA_c_
    abs_mean_simd(stream: stream, elig, scratch0, scratch1, eligmean, k_, n_)
    stream.dispatch(
        kernel: "get_eta",
        args: [
            .buffer(eligmean.buffer),
            .buffer(eta.buffer),
            bytes(&eta_max),
            bytes(&n),
            bytes(&DA_c),
            bytes(&alpha)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func chem_decay_0(
    stream: ComputeStream,
    _ ACh: GPUBuffer <Float>,
    _ DA: GPUBuffer <Float>,
    _ NE: GPUBuffer <Float>,
    _ n_: UInt32,
    _ w_ACh_: Float,
    _ w_DA_: Float,
    _ w_NE_: Float
){
    precondition(NE.count == Int(n_), "NE has wrong size")
    precondition(ACh.count == Int(n_), "ACh has wrong size")
    precondition(DA.count == Int(n_), "DA has wrong size")
    var n=n_
    var w_NE=w_NE_
    var w_ACh=w_ACh_
    var w_DA=w_DA_
    stream.dispatch(
        kernel: "chem_decay",
        args: [
            .buffer(NE.buffer),
            .buffer(ACh.buffer),
            .buffer(DA.buffer),
            bytes(&n),
            bytes(&w_NE),
            bytes(&w_ACh),
            bytes(&w_DA)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func bg_forward(
    stream: ComputeStream,
    _ S: GPUBuffer<Float>,
    _ G: GPUBuffer<Float>,
    _ M: GPUBuffer<Float>,
    _ ST: GPUBuffer<Float>,
    _ scratch0: GPUBuffer <Float>,
    _ scratch1: GPUBuffer <Float>,
    _ n_: UInt32,
    _ DA_c_: Float,
    _ alpha_: Float,
    _ beta_: Float,
    _ kappa_: Float,
    _ gamma_: Float,
    _ k_low_: Float,
    _ k_high_: Float
){
    precondition(S.count == Int(n_), "S has wrong size")
    precondition(G.count == Int(n_), "G has wrong size")
    var n=n_
    var DA_c=DA_c_
    var alpha=alpha_
    var beta=beta_
    var kappa=kappa_
    var gamma=gamma_
    var k_low=k_low_
    var k_high=k_high_
    stream.dispatch(
        kernel: "bg_step",
        args: [
            .buffer(S.buffer),
            .buffer(G.buffer),
            bytes(&n),
            bytes(&DA_c),
            bytes(&alpha),
            bytes(&beta),
            bytes(&kappa)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
    mean_simd(stream: stream, G, scratch0, scratch1, M, n_, 1)
    variance_simd(stream: stream, G, scratch0, scratch1, ST, M, n_, 1)
    sqrt(stream: stream, ST, ST, 1, 1)
    stream.dispatch(
        kernel: "really_specific_kernel",
        args: [
            .buffer(G.buffer),
            .buffer(M.buffer),
            .buffer(ST.buffer),
            bytes(&n),
            bytes(&gamma),
            bytes(&DA_c),
            bytes(&k_low),
            bytes(&k_high)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func bg_oja(
    stream: ComputeStream,
    _ g: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ eta_: Float,
    _ w_max_: Float
){
    precondition(g.count == Int(b_), "g has wrong size")
    precondition(W.count == Int(n_*b_), "W has wrong size")
    var n=n_
    var b=b_
    var eta=eta_
    var w_max=w_max_
    stream.dispatch(
        kernel: "bg_oja",
        args: [
            .buffer(g.buffer),
            .buffer(W.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&eta),
            bytes(&w_max)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func write_point(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ x_: Float,
    _ y_: Float
) {
    precondition(A.count==2, "A has wrong size")
    var x=x_
    var y=y_
    stream.dispatch(
        kernel: "write_point",
        args: [
            .buffer(A.buffer),
            bytes(&x),
            bytes(&y)
        ],
        grid: MTLSize(
            width: 1,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 1,
            height: 1,
            depth: 1
        )
    )
}
public func fill(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B_: Float,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    var B=B_
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "fill",
        args: [
            .buffer(A.buffer),
            bytes(&B),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func fill_random(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ x_: Float,
    _ y_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    var n=n_
    var b=b_
    var x=x_
    var y=y_
    stream.dispatch(
        kernel: "fill_random",
        args: [
            .buffer(A.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&x),
            bytes(&y)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func outer_prod_learn(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ C: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ alpha_: Float
){
    precondition(A.count == Int(n_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    var alpha=alpha_
    stream.dispatch(
        kernel: "outer_prod_learn",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&alpha)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}