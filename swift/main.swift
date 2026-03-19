import Metal
import Foundation
import Darwin
let context=MetalContext(kernelsDirectory: URL(fileURLWithPath: "../kernels"))
let device=context.device
let stream = ComputeStream(context: context)
