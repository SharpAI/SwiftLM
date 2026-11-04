import XCTest
import MLX
import MLXNN
@testable import SwiftLM

/// KimiLinear's MLP is SwiGLU: down(silu(gate(x)) * up(x)). The activation must sit on
/// gate_proj; putting it on up_proj silently produces wrong activations (no shape error).
final class KimiLinearMLPTests: XCTestCase {

    func testMLPIsSwiGLUWithSiluOnGate() {
        let dim = 4, hidden = 6
        let mlp = KimiMLP(dimensions: dim, hiddenDimensions: hidden)

        // Fixed, non-symmetric weights so gate/up are distinguishable.
        let g = (0..<(hidden * dim)).map { Float(sin(Double($0) * 0.7)) }
        let u = (0..<(hidden * dim)).map { Float(cos(Double($0) * 0.4 + 1)) }
        let d = (0..<(dim * hidden)).map { Float(sin(Double($0) * 0.3 + 2)) }
        mlp.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": MLXArray(g, [hidden, dim]),
            "up_proj.weight": MLXArray(u, [hidden, dim]),
            "down_proj.weight": MLXArray(d, [dim, hidden]),
        ]))

        let xs: [Float] = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.1, -0.3]
        let x = MLXArray(xs, [2, dim])
        let got = Device.withDefaultDevice(.cpu) { mlp(x).asArray(Float.self) }

        // Hand-computed SwiGLU on the CPU.
        func silu(_ v: Float) -> Float { v / (1 + exp(-v)) }
        var expected = [Float]()
        for r in 0..<2 {
            var h = [Float](repeating: 0, count: hidden)
            for j in 0..<hidden {
                var gv: Float = 0, uv: Float = 0
                for k in 0..<dim {
                    gv += g[j * dim + k] * xs[r * dim + k]
                    uv += u[j * dim + k] * xs[r * dim + k]
                }
                h[j] = silu(gv) * uv
            }
            for i in 0..<dim {
                var o: Float = 0
                for j in 0..<hidden { o += d[i * hidden + j] * h[j] }
                expected.append(o)
            }
        }
        XCTAssertEqual(got.count, expected.count)
        for (a, b) in zip(got, expected) {
            XCTAssertEqual(a, b, accuracy: 1e-4)
        }
    }
}
