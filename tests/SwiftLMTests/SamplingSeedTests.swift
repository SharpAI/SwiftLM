import XCTest
import Foundation
import MLX
import MLXLMCommon
@testable import SwiftLM

// MARK: - The `seed` request field
//
// Two defects hid behind one line. `MLXRandom.seed(UInt64(seed))` seeded the *global*
// generator, but the samplers draw from their own `RandomState`, which is only seeded
// through `GenerateParameters.seed` — so the same seed produced different output on every
// call. And `UInt64(seed)` traps on a negative value, so `"seed": -1` took the whole
// server down with "Negative value is not representable".
final class SamplingSeedTests: XCTestCase {

    func testAbsentSeedStaysAbsent() {
        XCTAssertNil(samplingSeed(nil), "no seed means entropy-seeded sampling, as before")
    }

    func testNonNegativeSeedIsPassedThrough() {
        XCTAssertEqual(samplingSeed(0), 0)
        XCTAssertEqual(samplingSeed(42), 42)
        XCTAssertEqual(samplingSeed(Int.max), UInt64(Int.max))
    }

    /// OpenAI's `seed` is a plain integer, so clients can and do send negative values.
    /// They must map to *some* seed rather than crash the process.
    func testNegativeSeedDoesNotTrap() {
        XCTAssertEqual(samplingSeed(-1), UInt64(bitPattern: -1))
        XCTAssertEqual(samplingSeed(Int.min), UInt64(bitPattern: Int64(Int.min)))
        XCTAssertNotEqual(samplingSeed(-1), samplingSeed(-2), "distinct seeds stay distinct")
    }

    /// The seed must reach the sampler, not just the global generator: two samplers built
    /// from the same parameters have to agree, and a different seed has to be able to
    /// disagree. This is the property the server relies on for reproducible requests.
    func testSeededParametersGiveReproducibleSampling() {
        func sample(seed: UInt64?, draws: Int = 24) -> [Int] {
            var params = GenerateParameters(temperature: 1.0, topP: 1.0)
            params.seed = seed
            let sampler = params.sampler()
            // A flat distribution over a handful of tokens, so the draw is all RNG.
            let logits = MLXArray([Float](repeating: 0, count: 64)).reshaped(1, 64)
            return (0..<draws).map { _ in sampler.sample(logits: logits).item(Int.self) }
        }
        XCTAssertEqual(sample(seed: 42), sample(seed: 42), "same seed, same tokens")
        XCTAssertNotEqual(sample(seed: 42), sample(seed: 43), "different seed, different tokens")
    }
}
