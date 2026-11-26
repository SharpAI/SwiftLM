import XCTest
import Foundation
import Jinja
@testable import SwiftLM

/// Issue #168: JSON `null` (NSNull) in tool schemas used to abort Jinja.Value conversion
/// with a misleading "Optional<Any>" error, mislabeled as a broken chat template.
final class JinjaSanitizerTests: XCTestCase {

    func testDropsTopLevelNSNull() {
        XCTAssertNil(sanitizeForJinja(NSNull()))
    }

    func testDropsNestedObjectNulls() throws {
        let tool: [String: any Sendable] = [
            "type": "function",
            "function": [
                "name": "probe",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "query": [
                            "type": "string",
                            "default": NSNull() as any Sendable,
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        let cleaned = try XCTUnwrap(sanitizeForJinja(tool) as? [String: any Sendable])
        let fn = try XCTUnwrap(cleaned["function"] as? [String: any Sendable])
        let params = try XCTUnwrap(fn["parameters"] as? [String: any Sendable])
        let props = try XCTUnwrap(params["properties"] as? [String: any Sendable])
        let query = try XCTUnwrap(props["query"] as? [String: any Sendable])

        XCTAssertNil(query["default"], "null default must be stripped")
        XCTAssertEqual(query["type"] as? String, "string")
        XCTAssertEqual(params["type"] as? String, "object")
    }

    // MARK: - A null inside an array keeps its place (#264)

    /// The array went through `compactMap`, so `[1, null, 2]` came out as `[1, 2]` and a
    /// template reading by index got the value one position over.
    func testKeepsNullArrayElementsInPlace() throws {
        let value: [String: any Sendable] = [
            "enum": [1, NSNull(), 2] as [any Sendable]
        ]
        let cleaned = try XCTUnwrap(sanitizeForJinja(value) as? [String: any Sendable])
        let enumValues = try XCTUnwrap(cleaned["enum"] as? [Any])
        guard enumValues.count == 3 else {
            return XCTFail("expected 3 elements, got \(enumValues.count)")
        }
        XCTAssertEqual(enumValues[0] as? Int, 1)
        XCTAssertNil(sanitizeForJinja(enumValues[1] as any Sendable), "the middle element is a plain nil")
        XCTAssertEqual(enumValues[2] as? Int, 2)
        XCTAssertFalse(enumValues.contains { $0 is NSNull }, "NSNull itself must never reach swift-jinja")
    }

    func testNullsNestedInsideAnArrayFollowTheSameRules() throws {
        let value: [any Sendable] = [
            ["keep": 1, "drop": NSNull()] as [String: any Sendable],
            NSNull(),
            [NSNull(), "x"] as [any Sendable],
        ]
        let cleaned = try XCTUnwrap(sanitizeForJinja(value) as? [Any])
        guard cleaned.count == 3 else {
            return XCTFail("expected 3 elements, got \(cleaned.count)")
        }
        let object = try XCTUnwrap(cleaned[0] as? [String: any Sendable])
        XCTAssertEqual(object["keep"] as? Int, 1)
        XCTAssertNil(object["drop"], "a null dictionary entry is still dropped")
        let inner = try XCTUnwrap(cleaned[2] as? [Any])
        XCTAssertEqual(inner.count, 2)
        XCTAssertEqual(inner[1] as? String, "x")
    }

    /// The same conversion swift-transformers does before it renders (`Value(any:)`), on a
    /// `chat_template_kwargs` list: the template sees three elements with `none` in the middle.
    func testTemplateReadsTheNullElementAsNone() throws {
        let kwargs: [String: any Sendable] = ["ids": [1, NSNull(), 2] as [any Sendable]]
        let context = try kwargs.mapValuesDeep(sanitizeForJinja).mapValues { try Value(any: $0) }
        let template = try Template("{{ ids | length }} {{ ids[2] }} {{ ids[1] is none }}")
        XCTAssertEqual(try template.render(context), "3 2 true")
    }

    /// Tool schemas take the same path. A nullable enum keeps its `null` option in the JSON
    /// the model is shown.
    func testToolSchemaEnumKeepsItsNullOption() throws {
        let tool: [String: any Sendable] = [
            "name": "set_unit",
            "parameters": [
                "unit": ["enum": ["c", NSNull(), "f"] as [any Sendable]] as [String: any Sendable]
            ] as [String: any Sendable],
        ]
        let context = ["tool": try Value(any: tool.mapValuesDeep(sanitizeForJinja))]
        let template = try Template("{{ tool.parameters.unit.enum | tojson }}")
        XCTAssertEqual(
            try template.render(context).replacingOccurrences(of: " ", with: ""), #"["c",null,"f"]"#)
    }

    func testSanitizingTwiceGivesTheSameResult() throws {
        let value: [any Sendable] = [1, NSNull(), ["a": NSNull(), "b": [NSNull()] as [any Sendable]] as [String: any Sendable]]
        let once = try XCTUnwrap(sanitizeForJinja(value))
        let twice = try XCTUnwrap(sanitizeForJinja(once))
        XCTAssertEqual(try Value(any: once), try Value(any: twice))
        XCTAssertEqual(try Template("{{ v | tojson }}").render(["v": try Value(any: twice)])
            .replacingOccurrences(of: " ", with: ""), #"[1,null,{"b":[null]}]"#)
    }

    func testPreservesScalarsAndStructure() throws {
        let value: [String: any Sendable] = [
            "type": "string",
            "minimum": 0,
            "required": ["command"] as [any Sendable],
            "flag": true,
        ]
        let cleaned = try XCTUnwrap(sanitizeForJinja(value) as? [String: any Sendable])
        XCTAssertEqual(cleaned["type"] as? String, "string")
        XCTAssertEqual(cleaned["minimum"] as? Int, 0)
        XCTAssertEqual(cleaned["flag"] as? Bool, true)
        XCTAssertEqual((cleaned["required"] as? [Any])?.count, 1)
    }

    func testUnwrapsNestedOptional() throws {
        let wrapped: any Sendable = Optional<String>.some("hi")
        let cleaned = sanitizeForJinja(wrapped)
        XCTAssertEqual(cleaned as? String, "hi")

        let empty: any Sendable = Optional<String>.none
        XCTAssertNil(sanitizeForJinja(empty))
    }

    func testMapValuesDeepOnToolDict() throws {
        let dict: [String: any Sendable] = [
            "keep": "x",
            "drop": NSNull(),
        ]
        let cleaned = dict.mapValuesDeep(sanitizeForJinja)
        XCTAssertEqual(cleaned["keep"] as? String, "x")
        XCTAssertNil(cleaned["drop"])
    }
}
