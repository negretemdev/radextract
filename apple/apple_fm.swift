// On-device Apple Foundation Model helper. Reads JSON on stdin:
//   {"system": ..., "prompt": ..., "finding_fields": ["a", "b"] | null, "single_finding": bool}
// With finding_fields the answer is constrained to an object whose properties are all Finding objects
// (mentioned: bool, present: bool, evidence: string?); with single_finding it is one Finding object; otherwise free text.
// Prints {"text": ...} or {"error": ...}. SystemLanguageModel runs on-device only; nothing leaves the machine.
import Foundation
import FoundationModels

struct Request: Decodable {
    let system: String
    let prompt: String
    let finding_fields: [String]?
    let single_finding: Bool?
}

struct Reply: Encodable {
    var text: String? = nil
    var error: String? = nil
}

func emit(_ reply: Reply) {
    let data = try! JSONEncoder().encode(reply)
    print(String(data: data, encoding: .utf8)!)
}

func findingSchema() -> DynamicGenerationSchema {
    DynamicGenerationSchema(name: "Finding", properties: [
        DynamicGenerationSchema.Property(name: "mentioned", schema: DynamicGenerationSchema(type: Bool.self)),
        DynamicGenerationSchema.Property(name: "present", schema: DynamicGenerationSchema(type: Bool.self)),
        DynamicGenerationSchema.Property(name: "evidence", schema: DynamicGenerationSchema(type: String.self), isOptional: true),
    ])
}

let input = FileHandle.standardInput.readDataToEndOfFile()
guard let request = try? JSONDecoder().decode(Request.self, from: input) else {
    emit(Reply(error: "stdin must be JSON with system and prompt"))
    exit(2)
}
let model = SystemLanguageModel(guardrails: .permissiveContentTransformations)
switch model.availability {
case .available:
    break
case .unavailable(let reason):
    emit(Reply(error: "on-device model unavailable: \(reason)"))
    exit(3)
}
let session = LanguageModelSession(model: model, instructions: request.system)
let options = GenerationOptions(samplingMode: .greedy)
do {
    if let fields = request.finding_fields, !fields.isEmpty {
        let root = DynamicGenerationSchema(name: "Group", properties: fields.map {
            DynamicGenerationSchema.Property(name: $0, schema: DynamicGenerationSchema(referenceTo: "Finding"))
        })
        let schema = try GenerationSchema(root: root, dependencies: [findingSchema()])
        let response = try await session.respond(to: request.prompt, schema: schema, options: options)
        emit(Reply(text: response.content.jsonString))
    } else if request.single_finding == true {
        let schema = try GenerationSchema(root: findingSchema(), dependencies: [])
        let response = try await session.respond(to: request.prompt, schema: schema, options: options)
        emit(Reply(text: response.content.jsonString))
    } else {
        let response = try await session.respond(to: request.prompt, options: options)
        emit(Reply(text: response.content))
    }
} catch {
    emit(Reply(error: "\(error)"))
    exit(4)
}
