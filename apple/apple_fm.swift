// On-device Apple Foundation Model helper. Reads {"system": ..., "prompt": ...} as JSON on stdin,
// prints {"text": ...} or {"error": ...} as JSON on stdout. Nothing leaves the machine: SystemLanguageModel is on-device only.
import Foundation
import FoundationModels

struct Request: Decodable {
    let system: String
    let prompt: String
}

struct Reply: Encodable {
    var text: String? = nil
    var error: String? = nil
}

func emit(_ reply: Reply) {
    let data = try! JSONEncoder().encode(reply)
    print(String(data: data, encoding: .utf8)!)
}

let input = FileHandle.standardInput.readDataToEndOfFile()
guard let request = try? JSONDecoder().decode(Request.self, from: input) else {
    emit(Reply(error: "stdin must be JSON with system and prompt"))
    exit(2)
}
let model = SystemLanguageModel.default
switch model.availability {
case .available:
    break
case .unavailable(let reason):
    emit(Reply(error: "on-device model unavailable: \(reason)"))
    exit(3)
}
let session = LanguageModelSession(model: model, instructions: request.system)
do {
    let response = try await session.respond(to: request.prompt, options: GenerationOptions(samplingMode: .greedy))
    emit(Reply(text: response.content))
} catch {
    emit(Reply(error: "\(error)"))
    exit(4)
}
