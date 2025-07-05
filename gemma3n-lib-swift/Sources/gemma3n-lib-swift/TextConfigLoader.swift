import Foundation

extension TextConfig {
    /// Load TextConfig from a model directory
    public static func load(from modelPath: String) throws -> TextConfig {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        
        // Parse JSON to extract just the text_config section
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let textConfigDict = json["text_config"] as? [String: Any] else {
            throw NSError(domain: "TextConfigLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "text_config not found in JSON"])
        }
        
        // Encode the text_config dict back to Data and decode as TextConfig
        let textConfigData = try JSONSerialization.data(withJSONObject: textConfigDict)
        return try JSONDecoder().decode(TextConfig.self, from: textConfigData)
    }
}

extension Gemma3nConfig {
    /// Load full Gemma3n config from a model directory
    public static func load(from modelPath: String) throws -> Gemma3nConfig {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(Gemma3nConfig.self, from: data)
    }
}