import Foundation
import Hub

public class Gemma3nConfigurationLoader {
    
    public static func loadConfig(from path: String) throws -> Config {
        let configPath = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        let data = try Data(contentsOf: configPath)
        let json = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = json as? [String: Any] else {
            throw ConfigurationError.invalidFormat
        }
        // Convert String keys to NSString keys for Config initializer
        let nsDict = dictionary.reduce(into: [NSString: Any]()) { result, pair in
            result[pair.key as NSString] = pair.value
        }
        return Config(nsDict)
    }
    
    public enum ConfigurationError: Error {
        case invalidFormat
    }
}

extension Config {
    
    public var isGemma3n: Bool {
        self.modelType.string() == "gemma3n"
    }
    
    public var textVocabSize: Int? {
        self.textConfig["vocab_size"].integer()
    }
    
    public var textHiddenSize: Int? {
        self.textConfig["hidden_size"].integer()
    }
    
    public var textNumLayers: Int? {
        self.textConfig["num_hidden_layers"].integer()
    }
    
    public var textNumHeads: Int? {
        self.textConfig["num_attention_heads"].integer()
    }
    
    public var textNumKVHeads: Int? {
        self.textConfig["num_key_value_heads"].integer()
    }
    
    public var textIntermediateSize: [Int]? {
        guard let array = self.textConfig["intermediate_size"].array() else { return nil }
        return array.compactMap { $0.integer() }
    }
    
    public var textRopeTheta: Double? {
        if let float = self.textConfig["rope_theta"].floating() {
            return Double(float)
        }
        return nil
    }
    
    public var textSlidingWindow: Int? {
        self.textConfig["sliding_window"].integer()
    }
    
    public var textLayerTypes: [String]? {
        guard let array = self.textConfig["layer_types"].array() else { return nil }
        return array.compactMap { $0.string() }
    }
    
    public var textActivationSparsityPattern: [Double]? {
        guard let array = self.textConfig["activation_sparsity_pattern"].array() else { return nil }
        return array.compactMap { config in
            if let float = config.floating() {
                return Double(float)
            }
            return nil
        }
    }
    
    public var visionVocabSize: Int? {
        self.visionConfig["vocab_size"].integer()
    }
    
    public var visionVocabOffset: Int? {
        self.visionConfig["vocab_offset"].integer()
    }
    
    public var visionHiddenSize: Int? {
        self.visionConfig["hidden_size"].integer()
    }
    
    public var visionArchitecture: String? {
        self.visionConfig["architecture"].string()
    }
    
    public var audioVocabSize: Int? {
        self.audioConfig["vocab_size"].integer()
    }
    
    public var audioVocabOffset: Int? {
        self.audioConfig["vocab_offset"].integer()
    }
    
    public var audioHiddenSize: Int? {
        self.audioConfig["hidden_size"].integer()
    }
    
    public var audioNumLayers: Int? {
        self.audioConfig["conf_num_hidden_layers"].integer()
    }
    
    public var visionSoftTokensPerImage: Int? {
        self["vision_soft_tokens_per_image"].integer()
    }
    
    public var audioSoftTokensPerImage: Int? {
        self["audio_soft_tokens_per_image"].integer()
    }
    
    public var modelName: String {
        if let numLayers = textNumLayers {
            switch numLayers {
            case 30: return "E2B"
            case 35: return "E4B"
            default: return "Unknown"
            }
        }
        return "Unknown"
    }
}