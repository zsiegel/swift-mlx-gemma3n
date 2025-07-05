import Foundation
import Tokenizers
import Hub

/// Gemma3n tokenizer wrapper that provides additional functionality for the Gemma3n model
/// This wraps a standard tokenizer loaded through the swift-transformers framework
public class Gemma3nTokenizer: @unchecked Sendable {
    
    // Text-only vocabulary size (excluding multimodal tokens)
    public static let textVocabularySize = 262144
    
    // Special token IDs
    public static let padTokenId = 0
    public static let eosTokenId = 1
    public static let alternateEosTokenId = 106  // Gemma3n uses token 106 as an additional EOS
    public static let bosTokenId = 2
    public static let unkTokenId = 3
    
    // Multimodal token boundaries (tokens beyond this are for images/audio)
    public static let multimodalTokenStart = 262144
    
    /// The underlying tokenizer from swift-transformers
    private let tokenizer: Tokenizer
    
    /// Multiple EOS token IDs used by Gemma3n
    public var eosTokenIds: [Int] {
        return [Self.eosTokenId, Self.alternateEosTokenId]
    }
    
    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }
    
    /// Encode text to token IDs, filtering out multimodal tokens for text-only generation
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        let encoded = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        
        // Filter out any multimodal tokens if they appear
        return encoded.map { tokenId in
            if tokenId >= Self.multimodalTokenStart {
                // Replace multimodal tokens with UNK token
                return Self.unkTokenId
            }
            return tokenId
        }
    }
    
    /// Decode token IDs back to text
    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        return tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }
    
    /// Check if a token is an EOS token (handles multiple EOS tokens)
    public func isEosToken(_ tokenId: Int) -> Bool {
        return eosTokenIds.contains(tokenId)
    }
    
    /// Get the text-only vocabulary size
    public var textVocabularySize: Int {
        return Self.textVocabularySize
    }
    
    // Forward other tokenizer properties
    public var bosToken: String? { tokenizer.bosToken }
    public var bosTokenId: Int? { tokenizer.bosTokenId }
    public var eosToken: String? { tokenizer.eosToken }
    public var eosTokenId: Int? { tokenizer.eosTokenId }
    public var unknownToken: String? { tokenizer.unknownToken }
    public var unknownTokenId: Int? { tokenizer.unknownTokenId }
    
    // Forward tokenize method
    public func tokenize(text: String) -> [String] {
        return tokenizer.tokenize(text: text)
    }
    
    // Forward token/ID conversion methods
    public func convertTokenToId(_ token: String) -> Int? {
        return tokenizer.convertTokenToId(token)
    }
    
    public func convertIdToToken(_ id: Int) -> String? {
        return tokenizer.convertIdToToken(id)
    }
}

// MARK: - Convenience Loading

extension Gemma3nTokenizer {
    /// Load a Gemma3n tokenizer from a model directory using the swift-transformers framework
    public static func from(modelPath: String) async throws -> Gemma3nTokenizer {
        let modelURL = URL(fileURLWithPath: modelPath)
        
        // Use AutoTokenizer from swift-transformers to load the tokenizer
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
        
        return Gemma3nTokenizer(tokenizer: tokenizer)
    }
    
    /// Load a Gemma3n tokenizer from a pretrained model name
    public static func from(pretrained modelName: String) async throws -> Gemma3nTokenizer {
        // Use AutoTokenizer from swift-transformers to load the tokenizer
        let tokenizer = try await AutoTokenizer.from(pretrained: modelName)
        
        return Gemma3nTokenizer(tokenizer: tokenizer)
    }
}