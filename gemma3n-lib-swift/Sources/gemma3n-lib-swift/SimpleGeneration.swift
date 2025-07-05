import Foundation
import MLX
import MLXNN
import MLXRandom

/// Simple generation implementation with optional KV cache support
public class SimpleGeneration {
    
    private let model: Gemma3nLanguageModel
    private let tokenizer: Gemma3nTokenizer
    
    public init(model: Gemma3nLanguageModel, tokenizer: Gemma3nTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }
    
    /// Token generation result containing the token ID and decoded text
    public struct TokenResult: Sendable {
        public let tokenId: Int
        public let text: String
    }
    
    /// Create KV caches for all model layers matching Python's make_cache() implementation
    private func createCaches() -> [KVCache] {
        var caches: [KVCache] = []
        
        // Only create caches for layers before first_kv_shared_layer_idx
        // This matches Python: self.config.layer_types[: self.model.first_kv_shared_layer_idx]
        let firstKvSharedLayerIdx = model.config.numHiddenLayers - model.config.numKvSharedLayers
        
        for layerIdx in 0..<firstKvSharedLayerIdx {
            let layerType = model.config.layerTypes[layerIdx]
            
            if layerType == "full_attention" {
                // Use standard KVCache for full attention layers
                caches.append(KVCacheSimple())
            } else if layerType == "sliding_attention" {
                // Use RotatingKVCache for sliding attention layers
                caches.append(
                    RotatingKVCache(
                        maxSize: model.config.slidingWindow,
                        keep: 0,
                        step: 256
                    )
                )
            } else {
                fatalError("Unknown layer type: \(layerType)")
            }
        }
        
        // Pad with nil for shared KV layers
        for _ in firstKvSharedLayerIdx..<model.config.numHiddenLayers {
            caches.append(KVCacheSimple()) // These won't be used, but need placeholders
        }
        
        return caches
    }
    
    
    /// Generate the next token using autoregressive generation with KV cache
    /// This processes only the new token and uses cached keys/values
    private func generateNextTokenWithCache(
        inputIds: [Int],
        caches: inout [KVCache],
        temperature: Float = 0.0,
        verbose: Bool = false
    ) -> Int {
        autoreleasepool {
            // Determine which tokens to process and initialize cache if needed
            let isFirstToken = caches.isEmpty
            if isFirstToken {
                caches = createCaches()
            }
            let inputTokens = isFirstToken ? inputIds : [inputIds.last!]
            
            // print("\n[SWIFT _STEP] Processing token(s) with shape: [\(inputTokens.count)]")
            // if !caches.isEmpty && caches.count > 10 {
                // print("  [SWIFT _STEP] Cache offset before (L0): \(caches[0].offset)")
                // print("  [SWIFT _STEP] Cache offset before (L10): \(caches[10].offset)")
            // }
            
            // Convert to MLXArray and add batch dimension
            let inputArray = MLXArray(inputTokens.map { Int32($0) })
            let batchedInput = inputArray.reshaped([1, inputTokens.count])
            
            // Run forward pass through model with cache
            let logits = model(batchedInput, cache: &caches)
            
            // if !caches.isEmpty && caches.count > 10 {
                // print("  [SWIFT _STEP] Cache offset after (L0): \(caches[0].offset)")
                // print("  [SWIFT _STEP] Cache offset after (L10): \(caches[10].offset)")
            // }
            
            // Get logits for the last token
            let lastTokenLogits = logits[0, -1, 0...]
            
            // Apply temperature and sampling
            let nextTokenId: MLXArray
            
            if temperature == 0 {
                // Greedy decoding
                nextTokenId = argMax(lastTokenLogits, axis: -1)
            } else {
                // Temperature sampling
                let scaledLogits = lastTokenLogits / temperature
                nextTokenId = MLXRandom.categorical(scaledLogits, axis: -1, shape: [1])
            }
            
            // Force evaluation and convert to Int
            MLX.eval(nextTokenId)
            let tokenValue = nextTokenId.item(Int32.self)
            
            return Int(tokenValue)
        }
    }
    
    /// Generate the next token given a sequence of input tokens
    /// This processes the entire sequence from scratch (no KV cache)
    private func generateNextToken(inputIds: [Int], temperature: Float = 0.0) -> Int {
        autoreleasepool {
            // NOTE: This implementation processes the full sequence on each step
            // The Python implementation uses KV caching and only processes the last token
            // This difference may cause divergent behavior. A proper fix requires:
            // 1. Implementing KV cache support
            // 2. Processing only the new token on each step
            // 3. Using cached keys/values from previous steps
            
            // Convert to MLXArray
            let inputArray = MLXArray(inputIds.map { Int32($0) })
            
            // Add batch dimension: [sequence_length] -> [1, sequence_length]
            let batchedInput = inputArray.reshaped([1, inputIds.count])
            
            // Run forward pass through model
            var emptyCache = [KVCache]()
            let logits = model(batchedInput, cache: &emptyCache)
            
            // Get logits for the last token: [batch, sequence, vocab] -> [vocab]
            let lastTokenLogits = logits[0, inputIds.count - 1, 0...]
            
            // Apply temperature and sampling
            let nextTokenId: MLXArray
            
            if temperature == 0 {
                // Greedy decoding - just take argmax
                nextTokenId = argMax(lastTokenLogits, axis: -1)
            } else {
                // Temperature sampling
                let scaledLogits = lastTokenLogits / temperature
                
                // Convert to probabilities with softmax
                let probs = MLX.softmax(scaledLogits, axis: -1)
                
                // Sample from the multinomial distribution
                // MLXRandom.categorical returns indices sampled according to probabilities
                nextTokenId = MLXRandom.categorical(scaledLogits, axis: -1, shape: [1])
            }
            
            // Force evaluation and convert to Int
            MLX.eval(nextTokenId)
            let tokenValue = nextTokenId.item(Int32.self)
            
            return Int(tokenValue)
        }
    }
    
    /// Format a user prompt with the Gemma3n chat template
    public func formatPrompt(_ userMessage: String) -> String {
        return "<bos><start_of_turn>user\n\(userMessage)<end_of_turn>\n<start_of_turn>model\n"
    }
    
    /// Generate a response using autoregressive generation with KV cache
    public func generateWithCache(
        prompt: String,
        maxTokens: Int = 50,
        temperature: Float = 0.0,
        verbose: Bool = false
    ) -> String {
        autoreleasepool {
            // Start with an empty cache; it will be created on the first run
            var caches = [KVCache]()
            
            // Format the prompt with chat template
            let formattedPrompt = formatPrompt(prompt)
            
            
            // Tokenize the prompt
            var tokenIds = tokenizer.encode(formattedPrompt, addSpecialTokens: false)
            
            // Generation loop
            for i in 0..<maxTokens {
                // Generate next token with cache
                let nextToken = generateNextTokenWithCache(
                    inputIds: tokenIds,
                    caches: &caches,
                    temperature: temperature,
                    verbose: verbose
                )
                
                // Add to sequence
                tokenIds.append(nextToken)
                
                // Check for EOS tokens (1 or 106 for Gemma3n)
                if nextToken == Gemma3nTokenizer.eosTokenId || 
                   nextToken == Gemma3nTokenizer.alternateEosTokenId {
                    break
                }
            }
            
            // Decode the full sequence
            let fullText = tokenizer.decode(tokens: tokenIds, skipSpecialTokens: false)
            
            // Extract just the model's response (after the last "model\n")
            if let modelStart = fullText.range(of: "<start_of_turn>model\n", options: .backwards) {
                let response = String(fullText[modelStart.upperBound...])
                // Remove any trailing end_of_turn token
                return response.replacingOccurrences(of: "<end_of_turn>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            return fullText
        }
    }
    
    /// Generate a response to a prompt
    public func generate(
        prompt: String,
        maxTokens: Int = 50,
        temperature: Float = 0.0,
        verbose: Bool = false
    ) -> String {
        autoreleasepool {
            // Format the prompt with chat template
            let formattedPrompt = formatPrompt(prompt)
            
            
            // Tokenize the prompt
            var tokenIds = tokenizer.encode(formattedPrompt, addSpecialTokens: false)
            
            print("\n=== SWIFT TOKENIZATION DEBUG ===")
            print("Formatted prompt: \(formattedPrompt)")
            print("Initial token IDs: \(tokenIds)")
            print("Number of initial tokens: \(tokenIds.count)")
            
            // Generation loop
            for i in 0..<maxTokens {
                if verbose && i < 3 {  // Only log first few iterations
                    print("\n=== Generation step \(i+1) ===")
                    print("  Current sequence length: \(tokenIds.count)")
                }
                
                // Generate next token
                let nextToken = generateNextToken(inputIds: tokenIds, temperature: temperature)
                
                // Add to sequence
                tokenIds.append(nextToken)
                
                // Check for EOS tokens (1 or 106 for Gemma3n)
                if nextToken == Gemma3nTokenizer.eosTokenId || 
                   nextToken == Gemma3nTokenizer.alternateEosTokenId {
                    break
                }
            }
            
            // Decode the full sequence
            let fullText = tokenizer.decode(tokens: tokenIds, skipSpecialTokens: false)
            
            // Extract just the model's response (after the last "model\n")
            if let modelStart = fullText.range(of: "<start_of_turn>model\n", options: .backwards) {
                let response = String(fullText[modelStart.upperBound...])
                // Remove any trailing end_of_turn token
                return response.replacingOccurrences(of: "<end_of_turn>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            return fullText
        }
    }
    
    /// Generate a response with streaming using a callback for each token
    public func generateStreaming(
        prompt: String,
        maxTokens: Int = 50,
        temperature: Float = 0.0,
        verbose: Bool = false,
        onToken: @escaping @Sendable (TokenResult) -> Void
    ) -> String {
        autoreleasepool {
            var caches = [KVCache]()
            
            // Format the prompt with chat template
            let formattedPrompt = formatPrompt(prompt)
            
            // Tokenize the prompt
            var tokenIds = tokenizer.encode(formattedPrompt, addSpecialTokens: false)
            var generatedTokenIds: [Int] = []
            
            // Generation loop
            for _ in 0..<maxTokens {
                // Generate next token with cache
                let nextToken = generateNextTokenWithCache(
                    inputIds: tokenIds,
                    caches: &caches,
                    temperature: temperature,
                    verbose: verbose
                )
                
                // Add to sequences
                tokenIds.append(nextToken)
                generatedTokenIds.append(nextToken)
                
                // Decode just the new token
                let tokenText = tokenizer.decode(tokens: [nextToken], skipSpecialTokens: false)
                
                // Call the callback with the new token
                onToken(TokenResult(tokenId: nextToken, text: tokenText))
                
                // Check for EOS tokens
                if nextToken == Gemma3nTokenizer.eosTokenId || 
                   nextToken == Gemma3nTokenizer.alternateEosTokenId {
                    break
                }
            }
            
            // Decode the full generated sequence
            let generatedText = tokenizer.decode(tokens: generatedTokenIds, skipSpecialTokens: false)
            
            // Remove any trailing end_of_turn token
            return generatedText.replacingOccurrences(of: "<end_of_turn>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
    
}