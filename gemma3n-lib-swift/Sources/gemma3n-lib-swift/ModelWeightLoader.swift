import Foundation
import MLX
import MLXNN

/// Enhanced weight loader for Gemma3n models with proper sanitization
public class Gemma3nModelWeightLoader {
    
    /// Load and sanitize all weights from a model directory
    /// - Parameter directory: Path to model directory containing safetensors files
    /// - Returns: Sanitized weights dictionary
    public static func loadAndSanitizeWeights(from directory: URL) throws -> [String: MLXArray] {
        // Load raw weights
        let rawWeights = try loadRawWeights(from: directory)
        
        // Apply Gemma3n-specific sanitization
        let sanitized = sanitizeWeights(rawWeights)
        
        // Handle tied embeddings if needed
        let finalWeights = handleTiedEmbeddings(sanitized)
        
        return finalWeights
    }
    
    /// Load raw weights from all safetensors files
private static func loadRawWeights(from directory: URL) throws -> [String: MLXArray] {
        var allWeights: [String: MLXArray] = [:]
        
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        for file in safetensorFiles {
            let weights = try MLX.loadArrays(url: file)
            allWeights.merge(weights) { _, new in new }
        }
        
        return allWeights
    }
    
    /// Sanitize weight keys according to Gemma3n conventions
    public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            // Skip non-language model weights (audio, vision)
            guard key.hasPrefix("language_model.") else { continue }
            
            // Skip rotary embeddings (they're computed, not loaded)
            if key.contains("rotary_emb.inv_freq") {
                continue
            }
            
            // Remove language_model prefix
            var cleanKey = String(key.dropFirst("language_model.".count))
            
            // For weights not under model. or lm_head., add model. prefix
            // This matches the Python sanitization logic
            if !cleanKey.hasPrefix("model.") && !cleanKey.hasPrefix("lm_head.") {
                cleanKey = "model." + cleanKey
            }
            
            sanitized[cleanKey] = value
        }
        
        return sanitized
    }
    
    /// Handle tied embeddings - share embed_tokens with lm_head if configured
    public static func handleTiedEmbeddings(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var finalWeights = weights
        
        let embedKey = "model.embed_tokens.weight"
        let lmHeadKey = "lm_head.weight"
        
        // If lm_head is missing but embeddings exist, they're tied
        if finalWeights[lmHeadKey] == nil, let embedWeight = finalWeights[embedKey] {
            finalWeights[lmHeadKey] = embedWeight
        }
        
        return finalWeights
    }
    
    /// Load weights for the complete Gemma3n model
    public static func loadModelWeights(
        model: Gemma3nModel,
        from directory: URL
    ) throws {
        let weights = try loadAndSanitizeWeights(from: directory)
        
        // Separate model weights from lm_head weights
        var modelWeights: [String: MLXArray] = [:]
        var lmHeadWeights: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            if key.hasPrefix("model.") {
                // Remove "model." prefix for the model's internal structure
                let modelKey = String(key.dropFirst("model.".count))
                modelWeights[modelKey] = value
            } else if key.hasPrefix("lm_head.") {
                // LM head weights are handled separately (in LanguageModel)
                lmHeadWeights[key] = value
            }
        }
        
        // Convert to nested parameters structure
        let parameters = ModuleParameters.unflattened(modelWeights)
        
        // Update model with verification
        try model.update(parameters: parameters, verify: [.noUnusedKeys])
        
        // Evaluate to materialize lazy arrays
        MLX.eval(model)
    }
    
    /// Load weights for the language model (includes LM head)
    public static func loadLanguageModelWeights(
        model: Gemma3nLanguageModel,
        from directory: URL
    ) throws {
        let weights = try loadAndSanitizeWeights(from: directory)
        
        // Convert to nested parameters structure
        let parameters = ModuleParameters.unflattened(weights)
        
        // Update model with verification
        try model.update(parameters: parameters, verify: [.noUnusedKeys])
        
        // Evaluate to materialize lazy arrays
        MLX.eval(model)
    }
    
    /// Verify model weights are properly loaded
    public static func verifyModelWeights(
        model: Gemma3nModel,
        config: TextConfig
    ) -> (loaded: Int, missing: [String]) {
        let params = model.parameters()
        let flattened = params.flattened()
        
        // Expected major components
        let expectedComponents = [
            "embed_tokens.weight",
            "embed_tokens_per_layer.weight",
            "norm.weight",
            "per_layer_model_projection.weight",
            "per_layer_projection_norm.weight"
        ]
        
        // Add layer-specific components
        var allExpected: [String] = expectedComponents
        for i in 0..<config.numHiddenLayers {
            allExpected.append("layers.\(i).altup.correct_output_scale")
            allExpected.append("layers.\(i).self_attn.q_proj.weight")
            allExpected.append("layers.\(i).mlp.gate_proj.weight")
            // Add more as needed...
        }
        
        // Check what's loaded
        let loadedKeys = Set(flattened.map { $0.0 })
        let missing = allExpected.filter { !loadedKeys.contains($0) }
        
        return (loaded: loadedKeys.count, missing: missing)
    }
    
}