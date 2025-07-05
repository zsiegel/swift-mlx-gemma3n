import Foundation
import MLX
import MLXNN

/// Utility for loading Gemma3n weights from safetensors files
public class Gemma3nWeightLoader {
    
    /// Load all weights from a model directory
    public static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        var allWeights: [String: MLXArray] = [:]
        
        // Find all safetensors files
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }
        
        // Load each safetensors file
        for file in safetensorFiles {
            let weights = try MLX.loadArrays(url: file)
            allWeights.merge(weights) { _, new in new }
        }
        
        return allWeights
    }
    
    /// Sanitize weight keys for Gemma3n model
    /// Removes prefixes and maps to our module structure
    public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            // Skip non-language model weights for now (audio, vision)
            if !key.contains("language_model") {
                continue
            }
            
            // Remove "language_model.model." prefix if present
            let cleanKey: String
            if key.hasPrefix("language_model.model.") {
                cleanKey = String(key.dropFirst(21))  // "language_model.model.".count = 21
            } else if key.hasPrefix("language_model.") {
                cleanKey = String(key.dropFirst(15))  // "language_model.".count = 15
            } else {
                cleanKey = key
            }
            
            sanitized[cleanKey] = value
        }
        
        return sanitized
    }
    
    /// Extract weights for a specific transformer block
    public static func extractBlockWeights(
        from weights: [String: MLXArray],
        layerIdx: Int
    ) -> [String: MLXArray] {
        let prefix = "layers.\(layerIdx)."
        var blockWeights: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            if key.hasPrefix(prefix) {
                // Remove the layer prefix to get component keys
                let componentKey = String(key.dropFirst(prefix.count))
                blockWeights[componentKey] = value
            }
        }
        
        return blockWeights
    }
    
    /// Load weights for a specific transformer block
    public static func loadBlockWeights(
        block: Gemma3nTransformerBlock,
        from directory: URL
    ) throws {
        // Load all weights
        let allWeights = try loadWeights(from: directory)
        let sanitizedWeights = sanitizeWeights(allWeights)
        
        // Extract weights for this specific block
        let blockWeights = extractBlockWeights(from: sanitizedWeights, layerIdx: block.layerIdx)
        
        // Convert to module parameters
        let parameters = ModuleParameters.unflattened(blockWeights)
        
        // Update the block with verification
        try block.update(parameters: parameters, verify: [.all])
        
        // Evaluate to materialize lazy arrays
        MLX.eval(block)
    }
    
    /// Verify that all expected weights are present for a transformer block
    public static func verifyBlockWeights(
        weights: [String: MLXArray],
        config: TextConfig
    ) -> (missing: [String], unexpected: [String]) {
        // Expected weight keys for a transformer block
        let expectedKeys = [
            // AltUp weights
            "altup.correct_output_scale",
            "altup.correction_coefs.weight",
            "altup.modality_router.weight",
            "altup.prediction_coefs.weight",
            "altup.router_norm.weight",
            
            // Normalization weights
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "post_per_layer_input_norm.weight",
            
            // LAUREL weights
            "laurel.linear_left.weight",
            "laurel.linear_right.weight",
            "laurel.post_laurel_norm.weight",
            
            // Attention weights
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            
            // MLP weights
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight"
        ]
        
        // Add per-layer embedding weights if configured
        var allExpectedKeys = expectedKeys
        // Always expect per-layer weights (default 256)
        if true {
            allExpectedKeys.append("per_layer_projection.weight")
            allExpectedKeys.append("per_layer_input_gate.weight")
        }
        
        let weightKeys = Set(weights.keys)
        let expectedSet = Set(allExpectedKeys)
        
        let missing = Array(expectedSet.subtracting(weightKeys))
        let unexpected = Array(weightKeys.subtracting(expectedSet))
        
        return (missing: missing, unexpected: unexpected)
    }
}