import Foundation
import MLX
import MLXNN
import CryptoKit

/// Main Gemma3n model combining all transformer components
public class Gemma3nModel: Module {
    
    // Embeddings
    @ModuleInfo(key: "embed_tokens") public var embedTokens: TokenEmbedding
    @ModuleInfo(key: "embed_tokens_per_layer") public var embedTokensPerLayer: PerLayerEmbedding
    
    // Transformer layers
    @ModuleInfo(key: "layers") public var layers: [Gemma3nTransformerBlock]
    
    // Final normalization
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    
    // Per-layer projections
    @ModuleInfo(key: "per_layer_model_projection") public var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") public var perLayerProjectionNorm: Gemma3nRMSNorm
    
    // AltUp projections for multi-stream expansion
    @ModuleInfo(key: "altup_projections") public var altupProjections: [Linear]
    @ModuleInfo(key: "altup_unembed_projections") public var altupUnembedProjections: [Linear]
    
    // RoPE embeddings
    public let ropeEmbedding: RotaryPositionEmbedding
    public let ropeEmbeddingLocal: RotaryPositionEmbedding
    
    // Configuration
    public let config: TextConfig
    
    // Scales
    private let embedTokensScale: Float
    private let embedTokensPerLayerScale: Float
    private let perLayerProjectionScale: Float
    private let perLayerInputScale: Float
    var passNumber = 0

    public init(config: TextConfig) {
        self.config = config
        
        // Initialize scales
        self.embedTokensScale = sqrt(Float(config.hiddenSize))
        self.embedTokensPerLayerScale = sqrt(Float(config.hiddenSizePerLayerInput))
        self.perLayerProjectionScale = 1.0 / sqrt(Float(config.hiddenSize))
        self.perLayerInputScale = 1.0 / sqrt(2.0)
        
        // Initialize embeddings
        self._embedTokens = ModuleInfo(
            wrappedValue: TokenEmbedding(
                vocabSize: config.vocabSize,
                hiddenSize: config.hiddenSize
            ),
            key: "embed_tokens"
        )
        
        self._embedTokensPerLayer = ModuleInfo(
            wrappedValue: PerLayerEmbedding(
                vocabSize: config.vocabSizePerLayerInput,
                hiddenSize: config.hiddenSizePerLayerInput,
                numLayers: config.numHiddenLayers
            ),
            key: "embed_tokens_per_layer"
        )
        
        // Initialize transformer layers
        var layers: [Gemma3nTransformerBlock] = []
        for layerIdx in 0..<config.numHiddenLayers {
            layers.append(Gemma3nTransformerBlock(config: config, layerIdx: layerIdx))
        }
        self._layers = ModuleInfo(wrappedValue: layers, key: "layers")
        
        // Initialize final norm
        self._norm = ModuleInfo(
            wrappedValue: RMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps)
            ),
            key: "norm"
        )
        
        // Initialize per-layer projections
        self._perLayerModelProjection = ModuleInfo(
            wrappedValue: Linear(
                config.hiddenSize,
                config.numHiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            ),
            key: "per_layer_model_projection"
        )
        
        self._perLayerProjectionNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: config.hiddenSizePerLayerInput,
                eps: Float(config.rmsNormEps)
            ),
            key: "per_layer_projection_norm"
        )
        
        // Initialize AltUp projections (n-1 projections for n streams)
        var altupProjs: [Linear] = []
        var altupUnembedProjs: [Linear] = []
        for _ in 1..<config.altupNumInputs {
            altupProjs.append(Linear(config.hiddenSize, config.hiddenSize, bias: false))
            altupUnembedProjs.append(Linear(config.hiddenSize, config.hiddenSize, bias: false))
        }
        self._altupProjections = ModuleInfo(wrappedValue: altupProjs, key: "altup_projections")
        self._altupUnembedProjections = ModuleInfo(wrappedValue: altupUnembedProjs, key: "altup_unembed_projections")
        
        // Initialize RoPE embeddings
        self.ropeEmbedding = RotaryPositionEmbedding(
            dim: config.headDim,
            base: Float(config.ropeTheta)
        )
        
        // Create local RoPE with different theta
        self.ropeEmbeddingLocal = RotaryPositionEmbedding(
            dim: config.headDim,
            base: Float(config.ropeLocalBaseFreq)
        )
        
        super.init()
    }
    
    
    /// Forward pass through the model
    /// - Parameters:
    ///   - inputs: Input token IDs [batch_size, sequence_length]
    ///   - mask: Attention mask (optional)
    ///   - cache: KV cache for each layer (optional)
    ///   - perLayerInputs: Pre-computed per-layer inputs (optional)
    /// - Returns: Hidden states after all transformer layers [batch_size, sequence_length, hidden_size]
    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: inout [KVCache],
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        // Get input embeddings
        var h: MLXArray
        passNumber += 1
        
        // Update DebugLayers with current pass
        DebugLayers.currentPass = passNumber

        // print("\n\n[SWIFT] FORWARD PASS WITH INPUTS \(passNumber)")
        // print("  raw inputs shape: \(inputs.shape)")
        // print("  raw inputs dtype: \(inputs.dtype)")
        // print("  First 5 items from raw inputs: \(inputs)")  

        // This fixes an issue because our Swift model is a language model ONLY
        // This compensates for pre-scaled weights (ie we dont have multi-modal)
        if passNumber >= 2 {
            let embeddings = embedTokens.weight[inputs]
            let scaledEmbeddings = embeddings.asType(.float32) * self.embedTokensScale
            h = scaledEmbeddings.asType(.bfloat16)
            
            // Debug embedding for token 818
            // if passNumber == 2 {
            //     print("\n[DEBUG] Pass 2 - Token 818 embedding check:")
            //     let tokenId = inputs[0, 0].item(Int.self)
            //     print("  Token ID from input: \(tokenId)")
            //     let rawEmbed818 = embedTokens.weight[818]
            //     print("  Raw embedding for token 818 first 5: \(rawEmbed818[0..<5])")
            //     print("  Scaled embedding first 5: \(h[0, 0, 0..<5])")
            //     print("  Embedding scale: \(self.embedTokensScale)")
            // }
        } else {
            h = embedTokens(inputs)
            
            // Debug: Check raw embedding values on first pass
            // if passNumber == 1 {
            //     print("\n[DEBUG] Pass 1 - Initial embedding check:")
            //     // Check each token in the sequence
            //     let tokenCount = min(5, inputs.shape[1])
            //     for i in 0..<tokenCount {
            //         let tokenId = inputs[0, i].item(Int.self)
            //         let rawEmbed = embedTokens.weight[tokenId]
            //         print("  Token [\(i)] ID \(tokenId) raw embed first 5: \(rawEmbed[0..<5])")
            //         print("  Token [\(i)] ID \(tokenId) scaled embed: \(h[0, i, 0..<5])")
            //     }
            //     print("  Embedding scale: \(self.embedTokensScale)")
            //     print("  Full h shape: \(h.shape)")
            //     print("  h[0,0] first 5 (token 0): \(h[0, 0, 0..<5])")
            //     print("  h[0,1] first 5 (token 1): \(h[0, 1, 0..<5])")
            // }
        }
        
        // print("[SWIFT] EMBED \(passNumber)")
        // let firstFive: MLXArray = h[0, 0, 0..<5]
        // print("  First 5 items from scaled_inputs: \(firstFive)")  
        
        
        // Get per-layer inputs if not provided
        var perLayerInputsComputed: MLXArray?
        if perLayerInputs == nil {
            perLayerInputsComputed = getPerLayerInputs(inputs)
        } else {
            perLayerInputsComputed = perLayerInputs
        }

        // Project per-layer inputs
        var projectedPerLayerInputs = perLayerInputsComputed
        if let perLayerInputs = projectedPerLayerInputs {
            projectedPerLayerInputs = projectPerLayerInputs(h, perLayerInputs: perLayerInputs)
        }
        // print(" projectedPerLayerInputs values: \(projectedPerLayerInputs?[0, 0, 0])")

        // Initialize cache position
        let cachePosition: MLXArray
        var pastSeenTokens = 0
        if !cache.isEmpty {
            pastSeenTokens = cache[0].offset
        }
        cachePosition = MLXArray(pastSeenTokens..<(pastSeenTokens + h.shape[1]))

        // Expand hidden states to multiple streams
        let h0 = h
        
        // Initialize position embeddings
        let positionIds = cachePosition.expandedDimensions(axis: 0)
        // RoPE needs shape info from the actual hidden states
        let positionEmbeddingsGlobal = ropeEmbedding(h0, positionIds: positionIds)
        let positionEmbeddingsLocal = ropeEmbeddingLocal(h0, positionIds: positionIds)
        
        // Debug RoPE embeddings
        // if passNumber == 1 {
        //     print("\n[RoPE Debug]")
        //     print("  Cache position: \(cachePosition)")
        //     print("  Position IDs shape: \(positionIds.shape)")
        //     print("  Position IDs: \(positionIds)")
        //     print("  Global RoPE base: \(config.ropeTheta)")
        //     print("  Local RoPE base: \(config.ropeLocalBaseFreq)")
        //     print("  Global RoPE cos shape: \(positionEmbeddingsGlobal.0.shape)")
        //     print("  Global RoPE cos first 5: \(positionEmbeddingsGlobal.0[0, 0, 0..<5])")
        //     print("  Global RoPE sin first 5: \(positionEmbeddingsGlobal.1[0, 0, 0..<5])")
        //     print("  Local RoPE cos first 5: \(positionEmbeddingsLocal.0[0, 0, 0..<5])")
        //     print("  Local RoPE sin first 5: \(positionEmbeddingsLocal.1[0, 0, 0..<5])")
        // }
        let targetMagnitude = MLX.mean(MLX.pow(h0, 2), axes: [-1], keepDims: true).sqrt()
        // Match Python's mx.finfo(h0.dtype).min exactly
        // This is the most negative value for the dtype
        let epsilonTensor = MLXArray(-Float.greatestFiniteMagnitude, dtype: h0.dtype)
        
        // print("     h0: \(h0)")
        // print("     h0 shape: \(h0.shape)")
        // print("     h0 dtype: \(h0.dtype)")
        // Debug initial embeddings and multi-stream setup
        // print("\n[Initial Setup]")
        // print("  h0 first 5 values: \(h0[0, 0, 0..<5])")
        // print("  targetMag: \(targetMagnitude[0, 0])")
        // print("  epsilonT: \(epsilonTensor)")

        // Create multiple streams - start with duplicates of h0
        var hList: [MLXArray] = []
        for _ in 0..<config.altupNumInputs {
            hList.append(h0)
        }
        
        // Apply AltUp projections to modify streams 1 through n-1
        for i in 1..<config.altupNumInputs {
            let altupProj = altupProjections[i - 1](hList[i])
            hList[i] = altupProj.asType(h0.dtype)
            let newMagnitude = MLX.mean(MLX.pow(hList[i], 2), axes: [-1], keepDims: true).sqrt()
            hList[i] = hList[i] * (targetMagnitude / MLX.maximum(newMagnitude, epsilonTensor))
        }
        
        // Stack streams: [num_streams, batch_size, seq_len, hidden_size]
        var hStacked = MLX.stacked(hList, axis: 0)
        
        // print("\n[Multi-stream Setup]")
        // for i in 0..<min(config.altupNumInputs, 4) {
        //     print("  Stream \(i) first 5 values: \(hStacked[i, 0, 0, 0..<5])")
        // }
        
        
        // Create masks for different layer types
        let fullMask: MLXArray?
        
        if mask == nil {
            let seqLen = hStacked.shape[2]
            
            // Create full causal mask matching Python's implementation
            // Python uses float mask with 0 for attended positions and -inf for masked positions
            
            // Create indices for rows and columns
            let rowIndices = MLXArray(0..<seqLen).expandedDimensions(axis: 1)  // [seqLen, 1]
            let colIndices = MLXArray(0..<seqLen).expandedDimensions(axis: 0)  // [1, seqLen]
            
            // Create boolean mask where True means "can attend"
            // For causal mask: can attend to positions <= current position
            let canAttend = rowIndices .>= colIndices  // [seqLen, seqLen]
            
            // Provide boolean causal mask (Python uses bool for global full attention)
            fullMask = canAttend.expandedDimensions(axis: 0)  // [1, seq_len, seq_len] – dtype = Bool
            
        } else {
            fullMask = mask
        }
        // print(" fullMask shape: \(fullMask?.shape ?? [])")
        // print(" fullMask dtype: \(fullMask?.dtype ?? .float32)")
        // print(" fullMask values: \(fullMask?[0, 0, 0..<5] ?? MLXArray([]))")

        // Process through transformer layers
        for i in 0..<layers.count {
            // Remove unused input state capture
            
            // Extract per-layer input for this layer
            let perLayerInput: MLXArray?
            if let perLayerInputs = projectedPerLayerInputs {
                // Extract slice for current layer: [..., layer_idx, :]
                perLayerInput = perLayerInputs[.ellipsis, i, 0...]
            } else {
                perLayerInput = nil
            }
            
            // Determine if this is a global attention layer
            let isGlobal = (config.layerTypes[i] == "full_attention")
            let isSliding = (config.layerTypes[i] == "sliding_attention")
            
            // Use the same float mask for all layers (0/-inf)
            let layerMask = fullMask
            
            if DebugLayers.shouldDebug(layer: i) {
                print("  [Layer 3] Sliding window - using float mask")
                print("  [Layer 3] Mask dtype: \(layerMask?.dtype ?? .float32)")
                print("  [Layer 4] Full attention - using float mask")
                print("  [Layer 4] Mask dtype: \(layerMask?.dtype ?? .float32)")
                print("  [Layer 4] isSliding: \(isSliding), isGlobal: \(isGlobal)")
                print("  [Layer 4] Layer type from config: \(config.layerTypes[i])")
            }
            
            let positionEmbeddings = isGlobal ? positionEmbeddingsGlobal : positionEmbeddingsLocal
            
            // Get window size for sliding attention layers
            let windowSize = isSliding ? config.slidingWindow : nil
            
            // Apply transformer layer
            hStacked = layers[i](
                hStacked,
                mask: layerMask,
                cache: &cache[i],
                perLayerInput: perLayerInput,
                caches: cache,
                cachePosition: cachePosition,
                positionEmbeddings: positionEmbeddings
            )
            // Debug layer outputs to track divergence
            if DebugLayers.shouldDebug(layer: i) {
                print("  Layer \(i) output: \(hStacked[0, 0, 0, 0..<5])")
            }
        
        }
        
        // Reduce multiple streams back to single output
        let finalTargetMagnitude = MLX.mean(MLX.pow(hStacked[0], 2), axes: [-1], keepDims: true).sqrt()
        // print(" finalTargetMagnitude: \(finalTargetMagnitude)")
        
        // if passNumber == 1 {
        //     print("\n[DEBUG] Final processing:")
        //     print("  hStacked[0] first 5 positions first 5 values:")
        //     for pos in 0..<min(5, hStacked.shape[2]) {
        //         print("    Pos \(pos): \(hStacked[0, 0, pos, 0..<5])")
        //     }
        // }

        // Apply unembed projections
        var hListFinal = [hStacked[0]]
        for i in 0..<(config.altupNumInputs - 1) {
            let altupUnembedProj = altupUnembedProjections[i](hStacked[i + 1])
            let newMagnitude = MLX.mean(MLX.pow(altupUnembedProj, 2), axes: [-1], keepDims: true).sqrt()
            let normalizedProj = altupUnembedProj * (finalTargetMagnitude / MLX.maximum(newMagnitude, epsilonTensor))
            hListFinal.append(normalizedProj.asType(h0.dtype))
        }
        
        // Average all streams
        let hFinal = MLX.mean(MLX.stacked(hListFinal, axis: 0), axis: 0)
        
        // if passNumber == 1 {
        //     print("  hFinal (after averaging) first 5 positions first 5 values:")
        //     for pos in 0..<min(5, hFinal.shape[1]) {
        //         print("    Pos \(pos): \(hFinal[0, pos, 0..<5])")
        //     }
        // }
        
        // Apply final normalization
        let output = norm(hFinal)
        
        // if passNumber == 1 {
        //     print("  Final normalized output first 5 positions first 5 values:")
        //     for pos in 0..<min(5, output.shape[1]) {
        //         print("    Pos \(pos): \(output[0, pos, 0..<5])")
        //     }
        // }
        
        // print("=== OUTPUT PASS \(self.passNumber) ===")
        // print("output shape: \(output.shape)")
        // print(output)
        // print("=== OUTPUT PASS \(self.passNumber) ===\n\n")

        // Removed early exit to see full output

        return output
    }
    
    /// Get per-layer inputs from token IDs
    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        // Mask for valid per-layer vocabulary
        let mask = inputIds .< config.vocabSizePerLayerInput
        let tokens = MLX.where(mask, inputIds, MLXArray(0, dtype: inputIds.dtype))
        
        // Get embeddings (already scaled internally by PerLayerEmbedding)
        let result = embedTokensPerLayer(tokens)
        
        // Reshape to [batch_size, seq_len, num_layers, hidden_size_per_layer]
        let shape = inputIds.shape + [config.numHiddenLayers, config.hiddenSizePerLayerInput]
        return result.reshaped(shape)
    }
    
    /// Project per-layer inputs
    private func projectPerLayerInputs(_ inputsEmbeds: MLXArray, perLayerInputs: MLXArray) -> MLXArray {
        // Project input embeddings to per-layer space
        var perLayerProjection = perLayerModelProjection(inputsEmbeds)
        perLayerProjection = perLayerProjection * perLayerProjectionScale
        
        // Reshape to [batch_size, seq_len, num_layers, hidden_size_per_layer]
        let shape = Array(inputsEmbeds.shape.dropLast()) + [config.numHiddenLayers, config.hiddenSizePerLayerInput]
        perLayerProjection = perLayerProjection.reshaped(shape)
        
        // Apply normalization
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)
        
        // Add per-layer inputs if provided
        var result = perLayerProjection
        if perLayerInputs.shape == perLayerProjection.shape {
            result = (perLayerProjection + perLayerInputs) * perLayerInputScale
        }
        
        return result
    }
}

/// Language model wrapper with LM head for text generation
public class Gemma3nLanguageModel: Module {
    @ModuleInfo(key: "model") public var model: Gemma3nModel
    @ModuleInfo(key: "lm_head") public var lmHead: Linear
    
    public let config: TextConfig
    public let textVocabSize: Int
    public let finalLogitSoftcapping: Float
    
    public init(config: TextConfig) {
        self.config = config
        self.textVocabSize = config.vocabSizePerLayerInput  // 262144
        self.finalLogitSoftcapping = Float(config.finalLogitSoftcapping)
        
        // Initialize model
        self._model = ModuleInfo(
            wrappedValue: Gemma3nModel(config: config),
            key: "model"
        )
        
        // Initialize language modeling head
        self._lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head"
        )
        
        super.init()
    }
    
    /// Forward pass through language model
    /// - Parameters:
    ///   - inputs: Input token IDs
    ///   - mask: Attention mask (optional)
    ///   - cache: KV cache (optional)
    /// - Returns: Logits over vocabulary [batch_size, sequence_length, vocab_size]
    
    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: inout [KVCache]
    ) -> MLXArray {
        
        // Get model output
        let out = model(inputs, mask: mask, cache: &cache)
        
        // Apply LM head
        var logits = lmHead(out)
        
        // Apply logit soft-capping if configured
        if finalLogitSoftcapping > 0 {
            logits = MLX.tanh(logits / finalLogitSoftcapping) * finalLogitSoftcapping
        }
        
        return logits
    }
}