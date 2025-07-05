import Foundation
import MLX
import MLXNN

/// Gemma3n Transformer Block - Core building block of the model
public class Gemma3nTransformerBlock: Module {
    public let layerIdx: Int
    public let config: TextConfig
    
    // AltUp components
    @ModuleInfo(key: "altup") public var altUp: AltUp
    
    // Normalization layers (6 total per block)
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") public var preFeedforwardLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFeedforwardLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_per_layer_input_norm") public var postPerLayerInputNorm: Gemma3nRMSNorm
    
    // LAUREL augmented residual
    @ModuleInfo(key: "laurel") public var laurel: LAURELBlock
    
    // Multi-head attention
    @ModuleInfo(key: "self_attn") public var selfAttention: Gemma3nAttention
    
    // MLP feedforward
    @ModuleInfo(key: "mlp") public var mlp: Gemma3nMLP
    
    // Per-layer embeddings
    @ModuleInfo(key: "per_layer_projection") public var perLayerProjection: Linear?
    @ModuleInfo(key: "per_layer_input_gate") public var perLayerInputGate: Linear?
    
    // Layer configuration
    public var layerType: String {
        // Get layer type from config
        if layerIdx < config.layerTypes.count {
            return config.layerTypes[layerIdx]
        }
        // Default to sliding attention
        return "sliding_attention"
    }
    
    public var slidingWindow: Int? {
        return layerType == "sliding_attention" ? config.slidingWindow : nil
    }
    
    public init(config: TextConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        
        // Initialize AltUp using config values
        let altUpInstance = AltUp(
            hiddenSize: config.hiddenSize,
            numInputs: config.altupNumInputs,
            activeIdx: config.altupActiveIdx,
            coefClip: Float(config.altupCoefClip),
            eps: Float(config.rmsNormEps),
            layerIdx: layerIdx
        )
        self._altUp = ModuleInfo(
            wrappedValue: altUpInstance,
            key: "altup"
        )
        
        // Initialize normalization layers
        self._inputLayerNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps)
            ),
            key: "input_layernorm"
        )
        
        self._postAttentionLayerNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps)
            ),
            key: "post_attention_layernorm"
        )
        
        self._preFeedforwardLayerNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps)
            ),
            key: "pre_feedforward_layernorm"
        )
        
        self._postFeedforwardLayerNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps)
            ),
            key: "post_feedforward_layernorm"
        )
        
        self._postPerLayerInputNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps)
            ),
            key: "post_per_layer_input_norm"
        )
        
        // Initialize LAUREL
        let laurelBlock = LAURELBlock(
            hiddenSize: config.hiddenSize,
            laurelRank: config.laurelRank,
            eps: Float(config.rmsNormEps)
        )
        laurelBlock.layerIdx = layerIdx  // Set layer index for debugging
        self._laurel = ModuleInfo(
            wrappedValue: laurelBlock,
            key: "laurel"
        )
        
        // Initialize attention
        // Determine layer type inline since we can't use computed property before super.init
        let layerTypeForInit = layerIdx < config.layerTypes.count ? config.layerTypes[layerIdx] : "sliding_attention"
        self._selfAttention = ModuleInfo(
            wrappedValue: Gemma3nAttention(
                hiddenSize: config.hiddenSize,
                numHeads: config.numAttentionHeads,
                numKVHeads: config.numKeyValueHeads,
                headDim: config.headDim,
                eps: Float(config.rmsNormEps),
                attnLogitSoftcapping: Float(config.attnLogitSoftcapping),
                layerIdx: layerIdx,
                numHiddenLayers: config.numHiddenLayers,
                numKvSharedLayers: config.numKvSharedLayers,
                layerType: layerTypeForInit
            ),
            key: "self_attn"
        )
        
        // Initialize MLP with layer-specific sparsity
        self._mlp = ModuleInfo(
            wrappedValue: Gemma3nMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: layerIdx < config.intermediateSize.count ? config.intermediateSize[layerIdx] : 16384,
                layerIdx: layerIdx,
                activationSparsityPattern: config.activationSparsityPattern.map { Float($0) }
            ),
            key: "mlp"
        )
        
        // Initialize per-layer embeddings if needed
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerProjection = ModuleInfo(
                wrappedValue: Linear(
                    config.hiddenSizePerLayerInput,
                    config.hiddenSize,
                    bias: false
                ),
                key: "per_layer_projection"
            )
            
            self._perLayerInputGate = ModuleInfo(
                wrappedValue: Linear(
                    config.hiddenSize,
                    config.hiddenSizePerLayerInput,
                    bias: false
                ),
                key: "per_layer_input_gate"
            )
        } else {
            self._perLayerProjection = ModuleInfo(wrappedValue: nil, key: "per_layer_projection")
            self._perLayerInputGate = ModuleInfo(wrappedValue: nil, key: "per_layer_input_gate")
        }
        
        super.init()
    }
    
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        cache: inout KVCache,
        perLayerInput: MLXArray? = nil,
        caches: [KVCache],
        cachePosition: MLXArray? = nil,
        positionEmbeddings: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        // hiddenStates shape: [num_streams, batch_size, seq_len, hidden_size]
        
        // Debug input to layer 0 and layer 1
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("\n[Layer 0 Input]")
            for i in 0..<min(hiddenStates.shape[0], 4) {
                print("  Stream \(i) first 5 values: \(hiddenStates[i, 0, 0, 0..<5])")
            }
        }
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("\n[Layer 1 Input - All Streams]")
            for i in 0..<min(hiddenStates.shape[0], 4) {
                print("  Stream \(i) first 5 values: \(hiddenStates[i, 0, 0, 0..<5])")
            }
        }
        
        // 1. AltUp predict phase
        let predictions = altUp.predict(hiddenStates)
        
        // Extract the active stream (configurable, default 0)
        let activePrediction = predictions[config.altupActiveIdx]
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  Active prediction first 5: \(activePrediction[0, 0, 0..<5])")
        }
        
        // 2. Pre-attention norm on active prediction
        let activePredictionNormed = inputLayerNorm(activePrediction)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  After norm first 5: \(activePredictionNormed[0, 0, 0..<5])")
        }
        
        // 3. LAUREL transformation
        let laurelOutput = laurel(activePredictionNormed)
        
        // Debug Layer 1 and 4 in detail to track divergence
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("\n[Layer \(layerIdx) DETAILED DEBUG - Finding divergence]")
            print("  1. Input (activePrediction): \(activePrediction[0, 0, 0..<5])")
            print("  2. After input norm: \(activePredictionNormed[0, 0, 0..<5])")
            print("  3. LAUREL output: \(laurelOutput[0, 0, 0..<5])")
            
            // Add attention type debug
            print("  Layer type: \(layerType)")
            print("  Sliding window: \(slidingWindow ?? -1)")
            
            // Debug position embeddings
            if let posEmb = positionEmbeddings {
                print("  RoPE cos shape: \(posEmb.0.shape)")
                print("  RoPE cos [0,0,0:5]: \(posEmb.0[0, 0, 0..<5])")
                print("  RoPE sin [0,0,0:5]: \(posEmb.1[0, 0, 0..<5])")
                // Only access position 15 if it exists
                if posEmb.0.shape[1] > 15 {
                    print("  RoPE cos [0,15,0:5]: \(posEmb.0[0, 15, 0..<5])") // Last position
                    print("  RoPE sin [0,15,0:5]: \(posEmb.1[0, 15, 0..<5])")
                }
                
                // Extra debug for layer 4 to verify global RoPE
                if DebugLayers.shouldDebug(layer: layerIdx) {
                    print("  [Layer 4] Verifying global RoPE is being used")
                    print("  [Layer 4] RoPE base should be 1000000 for global attention")
                }
            }
        }
        
        // 4. Self-attention
        // ------------------------------------------------------------------
        // Handle KV sharing (last N layers reuse the KV from an earlier layer)
        var sharedKeys: MLXArray? = nil
        var sharedValues: MLXArray? = nil
        if selfAttention.isKVSharedLayer,
           let sharedIdx = selfAttention.kvSharedLayerIndex,
           sharedIdx < caches.count {
            let kvState = caches[sharedIdx].state
            if kvState.count == 2 {
                // KVCache stores as [B, KVHeads, Seq, D]; transform to [B, Seq, KVHeads, D]
                sharedKeys   = kvState[0].transposed(0, 2, 1, 3)
                sharedValues = kvState[1].transposed(0, 2, 1, 3)
            }
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("  [Layer \(layerIdx)] Using shared KV from layer \(sharedIdx). sharedKeys shape: \(sharedKeys?.shape ?? [])")
            }
        }
        let attentionOutput = selfAttention(
            activePredictionNormed,
            rope: nil,  // RoPE is pre-computed and passed via positionEmbeddings
            mask: mask,
            windowSize: slidingWindow,  // Pass sliding window size for sliding attention layers
            cache: &cache,
            positionEmbeddings: positionEmbeddings,
            sharedKeys: sharedKeys,
            sharedValues: sharedValues,
            cachePosition: cachePosition
        )
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  4. Attention output: \(attentionOutput[0, 0, 0..<5])")
        }
        
        // 5. Post-attention norm – debug scale weight & output for critical layers
        let attnNormed = postAttentionLayerNorm(attentionOutput)
        if DebugLayers.shouldDebug(layer: layerIdx) {
            let wSlice = postAttentionLayerNorm.weight[0..<5]
            print("  5. Post-attention norm: \(attnNormed[0, 0, 0..<5])")
            print("     [Swift] post_attn_layernorm.weight first 5: \(wSlice)")
        }
        
        // 6. Attention residual + gating
        let attnGated = activePrediction + attnNormed
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  6. Attention residual (activePred + attnNormed): \(attnGated[0, 0, 0..<5])")
        }
        
        // 7. Combine with LAUREL and normalize by sqrt(2)
        // Ensure sqrt(2) has same dtype as tensors
        let sqrt2 = MLXArray(sqrt(2.0), dtype: attnGated.dtype)
        let attnLaurel = (attnGated + laurelOutput) / sqrt2
        
        // 8. Pre-feedforward norm
        let attnNorm = preFeedforwardLayerNorm(attnLaurel)
        
        // 9. MLP
        let attnFFW = mlp(attnNorm)
        
        // 10. Post-feedforward norm
        let attnFFWNorm = postFeedforwardLayerNorm(attnFFW)
        
        // 11. Feedforward residual
        let attnFFWLaurelGated = attnLaurel + attnFFWNorm
        
        // 12. AltUp correct phase
        let correctedPredictions = altUp.correct(predictions, activated: attnFFWLaurelGated)
        
        // 13. Per-layer input processing
        if let perLayerInput = perLayerInput,
           let perLayerGate = perLayerInputGate,
           let perLayerProj = perLayerProjection {
            // Extract first prediction for per-layer processing
            var firstPrediction = correctedPredictions[config.altupActiveIdx]
            
            if config.altupCorrectScale {
                firstPrediction = altUp.scaleOutput(firstPrediction)
            }
            
            // Work in Float32 for numerical parity with Python
            var fp32 = firstPrediction.asType(.float32)
            var perLayerInF32 = perLayerInput.asType(.float32)

            // 1. Gate + GELU  (no transpose – Linear.weight is already (in,out))
            fp32 = perLayerGate(fp32)
            fp32 = MLXNN.geluApproximate(fp32)

            // 2. Element-wise multiply with per-layer input
            fp32 = fp32 * perLayerInF32

            // 3. Project back to hidden size (no transpose)
            fp32 = perLayerProj(fp32)

            // 4. RMS-norm
            fp32 = postPerLayerInputNorm(fp32)

            // Cast back to original dtype once
            firstPrediction = fp32.asType(firstPrediction.dtype)

            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("  Per-layer branch output first 5: \(firstPrediction[0,0,0..<5])")
            }
            
            // Update only non-active streams with the per-layer contribution
            // NOTE: Stream 0 keeps its original value from AltUp.correct, matching Python behavior
            for i in 1..<correctedPredictions.shape[0] {
                correctedPredictions[i] = correctedPredictions[i] + firstPrediction
            }
        }
        
        return correctedPredictions
    }
}

