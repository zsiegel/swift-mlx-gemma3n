import Foundation
import MLX
import MLXNN

/// Multi-head attention with Grouped Query Attention (GQA) for Gemma3n
public class Gemma3nAttention: Module {
    
    // Configuration
    public let hiddenSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let repeats: Int  // numHeads / numKVHeads
    public let eps: Float
    public let attnLogitSoftcapping: Float
    
    // KV Sharing properties
    public let layerIdx: Int
    public let isKVSharedLayer: Bool
    public let kvSharedLayerIndex: Int?
    
    // Scale factor (Gemma3n uses 1.0, not 1/sqrt(headDim))
    public let scale: Float = 1.0
    
    // Projection layers
    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    
    // Pre-RoPE normalization layers
    @ModuleInfo(key: "q_norm") public var qNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: Gemma3nRMSNorm
    public var vNorm: Gemma3nRMSNorm  // No scale weights for values, not saved
    
    /// Initialize Gemma3n attention layer
    /// - Parameters:
    ///   - hiddenSize: Hidden dimension size
    ///   - numHeads: Number of query heads
    ///   - numKVHeads: Number of key/value heads (for GQA)
    ///   - headDim: Optional head dimension (calculated if not provided)
    ///   - eps: Epsilon for RMS normalization
    ///   - attnLogitSoftcapping: Attention logit softcapping value (0 to disable)
    ///   - layerIdx: Layer index for KV sharing determination
    ///   - numHiddenLayers: Total number of hidden layers
    ///   - numKvSharedLayers: Number of layers that share KV
    ///   - layerType: Type of attention layer ("full_attention" or "sliding_attention")
    public init(
        hiddenSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int? = nil,
        eps: Float = Float(1e-6),
        attnLogitSoftcapping: Float = 0.0,
        layerIdx: Int = 0,
        numHiddenLayers: Int = 30,
        numKvSharedLayers: Int = 10,
        layerType: String = "sliding_attention"
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        // If headDim is provided, use it; otherwise calculate from hiddenSize/numHeads
        if let headDim = headDim {
            self.headDim = headDim
        } else {
            assert(hiddenSize % numHeads == 0, "Hidden size must be divisible by number of heads when headDim is not provided")
            self.headDim = hiddenSize / numHeads
        }
        self.repeats = numHeads / numKVHeads
        self.eps = eps
        self.attnLogitSoftcapping = attnLogitSoftcapping
        self.layerIdx = layerIdx
        
        // Determine if this layer shares KV
        let firstKvSharedLayerIdx = numHiddenLayers - numKvSharedLayers
        self.isKVSharedLayer = layerIdx >= firstKvSharedLayerIdx
        
        // Compute the layer index from which shared KV cache values will be retrieved
        if !isKVSharedLayer {
            self.kvSharedLayerIndex = nil
        } else if layerType == "sliding_attention" {
            // The last layer that computes local sliding attention is always 2 before sharing starts
            self.kvSharedLayerIndex = firstKvSharedLayerIdx - 2
        } else {
            // The last layer before sharing starts is always the last that computes global attention layer
            self.kvSharedLayerIndex = firstKvSharedLayerIdx - 1
        }
        
        // Verify configuration
        assert(numHeads % numKVHeads == 0, "Number of heads must be divisible by number of KV heads")
        
        // Initialize projections
        // Q projection: hidden_size -> num_heads * head_dim
        self._qProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numHeads * self.headDim, bias: false),
            key: "q_proj"
        )
        
        // K, V projections: hidden_size -> num_kv_heads * head_dim
        self._kProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numKVHeads * self.headDim, bias: false),
            key: "k_proj"
        )
        self._vProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numKVHeads * self.headDim, bias: false),
            key: "v_proj"
        )
        
        // Output projection: num_heads * head_dim -> hidden_size
        self._oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * self.headDim, hiddenSize, bias: false),
            key: "o_proj"
        )
        
        // Initialize normalization layers
        // According to Python implementation, normalization dimension is headDim
        // and it's applied after reshaping to [..., headDim]
        self._qNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: self.headDim,
                eps: eps,
                scaleShift: 0.0,
                withScale: true
            ),
            key: "q_norm"
        )
        
        self._kNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: self.headDim,
                eps: eps,
                scaleShift: 0.0,
                withScale: true
            ),
            key: "k_norm"
        )
        
        // Value normalization without scale weights
        self.vNorm = Gemma3nRMSNorm(
            dimensions: self.headDim,
            eps: eps,
            scaleShift: 0.0,
            withScale: false
        )
        
        super.init()
    }
    
    /// Create sliding window mask
    /// - Parameters:
    ///   - seqLen: Sequence length
    ///   - windowSize: Size of the sliding window
    /// - Returns: Mask tensor where True values are attended to
    private func makeSlidingWindowMask(seqLen: Int, windowSize: Int) -> MLXArray {
        // Create position indices
        let positions = MLXArray(0..<seqLen)
        let rowPositions = positions.expandedDimensions(axis: 1)  // [seqLen, 1]
        let colPositions = positions.expandedDimensions(axis: 0)  // [1, seqLen]
        
        // Create causal sliding window mask:
        // - Must be causal (can't attend to future)
        // - Can only attend to windowSize positions in the past
        let causalMask = rowPositions .>= colPositions  // Can't look ahead
        let windowMask = (rowPositions - colPositions) .<= MLXArray(windowSize - 1)  // Within window
        
        // Combine both constraints
        let slidingWindowMask = causalMask .&& windowMask
        
        return slidingWindowMask
    }
    
    /// Apply attention with optional masking
    /// - Parameters:
    ///   - queries: Query tensor [batch, seqLen, numHeads, headDim]
    ///   - keys: Key tensor [batch, seqLen, numKVHeads, headDim]
    ///   - values: Value tensor [batch, seqLen, numKVHeads, headDim]
    ///   - mask: Optional attention mask
    ///   - windowSize: Optional sliding window size (nil for full attention)
    /// - Returns: Attention output [batch, seqLen, numHeads, headDim]
    private func applyAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXArray? = nil,
        windowSize: Int? = nil
    ) -> MLXArray {
        let batchSize = queries.shape[0]
        let seqLen = queries.shape[1]
        
        // [batch, seqLen, numHeads, headDim] -> [batch, numHeads, seqLen, headDim]
        let q = queries.transposed(0, 2, 1, 3)
        var k = keys.transposed(0, 2, 1, 3)     // [batch, numKVHeads, seqLen, headDim]
        var v = values.transposed(0, 2, 1, 3)   // [batch, numKVHeads, seqLen, headDim]
        
        // Repeat KV heads AFTER transpose (matching Python)
        if repeats > 1 {
            k = MLX.repeated(k, count: repeats, axis: 1)  // [batch, numHeads, seqLen, headDim]
            v = MLX.repeated(v, count: repeats, axis: 1)  // [batch, numHeads, seqLen, headDim]
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 4 DEBUG] K after repeat first 5: \(k[0, 0, 0, 0..<5])")
                print("    [Layer 4 DEBUG] V after repeat first 5: \(v[0, 0, 0, 0..<5])")
            }
        }
        
        // Remove debug - using focused debug in transformer block
        
        // Make contiguous copies of Q and K to avoid potential strides-related
        // issues in the low-level einsum implementation (observed -Inf outputs
        // when operating on non-contiguous tensors).
        let qF32 = (q.asType(.float32) + MLXArray(0)).asType(.float32)
        let kF32 = (k.asType(.float32) + MLXArray(0)).asType(.float32)
        
        // Work around MLX matmul bug: compute logits via explicit einsum in fp32
        // q  : [B, H, L_q, D], k : [B, H, L_k, D]
        // einsum pattern "bhld,bhmd->bhlm" gives [B, H, L_q, L_k]
        var scores = einsum("bhld,bhmd->bhlm", qF32, kF32) * scale  // fp32 result
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer \(layerIdx) Attention Swift] Q shape: \(q.shape), K shape: \(k.shape)")
            print("    [Layer \(layerIdx) Attention Swift] Scores shape: \(scores.shape)")
            print("    [Layer \(layerIdx) Attention Swift] Scale: \(scale)")
            print("    [Layer \(layerIdx) Attention Swift] Is sliding: \(windowSize != nil)")
            print("    [Layer \(layerIdx) Attention Swift] Q first 5 values: \(q.flattened()[0..<5])")
            print("    [Layer \(layerIdx) Attention Swift] K first 5 values: \(k.flattened()[0..<5])")
            print("    [Layer \(layerIdx) Attention Swift] Scores first 5 values: \(scores.flattened()[0..<5])")
        }
        
        // Removed debug - focusing on layer inputs
        
        // Remove debug - using focused debug in transformer block
        
        // Apply attention logit softcapping if enabled
        if attnLogitSoftcapping > 0 {
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 0 Attention] Before softcapping first 5 scores: \(scores[0, 0, 0, 0..<5])")
            }
            // Convert to Float32 for precision
            let scoresF32 = scores.asType(.float32)
            // Divide by softcapping value
            let normalized = scoresF32 / attnLogitSoftcapping
            // Apply tanh
            let tanhed = MLX.tanh(normalized)
            // Multiply back by softcapping value
            scores = (tanhed * attnLogitSoftcapping).asType(queries.dtype)
            
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 0 Attention] After softcapping first 5 scores: \(scores[0, 0, 0, 0..<5])")
            }
        }
        
        // Apply sliding window mask if specified and needed
        if let windowSize = windowSize, windowSize < seqLen {
            // Remove debug - using focused debug in transformer block
            
            // Only apply sliding window if it actually restricts attention
            let slidingMask = makeSlidingWindowMask(seqLen: seqLen, windowSize: windowSize)
            // Expand mask to match scores shape
            let expandedMask = slidingMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            let broadcastMask = broadcast(expandedMask, to: [batchSize, numHeads, seqLen, seqLen])
            
            // Apply mask by setting non-attended positions to -inf to match Python
            scores = MLX.where(broadcastMask, scores, full(scores.shape, values: -Float.infinity))
            
            // Remove debug - using focused debug in transformer block
        }
        
        // Apply additional mask if provided
        if let mask = mask {
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer \(layerIdx) Attention Swift] Mask provided: true")
                print("    [Layer \(layerIdx) Attention Swift] Mask shape: \(mask.shape)")
                print("    [Layer \(layerIdx) Attention Swift] Mask dtype: \(mask.dtype)")
                if mask.shape.count >= 2 {
                    print("    [Layer \(layerIdx) Attention Swift] Mask first 5x5:")
                    let maskSlice = mask.shape.count == 2 ? mask[0..<5, 0..<5] : mask[0, 0..<5, 0..<5]
                    print(maskSlice)
                }
            }
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 0 Attention] Mask shape: \(mask.shape)")
                print("    [Layer 0 Attention] Mask dtype: \(mask.dtype)")
                print("    [Layer 0 Attention] Mask first row: \(mask[0, 0, 0...])")
                print("    [Layer 0 Attention] Scores before mask first 5: \(scores[0, 0, 0, 0..<5])")
            }
            // Slice mask to match keys shape for cache handling
            let causalMask = mask[0..., 0..<k.shape[2]]
            
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer \(layerIdx) Attention Swift] causalMask shape: \(causalMask.shape)")
                print("    [Layer \(layerIdx) Attention Swift] causalMask dtype: \(causalMask.dtype)")
                print("    [Layer \(layerIdx) Attention Swift] causalMask first 5 values: \(causalMask.flattened()[0..<5])")
                print("    [Layer \(layerIdx) Attention Swift] Scores before mask: \(scores[0, 0, 0, 0..<5])")
            }
            
            // Python distinguishes two cases:
            // 1) Boolean causal mask (True = attend) ➜ replace False positions with -inf
            // 2) Pre-computed float mask (0 / -inf) ➜ add directly.

            if causalMask.dtype == .bool {
                var boolMask = causalMask
                if boolMask.shape.count == 3 {      // add heads dim if needed
                    boolMask = boolMask.expandedDimensions(axis: 1)
                }
                // True  → 0.0  ,  False → -inf  (match Python)
                let allowed  = MLXArray(0.0, dtype: .float32)
                let blocked  = MLXArray(-Float.infinity, dtype: .float32)
                let fMask    = MLX.where(boolMask, allowed, blocked)
                let broadcastMask = broadcast(fMask, to: scores.shape)
                
                if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
                    print("    [L4] fMask sample:", fMask[0,0,0,0..<10])
                }
                
                if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
                    print("    [L4] scores pre-mask:", scores[0,0,0,0..<10])
                }
                scores = scores + broadcastMask
                if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
                    print("    [L4] scores post-mask:", scores[0,0,0,0..<10])
                }
            } else {
                // Float mask path (already contains 0 / -inf)
                var maskToApply = causalMask
                if maskToApply.dtype != scores.dtype {
                    maskToApply = maskToApply.asType(scores.dtype)
                }
                scores = scores + maskToApply
            }
            
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer \(layerIdx) Attention Swift] Scores after mask: \(scores[0, 0, 0, 0..<5])")
            }
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 0 Attention] Scores after mask first 5: \(scores[0, 0, 0, 0..<5])")
            }

            // --- Debug causalMask meta for Layer 4 while 'causalMask' is in scope ---
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [L4 SWIFT] mask.dtype  = \(causalMask.dtype)")
                print("    [L4 SWIFT] mask.shape  = \(causalMask.shape)")

                // count True vs False
                let nTrue  = MLX.sum(causalMask.asType(.int32)).item(Int.self)
                let nTotal = causalMask.size
                print("    [L4 SWIFT] mask True ratio: \(Float(nTrue)/Float(nTotal))")

                // visual 6×6 corner to spot shifts
                let corner = causalMask.shape.count == 2
                   ? causalMask[0..<6, 0..<6]
                   : causalMask[0, 0..<6, 0..<6]
                print("    [L4 SWIFT] mask[0..5,0..5] =\n\(corner)")
            }
        }
        
        // Removed debug - focusing on layer inputs
        
        // Compute attention weights (scores already fp32)
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer 4 Attention Swift] scores fp32 (pre-mask) row0[:10] = \(scores[0,0,0,0..<10])")
        }

        let weights = softmax(scores, axis: -1).asType(queries.dtype)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer \(layerIdx) Attention Swift] Weights shape: \(weights.shape)")
            print("    [Layer \(layerIdx) Attention Swift] Weights first 5: \(weights[0, 0, 0, 0..<5])")
            
            // Extra debug for layer 4 specifically
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 4 Attention Swift] Q shape: \(q.shape), K shape: \(k.shape)")
                print("    [Layer 4 Attention Swift] Q first 5: \(q[0, 0, 0, 0..<5])")
                print("    [Layer 4 Attention Swift] K first 5: \(k[0, 0, 0, 0..<5])")
                print("    [Layer 4 Attention Swift] Raw scores first 5: \(scores[0, 0, 0, 0..<5])")
                print("    [Layer 4 Attention Swift] Mask applied? \(mask != nil)")
                if mask != nil {
                    print("    [Layer 4 Attention Swift] Mask shape: \(mask!.shape)")
                    print("    [Layer 4 Attention Swift] Mask dtype: \(mask!.dtype)")
                    print("    [Layer 4 Attention Swift] Mask first row: \(mask![0, 0, 0...])")
                }
            }
        }
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer 0 Attention] Weights first 5: \(weights[0, 0, 0, 0..<5])")
            print("    [Layer 0 Attention] V first 5 values: \(v[0, 0, 0, 0..<5])")
        }
        
        // Apply attention weights to values in fp32 for numerical parity with Python
        // 1. Up-cast operands to Float32 so accumulation happens in high precision.
        // 2. Perform the matmul.
        // 3. Cast back to the original dtype (bf16) for subsequent layers.
        // Ensure V is materialised in contiguous memory by forcing a copy
        let vF32 = (v + MLXArray(0)).asType(.float32)
        let outputF32 = matmul(weights.asType(.float32), vF32)
        if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
            print("    [L4] matmul result dtype:", outputF32.dtype)
        }
        var output = outputF32.asType(queries.dtype)

        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer 0 Attention] Raw output first 5: \(output[0, 0, 0, 0..<5])")
            print("    [Layer 4 Attention] Raw output first 5: \(output[0, 0, 0, 0..<5])")
            print("    [Layer 4 DEBUG] Raw output FP32 first 5: \(outputF32[0,0,0,0..<5])")
        }
        
        // 3×3 patch so we can recompute manually
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("[L4] weights patch:", weights[0,0,0,0..<3])
            print("[L4] V patch:", v[0,0,0..<3, 0..<3])
        }

        // recompute row 0,col 0 "by hand"
        let hand = MLX.sum(weights[0,0,0,0..<16] * v[0,0,0..<16,0])
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("[L4] hand-calc(0,0):", hand)
            print("[L4] matmul(0,0):", outputF32[0,0,0,0])
        }
        
        // Transpose back
        output = output.transposed(0, 2, 1, 3)  // [batch, seqLen, numHeads, headDim]
        
        let checksum = MLX.mean(scores).item(Float.self)
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [L4 SWIFT] scores mean pre-mask = \(checksum)")
        }

        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [L4 SWIFT] scores fp32 (post-mask) 0,0 row[:10] = \(scores[0,0,0,0..<10])")
        }
        
        if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
            let w = weights[0,0,0,0..<10]
            print("    [L4] weights first row:", w)
            print("    [L4] weights sum:", MLX.sum(w).item(Float.self))
        }
        
        if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
            // take head-0, row-0, first 4 columns so we can compare with Python
            let dot = matmul(q[0,0,0,0..<256].asType(.float32),
                             k[0,0,0..<4,0..<256].swappedAxes(0,1).asType(.float32))
            print("    [L4] manual dot row0 cols0-3:", dot[0..<4])
            // also dump the first 6 raw elements of q & k that participate
            print("    [L4] q[0,:6]:", q[0,0,0,0..<6])
            print("    [L4] k[0,:6]:", k[0,0,0,0..<6])
        }
        
        if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
            // Up-cast both operands and recompute one scalar in fp32
            let wF32 = weights.asType(.float32)
            let vF32 = v.asType(.float32)
            let ref = MLX.sum(wF32[0,0,0,0..<16] * vF32[0,0,0..<16,0])
            print("    [L4] fp32 re-dot(0,0):", ref)
        }
        
        return output
    }
    
    /// Forward pass through attention layer
    /// - Parameters:
    ///   - x: Input tensor [batch, seqLen, hiddenSize]
    ///   - rope: Rotary position embedding to apply
    ///   - mask: Optional attention mask
    ///   - windowSize: Optional sliding window size (nil for full attention)
    ///   - cache: Optional KV cache
    ///   - positionEmbeddings: Pre-computed position embeddings (cos, sin)
    ///   - sharedKeys: Optional shared keys from another layer
    ///   - sharedValues: Optional shared values from another layer
    /// - Returns: Attention output [batch, seqLen, hiddenSize]
    public func callAsFunction(
        _ x: MLXArray,
        rope: RotaryPositionEmbedding? = nil,
        mask: MLXArray? = nil,
        windowSize: Int? = nil,
        cache: inout KVCache,
        positionEmbeddings: (MLXArray, MLXArray)? = nil,
        sharedKeys: MLXArray? = nil,
        sharedValues: MLXArray? = nil,
        cachePosition: MLXArray? = nil
    ) -> MLXArray {
        // Remove debug - using focused debug in transformer block
        
        let originalDtype = x.dtype
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]
        
        // Project to Q – promote both input and weight to fp32 so matmul accumulates in high precision
        let xF32 = x.asType(.float32)
        // Use the existing Linear layer for projection (it internally handles weight.T)
        var queries = qProj(xF32)
        queries = queries.reshaped([batchSize, seqLen, numHeads, headDim])
        queries = queries.asType(x.dtype) // back to original dtype (bfloat16)
        
        // Debug: print Q pre-norm and qNorm weights for layer 4
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer 4 DEBUG] Q pre-norm first 5: \(queries[0, 0, 0, 0..<5])")
            print("    [Layer 4 DEBUG] qNorm.weight first 5: \(qNorm.weight[0..<5])")
            // Also log raw weight slices of Q and K projection matrices for orientation check
            print("    [Layer 4 DEBUG] Q weight slice [0,0..<5]: \(qProj.weight[0, 0..<5])")
            print("    [Layer 4 DEBUG] K weight slice [0,0..<5]: \(kProj.weight[0, 0..<5])")
        }
        
        // For K,V: Check if we should use shared values
        var keys: MLXArray
        var values: MLXArray
        
        if isKVSharedLayer, let sharedKeys = sharedKeys, let sharedValues = sharedValues {
            // Use shared K,V from another layer
            keys = sharedKeys
            values = sharedValues
        } else {
            // Compute K,V normally
            keys = kProj(x)
            values = vProj(x)
            keys = keys.reshaped([batchSize, seqLen, numKVHeads, headDim])
            values = values.reshaped([batchSize, seqLen, numKVHeads, headDim])
            
            // New debug: inspect raw V and vNorm weight before normalization (Layer 4, first pass)
            if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
                print("    [Layer 4 DEBUG] V pre-norm first 5: \(values[0, 0, 0, 0..<5])")
                print("    [Layer 4 DEBUG] vNorm.weight first 5: \(vNorm.weight[0..<5])")
            }
            
            // Remove debug - using focused debug in transformer block
        }
        
        // Apply pre-RoPE normalization
        // Python applies normalization to 4D tensors [batch, seq, heads, headDim]
        // NOT flattened to 2D!
        queries = qNorm(queries)
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer 4 DEBUG] Q after norm first 5: \(queries[0, 0, 0, 0..<5])")
        }
        
        // Remove debug - using focused debug in transformer block
        
        keys = kNorm(keys)
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer 4 DEBUG] K after norm first 5: \(keys[0, 0, 0, 0..<5])")
        }
        
        values = vNorm(values)
        // Debug V after norm for layer 4
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("    [Layer 4 DEBUG] V after norm first 5: \(values[0, 0, 0, 0..<5])")
        }
        
        // Apply RoPE - use pre-computed embeddings if available
        if let (cos, sin) = positionEmbeddings {
            // Use pre-computed position embeddings
            queries = RotaryPositionEmbedding.applyRotaryPosEmb(queries, cos: cos, sin: sin, unsqueezeDim: 2)
            keys = RotaryPositionEmbedding.applyRotaryPosEmb(keys, cos: cos, sin: sin, unsqueezeDim: 2)
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 4 DEBUG] Q after RoPE first 5: \(queries[0, 0, 0, 0..<5])")
                print("    [Layer 4 DEBUG] K after RoPE first 5: \(keys[0, 0, 0, 0..<5])")
            }
        } else if let rope = rope {
            // Compute position embeddings on the fly
            let positionIds: MLXArray
            if let cachePos = cachePosition {
                // Use provided cache position
                positionIds = cachePos.expandedDimensions(axis: 0)
            } else {
                // Fallback to starting from 0 (only for testing)
                positionIds = MLXArray(0..<seqLen).expandedDimensions(axis: 0)
            }
            let posIds = broadcast(positionIds, to: [batchSize, seqLen])
            
            // Get cos and sin embeddings
            let (cos, sin) = rope(queries, positionIds: posIds)
            
            // Apply rotary embeddings to queries and keys
            // Note: unsqueezeDim=2 because our tensors are [batch, seq, heads, dim]
            queries = RotaryPositionEmbedding.applyRotaryPosEmb(queries, cos: cos, sin: sin, unsqueezeDim: 2)
            keys = RotaryPositionEmbedding.applyRotaryPosEmb(keys, cos: cos, sin: sin, unsqueezeDim: 2)
            if DebugLayers.shouldDebug(layer: layerIdx) {
                print("    [Layer 4 DEBUG] Q after RoPE first 5: \(queries[0, 0, 0, 0..<5])")
                print("    [Layer 4 DEBUG] K after RoPE first 5: \(keys[0, 0, 0, 0..<5])")
            }
        }
        
        // Remove debug - using focused debug in transformer block
        
        // Update KV cache only when we computed fresh K/V (i.e. not re-using shared values)
        if sharedKeys == nil {
            let keysForCache   = keys.transposed(0, 2, 1, 3)   // [B, KVHeads, Seq, D]
            let valuesForCache = values.transposed(0, 2, 1, 3)

            let (cachedKeys, cachedValues) = cache.update(keys: keysForCache, values: valuesForCache)

            // Transpose back to [B, Seq, KVHeads, D] for attention
            keys   = cachedKeys.transposed(0, 2, 1, 3)
            values = cachedValues.transposed(0, 2, 1, 3)
        }
        
        // Apply attention
        let attentionOutput = applyAttention(
            queries: queries,
            keys: keys,
            values: values,
            mask: mask,
            windowSize: windowSize
        )
        
        // Reshape and project output
        let reshapedOutput = attentionOutput.reshaped([batchSize, seqLen, numHeads * headDim])
        
        // Inspect output projection on Layer 4 first pass
        if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
            print("    [L4] oProj.weight[0,0:5]:", oProj.weight[0, 0..<5])
            print("    [L4] attnOut pre-oProj first 5:", reshapedOutput[0, 0, 0..<5])
        }

        // Up-cast to Float32 for the projection so that accumulation happens
        // in high precision (matches Python implementation). Then cast the
        // result back to the original dtype.
        let oProjInputF32   = reshapedOutput.asType(.float32)
        let oProjOutputF32  = oProj(oProjInputF32)
        let output          = oProjOutputF32.asType(originalDtype)

        if DebugLayers.shouldDebug(layer: layerIdx) && DebugLayers.currentPass == 1 {
            print("    [L4] attnOut post-oProj first 5:", output[0, 0, 0..<5])
        }

        return output
    }
}

/// Gemma3n E2B model attention configuration
public class Gemma3nE2BAttention: Gemma3nAttention {
    /// Initialize with E2B model defaults
    /// - Parameters:
    ///   - hiddenSize: Hidden dimension (default 2048)
    ///   - attnLogitSoftcapping: Attention logit softcapping value (default 0.0 - disabled)
    ///   - layerIdx: Layer index for KV sharing
    ///   - layerType: Type of attention layer
    public init(hiddenSize: Int = 2048, attnLogitSoftcapping: Float = 0.0, layerIdx: Int = 0, layerType: String = "sliding_attention") {
        super.init(
            hiddenSize: hiddenSize,
            numHeads: 8,
            numKVHeads: 2,
            headDim: 256,  // E2B uses explicit headDim of 256
            eps: Float(1e-6),
            attnLogitSoftcapping: attnLogitSoftcapping,
            layerIdx: layerIdx,
            numHiddenLayers: 30,  // E2B has 30 layers
            numKvSharedLayers: 10,  // E2B shares last 10 layers
            layerType: layerType
        )
    }
}

/// Gemma3n E4B model attention configuration
public class Gemma3nE4BAttention: Gemma3nAttention {
    /// Initialize with E4B model defaults
    /// - Parameters:
    ///   - hiddenSize: Hidden dimension (default 2048)
    ///   - attnLogitSoftcapping: Attention logit softcapping value (default 0.0 - disabled)
    ///   - layerIdx: Layer index for KV sharing
    ///   - layerType: Type of attention layer
    public init(hiddenSize: Int = 2048, attnLogitSoftcapping: Float = 0.0, layerIdx: Int = 0, layerType: String = "sliding_attention") {
        super.init(
            hiddenSize: hiddenSize,
            numHeads: 8,      // E4B uses same attention config as E2B
            numKVHeads: 2,
            headDim: 256,     // E4B also uses explicit headDim of 256
            eps: Float(1e-6),
            attnLogitSoftcapping: attnLogitSoftcapping,
            layerIdx: layerIdx,
            numHiddenLayers: 35,  // E4B has 35 layers
            numKvSharedLayers: 10,  // E4B also shares last 10 layers
            layerType: layerType
        )
    }
}