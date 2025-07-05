import Foundation
import MLX
import MLXNN

/// Rotary Position Embedding (RoPE) implementation for Gemma3n
/// This implements the rotary position embeddings as described in the RoFormer paper
/// with Gemma3n's specific parameters (θ=1,000,000)
public class RotaryPositionEmbedding: Module {
    
    public let dim: Int
    public let maxPositionEmbeddings: Int
    public let base: Float
    
    // Cached values
    private let invFreq: MLXArray
    
    /// Initialize Rotary Position Embedding
    /// - Parameters:
    ///   - dim: Dimension of the embeddings (head_dim)
    ///   - maxPositionEmbeddings: Maximum sequence length
    ///   - base: Base for computing frequencies (θ, default 1,000,000 for Gemma3n)
    public init(dim: Int, maxPositionEmbeddings: Int = 32768, base: Float = 1_000_000.0) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        
        // Compute inverse frequencies
        // freq = 1.0 / (base^(2i/dim)) for i in [0, dim/2)
        let halfDim = dim / 2
        let indices = MLXArray(0..<halfDim).asType(.float32)
        let exponents = indices * (Float(2.0) / Float(dim))
        self.invFreq = Float(1.0) / MLX.pow(MLXArray(base), exponents)
        
        super.init()
    }
    
    /// Compute rotary embeddings for given positions
    /// - Parameters:
    ///   - x: Input tensor (used only for dtype)
    ///   - positionIds: Position indices [batch_size, seq_len]
    /// - Returns: Tuple of (cos, sin) embeddings
    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {

        // Ensure position IDs are float32 for computation
        let posIds = positionIds.asType(.float32)
        
        // Expand dimensions for broadcasting
        // invFreq: [dim/2] -> [1, dim/2, 1]
        let invFreqExpanded = invFreq.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        
        // posIds: [batch, seq_len] -> [batch, 1, seq_len]
        let posIdsExpanded = posIds.expandedDimensions(axis: 1)
        
        // Compute frequencies: [batch, dim/2, seq_len]
        // Matrix multiply: [batch, dim/2, 1] @ [batch, 1, seq_len]
        let freqs = MLX.matmul(invFreqExpanded, posIdsExpanded)
        
        // Transpose to [batch, seq_len, dim/2]
        let freqsTransposed = freqs.transposed(0, 2, 1)
        
        // Concatenate to get full dimension [batch, seq_len, dim]
        let emb = MLX.concatenated([freqsTransposed, freqsTransposed], axis: -1)
        
        // Remove debug - using focused debug in transformer block
        
        // Compute cos and sin
        let cos = MLX.cos(emb)
        let sin = MLX.sin(emb)

        // Cast to input dtype
        return (cos.asType(x.dtype), sin.asType(x.dtype))
    }
}

/// Helper functions for applying rotary embeddings
public extension RotaryPositionEmbedding {
    
    /// Rotate half of the values in the last dimension
    /// For input [..., x1, x2, x3, x4], returns [..., -x3, -x4, x1, x2]
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let lastDim = x.dim(-1)
        let halfDim = lastDim / 2
        
        // Split into two halves
        let x1 = x[.ellipsis, ..<halfDim]
        let x2 = x[.ellipsis, halfDim..<lastDim]
        
        // Concatenate with negated second half first
        return MLX.concatenated([-x2, x1], axis: -1)
    }
    
    /// Apply rotary position embeddings to query or key tensors
    /// - Parameters:
    ///   - x: Input tensor [..., seq_len, dim]
    ///   - cos: Cosine embeddings from RoPE
    ///   - sin: Sine embeddings from RoPE
    ///   - unsqueezeDim: Dimension to unsqueeze for broadcasting (default 1 for multi-head attention)
    /// - Returns: Tensor with rotary embeddings applied
    static func applyRotaryPosEmb(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        unsqueezeDim: Int = 1
    ) -> MLXArray {
        
        // Expand cos and sin for broadcasting with multi-head tensors
        let cosExpanded = cos.expandedDimensions(axis: unsqueezeDim)
        let sinExpanded = sin.expandedDimensions(axis: unsqueezeDim)
        
        let rotated = rotateHalf(x)
        let result = (x * cosExpanded) + (rotated * sinExpanded)
        
        // Apply rotation: x * cos + rotate_half(x) * sin
        return result
    }
}

/// Gemma3n-specific Rotary Position Embedding
/// Uses the default parameters from Gemma3n configuration
public class Gemma3nRotaryEmbedding: RotaryPositionEmbedding {
    
    public init() {
        // Use Gemma3n defaults: dim=256 (head_dim), max_pos=32768, base=1,000,000
        super.init(dim: 256, maxPositionEmbeddings: 32768, base: 1_000_000.0)
    }
    
    public convenience init(headDim: Int) {
        self.init()
    }
}