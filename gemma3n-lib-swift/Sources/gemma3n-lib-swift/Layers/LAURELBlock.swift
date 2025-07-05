import Foundation
import MLX
import MLXNN

/// LAUREL (Learned Augmented Residual Layer) block for Gemma3n
/// This implements a low-rank factorization with learnable normalization
/// to enhance residual connections in the transformer architecture
public class LAURELBlock: Module, UnaryLayer {
    
    public let hiddenSize: Int
    public let laurelRank: Int
    public let eps: Float
    public var layerIdx: Int? = nil  // Set by parent layer for debugging
    
    // Linear layers for low-rank factorization
    @ModuleInfo(key: "linear_left") public var linearLeft: Linear
    @ModuleInfo(key: "linear_right") public var linearRight: Linear
    
    // Post-LAUREL normalization
    @ModuleInfo(key: "post_laurel_norm") public var postLaurelNorm: Gemma3nRMSNorm
    
    /// Initialize LAUREL block
    /// - Parameters:
    ///   - hiddenSize: Size of the hidden dimension
    ///   - laurelRank: Rank for the low-rank factorization (default 64 for Gemma3n)
    ///   - eps: Epsilon for RMS normalization (default 1e-6)
    public init(hiddenSize: Int, laurelRank: Int = 64, eps: Float = Float(1e-6)) {
        self.hiddenSize = hiddenSize
        self.laurelRank = laurelRank
        self.eps = eps
        
        // Create linear layers for factorization
        // Left: hiddenSize -> laurelRank
        self._linearLeft = ModuleInfo(
            wrappedValue: Linear(hiddenSize, laurelRank, bias: false),
            key: "linear_left"
        )
        
        // Right: laurelRank -> hiddenSize
        self._linearRight = ModuleInfo(
            wrappedValue: Linear(laurelRank, hiddenSize, bias: false),
            key: "linear_right"
        )
        
        // Create post-LAUREL normalization
        self._postLaurelNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: hiddenSize,
                eps: eps,
                scaleShift: 0.0,
                withScale: true
            ),
            key: "post_laurel_norm"
        )
        
        super.init()
    }
    
    /// Forward pass through LAUREL block
    /// - Parameter x: Input tensor [batch_size, seq_len, hidden_size]
    /// - Returns: Output tensor with residual connection [batch_size, seq_len, hidden_size]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Store original dtype for preservation
        let originalDtype = x.dtype
        
        // Low-rank factorization
        // Step 1: Project down to laurel_rank
        var laurelX = linearLeft(x)
        
        // Step 2: Project back up to hidden_size
        laurelX = linearRight(laurelX)
        
        // Step 3: Apply normalization
        let normedLaurelX = postLaurelNorm(laurelX)
        
        // Debug LAUREL computation only when enabled for this layer
        if let layerIdx = layerIdx, DebugLayers.shouldDebug(layer: layerIdx) {
            print("      [LAUREL DEBUG Layer \(layerIdx)] Input x first 10: \(x[0, 0, 0..<10])")
            print("      [LAUREL DEBUG Layer \(layerIdx)] linearLeft output first 10: \(laurelX[0, 0, 0..<10])")
            print("      [LAUREL DEBUG Layer \(layerIdx)] postLaurelNorm output first 10: \(normedLaurelX[0, 0, 0..<10])")
            print("      [LAUREL DEBUG Layer \(layerIdx)] Final (x + normed) first 10: \((x + normedLaurelX)[0, 0, 0..<10])")
        }
        
        // Step 4: Add residual connection
        let output = x + normedLaurelX
        
        // Preserve original dtype
        return output.asType(originalDtype)
    }
}

/// Gemma3n-specific LAUREL block with default parameters
public class Gemma3nLAURELBlock: LAURELBlock {
    
    /// Initialize with Gemma3n default parameters
    /// - Parameter hiddenSize: Hidden dimension size (2048 for E2B model)
    public init(hiddenSize: Int = 2048) {
        // Use Gemma3n defaults: laurelRank=64, eps=1e-6
        super.init(hiddenSize: hiddenSize, laurelRank: 64, eps: Float(1e-6))
    }
}