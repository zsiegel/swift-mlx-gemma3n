import Foundation
import MLX
import MLXNN

/// Token embedding layer for Gemma3n models
/// Implements standard token embeddings with sqrt(hidden_size) scaling
public class TokenEmbedding: Module {
    @ModuleInfo(key: "weight") public var weight: MLXArray
    public let vocabSize: Int
    public let hiddenSize: Int
    public let scale: Float
    
    /// Initialize token embeddings
    /// - Parameters:
    ///   - vocabSize: Size of the vocabulary (262400 for Gemma3n)
    ///   - hiddenSize: Size of hidden dimension (2048 for Gemma3n)
    public init(vocabSize: Int, hiddenSize: Int) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.scale = sqrt(Float(hiddenSize))
        
        // Initialize weights with zeros - will be replaced by loaded weights
        self._weight = ModuleInfo(
            wrappedValue: MLX.zeros([vocabSize, hiddenSize]),
            key: "weight"
        )
        
        super.init()
    }
    
    /// Forward pass - embed tokens
    /// - Parameter inputIds: Token IDs of shape [batch_size, sequence_length]
    /// - Returns: Embeddings of shape [batch_size, sequence_length, hidden_size]
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        // Get embeddings using array indexing
        let embeddings = weight[inputIds]
        
        // Apply scaling to match Python
        // Python: h = (h * mx.array(self._embed_tokens_scale, mx.float32)).astype(h.dtype)
        // where _embed_tokens_scale = sqrt(hidden_size) = sqrt(2048) ≈ 45.25
        // let scaledEmbeddings = embeddings.asType(.float32) * scale
        // let result = scaledEmbeddings.asType(embeddings.dtype)
        
        // print("\n[After TokenEmbedding Scaling]")
        // print("  Output shape: \(result.shape)")
        // print("  Output dtype: \(result.dtype)")
        // print("  L2 norm: \(String(format: "%.6f", MLX.sqrt(MLX.sum(result * result)).item(Float.self)))")
        // print("  Scale factor: \(scale)")
        
        return embeddings
    }
}

/// Per-layer embedding layer for Gemma3n models
/// Provides separate embeddings for each transformer layer
public class PerLayerEmbedding: Module {
    @ModuleInfo(key: "weight") public var weight: MLXArray
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numLayers: Int
    public let scale: Float
    
    /// Initialize per-layer embeddings
    /// - Parameters:
    ///   - vocabSize: Size of per-layer vocabulary (262144 for Gemma3n)
    ///   - hiddenSize: Size of per-layer hidden dimension (256 for Gemma3n)
    ///   - numLayers: Number of transformer layers (30 for 2B, 35 for 4B)
    public init(vocabSize: Int, hiddenSize: Int, numLayers: Int) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.scale = sqrt(Float(hiddenSize))
        
        // Initialize weights: [vocab_size, num_layers * hidden_size]
        let totalDim = numLayers * hiddenSize
        // Initialize with zeros - will be replaced by loaded weights
        self._weight = ModuleInfo(
            wrappedValue: MLX.zeros([vocabSize, totalDim]),
            key: "weight"
        )
        
        super.init()
    }
    
    /// Forward pass - embed tokens
    /// - Parameter inputIds: Token IDs of shape [batch_size, sequence_length]
    /// - Returns: Embeddings of shape [batch_size, sequence_length, num_layers * hidden_size]
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        // Get embeddings using array indexing
        let embeddings = weight[inputIds]
        
        // Apply scaling to match Python
        // Python applies: result * mx.array(self._embed_tokens_per_layer_scale, mx.float32)
        // where _embed_tokens_per_layer_scale = sqrt(hidden_size) = sqrt(256) = 16.0
        let scaledEmbeddings = embeddings.asType(.float32) * scale
        let result = scaledEmbeddings.asType(embeddings.dtype)
        
        return result
    }
    
    /// Extract embeddings for a specific layer
    /// - Parameters:
    ///   - embeddings: Full per-layer embeddings [batch_size, seq_len, num_layers * hidden_size]
    ///   - layerIdx: Index of the layer to extract
    /// - Returns: Embeddings for the specified layer [batch_size, seq_len, hidden_size]
    public func extractLayer(_ embeddings: MLXArray, layerIdx: Int) -> MLXArray {
        let startIdx = layerIdx * hiddenSize
        let endIdx = startIdx + hiddenSize
        return embeddings[.ellipsis, startIdx..<endIdx]
    }
}

/// Combined embeddings for Gemma3n model
/// Manages both token and per-layer embeddings
public class Gemma3nEmbeddings: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: TokenEmbedding
    public var embedTokensPerLayer: PerLayerEmbedding?
    
    public let config: TextConfig
    
    /// Initialize Gemma3n embeddings from configuration
    public init(config: TextConfig) {
        self.config = config
        
        // Initialize token embeddings
        self._embedTokens = ModuleInfo(
            wrappedValue: TokenEmbedding(
                vocabSize: config.vocabSize,
                hiddenSize: config.hiddenSize
            ),
            key: "embed_tokens"
        )
        
        // Initialize per-layer embeddings if configured
        if config.hiddenSizePerLayerInput > 0 {
            self.embedTokensPerLayer = PerLayerEmbedding(
                vocabSize: config.vocabSizePerLayerInput,
                hiddenSize: config.hiddenSizePerLayerInput,
                numLayers: config.numHiddenLayers
            )
        }
        
        super.init()
    }
    
    /// Get token embeddings
    /// - Parameter inputIds: Token IDs
    /// - Returns: Token embeddings with scaling applied
    public func getTokenEmbeddings(_ inputIds: MLXArray) -> MLXArray {
        return embedTokens(inputIds)
    }
    
    /// Get per-layer embeddings
    /// - Parameter inputIds: Token IDs (filtered for valid per-layer vocab)
    /// - Returns: Per-layer embeddings if available, nil otherwise
    public func getPerLayerEmbeddings(_ inputIds: MLXArray) -> MLXArray? {
        guard let perLayerEmbed = embedTokensPerLayer else { return nil }
        
        // Filter input IDs that are within per-layer vocabulary range
        let mask = inputIds .< config.vocabSizePerLayerInput
        let validIds = MLX.where(mask, inputIds, MLXArray(0, dtype: inputIds.dtype))
        
        return perLayerEmbed(validIds)
    }
    
}