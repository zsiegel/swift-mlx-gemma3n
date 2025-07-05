import Foundation
import MLX
import MLXNN

/// Gemma3n Multi-Layer Perceptron (MLP) with optional activation sparsity
public class Gemma3nMLP: Module {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerIdx: Int
    public let activationSparsity: Float
    
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear
    
    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        layerIdx: Int = 0,
        activationSparsityPattern: [Float]? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerIdx = layerIdx
        
        // Get activation sparsity for this layer
        if let pattern = activationSparsityPattern, layerIdx < pattern.count {
            self.activationSparsity = pattern[layerIdx]
        } else {
            self.activationSparsity = 0.0
        }
        
        // Initialize linear layers without bias
        self._gateProj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false), key: "gate_proj")
        self._upProj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false), key: "up_proj")
        self._downProj = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize, bias: false), key: "down_proj")
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalDtype = x.dtype
        let x_f32 = x.asType(.float32)
        
        // Gate projection
        var gateOutput = gateProj(x_f32)
        
        // Apply sparsity if needed
        if activationSparsity > 0.0 {
            gateOutput = gaussianTopK(gateOutput)
        }
        
        // GELU activation
        let activations = MLXNN.geluApproximate(gateOutput)
        
        // Up projection
        let upOutput = upProj(x_f32)
        
        // Element-wise multiply and down project
        let mlpOutput = activations * upOutput
        let output = downProj(mlpOutput)
        
        return output.asType(originalDtype)
    }
    
    /// Apply Gaussian top-k sparsity to activations
    private func gaussianTopK(_ inputs: MLXArray) -> MLXArray {
        // For normal distribution, icdf(p) = -sqrt(2) * erfinv(2p - 1)
        let p = MLXArray(activationSparsity).asType(.float32)
        let sqrtTwo = MLXArray(Float(sqrt(2.0))).asType(.float32)
        let stdMultiplier = sqrtTwo * MLX.erfInverse(2 * p - 1)
        
        // Calculate mean and std along the last dimension
        let inputsMean = MLX.mean(inputs, axes: [-1], keepDims: true)
        let inputsStd = MLX.std(inputs, axes: [-1], keepDims: true)
        
        // Calculate cutoff threshold
        let cutoff = inputsMean + inputsStd * stdMultiplier.asType(inputs.dtype)
        
        // Zero out values below threshold
        return MLX.maximum(MLXArray(Float(0)).asType(inputs.dtype), inputs - cutoff)
    }
}

/// E2B configuration for Gemma3n MLP
public class Gemma3nE2BMLP: Gemma3nMLP {
    public init(layerIdx: Int = 0) {
        // E2B uses 16384 intermediate size with specific sparsity pattern
        let sparsityPattern: [Float] = Array(repeating: 0.95, count: 10) + Array(repeating: 0.0, count: 25)
        
        super.init(
            hiddenSize: 2048,
            intermediateSize: 16384,
            layerIdx: layerIdx,
            activationSparsityPattern: sparsityPattern
        )
    }
}

/// E4B configuration for Gemma3n MLP
public class Gemma3nE4BMLP: Gemma3nMLP {
    public init(layerIdx: Int = 0) {
        // E4B uses same configuration as E2B
        let sparsityPattern: [Float] = Array(repeating: 0.95, count: 10) + Array(repeating: 0.0, count: 25)
        
        super.init(
            hiddenSize: 2048,
            intermediateSize: 16384,
            layerIdx: layerIdx,
            activationSparsityPattern: sparsityPattern
        )
    }
}