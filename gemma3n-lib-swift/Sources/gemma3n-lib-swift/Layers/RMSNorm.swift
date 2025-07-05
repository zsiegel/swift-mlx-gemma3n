import Foundation
import MLX
import MLXNN

/// Gemma3n-specific RMSNorm implementation
/// This implements Root Mean Square Layer Normalization with optional scale shift
public class Gemma3nRMSNorm: Module, UnaryLayer {
    
    @ParameterInfo(key: "weight") public var weight = MLXArray.ones([1])
    public let eps: Float
    public let scaleShift: Float
    public let withScale: Bool
    
    public init(dimensions: Int, eps: Float = Float(1e-6), scaleShift: Float = 0.0, withScale: Bool = true) {
        self.eps = eps
        self.scaleShift = scaleShift
        self.withScale = withScale
        
        // Initialize weight as a parameter
        self._weight = ParameterInfo(
            wrappedValue: MLXArray.ones([dimensions]).asType(.float32),
            key: "weight"
        )
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Cast input to float32 for normalization
        let originalDtype = x.dtype
        let x_f32 = x.asType(.float32)
        
        
        // Compute variance in fp32 for stability
        let variance = MLX.mean(MLX.square(x_f32), axes: [-1], keepDims: true)
        
        // Compute RMSNorm in fp32 then cast back
        let normed = x_f32 / MLX.sqrt(variance + eps)
        
        // Apply scale if enabled
        let output: MLXArray
        if withScale {
            // Apply learnable scale with shift: normed * (weight + scaleShift)
            let scaledWeight = weight.asType(.float32) + scaleShift
            output = normed * scaledWeight
        } else {
            // No scaling applied
            output = normed
        }
        
        // Cast back to original dtype
        let result = output.asType(originalDtype)
        return result
    }
}

/// Standard RMSNorm for comparison (without the scale shift behavior)
public class BasicRMSNorm: Module, UnaryLayer {
    
    @ParameterInfo(key: "weight") public var weight = MLXArray.ones([1])
    public let eps: Float
    
    public init(dimensions: Int, eps: Float = Float(1e-6)) {
        self.eps = eps
        
        // Initialize weight as a parameter
        self._weight = ParameterInfo(
            wrappedValue: MLXArray.ones([dimensions]).asType(.float32),
            key: "weight"
        )
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Cast input to float32 for normalization
        let originalDtype = x.dtype
        let x_f32 = x.asType(.float32)
        
        
        // Compute variance in fp32 for stability
        let variance = MLX.mean(MLX.square(x_f32), axes: [-1], keepDims: true)
        
        // Compute RMSNorm in fp32 then cast back
        let normed = x_f32 / MLX.sqrt(variance + eps)
        
        // Apply learnable scale
        let output = normed * weight
        
        // Cast back to original dtype
        let result = output.asType(originalDtype)
        return result
    }
}