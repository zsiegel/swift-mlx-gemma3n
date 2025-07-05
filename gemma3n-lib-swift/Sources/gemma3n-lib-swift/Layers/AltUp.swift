import Foundation
import MLX
import MLXNN

/// AltUp implementation that matches Python's dtype handling
/// This version keeps weights in Float32 during computation to avoid precision loss
public class AltUp: Module {
    
    public let hiddenSize: Int
    public let numInputs: Int
    public let activeIdx: Int
    public let coefClip: Float
    public let layerIdx: Int
    
    // Normalization layers
    @ModuleInfo(key: "predict_norm") public var predictNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "router_norm") public var routerNorm: Gemma3nRMSNorm
    
    // Prediction and correction coefficient layers
    @ModuleInfo(key: "prediction_coefs") public var predictionCoefs: Linear
    @ModuleInfo(key: "correction_coefs") public var correctionCoefs: Linear
    
    // Modality router
    @ModuleInfo(key: "modality_router") public var modalityRouter: Linear
    
    // Correct output scale parameter
    @ParameterInfo(key: "correct_output_scale") public var correctOutputScale = MLXArray.ones([1])
    
    public init(
        hiddenSize: Int,
        numInputs: Int = 4,
        activeIdx: Int = 0,
        coefClip: Float = 120.0,
        eps: Float = Float(1e-6),
        layerIdx: Int = -1
    ) {
        self.hiddenSize = hiddenSize
        self.numInputs = numInputs
        self.activeIdx = activeIdx
        self.coefClip = coefClip
        self.layerIdx = layerIdx
        
        // Initialize normalization layers
        self._predictNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: hiddenSize,
                eps: eps,
                scaleShift: 0.0,
                withScale: true
            ),
            key: "predict_norm"
        )
        
        self._routerNorm = ModuleInfo(
            wrappedValue: Gemma3nRMSNorm(
                dimensions: hiddenSize,
                eps: eps,
                scaleShift: 0.0,
                withScale: true
            ),
            key: "router_norm"
        )
        
        // Initialize coefficient layers
        self._predictionCoefs = ModuleInfo(
            wrappedValue: Linear(numInputs, numInputs * numInputs, bias: false),
            key: "prediction_coefs"
        )
        
        self._correctionCoefs = ModuleInfo(
            wrappedValue: Linear(numInputs, numInputs, bias: false),
            key: "correction_coefs"
        )
        
        self._modalityRouter = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numInputs, bias: false),
            key: "modality_router"
        )
        
        self._correctOutputScale = ParameterInfo(
            wrappedValue: MLXArray.ones([hiddenSize]),
            key: "correct_output_scale"
        )
        
        super.init()
    }
    
    private func computeRouterModalities(_ x: MLXArray) -> MLXArray {
        // Match Python's dtype preservation - keep scale in the same dtype as the input
        let routerInputScale = MLXArray(Float(1.0 / Double(hiddenSize))).asType(x.dtype)
        let normed = routerNorm(x)
        let routerInputs = normed * routerInputScale
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  AltUp computeRouterModalities Layer \(layerIdx)")
            print("  Router norm output first 5: \(normed[0, 0, 0..<5])")
            print("  Router input scale: \(routerInputScale)")
            print("  Router inputs first 5: \(routerInputs[0, 0, 0..<5])")
            print("  Router inputs dtype: \(routerInputs.dtype)")
            print("  Modality router weight shape: \(modalityRouter.weight.shape)")
            print("  Modality router weight dtype: \(modalityRouter.weight.dtype)")
            print("  Modality router weight[0,0:5]: \(modalityRouter.weight[0, 0..<5])")
        }
        
        // Router projection to float32 for tanh
        let routed = modalityRouter(routerInputs).asType(.float32)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  Routed (before tanh) first 5: \(routed[0, 0, 0..<5])")
        }
        
        let modalities = MLX.tanh(routed)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  Modalities (after tanh) shape: \(modalities.shape)")
        }
        
        return modalities
    }
    
    public func predict(_ x: MLXArray) -> MLXArray {
        
        let originalDtype = x.dtype
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("\n[AltUp Predict - Layer \(layerIdx)]")
            print("  Active stream (\(activeIdx)) first 5: \(x[activeIdx, 0, 0, 0..<5])")
        }
        
        let activeStream = x[activeIdx]
        let modalities = computeRouterModalities(activeStream)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  Modalities first 5: \(modalities[0, 0, 0..<5])")
        }
        
        // Work on a local Float32 copy of the weight (cannot mutate `weight` which is `let`).
        let wOrig = predictionCoefs.weight.asType(.float32)
        let wPred = coefClip > 0
            ? MLX.clip(wOrig, min: -coefClip, max:  coefClip)
            : wOrig
        let coefs = MLX.matmul(modalities.asType(.float32), wPred.T)
        
        // Reshape coefficients
        let batchSize = x.shape[1]
        let seqLen = x.shape[2]
        var reshapedCoefs = coefs.reshaped([batchSize, seqLen, numInputs, numInputs])
        reshapedCoefs = reshapedCoefs.transposed(axes: [0, 1, 3, 2])
        
        
        // Permute x for matrix multiplication - do everything in Float32
        let xF32 = x.asType(.float32)
        let xPermuted = xF32.transposed(axes: [1, 2, 3, 0])
        
        // Matrix multiply in Float32
        var predictions = MLX.matmul(xPermuted, reshapedCoefs)
        predictions = predictions.transposed(axes: [3, 0, 1, 2])
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  Predictions before residual - Stream 0 first 5: \(predictions[0, 0, 0, 0..<5])")
        }
        
        // Add residual connection
        predictions = predictions + x.asType(.float32)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  Predictions after residual - Stream 0 first 5: \(predictions[0, 0, 0, 0..<5])")
        }
        
        // Only convert back to original dtype at the very end
        return predictions.asType(originalDtype)
    }
    
    public func correct(_ predictions: MLXArray, activated: MLXArray) -> MLXArray {
        let originalDtype = predictions.dtype
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("\n[AltUp Correct - Layer \(layerIdx) DEBUG]")
            print("  Activated input first 5: \(activated[0, 0, 0..<5])")
            print("  Predictions[activeIdx] first 5: \(predictions[activeIdx, 0, 0, 0..<5])")
        }

        // Compute modalities from activated value
        let modalities = computeRouterModalities(activated)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  Modalities shape: \(modalities.shape)")
            print("  Modalities first 5: \(modalities[0, 0, 0..<5])")
        }
        
        let wCorr = coefClip > 0
            ? MLX.clip(correctionCoefs.weight.asType(.float32), min: -coefClip, max: coefClip)
            : correctionCoefs.weight.asType(.float32)

        // Compute correction coefficients (add 1.0 as Python does) and reshape
        var coefs = MLX.matmul(modalities.asType(.float32), wCorr.T) + Float(1.0) // (B,S,N)

        // In Python: all_coefs.permute(2, 1, 0).unsqueeze(1)
        // The resulting tensor has shape (N, 1, S, B). However, for robust
        // broadcasting (especially when batch > 1) we prefer the shape
        // (N, B, S, 1) so that the batch dimension aligns and the scale is
        // broadcast across the hidden dimension.  This still matches the
        // mathematical intent (scalar factor per-(stream,batch,seq)).

        coefs = coefs.transposed(axes: [2, 0, 1])          // (N, B, S)
        coefs = coefs.expandedDimensions(axis: 3)          // (N, B, S, 1)
        
        if DebugLayers.shouldDebug(layer: layerIdx) {
            print("  prediction_coefs.weight[0,:5]", predictionCoefs.weight[0, 0..<5])
            print("  wCorr[0,:5]", wCorr[0, 0..<5])
            print("  Stream-0 coef (layer \(layerIdx)):", coefs[0,0,0,0])
        }
        
        // Innovation: difference between activated stream and active prediction
        let innovation = activated.asType(.float32) - predictions[activeIdx].asType(.float32)

        // Element-wise scale innovation then add residual
        // This was the source of a major bug - the broadcasting logic
        // was not replicating the python version correctly
        var correctedStreams = [MLXArray]()
        for i in 0..<numInputs {
            let streamPrediction = predictions[i].asType(.float32)
            let streamCoefs = coefs[i, 0..., 0..., 0...].squeezed(axes: [0])
            
            let corrected = streamPrediction + (innovation * streamCoefs)
            correctedStreams.append(corrected)
        }
        
        let correctedF32 = MLX.stacked(correctedStreams)
        
        // Convert back to original dtype only at the end
        return correctedF32.asType(originalDtype)
    }
    
    public func scaleOutput(_ output: MLXArray) -> MLXArray {
        return output * correctOutputScale
    }
}

/// Gemma3n-specific AltUp
public class Gemma3nAltUp: AltUp {
    public init(hiddenSize: Int = 2048, layerIdx: Int = -1) {
        super.init(
            hiddenSize: hiddenSize,
            numInputs: 4,
            activeIdx: 0,
            coefClip: 120.0,
            eps: Float(1e-6),
            layerIdx: layerIdx
        )
    }
}