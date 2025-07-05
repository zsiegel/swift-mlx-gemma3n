import Foundation

/// Global configuration for layer debugging
public class DebugLayers {
    // Use private storage with thread-safe accessors
    private static let queue = DispatchQueue(label: "debuglayers.queue")
    
    nonisolated(unsafe) private static var _startLayer: Int = -1
    nonisolated(unsafe) private static var _endLayer: Int = -1
    nonisolated(unsafe) private static var _startPass: Int = 1
    nonisolated(unsafe) private static var _endPass: Int = 1
    nonisolated(unsafe) private static var _currentPass: Int = 1
    
    /// Starting layer index for detailed debug output (inclusive)
    public static var startLayer: Int {
        get { queue.sync { _startLayer } }
        set { queue.sync { _startLayer = newValue } }
    }
    
    /// Ending layer index for detailed debug output (inclusive)
    public static var endLayer: Int {
        get { queue.sync { _endLayer } }
        set { queue.sync { _endLayer = newValue } }
    }
    
    /// Starting pass number for debug output (inclusive)
    public static var startPass: Int {
        get { queue.sync { _startPass } }
        set { queue.sync { _startPass = newValue } }
    }
    
    /// Ending pass number for debug output (inclusive)
    public static var endPass: Int {
        get { queue.sync { _endPass } }
        set { queue.sync { _endPass = newValue } }
    }
    
    /// Current pass number being executed
    public static var currentPass: Int {
        get { queue.sync { _currentPass } }
        set { queue.sync { _currentPass = newValue } }
    }
    
    /// Check if a layer should be debugged
    public static func shouldDebug(layer: Int) -> Bool {
        return false
        return queue.sync {
            let layerInRange = layer >= _startLayer && layer <= _endLayer
            let passInRange = _currentPass >= _startPass && _currentPass <= _endPass
            return layerInRange && passInRange
        }
    }
    
    /// Update debug range for layers
    public static func setDebugRange(start: Int, end: Int) {
        queue.sync {
            _startLayer = start
            _endLayer = end
        }
    }
    
    /// Update debug range for passes
    public static func setPassRange(start: Int, end: Int) {
        queue.sync {
            _startPass = start
            _endPass = end
        }
    }
    
    /// Disable all debug output
    public static func disableDebug() {
        queue.sync {
            _startLayer = Int.max
            _endLayer = -1
            _startPass = Int.max
            _endPass = -1
        }
    }
    
    /// Enable debug for a single layer
    public static func debugSingleLayer(_ layer: Int) {
        queue.sync {
            _startLayer = layer
            _endLayer = layer
        }
    }
    
    /// Enable debug for a single pass
    public static func debugSinglePass(_ pass: Int) {
        queue.sync {
            _startPass = pass
            _endPass = pass
        }
    }
    
    /// Debug all layers
    public static func debugAllLayers(maxLayer: Int = 30) {
        queue.sync {
            _startLayer = 0
            _endLayer = maxLayer
        }
    }
    
    /// Debug all passes
    public static func debugAllPasses(maxPass: Int = 100) {
        queue.sync {
            _startPass = 1
            _endPass = maxPass
        }
    }
    
    /// Set the layer range to debug in a single call
    /// - Parameters:
    ///   - start: The first layer to debug (inclusive)
    ///   - end: The last layer to debug (inclusive)
    public static func setLayerRange(start: Int, end: Int) {
        queue.sync {
            _startLayer = start
            _endLayer = end
        }
    }
}