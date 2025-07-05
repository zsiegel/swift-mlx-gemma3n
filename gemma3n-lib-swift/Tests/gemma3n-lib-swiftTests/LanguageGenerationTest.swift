import Testing
import Foundation
import MLX
import MLXNN
@testable import gemma3n_lib_swift

@Suite("Language Generation Test")
struct LanguageGenerationTest {
    
    // Use a fully qualified path to the model directory
    private var modelPath: String {
        return "/Users/zsiegel/src/gitlab.com/mlx-gemma3n/gemma3n-lib-swift/models/gemma-3n-E2B-it-bf16"
    }
    
    @Test("Run generation matching Python mlx_vlm")
    func testPythonCompatibleGeneration() async throws {
        print("\n=== SWIFT MLX GENERATION TEST ===")

        DebugLayers.setLayerRange(start: 0, end: 0)

        print("Debug enabled for layers \(DebugLayers.startLayer) to \(DebugLayers.endLayer)")
        
        // Load tokenizer and config
        print("\nLoading tokenizer...")
        let tokenizer = try await Gemma3nTokenizer.from(modelPath: modelPath)
        
        print("Loading configuration...")
        let config = try TextConfig.load(from: modelPath)
        
        try autoreleasepool {
            // Create model and load weights
            print("\nCreating model...")
            let model = Gemma3nLanguageModel(config: config)
            
            print("Loading weights...")
            let modelDirectory = URL(fileURLWithPath: modelPath)
            try Gemma3nModelWeightLoader.loadLanguageModelWeights(model: model, from: modelDirectory)
            
            print("✓ Model loaded successfully")
            print("Model config: hidden_size=\(config.hiddenSize), num_layers=\(config.numHiddenLayers)")
            
            // Match Python prompt setup exactly
            let userPrompt = "What is the capital of France?"
            print("\n" + String(repeating: "═", count: 10))
            print("User prompt: \"\(userPrompt)\"")
            
            // Format with chat template to match Python
            let formattedPrompt = "<bos><start_of_turn>user\n\(userPrompt)<end_of_turn>\n<start_of_turn>model\n"
            print("Formatted prompt: \"\(formattedPrompt)\"")
            
            // Tokenize to match Python
            let tokenIds = tokenizer.encode(formattedPrompt, addSpecialTokens: false)
            print("\nTokenized prompt (token IDs): \(tokenIds)")
            print("Number of tokens: \(tokenIds.count)")
            
            // Decode tokens to verify
            let decodedTokens = tokenIds.map { tokenizer.decode(tokens: [$0], skipSpecialTokens: false) }
            print("\nIndividual tokens: \(decodedTokens)")
            
            // Create generator
            let generator = SimpleGeneration(model: model, tokenizer: tokenizer)
            
            print("\n" + String(repeating: "═", count: 60))
            print("Generating with parameters:")
            print("  - temperature: 0.7")
            print("  - max_tokens: 100")
            print("  - Using KV cache (matching mlx_vlm)")
            print(String(repeating: "═", count: 60))
            
            let response = generator.generateWithCache(
                prompt: userPrompt,
                maxTokens: 1000,
                temperature: 0.7,
                verbose: true
            )
            
            print("\n=== FINAL SWIFT MLX GENERATION RESULT ===")
            print("Generated response: \"\(response)\"")
            
            // Tokenize the generated response to see token IDs
            let fullOutput = formattedPrompt + response
            let fullTokenIds = tokenizer.encode(fullOutput, addSpecialTokens: false)
            let generatedTokenIds = Array(fullTokenIds.dropFirst(tokenIds.count))
            
            print("\nGenerated token IDs: \(generatedTokenIds)")
            print("Number of generated tokens: \(generatedTokenIds.count)")
            
            // Show token mapping for generated text
            print("\n" + String(repeating: "=", count: 50))
            print("GENERATED TOKEN MAPPING")
            print(String(repeating: "=", count: 50))
            for (i, tokenId) in generatedTokenIds.enumerated() {
                let tokenStr = tokenizer.decode(tokens: [tokenId], skipSpecialTokens: false)
                print("  [\(i)] ID: \(String(format: "%6d", tokenId)) -> '\(tokenStr)'")
            }
            
            print("\n=== END OF PYTHON-COMPATIBLE GENERATION TEST ===")
        }
    }
}