import Foundation

public struct TextConfig: Codable, Equatable {
    public let activationSparsityPattern: [Double]
    public let architectures: [String]?
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let confCrossLayerInterval: Int?
    public let confNumAttentionHeads: Int?
    public let confNumHiddenLayers: Int?
    public let confNumKvHeads: Int?
    public let eosTokenId: [Int]
    public let headDim: Int
    public let hiddenActivation: String
    public let hiddenSize: Int
    public let initializerisRange: Double
    public let intermediateSize: [Int]
    public let laurelRank: Int
    public let layerTypes: [String]
    public let finalLogitSoftcapping: Double
    public let maxPositionEmbeddings: Int
    public let modelType: String
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let queryPreAttnoScalar: Int
    public let residualMultiplierInitialScale: Double?
    public let rmsNormEps: Double
    public let ropeTheta: Double
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let tieWordEmbeddings: Bool
    public let tokenizer: String?
    public let torchDtype: String
    public let useCache: Bool
    public let vocabSize: Int
    
    // Additional fields found in actual configs
    public let bosTokenId: Int
    public let padTokenId: Int
    public let vocabSizePerLayerInput: Int
    public let hiddenSizePerLayerInput: Int
    public let numKvSharedLayers: Int
    public let altupActiveIdx: Int
    public let altupCoefClip: Double
    public let altupCorrectScale: Bool
    public let altupNumInputs: Int
    public let altupLrMultiplier: Double
    public let ropeLocalBaseFreq: Double
    public let attnLogitSoftcapping: Double
    // ropeScaling is always null in both models, so we can skip it
    
    private enum CodingKeys: String, CodingKey {
        case activationSparsityPattern = "activation_sparsity_pattern"
        case architectures
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case confCrossLayerInterval = "conf_cross_layer_interval"
        case confNumAttentionHeads = "conf_num_attention_heads"
        case confNumHiddenLayers = "conf_num_hidden_layers"
        case confNumKvHeads = "conf_num_kv_heads"
        case eosTokenId = "eos_token_id"
        case headDim = "head_dim"
        case hiddenActivation = "hidden_activation"
        case hiddenSize = "hidden_size"
        case initializerisRange = "initializer_range"
        case intermediateSize = "intermediate_size"
        case laurelRank = "laurel_rank"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case queryPreAttnoScalar = "query_pre_attn_scalar"
        case residualMultiplierInitialScale = "residual_multiplier_initial_scale"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case tieWordEmbeddings = "tie_word_embeddings"
        case tokenizer
        case torchDtype = "torch_dtype"
        case useCache = "use_cache"
        case vocabSize = "vocab_size"
        case bosTokenId = "bos_token_id"
        case padTokenId = "pad_token_id"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case numKvSharedLayers = "num_kv_shared_layers"
        case altupActiveIdx = "altup_active_idx"
        case altupCoefClip = "altup_coef_clip"
        case altupCorrectScale = "altup_correct_scale"
        case altupNumInputs = "altup_num_inputs"
        case altupLrMultiplier = "altup_lr_multiplier"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case attnLogitSoftcapping = "attn_logit_softcapping"
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        self.activationSparsityPattern = try container.decode([Double].self, forKey: .activationSparsityPattern)
        self.architectures = try container.decodeIfPresent([String].self, forKey: .architectures)
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.attentionDropout = try container.decode(Double.self, forKey: .attentionDropout)
        self.confCrossLayerInterval = try container.decodeIfPresent(Int.self, forKey: .confCrossLayerInterval)
        self.confNumAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .confNumAttentionHeads)
        self.confNumHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .confNumHiddenLayers)
        self.confNumKvHeads = try container.decodeIfPresent(Int.self, forKey: .confNumKvHeads)
        
        // Handle eos_token_id as either single Int or [Int]
        if let singleEosToken = try? container.decode(Int.self, forKey: .eosTokenId) {
            self.eosTokenId = [singleEosToken]
        } else {
            self.eosTokenId = try container.decode([Int].self, forKey: .eosTokenId)
        }
        
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.hiddenActivation = try container.decode(String.self, forKey: .hiddenActivation)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.initializerisRange = try container.decode(Double.self, forKey: .initializerisRange)
        self.intermediateSize = try container.decode([Int].self, forKey: .intermediateSize)
        self.laurelRank = try container.decode(Int.self, forKey: .laurelRank)
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.finalLogitSoftcapping = try container.decode(Double.self, forKey: .finalLogitSoftcapping)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.queryPreAttnoScalar = try container.decode(Int.self, forKey: .queryPreAttnoScalar)
        self.residualMultiplierInitialScale = try container.decodeIfPresent(Double.self, forKey: .residualMultiplierInitialScale)
        self.rmsNormEps = try container.decode(Double.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Double.self, forKey: .ropeTheta)
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        self.slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.tokenizer = try container.decodeIfPresent(String.self, forKey: .tokenizer)
        self.torchDtype = try container.decode(String.self, forKey: .torchDtype)
        self.useCache = try container.decode(Bool.self, forKey: .useCache)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        
        // Additional fields
        self.bosTokenId = try container.decode(Int.self, forKey: .bosTokenId)
        self.padTokenId = try container.decode(Int.self, forKey: .padTokenId)
        self.vocabSizePerLayerInput = try container.decode(Int.self, forKey: .vocabSizePerLayerInput)
        self.hiddenSizePerLayerInput = try container.decode(Int.self, forKey: .hiddenSizePerLayerInput)
        self.numKvSharedLayers = try container.decode(Int.self, forKey: .numKvSharedLayers)
        self.altupActiveIdx = try container.decode(Int.self, forKey: .altupActiveIdx)
        self.altupCoefClip = try container.decode(Double.self, forKey: .altupCoefClip)
        self.altupCorrectScale = try container.decode(Bool.self, forKey: .altupCorrectScale)
        self.altupNumInputs = try container.decode(Int.self, forKey: .altupNumInputs)
        self.altupLrMultiplier = try container.decode(Double.self, forKey: .altupLrMultiplier)
        self.ropeLocalBaseFreq = try container.decode(Double.self, forKey: .ropeLocalBaseFreq)
        // attnLogitSoftcapping might not be present in config files, default to 0.0 (disabled) matching Python
        self.attnLogitSoftcapping = try container.decodeIfPresent(Double.self, forKey: .attnLogitSoftcapping) ?? 0.0
        // ropeScaling is always null in both models, so we skip it
    }
}

// Helper struct to enable skipping unknown fields
struct AnyCodable: Codable {
    init(from decoder: Decoder) throws {
        // We don't need to decode anything, just skip
    }
    
    func encode(to encoder: Encoder) throws {
        // We don't need to encode anything
    }
}

public struct Gemma3nConfig: Codable, Equatable {
    public let architectures: [String]
    public let audioSoftTokensPerImage: Int
    public let eosTokenId: [Int]
    public let imageTokenId: Int
    public let modelType: String
    public let textConfig: TextConfig
    public let tieWordEmbeddings: Bool
    public let transformersVersion: String
    public let visionSoftTokensPerImage: Int
    
    private enum CodingKeys: String, CodingKey {
        case architectures
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case eosTokenId = "eos_token_id"
        case imageTokenId = "image_token_id"
        case modelType = "model_type"
        case textConfig = "text_config"
        case tieWordEmbeddings = "tie_word_embeddings"
        case transformersVersion = "transformers_version"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        self.architectures = try container.decode([String].self, forKey: .architectures)
        self.audioSoftTokensPerImage = try container.decode(Int.self, forKey: .audioSoftTokensPerImage)
        
        // Handle eos_token_id as either single Int or [Int]
        if let singleEosToken = try? container.decode(Int.self, forKey: .eosTokenId) {
            self.eosTokenId = [singleEosToken]
        } else {
            self.eosTokenId = try container.decode([Int].self, forKey: .eosTokenId)
        }
        
        self.imageTokenId = try container.decode(Int.self, forKey: .imageTokenId)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.textConfig = try container.decode(TextConfig.self, forKey: .textConfig)
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.transformersVersion = try container.decode(String.self, forKey: .transformersVersion)
        self.visionSoftTokensPerImage = try container.decode(Int.self, forKey: .visionSoftTokensPerImage)
        
        // Note: We intentionally skip audio_config and vision_config fields that exist in the JSON
    }
}