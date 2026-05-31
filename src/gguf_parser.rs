use candle_core::quantized::{gguf_file, QTensor};
use candle_core::Device;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;

/// The primary entry point for your inference library.
/// This holds everything needed to run a forward pass.
pub struct InferenceModel {
    /// Global hyperparameters and architecture details
    pub metadata: ModelMetadata,
    /// The actual weights ready for computation, indexed by their GGUF name
    pub tensors: HashMap<String, NamedTensor>,
}

/// Global configuration parsed from the GGUF metadata KV pairs
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    // General Info
    pub architecture: String,       // e.g., "qwen2" or "llama"
    pub alignment: u32,             // Tensor data alignment (usually 32)
    
    // Hyperparameters required to initialize layers/attention matrices
    pub context_length: u64,        // max sequence length
    pub embedding_length: u64,      // hidden dim (d_model)
    pub block_count: u32,           // number of transformer layers
    pub attention_head_count: u32,   // number of query heads
    pub attention_head_count_kv: u32,// number of KV heads (for GQA/MQA)
    pub rope_dim_theta: f32,        // RoPE base frequency
    
    // Vocabulary + BPE merge rules required to reconstruct the real tokenizer
    pub tokenizer_model: String,        // e.g., "gpt2" — identifies tokenizer family
    pub tokenizer_pre: String,          // e.g., "qwen2" — pre-tokenizer regex variant
    pub tokenizer_tokens: Vec<String>,
    pub tokenizer_scores: Vec<f32>,
    pub tokenizer_token_types: Vec<i32>, // per-token type tag (1=normal, 3=control, ...)
    pub tokenizer_merges: Vec<String>,   // BPE merge pairs, each "left right"
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,

    // Catch-all for any architecture-specific or unexpected metadata
    pub raw_metadata: HashMap<String, candle_core::quantized::gguf_file::Value>,
}

/// Represents a single weight matrix or bias vector loaded into memory.
/// We keep the original quantized `QTensor` so downstream code can build a
/// `QMatMul` for SIMD-accelerated matmul without ever dequantizing the
/// whole weight up front.
pub struct NamedTensor {
    pub name: String,
    pub qtensor: Arc<QTensor>,
}

impl NamedTensor {
    pub fn shape(&self) -> Vec<usize> {
        self.qtensor.shape().dims().to_vec()
    }
}

pub fn parse_gguf(model_path: impl AsRef<Path>) -> Result<InferenceModel> {
    let mut file = File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    // 1. Extract Metadata cleanly with fallbacks
    let architecture = match content.metadata.get("general.architecture") {
        Some(gguf_file::Value::String(s)) => s.clone(),
        _ => "unknown".to_string(),
    };

    let context_length = match content.metadata.get(&format!("{}.context_length", architecture)) {
        Some(gguf_file::Value::U32(v)) => *v as u64,
        Some(gguf_file::Value::U64(v)) => *v,
        _ => 2048, // safe default fallback
    };

    let embedding_length = match content.metadata.get(&format!("{}.embedding_length", architecture)) {
        Some(gguf_file::Value::U32(v)) => *v as u64,
        Some(gguf_file::Value::U64(v)) => *v,
        _ => 4096,
    };

    let block_count = match content.metadata.get(&format!("{}.block_count", architecture)) {
        Some(gguf_file::Value::U32(v)) => *v,
        _ => 0,
    };

    let attention_head_count = match content.metadata.get(&format!("{}.attention.head_count", architecture)) {
        Some(gguf_file::Value::U32(v)) => *v,
        _ => 32,
    };

    let attention_head_count_kv = match content.metadata.get(&format!("{}.attention.head_count_kv", architecture)) {
        Some(gguf_file::Value::U32(v)) => *v,
        _ => attention_head_count, // If absent, defaults to standard MHA
    };

    let alignment = match content.metadata.get("general.alignment") {
        Some(gguf_file::Value::U32(v)) => *v,
        _ => 32,
    };

    let rope_dim_theta = match content.metadata.get(&format!("{}.rope.freq_base", architecture)) {
        Some(gguf_file::Value::F32(v)) => *v,
        _ => 10000.0,
    };

    // Extract Tokenizer Arrays safely
    let tokenizer_model = match content.metadata.get("tokenizer.ggml.model") {
        Some(gguf_file::Value::String(s)) => s.clone(),
        _ => String::new(),
    };

    let tokenizer_pre = match content.metadata.get("tokenizer.ggml.pre") {
        Some(gguf_file::Value::String(s)) => s.clone(),
        _ => String::new(),
    };

    let mut tokenizer_tokens: Vec<String> = Vec::new();
    if let Some(gguf_file::Value::Array(tokens)) = content.metadata.get("tokenizer.ggml.tokens") {
        for v in tokens {
            if let gguf_file::Value::String(s) = v {
                tokenizer_tokens.push(s.clone());
            }
        }
    }

    let mut tokenizer_scores: Vec<f32> = Vec::new();
    if let Some(gguf_file::Value::Array(scores)) = content.metadata.get("tokenizer.ggml.scores") {
        for v in scores {
            if let gguf_file::Value::F32(s) = v {
                tokenizer_scores.push(*s);
            }
        }
    }

    let mut tokenizer_token_types: Vec<i32> = Vec::new();
    if let Some(gguf_file::Value::Array(types)) = content.metadata.get("tokenizer.ggml.token_type") {
        for v in types {
            if let gguf_file::Value::I32(t) = v {
                tokenizer_token_types.push(*t);
            }
        }
    }

    let mut tokenizer_merges: Vec<String> = Vec::new();
    if let Some(gguf_file::Value::Array(merges)) = content.metadata.get("tokenizer.ggml.merges") {
        for v in merges {
            if let gguf_file::Value::String(s) = v {
                tokenizer_merges.push(s.clone());
            }
        }
    }

    let bos_token_id = content.metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| if let gguf_file::Value::U32(x) = v { Some(*x) } else { None });
    let eos_token_id = content.metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| if let gguf_file::Value::U32(x) = v { Some(*x) } else { None });
    let pad_token_id = content.metadata
        .get("tokenizer.ggml.padding_token_id")
        .and_then(|v| if let gguf_file::Value::U32(x) = v { Some(*x) } else { None });

    let metadata = ModelMetadata {
        architecture,
        alignment,
        context_length,
        embedding_length,
        block_count,
        attention_head_count,
        attention_head_count_kv,
        rope_dim_theta,
        tokenizer_model,
        tokenizer_pre,
        tokenizer_tokens,
        tokenizer_scores,
        tokenizer_token_types,
        tokenizer_merges,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        raw_metadata: content.metadata.clone(),
    };

    // 2. Load Tensors into memory
    let mut tensors = HashMap::new();
    let device = Device::Cpu;

    for (name, _tensor_info) in content.tensor_infos.iter() {
        let name = name.clone();
        let qtensor = content.tensor(&mut file, &name, &device)?;
        let named_tensor = NamedTensor {
            name: name.clone(),
            qtensor: Arc::new(qtensor),
        };
        tensors.insert(name, named_tensor);
    }

    Ok(InferenceModel { metadata, tensors })
}