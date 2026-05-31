//! Shared inference runtime: load a GGUF once, keep weights + tokenizer in
//! memory, and run greedy generation. Both the `inference` CLI and the
//! `server` binary drive the model through this one path so their behavior
//! (tokenization, chat template, prefill/decode, sampling) stays identical.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use candle_core::{Device, Tensor};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::forward::TransformerModel;
use crate::gguf_parser::{parse_gguf, InferenceModel};

/// A loaded model ready to serve requests. Holds the transformer (with its
/// in-memory KV cache), the tokenizer, and the bits of metadata generation
/// needs. Construct once with [`Engine::load`]; call [`Engine::generate`] per
/// request. Note: the transformer carries a single KV cache + position
/// counter, so a single `Engine` handles one request at a time — callers that
/// share it across threads must serialize access (the server wraps it in a
/// `Mutex`).
pub struct Engine {
    transformer: TransformerModel,
    tokenizer: Tokenizer,
    eos_id: Option<u32>,
    arch: String,
}

/// Result of a generation request, with enough to report throughput.
pub struct GenOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

impl Engine {
    /// Parse the GGUF, build the transformer and tokenizer. This is the slow,
    /// one-time step (dequantizes the embedding, allocates the KV cache, etc.).
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        let model = parse_gguf(model_path)?;
        let arch = model.metadata.architecture.clone();
        let eos_id = model.metadata.eos_token_id;
        let tokenizer = build_tokenizer_from_gguf(&model)?;
        let transformer = TransformerModel::load(&model, device)?;
        Ok(Self {
            transformer,
            tokenizer,
            eos_id,
            arch,
        })
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Greedy generation for a single prompt. Resets the KV cache, optionally
    /// wraps the prompt in the Qwen3 ChatML template, runs one batched prefill,
    /// then decodes up to `max_new_tokens` (stopping on EOS). `on_token` is
    /// invoked with each newly-decoded UTF-8-safe text delta so callers can
    /// stream; pass `|_| {}` to ignore. The full text is also returned.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        chat_template: bool,
        mut on_token: impl FnMut(&str),
    ) -> Result<GenOutput> {
        let input = if chat_template {
            // Instruct template; the trailing empty <think> block selects
            // non-thinking mode so small models answer directly.
            format!(
                "<|im_start|>user\n{prompt}<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        } else {
            prompt.to_string()
        };

        let encoding = self
            .tokenizer
            .encode(input.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt produced no tokens");
        }

        // Fresh KV cache / position for this request.
        self.transformer.reset_state();

        // Batched prefill: whole prompt in one pass, logits for the last token.
        let prefill_start = Instant::now();
        let mut logits = self.transformer.forward_chunk(&prompt_ids)?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();

        // Greedy decode loop.
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut printed_bytes = 0usize;
        let decode_start = Instant::now();
        for _ in 0..max_new_tokens {
            let next = argmax_last_dim(&logits)?;
            if Some(next) == self.eos_id {
                break;
            }
            generated.push(next);

            // Re-decode the whole sequence and emit only the new, complete
            // suffix (skip deltas ending on a partial multi-byte char).
            let decoded = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
            if decoded.len() > printed_bytes && !decoded.ends_with('\u{FFFD}') {
                on_token(&decoded[printed_bytes..]);
                printed_bytes = decoded.len();
            }

            logits = self.transformer.forward(next)?;
        }
        let decode_secs = decode_start.elapsed().as_secs_f64();

        // Final flush of any bytes held back for UTF-8 boundary safety.
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
        if text.len() > printed_bytes {
            on_token(&text[printed_bytes..]);
        }

        Ok(GenOutput {
            text,
            prompt_tokens: prompt_ids.len(),
            generated_tokens: generated.len(),
            prefill_secs,
            decode_secs,
        })
    }
}

/// Greedy sampler: argmax over the last (vocab) dimension of a logits tensor.
fn argmax_last_dim(logits: &Tensor) -> Result<u32> {
    let flat = logits.flatten_all()?;
    let idx = flat.argmax(0)?;
    Ok(idx.to_scalar::<u32>()?)
}

/// Dispatch on the tokenizer family declared in the GGUF metadata. Only the
/// implemented families build a real `Tokenizer`; others fail loudly so we
/// never silently use the wrong algorithm.
pub fn build_tokenizer_from_gguf(model: &InferenceModel) -> Result<Tokenizer> {
    let family = model.metadata.tokenizer_model.as_str();
    match family {
        "gpt2" => build_bpe_tokenizer(model),
        "llama" => anyhow::bail!(
            "SentencePiece tokenizer ('llama') not yet implemented — add a builder \
             that reads tokenizer_tokens/scores as a unigram/SP model"
        ),
        "bert" => anyhow::bail!("WordPiece tokenizer ('bert') not yet implemented"),
        "" => {
            anyhow::bail!("GGUF has no tokenizer.ggml.model field — cannot pick a tokenizer family")
        }
        other => anyhow::bail!("Unsupported tokenizer.ggml.model: {other:?}"),
    }
}

/// Build a byte-level BPE tokenizer (the "gpt2" GGUF family). Vocab + merges
/// come from the GGUF; only the pre-tokenizer regex varies, dispatched on
/// `tokenizer.ggml.pre`.
fn build_bpe_tokenizer(model: &InferenceModel) -> Result<Tokenizer> {
    let meta = &model.metadata;
    if meta.tokenizer_tokens.is_empty() {
        anyhow::bail!("GGUF contains no tokenizer.ggml.tokens array");
    }
    if meta.tokenizer_merges.is_empty() {
        anyhow::bail!("GGUF contains no tokenizer.ggml.merges — cannot build BPE");
    }

    let vocab: HashMap<String, u32> = meta
        .tokenizer_tokens
        .iter()
        .enumerate()
        .map(|(i, tok)| (tok.clone(), i as u32))
        .collect();

    let merges: Vec<(String, String)> = meta
        .tokenizer_merges
        .iter()
        .filter_map(|line| {
            let mut it = line.splitn(2, ' ');
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build BPE model: {e}"))?;

    let mut tokenizer = Tokenizer::new(bpe);
    let pre = pre_tokenizer_for_bpe(&meta.tokenizer_pre)?;
    tokenizer
        .with_pre_tokenizer(Some(pre))
        .with_decoder(Some(ByteLevelDecoder::new(false, true, true)));

    // Register control tokens (token_type == 3) so they're matched as a unit.
    let specials: Vec<AddedToken> = meta
        .tokenizer_token_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == 3)
        .filter_map(|(idx, _)| meta.tokenizer_tokens.get(idx).cloned())
        .map(|s| AddedToken::from(s, true))
        .collect();
    if !specials.is_empty() {
        tokenizer.add_special_tokens(&specials);
    }

    Ok(tokenizer)
}

/// Map the GGUF `tokenizer.ggml.pre` tag to the matching pre-tokenizer.
/// Unknown tags fall back to the GPT-2 default with a warning.
fn pre_tokenizer_for_bpe(pre: &str) -> Result<PreTokenizerWrapper> {
    let byte_level_default = || PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, true));

    match pre {
        "default" | "gpt-2" | "qwen2" | "olmo" | "jais" | "smollm" | "" => Ok(byte_level_default()),
        "llama-bpe" => {
            let pattern = SplitPattern::Regex(
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                    .to_string(),
            );
            let split = Split::new(pattern, SplitDelimiterBehavior::Isolated, false)
                .map_err(|e| anyhow::anyhow!("llama-bpe split regex failed: {e}"))?;
            let byte_level = ByteLevel::new(false, true, false);
            Ok(PreTokenizerWrapper::Sequence(PreSequence::new(vec![
                PreTokenizerWrapper::Split(split),
                PreTokenizerWrapper::ByteLevel(byte_level),
            ])))
        }
        other => {
            eprintln!(
                "warning: unknown tokenizer.ggml.pre {other:?}; falling back to GPT-2 default pre-tokenizer"
            );
            Ok(byte_level_default())
        }
    }
}
