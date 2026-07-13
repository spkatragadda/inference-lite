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

use crate::forward::{TransformerModel, WeightPrecision};
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
    /// How many prompt tokens were served from the prefix cache (their K/V was
    /// already computed by an earlier request); only the rest were prefilled.
    pub reused_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

impl Engine {
    /// Parse the GGUF, build the transformer and tokenizer. This is the slow,
    /// one-time step (dequantizes the embedding, allocates the KV cache, etc.).
    /// `precision` selects how linear weights are stored / which matmul kernel
    /// runs (see [`WeightPrecision`]); `WeightPrecision::F16` is the faster CPU
    /// default at ~2x the weight RAM.
    pub fn load(model_path: &str, device: &Device, precision: WeightPrecision) -> Result<Self> {
        let model = parse_gguf(model_path)?;
        let arch = model.metadata.architecture.clone();
        let eos_id = model.metadata.eos_token_id;
        let tokenizer = build_tokenizer_from_gguf(&model)?;
        let transformer = TransformerModel::load(&model, device, precision)?;
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

    /// Greedy generation for a single prompt. Reuses the KV cache for any
    /// prefix shared with the previous request (prefix caching — in a chat
    /// session this skips re-prefilling the transcript), optionally wraps the
    /// prompt in the ChatML template, prefills the remaining suffix in one
    /// batch, then decodes up to `max_new_tokens` (stopping on EOS). `on_token` is
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
            // Plain Qwen2.5 ChatML template (no forced <think> block). VibeThinker
            // is a reasoning model built on Qwen2.5-Math; it emits its own
            // chain-of-thought, so injecting an empty think block would suppress
            // the reasoning this model exists to produce.
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
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

        // Prefix-cache reuse: in a chat session each turn's prompt extends the
        // previous transcript, so the K/V for the shared prefix is already in
        // the cache. Roll back to the longest common prefix and prefill only
        // the new suffix (at least the final token, so logits exist to sample
        // from). Falls back to a full prefill on the first request or after
        // StreamingLLM eviction has compacted the cache.
        let reused_tokens = self.transformer.reuse_prefix(&prompt_ids);

        // Batched prefill of the (remaining) prompt, logits for the last token.
        let prefill_start = Instant::now();
        let mut logits = self.transformer.forward_chunk(&prompt_ids[reused_tokens..])?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();

        // Greedy decode loop.
        //
        // Incremental detokenization: rather than re-decoding the whole sequence
        // every step (O(N) per token -> O(N^2) over a generation, which on the
        // multi-thousand-token reasoning traces this model produces is real CPU
        // cost serialized into the decode loop), we decode only the
        // not-yet-emitted tail. `pending` holds the tokens whose bytes haven't
        // formed a complete UTF-8 string yet. The byte-level BPE decoder maps
        // each token to raw bytes and concatenates them with no cross-token
        // contextual stripping, so decode(tail) is exactly the corresponding
        // byte-slice of the full decode — provided we only ever split on a valid
        // UTF-8 boundary. We hold a chunk back whenever it ends mid-multi-byte
        // char (trailing U+FFFD) and flush it once the next token completes it.
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut pending: Vec<u32> = Vec::new();
        let mut text = String::new();
        let decode_start = Instant::now();
        for _ in 0..max_new_tokens {
            let next = argmax_last_dim(&logits)?;
            if Some(next) == self.eos_id {
                break;
            }
            generated.push(next);
            pending.push(next);

            let chunk = self
                .tokenizer
                .decode(&pending, true)
                .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
            // Emit only once the pending tokens decode to a complete string:
            // non-empty and not ending on a partial multi-byte char. Skipped
            // special tokens decode to "" and simply stay in `pending` until a
            // real token follows — harmless, as they contribute no bytes.
            if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                on_token(&chunk);
                text.push_str(&chunk);
                pending.clear();
            }

            logits = self.transformer.forward(next)?;
        }
        let decode_secs = decode_start.elapsed().as_secs_f64();

        // Final flush: emit any bytes still held back for UTF-8 boundary safety
        // (an unfinished multi-byte char at the EOS / length cutoff).
        if !pending.is_empty() {
            let tail = self
                .tokenizer
                .decode(&pending, true)
                .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
            if !tail.is_empty() {
                on_token(&tail);
                text.push_str(&tail);
            }
        }

        Ok(GenOutput {
            text,
            prompt_tokens: prompt_ids.len(),
            reused_tokens,
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
