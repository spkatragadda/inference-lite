//! Generalized decoder-only Transformer forward pass.
//!
//! Everything in this file is driven by metadata extracted from the GGUF:
//! tensor names, hyperparameters (n_heads / n_kv_heads / head_dim / rope_theta /
//! rms_norm_eps), and the `general.architecture` tag. We currently target the
//! LLaMA-family template (RMSNorm + GQA self-attention with RoPE + SwiGLU MLP),
//! which covers llama / qwen2 / qwen3 / mistral / yi / smollm / deepseek and
//! every other model that follows the same block layout. Non-LLaMA-family
//! architectures (`gpt2`, `bert`, `mamba`, ...) bail with an explicit error
//! rather than producing wrong output silently.
//!
//! Layout of the per-block tensors we look up (all optional bias terms are
//! probed with `_or_none`):
//!     blk.{i}.attn_norm.weight
//!     blk.{i}.attn_q.weight   (+ optional .bias)
//!     blk.{i}.attn_k.weight   (+ optional .bias)
//!     blk.{i}.attn_v.weight   (+ optional .bias)
//!     blk.{i}.attn_output.weight
//!     blk.{i}.ffn_norm.weight
//!     blk.{i}.ffn_gate.weight
//!     blk.{i}.ffn_up.weight
//!     blk.{i}.ffn_down.weight
//! Plus globals: token_embd.weight, output_norm.weight, output.weight
//! (output.weight may be absent if tied to token_embd).

use std::cell::RefCell;

use anyhow::{anyhow, bail, Context};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Tensor, D};
use rayon::prelude::*;

use crate::gguf_parser::{InferenceModel, NamedTensor};

/// Upper bound on the cached sequence length. The KV buffers are preallocated
/// eagerly at this size per layer, so it directly sets the RAM cost:
/// `n_layers * 2 * n_kv_heads * MAX_SEQ_LEN * head_dim * 4 bytes`. Models often
/// advertise a far larger `context_length` (Qwen3-0.6B reports 40960), which we
/// clamp to this. Growing past it is the job of the paged BlockManager; for now
/// we trade a fixed window for zero-allocation decoding.
const MAX_SEQ_LEN: usize = 4096;

/// StreamingLLM (attention-sink) parameters. When the cache fills, the first
/// `DEFAULT_N_SINK` tokens are kept as permanent "sinks" and the oldest recent
/// tokens are evicted in blocks so generation can continue past the window
/// instead of erroring. Each overflow drops `(max_seq - n_sink) / EVICT_FRACTION`
/// recent tokens at once (amortizes the compaction memmove + key re-rotation
/// over many decode steps).
const DEFAULT_N_SINK: usize = 4;
const EVICT_FRACTION: usize = 4;

/// Which QKV layout this block uses. Older LLaMA-style models keep Q, K, V
/// as three separate projection tensors; newer Qwen variants (Qwen3.5,
/// Qwen3-Next) fuse them into a single `attn_qkv.weight` of shape
/// `[(n_q + 2 * n_kv) * head_dim, hidden]` and split along the last axis
/// after the projection.
enum QkvProjection {
    Split {
        q: QMatMul,
        k: QMatMul,
        v: QMatMul,
        q_bias: Option<Tensor>,
        k_bias: Option<Tensor>,
        v_bias: Option<Tensor>,
    },
    Fused {
        qkv: QMatMul,
        bias: Option<Tensor>,
    },
}

/// One layer's KV cache: two flat, padded f32 buffers laid out
/// `[n_kv_heads, max_seq, head_dim]` row-major. The new chunk's K/V is written
/// in place at the current position (an O(q_len) scatter, no reallocation), and
/// the attention kernel reads these buffers *directly* with the right strides —
/// no copy of the prior history out of the cache each decode step. `RefCell`
/// gives the interior mutability the write needs behind the `&self` forward
/// pass; reads borrow immutably for the duration of the kernel.
struct KvLayer {
    k: RefCell<Vec<f32>>,
    v: RefCell<Vec<f32>>,
}

/// One transformer block's worth of weights, in QMatMul / Tensor form.
struct Block {
    attn_norm: Tensor, // f32, shape [hidden]
    qkv: QkvProjection,
    o_proj: QMatMul,
    // Optional per-head Q/K RMSNorms (present in Qwen3 / Qwen3.5).
    // Shape [head_dim]; applied right after Q/K reshape, before RoPE.
    q_head_norm: Option<Tensor>,
    k_head_norm: Option<Tensor>,
    ffn_norm: Tensor,
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

pub struct TransformerModel {
    device: Device,
    embed: Tensor,        // [vocab, hidden], f32
    blocks: Vec<Block>,
    final_norm: Tensor,   // [hidden]
    lm_head: QMatMul,     // [vocab, hidden] — may be tied to embed
    // Hyperparameters
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    rms_eps: f64,
    vocab_size: usize,
    // Preallocated per-layer KV cache, one `KvLayer` per block. Each holds flat
    // f32 buffers of shape [n_kv_heads, max_seq, head_dim]. The new token's K/V
    // is written at slot `position` in place, and the fused attention kernel
    // reads these buffers *directly* via strides — no per-step copy of the
    // history out of the cache, and no `.contiguous()`. This replaces both the
    // original `Tensor::cat(...).contiguous()` cache (O(N^2) recopy) and the
    // interim padded-Tensor cache that still did a `narrow().contiguous()` plus
    // `to_vec1::<f32>()` of the whole prefix every decode step.
    kv_cache: Vec<KvLayer>,
    // Sequence length the buffers above were sized for (the write/read bound).
    max_seq: usize,
    // Precomputed RoPE sin/cos tables, shape [max_seq, head_dim/2], built once
    // at load. `apply_rope_range` slices rows [pos0..pos0+q_len) instead of
    // recomputing powf/cos/sin per call, per layer, per forward.
    rope_cos: Tensor,
    rope_sin: Tensor,
    // Absolute position of the next token to be processed: both the RoPE
    // position and the KV-cache write slot. Under streaming this also moves
    // *backward* on eviction (the cache is compacted so slot == position holds).
    position: RefCell<usize>,
    // StreamingLLM (attention-sink) config. When `enable_streaming` is set, a
    // full cache evicts the oldest recent tokens instead of erroring; the first
    // `n_sink` tokens are never evicted.
    n_sink: usize,
    enable_streaming: bool,
}

impl TransformerModel {
    /// Reset any per-request state. Call this between independent prompts.
    pub fn reset_state(&self) {
        // Only the position needs resetting: the next prompt overwrites slots
        // 0.. as it goes and reads never look past `position`, so any stale
        // data beyond it is never observed. No need to zero the buffers.
        *self.position.borrow_mut() = 0;
    }

    /// Build the model from a parsed GGUF. Hyperparameters come from
    /// `model.metadata`; per-block tensors are looked up by name pattern.
    pub fn load(model: &InferenceModel, device: &Device) -> anyhow::Result<Self> {
        let arch = model.metadata.architecture.as_str();
        // Whitelist of architectures known to follow the LLaMA-family template.
        match arch {
            "llama" | "qwen" | "qwen2" | "qwen3" | "mistral" | "yi" | "smollm"
            | "deepseek" | "deepseek2" | "internlm2" | "stablelm" | "qwen35" => {}
            other => bail!(
                "TransformerModel: architecture {other:?} is not in the LLaMA-family \
                 whitelist. Add it once you've verified its tensor naming matches \
                 (attn_norm/attn_q/.../ffn_gate/ffn_up/ffn_down) and its FFN is SwiGLU."
            ),
        }

        let meta = &model.metadata;
        let n_heads = meta.attention_head_count as usize;
        let n_kv_heads = meta.attention_head_count_kv as usize;
        let hidden_dim = meta.embedding_length as usize;
        // head_dim isn't always hidden/n_heads (Qwen3 sometimes overrides it).
        // Prefer the explicit metadata key if present.
        let head_dim = read_u32(&meta.raw_metadata, &format!("{arch}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_dim / n_heads);
        let rope_theta = meta.rope_dim_theta;
        let rms_eps = read_f32(
            &meta.raw_metadata,
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
        )
        .map(|v| v as f64)
        .unwrap_or(1e-6);

        let embed_q = get_tensor(model, "token_embd.weight")?;
        let embed = embed_q.qtensor.dequantize(device)?.to_dtype(DType::F32)?;
        let vocab_size = embed.dim(0)?;

        let final_norm = get_tensor(model, "output_norm.weight")?
            .qtensor
            .dequantize(device)?
            .to_dtype(DType::F32)?;

        // `output.weight` is sometimes absent and tied to the embedding.
        let lm_head = match model.tensors.get("output.weight") {
            Some(t) => QMatMul::from_arc(t.qtensor.clone())?,
            None => QMatMul::from_arc(embed_q.qtensor.clone())?,
        };

        let n_blocks = meta.block_count as usize;
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            blocks.push(load_block(model, i, device)?);
        }

        // Preallocate the fixed-size KV buffers, one KvLayer per block. Each
        // buffer is a flat [n_kv_heads * max_seq * head_dim] f32 vec.
        let max_seq = (meta.context_length as usize).clamp(1, MAX_SEQ_LEN);

        // StreamingLLM config. Streaming is on by default; n_sink is clamped so
        // at least one recent slot remains evictable.
        let enable_streaming = true;
        let n_sink = DEFAULT_N_SINK.min(max_seq.saturating_sub(1));

        // Precompute the position-independent RoPE tables once for all positions
        // the KV cache can ever address (0..max_seq).
        let (rope_cos, rope_sin) = build_rope_tables(max_seq, head_dim, rope_theta, device)?;
        let layer_elems = n_kv_heads * max_seq * head_dim;
        let mut kv_cache = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            kv_cache.push(KvLayer {
                k: RefCell::new(vec![0f32; layer_elems]),
                v: RefCell::new(vec![0f32; layer_elems]),
            });
        }

        Ok(Self {
            device: device.clone(),
            embed,
            blocks,
            final_norm,
            lm_head,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            rms_eps,
            vocab_size,
            kv_cache,
            max_seq,
            rope_cos,
            rope_sin,
            position: RefCell::new(0),
            n_sink,
            enable_streaming,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Process `token_ids` as one chunk. Prefill feeds the whole prompt at
    /// once (q_len = prompt length); decode feeds a single id (q_len = 1). The
    /// chunk is appended to the KV cache starting at the current `position`,
    /// which is then advanced by `token_ids.len()`. Returns logits for the
    /// LAST position only, shape `[1, vocab]` — the only row that feeds
    /// next-token sampling.
    ///
    /// With streaming enabled, a chunk that would overrun the window is split
    /// into sub-chunks and the oldest recent tokens are evicted between them
    /// (StreamingLLM), so prompts and generations longer than `max_seq` keep
    /// going instead of erroring. With streaming off, the whole chunk is written
    /// in one pass and `write_cache` enforces the hard bound as before.
    pub fn forward_chunk(&self, token_ids: &[u32]) -> anyhow::Result<Tensor> {
        let total = token_ids.len();
        if total == 0 {
            bail!("forward_chunk called with an empty token slice");
        }

        if !self.enable_streaming {
            // Original single-pass path. Check the window bound up front so the
            // error is the informative "cache full" one rather than an opaque
            // RoPE-table narrow failure (table slicing happens before write_cache).
            let pos0 = *self.position.borrow();
            if pos0 + total > self.max_seq {
                bail!(
                    "KV cache full: chunk needs slots [{pos0}..{}) but max_seq is {}. \
                     Raise MAX_SEQ_LEN, lower the prompt/output length, or enable \
                     streaming.",
                    pos0 + total,
                    self.max_seq
                );
            }
            let last = self.run_chunk(token_ids, pos0)?;
            *self.position.borrow_mut() = pos0 + total;
            return self.finish_logits(&last);
        }

        // Streaming path: process in sub-chunks that each fit the remaining
        // window, evicting a block of recent tokens whenever the cache is full.
        // Only the final sub-chunk's last row feeds sampling, so lm_head runs
        // once at the end.
        let mut offset = 0;
        let mut last_hidden = None;
        while offset < total {
            if *self.position.borrow() >= self.max_seq {
                self.evict_block()?;
            }
            let pos0 = *self.position.borrow();
            let room = self.max_seq - pos0; // >= 1 after the eviction above
            let take = (total - offset).min(room);
            let sub = &token_ids[offset..offset + take];
            last_hidden = Some(self.run_chunk(sub, pos0)?);
            *self.position.borrow_mut() = pos0 + take;
            offset += take;
        }

        self.finish_logits(&last_hidden.expect("at least one sub-chunk runs"))
    }

    /// Embedding lookup + all transformer blocks for one sub-chunk whose tokens
    /// occupy absolute slots `[pos0, pos0 + token_ids.len())`. Writes this
    /// chunk's K/V into the cache in place but does NOT advance `position` (the
    /// caller owns that, since eviction can move it). Returns the LAST row's
    /// hidden state `[1, 1, hidden]` for the final norm + lm_head.
    fn run_chunk(&self, token_ids: &[u32], pos0: usize) -> anyhow::Result<Tensor> {
        let q_len = token_ids.len();
        // Embedding lookup for every token in the chunk -> [1, q_len, hidden].
        // One batched gather instead of a per-token row index.
        let ids = Tensor::from_slice(token_ids, (q_len,), &self.device)?;
        let mut x = self
            .embed
            .index_select(&ids, 0)?
            .reshape((1, q_len, self.hidden_dim))?;

        for (i, block) in self.blocks.iter().enumerate() {
            x = self.run_block(&x, block, i, pos0, q_len)?;
        }
        Ok(x.narrow(1, q_len - 1, 1)?) // [1, 1, hidden]
    }

    /// Final RMSNorm + vocab-wide lm_head on a single row -> logits `[1, vocab]`.
    fn finish_logits(&self, last_hidden: &Tensor) -> anyhow::Result<Tensor> {
        let normed = rms_norm(last_hidden, &self.final_norm, self.rms_eps)?;
        Ok(self.lm_head.forward(&normed)?.reshape((1, self.vocab_size))?)
    }

    /// Evict one block of the oldest recent tokens to free cache slots, keeping
    /// the `n_sink` sink tokens. Drops `(max_seq - n_sink) / EVICT_FRACTION`
    /// tokens (>=1), capped by how many recent tokens exist. Compacts each
    /// layer's cache and re-rotates the surviving keys so `slot == position`
    /// still holds, then moves `position` back by the evicted count.
    fn evict_block(&self) -> anyhow::Result<()> {
        let used = *self.position.borrow();
        let recent = used.saturating_sub(self.n_sink);
        if recent == 0 {
            bail!(
                "streaming eviction: nothing to evict (n_sink {} >= used {used}); \
                 raise MAX_SEQ_LEN or lower DEFAULT_N_SINK",
                self.n_sink
            );
        }
        let block = ((self.max_seq - self.n_sink) / EVICT_FRACTION).max(1);
        self.evict(block.min(recent))
    }

    /// Drop the `evict` oldest recent tokens from every layer's KV cache and
    /// re-rotate the survivors to their new positions (see
    /// [`evict_and_rotate_keys`]). Decrements `position` by `evict`.
    fn evict(&self, evict: usize) -> anyhow::Result<()> {
        let used = *self.position.borrow();
        debug_assert!(self.n_sink + evict <= used, "eviction would touch the sinks");
        // Row `evict` of the precomputed tables: cos/sin(evict * theta_d). One
        // lookup, shared across all layers and heads.
        let cos_e = self.rope_cos.narrow(0, evict, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let sin_e = self.rope_sin.narrow(0, evict, 1)?.flatten_all()?.to_vec1::<f32>()?;
        for layer in &self.kv_cache {
            let mut k = layer.k.borrow_mut();
            let mut v = layer.v.borrow_mut();
            evict_and_rotate_keys(
                &mut k, self.n_kv_heads, self.head_dim, self.max_seq, self.n_sink, used,
                evict, &cos_e, &sin_e,
            );
            evict_values(&mut v, self.n_kv_heads, self.head_dim, self.max_seq, self.n_sink, used, evict);
        }
        *self.position.borrow_mut() = used - evict;
        Ok(())
    }

    /// Single-token convenience wrapper over [`Self::forward_chunk`] for the
    /// decode loop. Returns logits of shape `[1, vocab]`.
    pub fn forward(&self, token_id: u32) -> anyhow::Result<Tensor> {
        self.forward_chunk(&[token_id])
    }

    fn run_block(
        &self,
        x: &Tensor,
        block: &Block,
        layer_idx: usize,
        pos0: usize,
        q_len: usize,
    ) -> anyhow::Result<Tensor> {
        // ---- Attention sub-block ----
        // x: [1, q_len, hidden]. q_len is the prompt length during prefill and
        // 1 during decode; every op below is written to handle both.
        let residual = x.clone();
        let h = rms_norm(x, &block.attn_norm, self.rms_eps)?;

        // QKV projections: produce Q [..., n_heads * head_dim],
        //                          K [..., n_kv_heads * head_dim],
        //                          V [..., n_kv_heads * head_dim].
        // Either three separate matmuls (Split) or one big matmul + narrow
        // along the last dim (Fused).
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let (q, k, v) = match &block.qkv {
            QkvProjection::Split {
                q,
                k,
                v,
                q_bias,
                k_bias,
                v_bias,
            } => (
                apply_linear(q, &h, q_bias.as_ref())?,
                apply_linear(k, &h, k_bias.as_ref())?,
                apply_linear(v, &h, v_bias.as_ref())?,
            ),
            QkvProjection::Fused { qkv, bias } => {
                let fused = apply_linear(qkv, &h, bias.as_ref())?;
                let q = fused.narrow(D::Minus1, 0, q_dim)?.contiguous()?;
                let k = fused.narrow(D::Minus1, q_dim, kv_dim)?.contiguous()?;
                let v = fused
                    .narrow(D::Minus1, q_dim + kv_dim, kv_dim)?
                    .contiguous()?;
                (q, k, v)
            }
        };

        // Reshape into heads. Shape: [1, n_heads, q_len, head_dim] etc.
        let q = q
            .reshape((1, q_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, q_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, q_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Optional per-head Q/K RMSNorms (Qwen3 / Qwen3.5). Weight shape is
        // [head_dim], which broadcasts over the leading [1, n_heads, q_len, _]
        // dims of `q` and `k`.
        let q = match &block.q_head_norm {
            Some(w) => rms_norm(&q, w, self.rms_eps)?,
            None => q,
        };
        let k = match &block.k_head_norm {
            Some(w) => rms_norm(&k, w, self.rms_eps)?,
            None => k,
        };

        // RoPE on Q and K. Row j along the seq axis is at absolute position
        // pos0 + j (so prefill rotates each prompt token by its own position).
        let q = apply_rope_range(&q, pos0, &self.rope_cos, &self.rope_sin)?;
        let k = apply_rope_range(&k, pos0, &self.rope_cos, &self.rope_sin)?;

        // Write this chunk's (k, v) into the per-layer KV cache at slots
        // [pos0..pos0+q_len). O(q_len) scatter — no read-back of the history.
        self.write_cache(layer_idx, pos0, q_len, &k, &v)?;

        // Fused causal SDPA with online softmax. Operates on raw f32 slices:
        // streams over the KV prefix once per (head, query row), maps Q heads
        // to KV heads on the fly (no repeat_kv_heads materialization), and
        // never builds the [n_heads, q_len, kv_len] score matrix or its
        // softmax temporaries. It reads the padded cache buffers
        // ([n_kv_heads, max_seq, head_dim]) directly: per-head stride is
        // `max_seq * head_dim`, per-position stride is `head_dim`. Only the
        // [0..pos0+q_len) prefix of each head is populated, and the kernel's
        // causal bound never reads past it.
        let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
        let kv = &self.kv_cache[layer_idx];
        let k_buf = kv.k.borrow();
        let v_buf = kv.v.borrow();
        let out = fused_sdpa(
            &q_vec,
            &k_buf,
            &v_buf,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            q_len,
            pos0,
            self.max_seq * self.head_dim,
            self.head_dim,
        );
        let attn = Tensor::from_vec(out, (1, self.n_heads, q_len, self.head_dim), &self.device)?;

        // Merge heads back: [1, q_len, hidden]
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, q_len, self.n_heads * self.head_dim))?;

        let attn_out = block.o_proj.forward(&attn)?;
        let x = (residual + attn_out)?;

        // ---- FFN sub-block (SwiGLU) ----
        let residual = x.clone();
        let h = rms_norm(&x, &block.ffn_norm, self.rms_eps)?;
        let gate = block.gate_proj.forward(&h)?;
        let up = block.up_proj.forward(&h)?;
        let gated = silu(&gate)?.mul(&up)?;
        let ffn_out = block.down_proj.forward(&gated)?;
        Ok((residual + ffn_out)?)
    }

    /// Scatter this chunk's K/V into the per-layer cache buffers in place.
    ///
    /// `k_new` / `v_new` are `[1, n_kv_heads, q_len, head_dim]`, contiguous
    /// (built via reshape/transpose/contiguous + RoPE upstream), so their flat
    /// layout is `[h * q_len * head_dim + j * head_dim + d]`. The cache buffers
    /// are padded `[n_kv_heads, max_seq, head_dim]`, so token `j` of head `h`
    /// lands at `h * max_seq * head_dim + (pos0 + j) * head_dim + d`. We copy a
    /// whole `head_dim`-length row at a time. No reallocation, no read-back of
    /// the existing history — that's the O(N^2)-copy elimination over a decode.
    fn write_cache(
        &self,
        layer_idx: usize,
        pos0: usize,
        q_len: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> anyhow::Result<()> {
        let kv_len = pos0 + q_len;
        if kv_len > self.max_seq {
            bail!(
                "KV cache full: chunk needs slots [{pos0}..{kv_len}) but max_seq \
                 is {} (MAX_SEQ_LEN). Raise MAX_SEQ_LEN or move to paged allocation.",
                self.max_seq
            );
        }

        let head_dim = self.head_dim;
        let row = head_dim; // elements per (head, position) row
        let dst_head_stride = self.max_seq * head_dim;
        let src_head_stride = q_len * head_dim;

        let k_src = k_new.flatten_all()?.to_vec1::<f32>()?;
        let v_src = v_new.flatten_all()?.to_vec1::<f32>()?;

        let kv = &self.kv_cache[layer_idx];
        let mut k_buf = kv.k.borrow_mut();
        let mut v_buf = kv.v.borrow_mut();

        for h in 0..self.n_kv_heads {
            for j in 0..q_len {
                let dst = h * dst_head_stride + (pos0 + j) * head_dim;
                let src = h * src_head_stride + j * head_dim;
                k_buf[dst..dst + row].copy_from_slice(&k_src[src..src + row]);
                v_buf[dst..dst + row].copy_from_slice(&v_src[src..src + row]);
            }
        }
        Ok(())
    }
}

fn load_block(model: &InferenceModel, i: usize, device: &Device) -> anyhow::Result<Block> {
    let p = format!("blk.{i}");
    let prefix = format!("{p}.");

    // Bail early on SSM / DeltaNet style blocks. These use a completely
    // different forward (input proj -> 1D conv -> SiLU -> state-space scan
    // -> output proj) and need their own implementation. We list EVERY
    // tensor under this block (not just ssm_*) so the next implementation
    // pass has the full layout — Q/K/V projections often live under names
    // outside the ssm_ namespace (e.g. attn_q.weight, linear_attn_*, or a
    // fused ssm_in.weight).
    let has_ssm = model
        .tensors
        .keys()
        .any(|k| k.starts_with(&prefix) && k.contains(".ssm_"));
    if has_ssm {
        let mut all_in_block: Vec<(String, Vec<usize>, candle_core::quantized::GgmlDType)> = model
            .tensors
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(k, t)| (k.clone(), t.shape(), t.qtensor.dtype()))
            .collect();
        all_in_block.sort_by(|a, b| a.0.cmp(&b.0));
        let listing = all_in_block
            .iter()
            .map(|(n, s, d)| format!("{n:<40} shape={s:?} dtype={d:?}"))
            .collect::<Vec<_>>()
            .join("\n  ");
        bail!(
            "blk.{i}: detected SSM/DeltaNet block. SSM forward is not \
             implemented yet — extend Block into an enum with an SSM variant \
             and add a scan kernel. Tensors under {prefix:?}:\n  {listing}"
        );
    }

    let attn_norm = require_first(
        model,
        &[
            format!("{p}.attn_norm.weight"),
            format!("{p}.input_layernorm.weight"),
        ],
        &prefix,
    )?
    .qtensor
    .dequantize(device)?
    .to_dtype(DType::F32)?;

    let ffn_norm = require_first(
        model,
        &[
            format!("{p}.ffn_norm.weight"),
            format!("{p}.post_attention_norm.weight"),
            format!("{p}.post_attention_layernorm.weight"),
        ],
        &prefix,
    )?
    .qtensor
    .dequantize(device)?
    .to_dtype(DType::F32)?;

    let qkv = build_qkv_projection(model, &p, device)?;

    Ok(Block {
        attn_norm,
        qkv,
        o_proj: qmatmul(model, &format!("{p}.attn_output.weight"))?,
        q_head_norm: optional_norm(model, &format!("{p}.attn_q_norm.weight"), device)?,
        k_head_norm: optional_norm(model, &format!("{p}.attn_k_norm.weight"), device)?,
        ffn_norm,
        gate_proj: qmatmul(model, &format!("{p}.ffn_gate.weight"))?,
        up_proj: qmatmul(model, &format!("{p}.ffn_up.weight"))?,
        down_proj: qmatmul(model, &format!("{p}.ffn_down.weight"))?,
    })
}

/// Pick between fused (`attn_qkv.weight`) and split (`attn_q/k/v.weight`)
/// QKV layouts, or fail with a full block-prefix listing if neither is
/// recognized.
fn build_qkv_projection(
    model: &InferenceModel,
    p: &str,
    device: &Device,
) -> anyhow::Result<QkvProjection> {
    let prefix = format!("{p}.");

    // 1) Fused: single `attn_qkv.weight` (optionally with `.bias`).
    if let Some(t) = model.tensors.get(&format!("{p}.attn_qkv.weight")) {
        return Ok(QkvProjection::Fused {
            qkv: QMatMul::from_arc(t.qtensor.clone())
                .map_err(|e| anyhow!("QMatMul build failed for {p}.attn_qkv.weight: {e}"))?,
            bias: optional_bias(model, &format!("{p}.attn_qkv.bias"), device)?,
        });
    }

    // 2) Split: three separate Q/K/V projections.
    let has_split = ["attn_q.weight", "attn_k.weight", "attn_v.weight"]
        .iter()
        .all(|s| model.tensors.contains_key(&format!("{p}.{s}")));
    if has_split {
        return Ok(QkvProjection::Split {
            q: qmatmul(model, &format!("{p}.attn_q.weight"))?,
            k: qmatmul(model, &format!("{p}.attn_k.weight"))?,
            v: qmatmul(model, &format!("{p}.attn_v.weight"))?,
            q_bias: optional_bias(model, &format!("{p}.attn_q.bias"), device)?,
            k_bias: optional_bias(model, &format!("{p}.attn_k.bias"), device)?,
            v_bias: optional_bias(model, &format!("{p}.attn_v.bias"), device)?,
        });
    }

    // 3) Neither layout matched — surface every tensor under this block so
    //    we can extend the loader by adding the actual name we find.
    let mut nearby: Vec<&String> = model
        .tensors
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .collect();
    nearby.sort();
    bail!(
        "{p}: no recognized QKV layout. Tried fused {:?} and split \
         {:?}/{:?}/{:?}. Tensors present under {:?}:\n  {}",
        format!("{p}.attn_qkv.weight"),
        format!("{p}.attn_q.weight"),
        format!("{p}.attn_k.weight"),
        format!("{p}.attn_v.weight"),
        prefix,
        nearby
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

// ---------- small helpers ----------

fn get_tensor<'a>(model: &'a InferenceModel, name: &str) -> anyhow::Result<&'a NamedTensor> {
    model
        .tensors
        .get(name)
        .with_context(|| format!("GGUF missing required tensor {name:?}"))
}

/// Try a list of candidate tensor names and return the first one that
/// exists. If none match, fail with an error that also lists every tensor
/// in the GGUF whose name starts with `layer_prefix` — that makes the
/// error actionable (you can immediately see what the GGUF actually calls
/// the missing piece and add a new candidate to the list).
fn require_first<'a>(
    model: &'a InferenceModel,
    candidates: &[String],
    layer_prefix: &str,
) -> anyhow::Result<&'a NamedTensor> {
    for name in candidates {
        if let Some(t) = model.tensors.get(name) {
            return Ok(t);
        }
    }
    let mut nearby: Vec<&String> = model
        .tensors
        .keys()
        .filter(|k| k.starts_with(layer_prefix))
        .collect();
    nearby.sort();
    bail!(
        "GGUF missing required tensor. Tried: {:?}.\nTensors present under {:?}:\n  {}",
        candidates,
        layer_prefix,
        nearby
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// Look up an optional norm tensor by name; dequantize to f32 if present,
/// return None otherwise. Used for the per-head Q/K RMSNorms that only
/// show up in some architectures (Qwen3 / Qwen3.5).
fn optional_norm(
    model: &InferenceModel,
    name: &str,
    device: &Device,
) -> anyhow::Result<Option<Tensor>> {
    match model.tensors.get(name) {
        None => Ok(None),
        Some(t) => Ok(Some(t.qtensor.dequantize(device)?.to_dtype(DType::F32)?)),
    }
}

fn qmatmul(model: &InferenceModel, name: &str) -> anyhow::Result<QMatMul> {
    let t = get_tensor(model, name)?;
    QMatMul::from_arc(t.qtensor.clone())
        .map_err(|e| anyhow!("QMatMul build failed for {name}: {e}"))
}

fn optional_bias(
    model: &InferenceModel,
    name: &str,
    device: &Device,
) -> anyhow::Result<Option<Tensor>> {
    match model.tensors.get(name) {
        None => Ok(None),
        Some(t) => Ok(Some(t.qtensor.dequantize(device)?.to_dtype(DType::F32)?)),
    }
}

fn apply_linear(w: &QMatMul, x: &Tensor, bias: Option<&Tensor>) -> anyhow::Result<Tensor> {
    let y = w.forward(x)?;
    Ok(match bias {
        None => y,
        Some(b) => y.broadcast_add(b)?,
    })
}

fn read_u32(meta: &std::collections::HashMap<String, gguf_file::Value>, key: &str) -> Option<u32> {
    match meta.get(key)? {
        gguf_file::Value::U32(v) => Some(*v),
        _ => None,
    }
}

fn read_f32(meta: &std::collections::HashMap<String, gguf_file::Value>, key: &str) -> Option<f32> {
    match meta.get(key)? {
        gguf_file::Value::F32(v) => Some(*v),
        _ => None,
    }
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> anyhow::Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let sq = x32.sqr()?;
    let mean = sq.mean_keepdim(D::Minus1)?;
    let inv = (mean + eps)?.sqrt()?.recip()?;
    let normed = x32.broadcast_mul(&inv)?;
    Ok(normed.broadcast_mul(weight)?)
}

fn silu(x: &Tensor) -> anyhow::Result<Tensor> {
    // x / (1 + exp(-x))
    let denom = x.neg()?.exp()?.affine(1.0, 1.0)?;
    Ok(x.div(&denom)?)
}

/// Apply rotary positional embeddings (split / NeoX convention used by LLaMA /
/// Qwen) to a tensor shaped `[1, n_heads, q_len, head_dim]`. Row `j` along the
/// seq axis is treated as absolute position `pos0 + j`, so during prefill each
/// prompt token is rotated by its own position in one call.
fn apply_rope_range(
    x: &Tensor,
    pos0: usize,
    cos_tab: &Tensor,
    sin_tab: &Tensor,
) -> anyhow::Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let q_len = x.dim(2)?;
    let half = head_dim / 2;

    // Slice the precomputed tables for positions pos0..pos0+q_len. Shape
    // [1, 1, q_len, half] broadcasts over the n_heads axis. The KV-cache bound
    // (pos0 + q_len <= max_seq, enforced in write_cache) guarantees this slice
    // stays in range, since the tables are sized to max_seq.
    let cos = cos_tab.narrow(0, pos0, q_len)?.reshape((1, 1, q_len, half))?;
    let sin = sin_tab.narrow(0, pos0, q_len)?.reshape((1, 1, q_len, half))?;

    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated_1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Ok(Tensor::cat(&[&rotated_1, &rotated_2], D::Minus1)?.contiguous()?)
}

/// Precompute the RoPE sin/cos tables for every position in `0..max_seq`,
/// returning two `[max_seq, head_dim/2]` f32 tensors. The frequency
/// `theta_base^(-2i/head_dim)` is position-independent, so building this once at
/// load replaces the per-call powf/cos/sin work in [`apply_rope_range`]. This is
/// also the seam for any RoPE scaling (YaRN / NTK / linear interpolation).
fn build_rope_tables(
    max_seq: usize,
    head_dim: usize,
    theta_base: f32,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos_vals = Vec::with_capacity(max_seq * half);
    let mut sin_vals = Vec::with_capacity(max_seq * half);
    for pos in 0..max_seq {
        for i in 0..half {
            let freq = (theta_base).powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * freq;
            cos_vals.push(angle.cos());
            sin_vals.push(angle.sin());
        }
    }
    let cos = Tensor::from_vec(cos_vals, (max_seq, half), device)?;
    let sin = Tensor::from_vec(sin_vals, (max_seq, half), device)?;
    Ok((cos, sin))
}

/// Slide the recent-window keys down by `evict` slots and re-rotate them to
/// their new (lower) RoPE positions — the StreamingLLM (attention-sink) cache
/// compaction step. Operates in place on one layer's flat K buffer, shaped
/// `[n_kv_heads, max_seq, head_dim]` (per-head stride `max_seq * head_dim`).
///
/// Layout invariant kept by this routine: `slot index == RoPE position`. Sink
/// slots `[0, n_sink)` are never touched. The `evict` oldest *recent* slots
/// `[n_sink, n_sink+evict)` are dropped; survivors `[n_sink+evict, used)` move
/// down to `[n_sink, used-evict)`.
///
/// Keys are stored already rotated by their absolute position `p` (RoPE is
/// applied before `write_cache`). After the shift a survivor must be rotated by
/// `p - evict` instead. Going from `p` to `p - evict` is a rotation by
/// `-evict * theta_d` per frequency pair `(d, d+half)`, which is *independent of
/// `p`* — so every survivor gets the same correction. The needed factors are
/// exactly row `evict` of the precomputed tables:
///   `cos_e[d] = cos(evict * theta_d)`, `sin_e[d] = sin(evict * theta_d)`.
/// For a stored pair `(a, b) = (k[d], k[half+d])` the inverse rotation `R(-Eθ)`
/// gives `(a*cos_e + b*sin_e, -a*sin_e + b*cos_e)`, which equals rotating the
/// raw key by `p - evict` (verified in `delta_rotate_matches_rerotation`).
///
/// In-place safety: the move is downward (`dst = src - evict*head_dim`), so a
/// destination row never overlaps its own source row. Iterating slots ascending
/// means each slot is read as a source before it can be overwritten as a
/// destination (its write happens `evict` steps later), so no live data is
/// clobbered.
#[allow(clippy::too_many_arguments)]
fn evict_and_rotate_keys(
    k_buf: &mut [f32],
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    n_sink: usize,
    used: usize,
    evict: usize,
    cos_e: &[f32],
    sin_e: &[f32],
) {
    let half = head_dim / 2;
    let head_stride = max_seq * head_dim;
    for h in 0..n_kv_heads {
        let base = h * head_stride;
        for slot in (n_sink + evict)..used {
            let src = base + slot * head_dim;
            let dst = base + (slot - evict) * head_dim;
            for d in 0..half {
                let a = k_buf[src + d];
                let b = k_buf[src + half + d];
                let c = cos_e[d];
                let s = sin_e[d];
                k_buf[dst + d] = a * c + b * s;
                k_buf[dst + half + d] = -a * s + b * c;
            }
        }
    }
}

/// V counterpart of [`evict_and_rotate_keys`]: V is never rotated, so eviction
/// is a pure downward shift of the recent window. Each head's survivor span is
/// contiguous, so it moves in one `copy_within` (memmove semantics handle the
/// overlap) rather than slot by slot.
fn evict_values(
    v_buf: &mut [f32],
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    n_sink: usize,
    used: usize,
    evict: usize,
) {
    let head_stride = max_seq * head_dim;
    for h in 0..n_kv_heads {
        let base = h * head_stride;
        let src_start = base + (n_sink + evict) * head_dim;
        let src_end = base + used * head_dim;
        let dst_start = base + n_sink * head_dim;
        v_buf.copy_within(src_start..src_end, dst_start);
    }
}

/// Fused causal scaled-dot-product attention with online (streaming) softmax.
///
/// One pass over the KV prefix per `(head, query row)`: never materializes the
/// `[n_heads, q_len, kv_len]` score matrix, never repeats KV heads (Q→KV head
/// mapping is done by index), and keeps softmax numerically stable via a
/// running max + running denominator. All f32, CPU.
///
/// Row-major indexing (the caller passes the strides so this works on either a
/// tight contiguous prefix or a padded buffer read in place):
///   q[h, qi, d]     = q[h * (q_len * head_dim) + qi * head_dim + d]
///   k/v[kvh, kj, d] = k[kvh * kv_head_stride + kj * kv_pos_stride + d]
///
/// Causal rule: query row `qi` (absolute position `pos0 + qi`) attends to key
/// positions `kj` in `0..=(pos0 + qi)`. Returns out laid out like `q`.
#[allow(clippy::too_many_arguments)]
fn fused_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_len: usize,
    pos0: usize,
    kv_head_stride: usize,
    kv_pos_stride: usize,
) -> Vec<f32> {
    let group = n_heads / n_kv_heads; // GQA: query heads per KV head
    let scale = (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * q_len * head_dim];

    // Per-head attention. `h` is the query head; `out_head` is its disjoint
    // `q_len * head_dim` output slice. Reads only the shared q/k/v slices, so
    // heads carry no cross-dependency. Indices into `out_head` are head-local
    // (the `h * q_len * head_dim` base is implicit in the chunk).
    let attend_head = |h: usize, out_head: &mut [f32]| {
        let k_base = (h / group) * kv_head_stride; // map Q head -> KV head
        let qh = h * q_len * head_dim;
        for qi in 0..q_len {
            let q_row = &q[qh + qi * head_dim..qh + qi * head_dim + head_dim];
            let last_key = pos0 + qi; // inclusive causal bound

            let mut m = f32::NEG_INFINITY; // running max
            let mut l = 0f32; // running denominator
            let mut acc = vec![0f32; head_dim]; // running sum of p * V

            for kj in 0..=last_key {
                let off = k_base + kj * kv_pos_stride;
                let k_row = &k[off..off + head_dim];
                let mut s = 0f32;
                for d in 0..head_dim {
                    s += q_row[d] * k_row[d];
                }
                s /= scale;

                // Online softmax: rescale the running totals by exp(m - new_m)
                // when the max grows, then fold in the new key. On the first
                // iteration m = -inf so corr = exp(-inf) = 0, which zeroes the
                // (empty) accumulators before adding the first contribution.
                let new_m = m.max(s);
                let corr = (m - new_m).exp();
                let p = (s - new_m).exp();
                l = l * corr + p;
                let v_row = &v[off..off + head_dim];
                for d in 0..head_dim {
                    acc[d] = acc[d] * corr + p * v_row[d];
                }
                m = new_m;
            }

            let inv = 1.0 / l;
            let o_row = &mut out_head[qi * head_dim..qi * head_dim + head_dim];
            for d in 0..head_dim {
                o_row[d] = acc[d] * inv;
            }
        }
    };

    // Heads are embarrassingly parallel, but the rayon fork/join costs a few
    // tens of microseconds — more than the whole kernel at short-context decode
    // (q_len=1, small kv_len), where multithreading measured ~2.4x *slower*.
    // It's a 5–6x win once the work is large (long-context decode, or any
    // prefill). Gate on a cheap estimate of total inner iterations,
    // `n_heads * Σ_qi (pos0 + qi + 1)`, and only fan out past the crossover.
    // (Threshold ~8k iters sits between decode kv=128 ≈ 2k and kv=1024 ≈ 16k,
    // measured on a 12-core box; see bench_sdpa.)
    const PAR_WORK_THRESHOLD: usize = 8192;
    let key_iters_per_head = q_len * (2 * pos0 + q_len + 1) / 2;
    let total_iters = n_heads * key_iters_per_head;

    if total_iters >= PAR_WORK_THRESHOLD {
        out.par_chunks_mut(q_len * head_dim)
            .enumerate()
            .for_each(|(h, out_head)| attend_head(h, out_head));
    } else {
        out.chunks_mut(q_len * head_dim)
            .enumerate()
            .for_each(|(h, out_head)| attend_head(h, out_head));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference: dense causal attention via candle ops
    /// (repeat-interleave KV -> Q·Kᵀ/√d -> +causal_mask -> softmax -> ·V).
    /// Deliberately uses a different code path than `fused_sdpa` so agreement
    /// is a real cross-check, not a tautology.
    fn reference_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        q_len: usize,
        kv_len: usize,
        pos0: usize,
    ) -> Tensor {
        let device = q.device();
        let group = n_heads / n_kv_heads;

        // Repeat-interleave KV heads from n_kv_heads to n_heads via cat (avoids
        // any expand/reshape contiguity subtleties — this is a test reference).
        let repeat = |t: &Tensor| -> Tensor {
            let mut heads = Vec::with_capacity(n_heads);
            for kvh in 0..n_kv_heads {
                let head = t.narrow(1, kvh, 1).unwrap(); // [1,1,kv_len,head_dim]
                for _ in 0..group {
                    heads.push(head.clone());
                }
            }
            Tensor::cat(&heads, 1).unwrap().contiguous().unwrap()
        };
        let k_r = repeat(k);
        let v_r = repeat(v);

        let scale = (head_dim as f64).sqrt();
        let scores =
            (q.matmul(&k_r.transpose(2, 3).unwrap().contiguous().unwrap()).unwrap() / scale)
                .unwrap(); // [1, n_heads, q_len, kv_len]

        // Causal mask: query row qi (abs pos pos0+qi) may attend to kj<=pos0+qi.
        let mut mask = vec![0f32; q_len * kv_len];
        for qi in 0..q_len {
            for kj in 0..kv_len {
                if kj > pos0 + qi {
                    mask[qi * kv_len + kj] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, q_len, kv_len), device).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();

        // Stable softmax over the last dim.
        let max = scores.max_keepdim(D::Minus1).unwrap();
        let exp = scores.broadcast_sub(&max).unwrap().exp().unwrap();
        let sum = exp.sum_keepdim(D::Minus1).unwrap();
        let weights = exp.broadcast_div(&sum).unwrap();
        weights.matmul(&v_r).unwrap() // [1, n_heads, q_len, head_dim]
    }

    fn check_case(n_heads: usize, n_kv_heads: usize, head_dim: usize, q_len: usize, pos0: usize) {
        let device = Device::Cpu;
        let kv_len = pos0 + q_len;
        let q = Tensor::rand(-1f32, 1f32, (1, n_heads, q_len, head_dim), &device).unwrap();
        let k = Tensor::rand(-1f32, 1f32, (1, n_kv_heads, kv_len, head_dim), &device).unwrap();
        let v = Tensor::rand(-1f32, 1f32, (1, n_kv_heads, kv_len, head_dim), &device).unwrap();

        let reference =
            reference_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, q_len, kv_len, pos0);
        let ref_vec = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let q_vec = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let k_vec = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v_vec = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let fused = fused_sdpa(
            &q_vec,
            &k_vec,
            &v_vec,
            n_heads,
            n_kv_heads,
            head_dim,
            q_len,
            pos0,
            kv_len * head_dim,
            head_dim,
        );

        assert_eq!(fused.len(), ref_vec.len());
        let max_diff = fused
            .iter()
            .zip(ref_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "mismatch (n_heads={n_heads}, n_kv={n_kv_heads}, d={head_dim}, \
             q_len={q_len}, pos0={pos0}): max_abs_diff={max_diff}"
        );
    }

    #[test]
    fn fused_sdpa_matches_dense_softmax() {
        // Decode step (q_len = 1), GQA, fresh and with history.
        check_case(16, 8, 128, 1, 0);
        check_case(16, 8, 128, 1, 23);
        // Batched prefill (q_len > 1) — exercises intra-chunk causal masking.
        check_case(16, 8, 128, 7, 0);
        check_case(16, 8, 128, 5, 11);
        // MHA (no GQA) and MQA (single KV head).
        check_case(8, 8, 64, 4, 3);
        check_case(8, 1, 64, 6, 2);
    }

    /// The StreamingLLM compaction identity: a key stored rotated by position
    /// `p`, then delta-rotated by `evict` via `evict_and_rotate_keys`, must equal
    /// the raw key rotated directly by `p - evict`. Exact up to fp rounding.
    #[test]
    fn delta_rotate_matches_rerotation() {
        let device = Device::Cpu;
        let head_dim = 8;
        let max_seq = 64;
        let theta = 10000.0f32;
        let (cos_tab, sin_tab) = build_rope_tables(max_seq, head_dim, theta, &device).unwrap();

        // Forward-rotate a raw head_dim vector by absolute position `p`, exactly
        // as apply_rope_range does at write time.
        let rotate_at = |raw: &[f32], p: usize| -> Vec<f32> {
            let x = Tensor::from_vec(raw.to_vec(), (1, 1, 1, head_dim), &device).unwrap();
            apply_rope_range(&x, p, &cos_tab, &sin_tab)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        // Row `e` of a table = cos/sin(e * theta_d), as `evict` extracts it.
        let row = |tab: &Tensor, e: usize| -> Vec<f32> {
            tab.narrow(0, e, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };

        let n_kv_heads = 2;
        let head_stride = max_seq * head_dim;
        // (position of the survivor, evict count, sink count). Requires
        // p in [n_sink + evict, used) with used = p + 1.
        for &(p, evict, n_sink) in &[(40usize, 7usize, 2usize), (12, 4, 0), (63, 1, 3)] {
            let used = p + 1;
            let mut k_buf = vec![0f32; n_kv_heads * head_stride];
            let mut raws = Vec::new();
            for h in 0..n_kv_heads {
                // Distinct raw key per head.
                let raw: Vec<f32> = (0..head_dim)
                    .map(|d| (h as f32 + 1.0) * (d as f32 - 3.5))
                    .collect();
                let stored = rotate_at(&raw, p); // as written into the cache
                let dst = h * head_stride + p * head_dim;
                k_buf[dst..dst + head_dim].copy_from_slice(&stored);
                raws.push(raw);
            }

            let cos_e = row(&cos_tab, evict);
            let sin_e = row(&sin_tab, evict);
            evict_and_rotate_keys(
                &mut k_buf, n_kv_heads, head_dim, max_seq, n_sink, used, evict, &cos_e, &sin_e,
            );

            for h in 0..n_kv_heads {
                let expected = rotate_at(&raws[h], p - evict);
                let off = h * head_stride + (p - evict) * head_dim;
                let got = &k_buf[off..off + head_dim];
                let max_diff = got
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max_diff < 1e-5, "p={p} evict={evict} h={h}: max_diff={max_diff}");
            }
        }
    }

    /// `evict_values` shifts the recent window down by `evict` slots, leaves the
    /// sinks in place, and lands old slot `s+evict` at new slot `s`.
    #[test]
    fn evict_values_shifts_window() {
        let head_dim = 4;
        let max_seq = 16;
        let n_kv_heads = 2;
        let head_stride = max_seq * head_dim;
        let (used, n_sink, evict) = (10usize, 2usize, 3usize);

        // Tag each (head, slot) row with a unique value head*100 + slot.
        let mut v = vec![0f32; n_kv_heads * head_stride];
        for h in 0..n_kv_heads {
            for slot in 0..used {
                let off = h * head_stride + slot * head_dim;
                for d in 0..head_dim {
                    v[off + d] = (h * 100 + slot) as f32;
                }
            }
        }

        evict_values(&mut v, n_kv_heads, head_dim, max_seq, n_sink, used, evict);

        for h in 0..n_kv_heads {
            // Sinks untouched.
            for slot in 0..n_sink {
                let off = h * head_stride + slot * head_dim;
                assert_eq!(v[off], (h * 100 + slot) as f32);
            }
            // Survivors shifted down: new slot `s` holds old slot `s + evict`.
            for s in n_sink..(used - evict) {
                let off = h * head_stride + s * head_dim;
                assert_eq!(v[off], (h * 100 + s + evict) as f32, "h={h} s={s}");
            }
        }
    }
}
