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
    rope_theta: f32,
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
    // Absolute position of the next token to be processed: both the RoPE
    // position and the KV-cache write slot.
    position: RefCell<usize>,
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
            rope_theta,
            rms_eps,
            vocab_size,
            kv_cache,
            max_seq,
            position: RefCell::new(0),
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
    pub fn forward_chunk(&self, token_ids: &[u32]) -> anyhow::Result<Tensor> {
        let q_len = token_ids.len();
        if q_len == 0 {
            bail!("forward_chunk called with an empty token slice");
        }
        let pos0 = *self.position.borrow();

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

        // Only the last position feeds sampling, so run the final RMSNorm and
        // the vocab-wide lm_head on a single row rather than all q_len rows.
        let last = x.narrow(1, q_len - 1, 1)?; // [1, 1, hidden]
        let normed = rms_norm(&last, &self.final_norm, self.rms_eps)?;
        let logits = self.lm_head.forward(&normed)?.reshape((1, self.vocab_size))?;

        *self.position.borrow_mut() = pos0 + q_len;
        Ok(logits)
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
        let q = apply_rope_range(&q, pos0, self.rope_theta)?;
        let k = apply_rope_range(&k, pos0, self.rope_theta)?;

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
fn apply_rope_range(x: &Tensor, pos0: usize, theta_base: f32) -> anyhow::Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let q_len = x.dim(2)?;
    let half = head_dim / 2;
    let device = x.device();

    // Build the [q_len, half] sin/cos table for positions pos0..pos0+q_len.
    let mut cos_vals = Vec::with_capacity(q_len * half);
    let mut sin_vals = Vec::with_capacity(q_len * half);
    for j in 0..q_len {
        let pos = (pos0 + j) as f32;
        for i in 0..half {
            let freq = (theta_base).powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos * freq;
            cos_vals.push(angle.cos());
            sin_vals.push(angle.sin());
        }
    }
    // Shape [1, 1, q_len, half] broadcasts over the n_heads axis.
    let cos = Tensor::from_vec(cos_vals, (1, 1, q_len, half), device)?.to_dtype(x.dtype())?;
    let sin = Tensor::from_vec(sin_vals, (1, 1, q_len, half), device)?.to_dtype(x.dtype())?;

    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated_1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Ok(Tensor::cat(&[&rotated_1, &rotated_2], D::Minus1)?.contiguous()?)
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
}
