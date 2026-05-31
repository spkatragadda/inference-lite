//! inference-lite: a from-scratch GGUF inference + serving engine built on
//! candle's quantized kernels. Parses a GGUF, runs a metadata-driven
//! LLaMA-family forward pass with a preallocated KV cache and a fused
//! online-softmax attention kernel, and exposes greedy generation.
//!
//! - [`gguf_parser`] — read GGUF metadata + quantized tensors.
//! - [`forward`] — the transformer forward pass (`TransformerModel`).
//! - [`engine`] — load-once runtime + tokenizer + generation (`Engine`).
//!
//! Binaries: `inference` (one-shot CLI) and `server` (HTTP on :8080).

pub mod engine;
pub mod forward;
pub mod gguf_parser;
