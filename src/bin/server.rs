//! HTTP server: load the GGUF once at startup, keep it resident, and serve
//! greedy completions on localhost:8080 so clients can hit the model
//! repeatedly without reloading weights.
//!
//!   POST /generate  {"prompt": "...", "max_tokens": 256, "chat_template": true}
//!                -> {"text": "...", "prompt_tokens": N, "generated_tokens": M,
//!                    "prefill_tps": f, "decode_tps": f}
//!   POST /generate/stream  (same body) -> newline-delimited JSON, one object per
//!                line, flushed as each token is produced:
//!                  {"token": "..."}            (repeated, live)
//!                  {"done": true, "prompt_tokens": N, ...}   (final summary)
//!                  {"error": "..."}            (on failure)
//!   GET  /health -> "ok"
//!
//! The transformer holds a single KV cache + position counter, so it is
//! inherently one-request-at-a-time. We wrap the `Engine` in a `Mutex` and run
//! each request on a blocking task; concurrent callers are serialized rather
//! than corrupting each other's cache.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Device;
use inference_lite::engine::Engine;
use inference_lite::forward::WeightPrecision;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
}

#[derive(Deserialize)]
struct GenerateReq {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_chat_template")]
    chat_template: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_chat_template() -> bool {
    true
}

#[derive(Serialize)]
struct GenerateResp {
    text: String,
    prompt_tokens: usize,
    /// Prompt tokens whose K/V was reused from the previous request's cache
    /// (prefix caching); only `prompt_tokens - reused_tokens` were prefilled.
    reused_tokens: usize,
    generated_tokens: usize,
    prefill_tps: f64,
    decode_tps: f64,
}

#[derive(Serialize)]
struct ErrResp {
    error: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Optional CLI override of the model path: `cargo run --bin server -- /path/model.gguf`.
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./Qwen3-0.6B-Q4_0.gguf".to_string());

    // Weight precision: f16 (faster CPU matmul) by default; set
    // WEIGHT_DTYPE=quantized to keep the native quantized kernel / smaller RAM.
    let precision = WeightPrecision::from_env();
    println!("Loading GGUF weights into memory: {model_path} (weights: {precision:?})");
    let engine = Engine::load(&model_path, &device, precision)?;
    println!("Loaded model (arch: {}).", engine.arch());

    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .route("/generate/stream", post(generate_stream))
        .with_state(state);

    let addr = "127.0.0.1:8080";
    let listener = TcpListener::bind(addr).await?;
    println!(
        "Serving on http://{addr}  (POST /generate, POST /generate/stream, GET /health)"
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateReq>,
) -> Result<Json<GenerateResp>, (StatusCode, Json<ErrResp>)> {
    let engine = state.engine.clone();

    // Inference is blocking + CPU-bound: run it off the async runtime, and take
    // the lock inside the blocking task so concurrent requests queue instead of
    // clobbering the shared KV cache. (Lock recovered on poison so one panicked
    // request doesn't take the whole server down.)
    let result = tokio::task::spawn_blocking(move || {
        let engine = engine.lock().unwrap_or_else(|p| p.into_inner());
        engine.generate(&req.prompt, req.max_tokens, req.chat_template, |_| {})
    })
    .await
    .map_err(|e| internal(format!("inference task failed to join: {e}")))?
    .map_err(|e| internal(format!("generation failed: {e}")))?;

    Ok(Json(GenerateResp {
        text: result.text,
        prompt_tokens: result.prompt_tokens,
        reused_tokens: result.reused_tokens,
        generated_tokens: result.generated_tokens,
        // Throughput over the tokens actually prefilled (reused ones cost ~0).
        prefill_tps: throughput(
            result.prompt_tokens - result.reused_tokens,
            result.prefill_secs,
        ),
        decode_tps: throughput(result.generated_tokens, result.decode_secs),
    }))
}

/// Streaming variant of [`generate`]: emits each token as it is produced rather
/// than buffering the whole completion. The response body is newline-delimited
/// JSON — one `{"token": "..."}` object per decoded delta (flushed live), then a
/// final `{"done": true, ...}` summary (or `{"error": "..."}`). Token text is
/// JSON-encoded, so embedded newlines stay inside their line and never break the
/// framing. Generation runs on a blocking task (CPU-bound, holds the engine
/// lock); its `on_token` callback pushes lines into an mpsc channel that becomes
/// the response stream, so the client sees tokens at decode speed.
async fn generate_stream(State(state): State<AppState>, Json(req): Json<GenerateReq>) -> Response {
    let engine = state.engine.clone();
    // Bounded so a slow/disconnected client applies backpressure on the decode
    // loop (blocking_send parks the worker) instead of buffering unboundedly.
    let (tx, rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(64);

    tokio::task::spawn_blocking(move || {
        let engine = engine.lock().unwrap_or_else(|p| p.into_inner());
        let tok_tx = tx.clone();
        let result = engine.generate(&req.prompt, req.max_tokens, req.chat_template, |delta| {
            let line = serde_json::json!({ "token": delta }).to_string();
            // If the receiver is gone (client hung up) the send errors; ignore —
            // generation will keep running to completion but produce no output.
            let _ = tok_tx.blocking_send(Ok(format!("{line}\n")));
        });
        let final_line = match result {
            Ok(out) => serde_json::json!({
                "done": true,
                "prompt_tokens": out.prompt_tokens,
                "reused_tokens": out.reused_tokens,
                "generated_tokens": out.generated_tokens,
                // Throughput over the tokens actually prefilled (reused ~0 cost).
                "prefill_tps": throughput(out.prompt_tokens - out.reused_tokens, out.prefill_secs),
                "decode_tps": throughput(out.generated_tokens, out.decode_secs),
            }),
            Err(e) => serde_json::json!({ "error": format!("generation failed: {e}") }),
        };
        let _ = tx.blocking_send(Ok(format!("{final_line}\n")));
        // tx dropped here -> channel closes -> stream ends -> connection closes.
    });

    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .expect("static header + stream body is always a valid response")
}

fn throughput(tokens: usize, secs: f64) -> f64 {
    if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    }
}

fn internal(msg: String) -> (StatusCode, Json<ErrResp>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrResp { error: msg }))
}
