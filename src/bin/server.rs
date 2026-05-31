//! HTTP server: load the GGUF once at startup, keep it resident, and serve
//! greedy completions on localhost:8080 so clients can hit the model
//! repeatedly without reloading weights.
//!
//!   POST /generate  {"prompt": "...", "max_tokens": 256, "chat_template": true}
//!                -> {"text": "...", "prompt_tokens": N, "generated_tokens": M,
//!                    "prefill_tps": f, "decode_tps": f}
//!   GET  /health -> "ok"
//!
//! The transformer holds a single KV cache + position counter, so it is
//! inherently one-request-at-a-time. We wrap the `Engine` in a `Mutex` and run
//! each request on a blocking task; concurrent callers are serialized rather
//! than corrupting each other's cache.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Device;
use inference_lite::engine::Engine;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

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
        .unwrap_or_else(|| "/mnt/c/Users/saipk/Downloads/Qwen3-0.6B-Q4_0.gguf".to_string());

    println!("Loading GGUF weights into memory: {model_path}");
    let engine = Engine::load(&model_path, &device)?;
    println!("Loaded model (arch: {}).", engine.arch());

    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .with_state(state);

    let addr = "127.0.0.1:8080";
    let listener = TcpListener::bind(addr).await?;
    println!("Serving on http://{addr}  (POST /generate, GET /health)");
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
        generated_tokens: result.generated_tokens,
        prefill_tps: throughput(result.prompt_tokens, result.prefill_secs),
        decode_tps: throughput(result.generated_tokens, result.decode_secs),
    }))
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
