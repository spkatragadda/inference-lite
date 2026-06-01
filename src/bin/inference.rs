use candle_core::Device;
use inference_lite::engine::Engine;
use std::io::Write;

/// One-shot CLI: load a GGUF, run a single prompt, stream the response, and
/// print prefill/decode throughput. The model + generation logic live in the
/// `inference_lite` library so this binary and the `server` binary share one
/// code path.
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    let model_path = "./Qwen3-0.6B-Q4_0.gguf";
    println!("Loading GGUF weights into memory: {model_path}");
    let engine = Engine::load(model_path, &device)?;
    println!("Loaded model (arch: {}).", engine.arch());

    // CLI override: `cargo run --bin inference -- "your prompt here"`.
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, world!".to_string());
    println!("\nUser prompt: {prompt:?}");

    print!("\nResponse: ");
    std::io::stdout().flush().ok();

    const MAX_NEW_TOKENS: usize = 64;
    let out = engine.generate(&prompt, MAX_NEW_TOKENS, true, |delta| {
        print!("{delta}");
        std::io::stdout().flush().ok();
    })?;
    println!();

    let prefill_tps = throughput(out.prompt_tokens, out.prefill_secs);
    let decode_tps = throughput(out.generated_tokens, out.decode_secs);
    println!(
        "Prefill: {} tok in {:.3}s ({:.2} tok/s)",
        out.prompt_tokens, out.prefill_secs, prefill_tps,
    );
    println!(
        "Decode:  {} tok in {:.3}s ({:.2} tok/s)",
        out.generated_tokens, out.decode_secs, decode_tps,
    );
    Ok(())
}

fn throughput(tokens: usize, secs: f64) -> f64 {
    if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    }
}
