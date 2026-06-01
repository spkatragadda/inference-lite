//! Interactive chat client for the `server` binary.
//!
//! Holds a multi-turn conversation in the terminal: it accumulates the message
//! history, renders it into the Qwen3 ChatML template client-side, and POSTs it
//! to the server's `/generate` endpoint with `chat_template: false` (the server
//! resets its KV cache every request, so sending the full transcript each turn
//! is what gives us context). After each reply it prints the model output plus
//! the prefill/decode throughput the server reported.
//!
//! Usage:
//!   cargo run --release --bin chat                 # talk to http://127.0.0.1:8080
//!   cargo run --release --bin chat -- 127.0.0.1:8080 --max-tokens 1024
//!
//! In-chat commands: /reset (clear history), /system <text> (set system prompt),
//! /help, /exit (or /quit, or Ctrl-D).

use std::io::{self, BufRead, Read, Write};
use std::net::TcpStream;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

// ANSI styling for a readable transcript. Kept tiny and self-contained.
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

/// The server's `/generate` response shape (mirrors `GenerateResp` in server.rs).
#[derive(Deserialize)]
struct GenerateResp {
    text: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_tps: f64,
    decode_tps: f64,
}

enum Role {
    User,
    Assistant,
}

struct Config {
    addr: String,
    max_tokens: usize,
}

fn parse_args() -> Config {
    let mut addr = "127.0.0.1:8080".to_string();
    let mut max_tokens = 512usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-tokens" | "-n" => {
                if let Some(v) = args.next() {
                    max_tokens = v.parse().unwrap_or(max_tokens);
                }
            }
            // Anything else is treated as the server address (host:port or URL).
            other => addr = normalize_addr(other),
        }
    }
    Config { addr, max_tokens }
}

/// Accept `host:port`, `http://host:port`, or a trailing slash and reduce it to
/// the bare `host:port` our raw TCP client needs.
fn normalize_addr(s: &str) -> String {
    s.trim_end_matches('/')
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .to_string()
}

fn main() -> Result<()> {
    let cfg = parse_args();

    // Confirm the server is up before dropping into the prompt, so a wrong
    // address fails fast with a clear message instead of on first message.
    match http_get(&cfg.addr, "/health") {
        Ok(body) if body.trim() == "ok" => {
            println!(
                "{GREEN}Connected to inference-lite server at {}{RESET}",
                cfg.addr
            );
        }
        Ok(body) => println!(
            "{DIM}Server at {} responded to /health with: {}{RESET}",
            cfg.addr,
            body.trim()
        ),
        Err(e) => {
            return Err(anyhow!(
                "could not reach server at {} (/health): {e}\n\
                 Start it first with:  cargo run --release --bin server",
                cfg.addr
            ));
        }
    }
    println!(
        "{DIM}max_tokens={} · commands: /reset  /system <text>  /help  /exit{RESET}\n",
        cfg.max_tokens
    );

    let mut system: Option<String> = None;
    // Full transcript of completed turns (user + assistant), oldest first.
    let mut history: Vec<(Role, String)> = Vec::new();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("{BOLD}{CYAN}you ▸ {RESET}");
        io::stdout().flush().ok();

        let line = match lines.next() {
            Some(line) => line.context("failed to read stdin")?,
            None => {
                // EOF (Ctrl-D): exit cleanly.
                println!("\n{DIM}bye{RESET}");
                break;
            }
        };
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }

        // Slash commands.
        if let Some(rest) = msg.strip_prefix('/') {
            let mut parts = rest.splitn(2, char::is_whitespace);
            let cmd = parts.next().unwrap_or("");
            let tail = parts.next().unwrap_or("").trim();
            match cmd {
                "exit" | "quit" | "q" => {
                    println!("{DIM}bye{RESET}");
                    break;
                }
                "reset" | "clear" => {
                    history.clear();
                    println!("{DIM}history cleared{RESET}\n");
                }
                "system" => {
                    if tail.is_empty() {
                        system = None;
                        println!("{DIM}system prompt cleared{RESET}\n");
                    } else {
                        system = Some(tail.to_string());
                        println!("{DIM}system prompt set{RESET}\n");
                    }
                }
                "help" | "h" => print_help(),
                other => println!("{DIM}unknown command /{other} — try /help{RESET}\n"),
            }
            continue;
        }

        // Build the ChatML prompt from system + history + this new user turn,
        // ending with the assistant generation prefix.
        history.push((Role::User, msg.to_string()));
        let prompt = render_chatml(system.as_deref(), &history);

        let req_body = serde_json::json!({
            "prompt": prompt,
            "max_tokens": cfg.max_tokens,
            "chat_template": false,
        })
        .to_string();

        print!("{DIM}…{RESET}\r");
        io::stdout().flush().ok();

        let resp = match http_post_json(&cfg.addr, "/generate", &req_body) {
            Ok(r) => r,
            Err(e) => {
                // Roll back the user turn we optimistically pushed.
                history.pop();
                eprintln!("\r{DIM}request failed: {e}{RESET}\n");
                continue;
            }
        };

        let reply = resp.text.trim();
        println!("\r{BOLD}{GREEN}bot ▸ {RESET}{reply}");
        println!(
            "{DIM}      {} prompt tok · {} gen tok · prefill {:.1} tok/s · decode {:.1} tok/s{RESET}\n",
            resp.prompt_tokens, resp.generated_tokens, resp.prefill_tps, resp.decode_tps,
        );

        history.push((Role::Assistant, reply.to_string()));
    }

    Ok(())
}

fn print_help() {
    println!(
        "{DIM}commands:\n  \
         /reset           clear the conversation history\n  \
         /system <text>   set a system prompt (empty to clear)\n  \
         /help            show this help\n  \
         /exit            quit (also Ctrl-D){RESET}\n"
    );
}

/// Render the conversation into the Qwen3 ChatML template. Matches the
/// single-turn template `Engine::generate` uses (empty `<think>` block selects
/// non-thinking mode) and extends it to multiple turns. The string ends with
/// the assistant prefix so the model continues from there.
fn render_chatml(system: Option<&str>, history: &[(Role, String)]) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }
    for (role, content) in history {
        match role {
            Role::User => {
                out.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n"));
            }
            Role::Assistant => {
                out.push_str(&format!(
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n"
                ));
            }
        }
    }
    // Open the assistant turn for the model to complete.
    out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    out
}

// --- Minimal HTTP/1.1 client over raw TCP (localhost, known endpoints) -------
//
// The server returns small JSON bodies and we send `Connection: close`, so we
// can just read to EOF and split off the body — no need for a full HTTP stack.

fn http_get(addr: &str, path: &str) -> Result<String> {
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    );
    let (_status, body) = http_roundtrip(addr, req.as_bytes())?;
    Ok(body)
}

fn http_post_json(addr: &str, path: &str, json: &str) -> Result<GenerateResp> {
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {len}\r\nConnection: close\r\n\r\n{json}",
        len = json.len(),
    );
    let (status, body) = http_roundtrip(addr, req.as_bytes())?;
    if status != 200 {
        return Err(anyhow!("server returned HTTP {status}: {body}"));
    }
    serde_json::from_str(&body).with_context(|| format!("invalid JSON response: {body}"))
}

/// Send a raw request, read the whole response, and return (status_code, body).
fn http_roundtrip(addr: &str, request: &[u8]) -> Result<(u16, String)> {
    let mut stream =
        TcpStream::connect(addr).with_context(|| format!("connect to {addr}"))?;
    stream.write_all(request).context("write request")?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).context("read response")?;
    let raw = String::from_utf8_lossy(&raw);

    let (head, body) = raw
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow!("malformed HTTP response (no header/body split)"))?;

    // Status line: "HTTP/1.1 200 OK".
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| anyhow!("could not parse status line: {head:?}"))?;

    Ok((status, body.to_string()))
}
