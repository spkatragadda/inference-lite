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

// ANSI styling for a readable transcript. Kept tiny and self-contained.
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

/// Summary the streaming client assembles from the final `{"done": ...}` line
/// of the `/generate/stream` response.
struct StreamResult {
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

        // Stream the reply, printing each token the instant the server emits it
        // so the model's progress is visible during the (slow) decode instead of
        // a multi-minute hang. `reply` accumulates the full text for the history.
        print!("{BOLD}{GREEN}bot ▸ {RESET}");
        io::stdout().flush().ok();

        let mut reply = String::new();
        let result = stream_generate(&cfg.addr, "/generate/stream", &req_body, |delta| {
            print!("{delta}");
            io::stdout().flush().ok();
            reply.push_str(delta);
        });

        let stats = match result {
            Ok(s) => s,
            Err(e) => {
                // Roll back the user turn we optimistically pushed.
                history.pop();
                eprintln!("\n{DIM}request failed: {e}{RESET}\n");
                continue;
            }
        };
        println!(); // terminate the streamed line
        println!(
            "{DIM}      {} prompt tok · {} gen tok · prefill {:.1} tok/s · decode {:.1} tok/s{RESET}\n",
            stats.prompt_tokens, stats.generated_tokens, stats.prefill_tps, stats.decode_tps,
        );

        history.push((Role::Assistant, reply.trim().to_string()));
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

/// Render the conversation into the Qwen2.5 ChatML template, matching the
/// single-turn template `Engine::generate` uses and extending it to multiple
/// turns. No `<think>` block is injected: VibeThinker is a reasoning model and
/// emits its own chain-of-thought, so forcing an empty think block would
/// suppress it. The string ends with the assistant prefix so the model continues
/// from there.
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
                out.push_str(&format!("<|im_start|>assistant\n{content}<|im_end|>\n"));
            }
        }
    }
    // Open the assistant turn for the model to complete.
    out.push_str("<|im_start|>assistant\n");
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

/// POST to the streaming endpoint and consume the newline-delimited JSON the
/// server flushes per token. Each `{"token": "..."}` line invokes `on_delta`
/// immediately (so the caller can print it live); the final `{"done": ...}` line
/// becomes the returned [`StreamResult`]. Reads the socket incrementally and
/// decodes HTTP/1.1 chunked transfer encoding by hand (the response body has no
/// known length), so tokens surface as they arrive rather than after EOF.
fn stream_generate(
    addr: &str,
    path: &str,
    json: &str,
    mut on_delta: impl FnMut(&str),
) -> Result<StreamResult> {
    let mut stream = TcpStream::connect(addr).with_context(|| format!("connect to {addr}"))?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {len}\r\nConnection: close\r\n\r\n{json}",
        len = json.len(),
    );
    stream.write_all(req.as_bytes()).context("write request")?;

    let mut buf: Vec<u8> = Vec::new(); // raw bytes not yet parsed
    let mut tmp = [0u8; 8192];
    let mut headers_done = false;
    let mut chunked = false;
    let mut status = 0u16;
    let mut line: Vec<u8> = Vec::new(); // current NDJSON line being assembled
    let mut done: Option<StreamResult> = None;
    let mut error: Option<String> = None;

    'read: loop {
        let n = stream.read(&mut tmp).context("read response")?;
        if n == 0 {
            break; // server closed the connection (we sent Connection: close)
        }
        buf.extend_from_slice(&tmp[..n]);

        // Split headers off once, recording the status and whether the body is
        // chunked (it always is for our streaming response).
        if !headers_done {
            let Some(pos) = find(&buf, b"\r\n\r\n") else {
                continue; // headers not fully received yet
            };
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            status = head
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|c| c.parse::<u16>().ok())
                .unwrap_or(0);
            chunked = head.lines().any(|l| {
                let l = l.to_ascii_lowercase();
                l.starts_with("transfer-encoding:") && l.contains("chunked")
            });
            buf.drain(..pos + 4);
            headers_done = true;
        }

        // Pull every complete body slice currently buffered, de-chunking if
        // needed, and feed it through the NDJSON line splitter.
        loop {
            let data: Vec<u8> = if chunked {
                let Some(rn) = find(&buf, b"\r\n") else { break };
                let size = std::str::from_utf8(&buf[..rn])
                    .ok()
                    .and_then(|s| usize::from_str_radix(s.trim(), 16).ok())
                    .ok_or_else(|| anyhow!("malformed chunk size"))?;
                if size == 0 {
                    break 'read; // terminating zero-length chunk
                }
                let need = rn + 2 + size + 2; // size line + data + trailing CRLF
                if buf.len() < need {
                    break; // wait for the rest of this chunk
                }
                let data = buf[rn + 2..rn + 2 + size].to_vec();
                buf.drain(..need);
                data
            } else if buf.is_empty() {
                break;
            } else {
                std::mem::take(&mut buf)
            };

            // '\n' never appears mid-UTF-8 and token text is JSON-escaped, so a
            // raw '\n' always terminates a complete NDJSON object.
            for b in data {
                if b == b'\n' {
                    handle_line(&line, &mut on_delta, &mut done, &mut error)?;
                    line.clear();
                } else {
                    line.push(b);
                }
            }
        }
    }

    if status != 0 && status != 200 {
        return Err(anyhow!(
            "server returned HTTP {status}: {}",
            String::from_utf8_lossy(&line)
        ));
    }
    if let Some(e) = error {
        return Err(anyhow!(e));
    }
    done.ok_or_else(|| anyhow!("stream ended without a 'done' summary"))
}

/// Parse one NDJSON line from the stream and dispatch it: a token is forwarded
/// to `on_delta`, the `done` summary is captured, an `error` is recorded.
fn handle_line(
    line: &[u8],
    on_delta: &mut impl FnMut(&str),
    done: &mut Option<StreamResult>,
    error: &mut Option<String>,
) -> Result<()> {
    let line = line.trim_ascii();
    if line.is_empty() {
        return Ok(());
    }
    let v: serde_json::Value = serde_json::from_slice(line)
        .with_context(|| format!("invalid NDJSON line: {}", String::from_utf8_lossy(line)))?;

    if let Some(t) = v.get("token").and_then(|t| t.as_str()) {
        on_delta(t);
    } else if v.get("done").and_then(|d| d.as_bool()) == Some(true) {
        let num = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let tps = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
        *done = Some(StreamResult {
            prompt_tokens: num("prompt_tokens"),
            generated_tokens: num("generated_tokens"),
            prefill_tps: tps("prefill_tps"),
            decode_tps: tps("decode_tps"),
        });
    } else if let Some(e) = v.get("error").and_then(|e| e.as_str()) {
        *error = Some(e.to_string());
    }
    Ok(())
}

/// First index of `needle` within `haystack`, if present.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
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
