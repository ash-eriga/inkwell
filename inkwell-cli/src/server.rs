//! OpenAI-compatible HTTP/1.1 + SSE inference server (`inkwell serve`).

use crate::chat_template::{parse_effort, render_chat_prompt, ChatMessage};
use crate::engine::{Engine, GenChunk, GenParams, GenStats};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

static STOP_SERVER: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn handle_sigint(_sig: libc::c_int) {
    STOP_SERVER.store(true, Ordering::SeqCst);
}

#[cfg(unix)]
fn install_sigint_handler() -> libc::sighandler_t {
    STOP_SERVER.store(false, Ordering::SeqCst);
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_sigint as *const () as libc::sighandler_t,
        )
    }
}

#[cfg(unix)]
fn restore_sigint_handler(prev: libc::sighandler_t) {
    unsafe {
        let _ = libc::signal(libc::SIGINT, prev);
    }
}

/// Server launch configuration.
#[derive(Debug, Clone)]
pub struct ServeOpts {
    /// Model directory or `.gguf` file.
    pub model: String,
    /// Bind host (default `127.0.0.1`).
    pub host: String,
    /// Bind port (default `8080`).
    pub port: u16,
    /// Maximum context window.
    pub ctx: usize,
    /// Default reasoning effort when a request omits `reasoning_effort`.
    pub default_effort: String,
}

static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn next_id(prefix: &str) -> String {
    let n = REQ_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{n}", unix_secs())
}

enum PromptInput {
    Chat(String),
    Raw(String),
}

enum JobEvent {
    Chunk(GenChunk),
    Done(GenStats),
    Err(String),
}

struct InferJob {
    prompt: PromptInput,
    params: GenParams,
    reply: SyncSender<JobEvent>,
}

struct ServerInfo {
    model_id: String,
    max_ctx: usize,
    default_effort: String,
    created: u64,
    jobs: SyncSender<InferJob>,
    active_conns: AtomicU64,
}

/// Start the OpenAI-compatible HTTP server, loading the model and returning the
/// warm [`Engine`] when stopped via Ctrl+C so the caller can switch modes without reloading.
pub fn run_server(opts: &ServeOpts) -> Result<Engine, String> {
    let _ = parse_effort(Some(&opts.default_effort))?;
    eprintln!(
        "inkwell serve: loading {} (ctx={})...",
        opts.model, opts.ctx
    );
    let mut engine = Engine::load(&opts.model, opts.ctx)?;
    run_server_with_engine(&mut engine, opts)?;
    Ok(engine)
}

/// Serve OpenAI-compatible HTTP/SSE requests using an already-loaded [`Engine`].
/// Stops cleanly on Ctrl+C (`SIGINT`), leaving the engine warm in VRAM.
pub fn run_server_with_engine(engine: &mut Engine, opts: &ServeOpts) -> Result<(), String> {
    let _ = parse_effort(Some(&opts.default_effort))?;
    let model_id = engine.model_id().to_owned();
    let max_ctx = engine.max_ctx();

    let addr = format!("{}:{}", opts.host, opts.port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("failed to bind {addr}: {e}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("failed to configure listener: {e}"))?;

    #[cfg(unix)]
    let prev_sig = install_sigint_handler();

    const MAX_PENDING_JOBS: usize = 64;
    let (job_tx, job_rx) = mpsc::sync_channel::<InferJob>(MAX_PENDING_JOBS);
    let info = Arc::new(ServerInfo {
        model_id: model_id.clone(),
        max_ctx,
        default_effort: opts.default_effort.clone(),
        created: unix_secs(),
        jobs: job_tx,
        active_conns: AtomicU64::new(0),
    });

    eprintln!(
        "\ninkwell server listening on http://{addr}\n  model:     {model_id}\n  max_ctx:   {max_ctx}\n  endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions\n"
    );

    let accept_info = Arc::clone(&info);
    let accept_handle = std::thread::spawn(move || {
        while !STOP_SERVER.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    const MAX_ACTIVE_CONNS: u64 = 256;
                    let prev = accept_info.active_conns.fetch_add(1, Ordering::Relaxed);
                    if prev >= MAX_ACTIVE_CONNS {
                        accept_info.active_conns.fetch_sub(1, Ordering::Relaxed);
                        drop(stream);
                        continue;
                    }
                    let _ = stream.set_nonblocking(false);
                    let st = Arc::clone(&accept_info);
                    std::thread::spawn(move || {
                        let _ = handle_conn(stream, &st);
                        st.active_conns.fetch_sub(1, Ordering::Relaxed);
                    });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(40));
                }
                Err(_) => break,
            }
        }
    });

    while !STOP_SERVER.load(Ordering::Relaxed) {
        match job_rx.recv_timeout(Duration::from_millis(60)) {
            Ok(job) => {
                let ids = match &job.prompt {
                    PromptInput::Chat(s) => engine.tokenizer().encode_with_special(s),
                    PromptInput::Raw(s) => engine.tokenizer().encode(s),
                };
                let reply = job.reply;
                match engine.generate(&ids, &job.params, |chunk| {
                    !STOP_SERVER.load(Ordering::Relaxed)
                        && reply.send(JobEvent::Chunk(chunk)).is_ok()
                }) {
                    Ok(stats) => {
                        let _ = reply.send(JobEvent::Done(stats));
                    }
                    Err(e) => {
                        let _ = reply.send(JobEvent::Err(e));
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    drop(info);
    let _ = accept_handle.join();
    #[cfg(unix)]
    restore_sigint_handler(prev_sig);
    eprintln!("\ninkwell server stopped.");
    Ok(())
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_line_bounded<R: BufRead>(
    reader: &mut R,
    limit: usize,
    out: &mut String,
) -> Result<usize, String> {
    out.clear();
    let n = reader
        .by_ref()
        .take((limit + 1) as u64)
        .read_line(out)
        .map_err(|e| e.to_string())?;
    if n > limit {
        return Err("line exceeds length limit".to_owned());
    }
    Ok(n)
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut req_line = String::new();
    let n = read_line_bounded(&mut reader, 8192, &mut req_line)?;
    if n == 0 {
        return Err("empty request".to_owned());
    }
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_owned();
    let raw_path = parts.next().unwrap_or("/");
    let path = raw_path.split('?').next().unwrap_or("/").to_owned();

    let mut content_len: Option<usize> = None;
    let mut chunked = false;
    let mut header_bytes = 0usize;
    let mut terminated = false;
    for _ in 0..128 {
        let mut hline = String::new();
        let hn = read_line_bounded(&mut reader, 8192, &mut hline)?;
        if hn == 0 || hline == "\r\n" || hline == "\n" {
            terminated = true;
            break;
        }
        header_bytes += hn;
        if header_bytes > 64 * 1024 {
            return Err("headers exceed 64 KiB".to_owned());
        }
        if let Some((k, v)) = hline.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                if content_len.is_some() {
                    return Err("duplicate content-length header".to_owned());
                }
                content_len = Some(
                    v.trim()
                        .parse()
                        .map_err(|_| "invalid content-length header".to_owned())?,
                );
            } else if k.trim().eq_ignore_ascii_case("transfer-encoding")
                && v.trim().eq_ignore_ascii_case("chunked")
            {
                chunked = true;
            }
        }
    }
    if !terminated {
        return Err("too many headers or missing header terminator".to_owned());
    }
    if chunked {
        return Err("chunked request bodies are not supported".to_owned());
    }

    let content_len = content_len.unwrap_or(0);
    if content_len > 16 * 1024 * 1024 {
        return Err("request body exceeds 16 MiB".to_owned());
    }
    let mut body = Vec::new();
    if content_len > 0 {
        let got = reader
            .by_ref()
            .take(content_len as u64)
            .read_to_end(&mut body)
            .map_err(|e| e.to_string())?;
        if got != content_len {
            return Err("request body truncated".to_owned());
        }
    }
    Ok(HttpRequest { method, path, body })
}

fn write_json(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    val: &Value,
) -> Result<(), String> {
    let payload = serde_json::to_vec(val).map_err(|e| e.to_string())?;
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(header.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(&payload).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn write_error(stream: &mut TcpStream, status: u16, reason: &str, msg: &str) -> Result<(), String> {
    write_json(
        stream,
        status,
        reason,
        &json!({
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "code": status,
            }
        }),
    )
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<(), String> {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n";
    stream
        .write_all(header.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn write_sse_event(stream: &mut TcpStream, val: &Value) -> bool {
    let Ok(s) = serde_json::to_string(val) else {
        return false;
    };
    stream
        .write_all(format!("data: {s}\n\n").as_bytes())
        .and_then(|()| stream.flush())
        .is_ok()
}

fn handle_conn(mut stream: TcpStream, state: &ServerInfo) -> Result<(), String> {
    let req = match read_request(&mut stream) {
        Ok(req) => req,
        Err(e) => return write_error(&mut stream, 400, "Bad Request", &e),
    };
    if req.method == "OPTIONS" {
        let resp = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n";
        stream
            .write_all(resp.as_bytes())
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => write_json(
            &mut stream,
            200,
            "OK",
            &json!({
                "status": "ok",
                "model": state.model_id,
                "max_ctx": state.max_ctx,
            }),
        ),
        ("GET", "/v1/models") => write_json(
            &mut stream,
            200,
            "OK",
            &json!({
                "object": "list",
                "data": [{
                    "id": state.model_id,
                    "object": "model",
                    "created": state.created,
                    "owned_by": "inkwell",
                }],
            }),
        ),
        ("POST", "/v1/chat/completions") => handle_chat(&mut stream, &req.body, state),
        ("POST", "/v1/completions") => handle_completions(&mut stream, &req.body, state),
        _ => write_error(
            &mut stream,
            404,
            "Not Found",
            &format!("unknown endpoint: {} {}", req.method, req.path),
        ),
    }
}

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    reasoning_effort: Option<String>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: Value,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Serialize)]
struct UsageObj {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

fn extract_content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(s) = p.as_str() {
                    out.push_str(s);
                } else if let Some(t) = p.get("text").and_then(Value::as_str) {
                    out.push_str(t);
                }
            }
            out
        }
        _ => String::new(),
    }
}

fn parse_stop(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn handle_chat(stream: &mut TcpStream, body: &[u8], state: &ServerInfo) -> Result<(), String> {
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return write_error(stream, 400, "Bad Request", &format!("invalid JSON: {e}")),
    };
    if req.messages.is_empty() {
        return write_error(stream, 400, "Bad Request", "messages must not be empty");
    }

    let effort_str = req
        .reasoning_effort
        .as_deref()
        .unwrap_or(&state.default_effort);
    let effort = match parse_effort(Some(effort_str)) {
        Ok(v) => v,
        Err(e) => return write_error(stream, 400, "Bad Request", &e),
    };

    let msgs: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: extract_content_text(&m.content),
            reasoning_content: m.reasoning_content.clone(),
        })
        .collect();

    let rendered = match render_chat_prompt(&msgs, effort) {
        Ok(s) => s,
        Err(e) => return write_error(stream, 400, "Bad Request", &e),
    };

    let params = GenParams {
        max_tokens: req.max_completion_tokens.or(req.max_tokens).unwrap_or(2048),
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: req.top_k.unwrap_or(40),
        min_p: req.min_p.unwrap_or(0.05),
        repeat_penalty: req.repeat_penalty.unwrap_or(1.0),
        seed: req.seed,
        stop: parse_stop(req.stop.as_ref()),
        chat_mode: true,
    };

    let model_name = req.model.unwrap_or_else(|| state.model_id.clone());
    let id = next_id("chatcmpl");
    let created = unix_secs();

    let (ev_tx, ev_rx) = mpsc::sync_channel(256);
    if state
        .jobs
        .try_send(InferJob {
            prompt: PromptInput::Chat(rendered),
            params,
            reply: ev_tx,
        })
        .is_err()
    {
        return write_error(
            stream,
            503,
            "Service Unavailable",
            "inference queue is full",
        );
    }

    if req.stream {
        write_sse_headers(stream)?;
        let init_chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": Value::Null,
            }],
        });
        if !write_sse_event(stream, &init_chunk) {
            return Ok(());
        }

        for ev in ev_rx {
            match ev {
                JobEvent::Chunk(chunk) => {
                    let delta = match chunk {
                        GenChunk::Content(s) => json!({"content": s}),
                        GenChunk::Reasoning(s) => json!({"reasoning_content": s}),
                    };
                    let msg = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": Value::Null,
                        }],
                    });
                    if !write_sse_event(stream, &msg) {
                        return Ok(());
                    }
                }
                JobEvent::Done(stats) => {
                    let mut final_ev = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": stats.finish_reason,
                        }],
                    });
                    if req.stream_options.as_ref().is_some_and(|o| o.include_usage) {
                        final_ev["usage"] = json!(UsageObj {
                            prompt_tokens: stats.prompt_tokens,
                            completion_tokens: stats.completion_tokens,
                            total_tokens: stats.prompt_tokens + stats.completion_tokens,
                        });
                    }
                    let _ = write_sse_event(stream, &final_ev);
                    let _ = stream.write_all(b"data: [DONE]\n\n");
                    let _ = stream.flush();
                    eprintln!(
                        "POST /v1/chat/completions stream=true prompt={} ({:.1} tok/s) gen={} ({:.1} tok/s) finish={}",
                        stats.prompt_tokens,
                        stats.prefill_tps(),
                        stats.completion_tokens,
                        stats.decode_tps(),
                        stats.finish_reason
                    );
                    return Ok(());
                }
                JobEvent::Err(e) => {
                    let _ = write_sse_event(stream, &json!({"error": {"message": e}}));
                    return Ok(());
                }
            }
        }
        return Ok(());
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut final_stats = None;
    for ev in ev_rx {
        match ev {
            JobEvent::Chunk(GenChunk::Content(s)) => content.push_str(&s),
            JobEvent::Chunk(GenChunk::Reasoning(s)) => reasoning.push_str(&s),
            JobEvent::Done(stats) => {
                final_stats = Some(stats);
                break;
            }
            JobEvent::Err(e) => return write_error(stream, 400, "Bad Request", &e),
        }
    }
    let stats = final_stats.ok_or_else(|| "engine stopped before completion".to_owned())?;

    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning);
    }

    eprintln!(
        "POST /v1/chat/completions stream=false prompt={} ({:.1} tok/s) gen={} ({:.1} tok/s) finish={}",
        stats.prompt_tokens,
        stats.prefill_tps(),
        stats.completion_tokens,
        stats.decode_tps(),
        stats.finish_reason
    );

    write_json(
        stream,
        200,
        "OK",
        &json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": stats.finish_reason,
            }],
            "usage": UsageObj {
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.completion_tokens,
                total_tokens: stats.prompt_tokens + stats.completion_tokens,
            },
        }),
    )
}

fn handle_completions(
    stream: &mut TcpStream,
    body: &[u8],
    state: &ServerInfo,
) -> Result<(), String> {
    let req: CompletionRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return write_error(stream, 400, "Bad Request", &format!("invalid JSON: {e}")),
    };
    let prompt_text = match &req.prompt {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr.first().and_then(Value::as_str).unwrap_or("").to_owned(),
        _ => return write_error(stream, 400, "Bad Request", "prompt must be a string"),
    };
    if prompt_text.is_empty() {
        return write_error(stream, 400, "Bad Request", "prompt must not be empty");
    }

    let params = GenParams {
        max_tokens: req.max_tokens.unwrap_or(128),
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: req.top_k.unwrap_or(40),
        min_p: req.min_p.unwrap_or(0.05),
        repeat_penalty: req.repeat_penalty.unwrap_or(1.0),
        seed: req.seed,
        stop: parse_stop(req.stop.as_ref()),
        chat_mode: false,
    };

    let model_name = req.model.unwrap_or_else(|| state.model_id.clone());
    let id = next_id("cmpl");
    let created = unix_secs();

    let (ev_tx, ev_rx) = mpsc::sync_channel(256);
    if state
        .jobs
        .try_send(InferJob {
            prompt: PromptInput::Raw(prompt_text),
            params,
            reply: ev_tx,
        })
        .is_err()
    {
        return write_error(
            stream,
            503,
            "Service Unavailable",
            "inference queue is full",
        );
    }

    if req.stream {
        write_sse_headers(stream)?;
        for ev in ev_rx {
            match ev {
                JobEvent::Chunk(GenChunk::Content(s)) => {
                    let ok = write_sse_event(
                        stream,
                        &json!({
                            "id": id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "text": s,
                                "finish_reason": Value::Null,
                            }],
                        }),
                    );
                    if !ok {
                        return Ok(());
                    }
                }
                JobEvent::Chunk(GenChunk::Reasoning(_)) => {}
                JobEvent::Done(stats) => {
                    let _ = write_sse_event(
                        stream,
                        &json!({
                            "id": id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "text": "",
                                "finish_reason": stats.finish_reason,
                            }],
                        }),
                    );
                    let _ = stream.write_all(b"data: [DONE]\n\n");
                    let _ = stream.flush();
                    return Ok(());
                }
                JobEvent::Err(e) => {
                    let _ = write_sse_event(stream, &json!({"error": {"message": e}}));
                    return Ok(());
                }
            }
        }
        return Ok(());
    }

    let mut text = String::new();
    let mut final_stats = None;
    for ev in ev_rx {
        match ev {
            JobEvent::Chunk(GenChunk::Content(s)) => text.push_str(&s),
            JobEvent::Chunk(GenChunk::Reasoning(_)) => {}
            JobEvent::Done(stats) => {
                final_stats = Some(stats);
                break;
            }
            JobEvent::Err(e) => return write_error(stream, 400, "Bad Request", &e),
        }
    }
    let stats = final_stats.ok_or_else(|| "engine stopped before completion".to_owned())?;

    write_json(
        stream,
        200,
        "OK",
        &json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": stats.finish_reason,
            }],
            "usage": UsageObj {
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.completion_tokens,
                total_tokens: stats.prompt_tokens + stats.completion_tokens,
            },
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Shutdown;

    fn parse_request_bytes(bytes: &[u8]) -> Result<HttpRequest, String> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let addr = listener.local_addr().map_err(|e| e.to_string())?;
        let mut client = TcpStream::connect(addr).map_err(|e| e.to_string())?;
        let (mut server_side, _) = listener.accept().map_err(|e| e.to_string())?;
        client.write_all(bytes).map_err(|e| e.to_string())?;
        client
            .shutdown(Shutdown::Write)
            .map_err(|e| e.to_string())?;
        read_request(&mut server_side)
    }

    #[test]
    fn parses_bounded_valid_request() {
        let req = parse_request_bytes(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
        )
        .expect("valid request");
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions");
        assert_eq!(req.body, b"{}");
    }

    #[test]
    fn rejects_overlong_request_line_without_allocating_it() {
        let mut bytes = b"GET /".to_vec();
        bytes.extend(std::iter::repeat_n(b'a', 9000));
        bytes.extend_from_slice(b" HTTP/1.1\r\n\r\n");
        assert!(parse_request_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_unterminated_header_block() {
        let mut bytes = b"POST / HTTP/1.1\r\n".to_vec();
        for i in 0..128 {
            bytes.extend_from_slice(format!("x-{i}: y\r\n").as_bytes());
        }
        assert!(parse_request_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_conflicting_or_chunked_framing() {
        assert!(parse_request_bytes(
            b"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx"
        )
        .is_err());
        assert!(parse_request_bytes(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
        )
        .is_err());
    }
}
