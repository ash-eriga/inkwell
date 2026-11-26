//! Terminal UI/UX for `inkwell run` (one-shot) and `inkwell chat` (interactive REPL).

use crate::chat_template::{parse_effort, render_chat_prompt, ChatMessage};
use crate::engine::{Engine, GenChunk, GenParams};
use std::io::{self, BufRead, IsTerminal, Write};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";

/// Options shared by `inkwell run` and `inkwell chat`.
#[derive(Debug, Clone)]
pub struct CliRunOpts {
    /// Model directory or `.gguf` file.
    pub model: String,
    /// Maximum context window.
    pub ctx: usize,
    /// Maximum tokens to generate per turn.
    pub max_tokens: usize,
    /// Sampling temperature (`0.0` = greedy).
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k cap.
    pub top_k: usize,
    /// Optional system prompt.
    pub system: Option<String>,
    /// Reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `max`).
    pub effort: String,
    /// One-shot prompt text (`inkwell run`).
    pub prompt: Option<String>,
    /// Skip chat template wrapping (`inkwell run --raw`).
    pub raw: bool,
}

impl Default for CliRunOpts {
    fn default() -> Self {
        Self {
            model: String::new(),
            ctx: 4096,
            max_tokens: 2048,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            system: None,
            effort: "none".to_owned(),
            prompt: None,
            raw: false,
        }
    }
}

fn color(tty: bool, code: &'static str) -> &'static str {
    if tty { code } else { "" }
}

#[cfg(unix)]
fn flush_stdin_tty() {
    if io::stdin().is_terminal() {
        unsafe {
            let _ = libc::tcflush(0, libc::TCIFLUSH);
        }
    }
}

#[cfg(not(unix))]
fn flush_stdin_tty() {}

#[cfg(unix)]
struct RawTtyGuard {
    orig: libc::termios,
}

#[cfg(unix)]
impl RawTtyGuard {
    fn enter() -> Option<Self> {
        let mut orig: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(0, &mut orig) } != 0 {
            return None;
        }
        let mut raw = orig;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(0, libc::TCSANOW, &raw) } != 0 {
            return None;
        }
        let _ = io::stdout().write_all(b"\x1b[?2004h");
        let _ = io::stdout().flush();
        Some(Self { orig })
    }
}

#[cfg(unix)]
impl Drop for RawTtyGuard {
    fn drop(&mut self) {
        let _ = io::stdout().write_all(b"\x1b[?2004l");
        let _ = io::stdout().flush();
        unsafe {
            let _ = libc::tcsetattr(0, libc::TCSANOW, &self.orig);
        }
    }
}

#[cfg(unix)]
fn stdin_has_more(timeout_ms: i32) -> bool {
    let mut pfd = libc::pollfd {
        fd: 0,
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe { libc::poll(&mut pfd, 1, timeout_ms) > 0 && (pfd.revents & libc::POLLIN) != 0 }
}

#[cfg(unix)]
fn read_fd0_byte() -> Result<Option<u8>, String> {
    let mut b = 0u8;
    loop {
        let n = unsafe { libc::read(0, (&mut b as *mut u8).cast(), 1) };
        if n == 1 {
            return Ok(Some(b));
        }
        if n == 0 {
            return Ok(None);
        }
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(format!("stdin read error: {err}"));
    }
}

/// Read one user turn from stdin.
/// On a Unix TTY, reads unbuffered from fd 0 with bracketed paste (`\x1b[?2004h`)
/// so `Ctrl+Shift+V` (single or multi-line) and `Ctrl+C` behave cleanly without premature sends.
fn read_chat_turn(tty_out: bool) -> Result<Option<String>, String> {
    if !io::stdin().is_terminal() {
        let mut line = String::new();
        let n = io::stdin()
            .lock()
            .read_line(&mut line)
            .map_err(|e| format!("stdin read error: {e}"))?;
        if n == 0 {
            return Ok(None);
        }
        return Ok(Some(line));
    }

    #[cfg(unix)]
    {
        flush_stdin_tty();
        let _guard = RawTtyGuard::enter();
        let mut out = io::stdout();
        let mut buf: Vec<u8> = Vec::new();
        let mut in_paste = false;
        let mut paste_buf: Vec<u8> = Vec::new();
        let mut paste_lines = 0usize;

        loop {
            let Some(byte) = read_fd0_byte()? else {
                return Ok(None);
            };

            // Escape sequence handling (bracketed paste start/end, arrow keys)
            if byte == 0x1b {
                let mut seq = vec![0x1b];
                if stdin_has_more(30) {
                    if let Some(next) = read_fd0_byte()? {
                        seq.push(next);
                        if next == b'[' {
                            while seq.len() < 12 && stdin_has_more(20) {
                                if let Some(ch) = read_fd0_byte()? {
                                    seq.push(ch);
                                    if (0x40..=0x7e).contains(&ch) {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
                if seq == b"\x1b[200~" {
                    in_paste = true;
                    paste_buf.clear();
                    paste_lines = 0;
                    continue;
                }
                if seq == b"\x1b[201~" {
                    in_paste = false;
                    buf.extend_from_slice(&paste_buf);
                    if paste_lines == 0 {
                        let _ = out.write_all(&paste_buf);
                    } else {
                        let _ = write!(
                            out,
                            "{}[pasted {} lines — press Enter to send]{} ",
                            color(tty_out, DIM),
                            paste_lines + 1,
                            color(tty_out, RESET)
                        );
                    }
                    paste_buf.clear();
                    let _ = out.flush();
                    continue;
                }
                continue;
            }

            if !in_paste {
                match byte {
                    // Ctrl+C: clear current input if non-empty, else exit chat cleanly (into 4s warm handoff)
                    3 => {
                        if buf.is_empty() {
                            let _ = writeln!(out, "^C");
                            let _ = out.flush();
                            return Ok(None);
                        }
                        buf.clear();
                        let _ = write!(
                            out,
                            "^C\n{}{}you ›{} ",
                            color(tty_out, BOLD),
                            color(tty_out, GREEN),
                            color(tty_out, RESET)
                        );
                        let _ = out.flush();
                    }
                    // Ctrl+D on empty line -> EOF
                    4 if buf.is_empty() => {
                        let _ = writeln!(out);
                        return Ok(None);
                    }
                    // Enter
                    b'\r' | b'\n' => {
                        // Fallback for terminals without bracketed paste: if more bytes
                        // arrive immediately (<20ms), treat as multi-line paste continuation.
                        if stdin_has_more(20) {
                            buf.push(b'\n');
                            let _ = write!(out, "\n");
                            let _ = out.flush();
                            continue;
                        }
                        let _ = writeln!(out);
                        let _ = out.flush();
                        return Ok(Some(String::from_utf8_lossy(&buf).into_owned()));
                    }
                    // Backspace / DEL
                    0x7f | 0x08 => {
                        if let Some(last) = buf.pop() {
                            if (last & 0xC0) == 0x80 {
                                while let Some(&prev) = buf.last() {
                                    buf.pop();
                                    if (prev & 0xC0) != 0x80 {
                                        break;
                                    }
                                }
                            }
                            let _ = write!(out, "\x08 \x08");
                            let _ = out.flush();
                        }
                    }
                    // Ctrl+U: clear current input
                    21 => {
                        buf.clear();
                        let _ = write!(
                            out,
                            "\r\x1b[2K{}{}you ›{} ",
                            color(tty_out, BOLD),
                            color(tty_out, GREEN),
                            color(tty_out, RESET)
                        );
                        let _ = out.flush();
                    }
                    other => {
                        buf.push(other);
                        let _ = out.write_all(&[other]);
                        let _ = out.flush();
                    }
                }
            } else {
                // Inside bracketed paste: normalize CR/CRLF -> LF
                if byte == b'\r' {
                    paste_buf.push(b'\n');
                    paste_lines += 1;
                } else if byte == b'\n' {
                    if paste_buf.last() != Some(&b'\n') {
                        paste_buf.push(b'\n');
                        paste_lines += 1;
                    }
                } else {
                    paste_buf.push(byte);
                }
            }
        }
    }

    #[cfg(not(unix))]
    {
        let mut line = String::new();
        let n = io::stdin()
            .lock()
            .read_line(&mut line)
            .map_err(|e| format!("stdin read error: {e}"))?;
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(line))
    }
}

/// One-shot prompt execution with live token streaming.
pub fn run_oneshot(opts: &CliRunOpts) -> Result<(), String> {
    let effort = parse_effort(Some(&opts.effort))?;
    let prompt = opts
        .prompt
        .as_deref()
        .ok_or_else(|| "missing prompt: pass -p \"<prompt>\"".to_owned())?;
    let tty_out = io::stdout().is_terminal();
    let tty_err = io::stderr().is_terminal();

    eprintln!(
        "{}inkwell{} loading {} (ctx={})...",
        color(tty_err, BOLD),
        color(tty_err, RESET),
        opts.model,
        opts.ctx
    );
    let mut engine = Engine::load(&opts.model, opts.ctx)?;

    let ids = if opts.raw {
        engine.tokenizer().encode(prompt)
    } else {
        let mut msgs = Vec::new();
        if let Some(sys) = &opts.system {
            msgs.push(ChatMessage {
                role: "system".to_owned(),
                content: sys.clone(),
                reasoning_content: None,
            });
        }
        msgs.push(ChatMessage {
            role: "user".to_owned(),
            content: prompt.to_owned(),
            reasoning_content: None,
        });
        let rendered = render_chat_prompt(&msgs, effort)?;
        engine.tokenizer().encode_with_special(&rendered)
    };

    let params = GenParams {
        max_tokens: opts.max_tokens,
        temperature: opts.temperature,
        top_p: opts.top_p,
        top_k: opts.top_k,
        chat_mode: !opts.raw,
        ..GenParams::default()
    };

    let mut out = io::stdout().lock();
    let mut in_thinking = false;
    let stats = engine.generate(&ids, &params, |chunk| {
        match chunk {
            GenChunk::Reasoning(s) => {
                if !in_thinking {
                    let _ = write!(
                        out,
                        "{}{}[thinking] ",
                        color(tty_out, DIM),
                        color(tty_out, ITALIC)
                    );
                    in_thinking = true;
                }
                let _ = write!(out, "{s}");
            }
            GenChunk::Content(s) => {
                if in_thinking {
                    let _ = write!(out, "{}\n\n", color(tty_out, RESET));
                    in_thinking = false;
                }
                let _ = write!(out, "{s}");
            }
        }
        let _ = out.flush();
        true
    })?;
    if in_thinking {
        let _ = write!(out, "{}", color(tty_out, RESET));
    }
    let _ = writeln!(out);
    eprintln!(
        "{}prefill {} tok ({:.1} tok/s) · decode {} tok ({:.1} tok/s) · finish={}{}",
        color(tty_err, DIM),
        stats.prompt_tokens,
        stats.prefill_tps(),
        stats.completion_tokens,
        stats.decode_tps(),
        stats.finish_reason,
        color(tty_err, RESET)
    );
    Ok(())
}

/// Interactive multi-turn terminal chat REPL (loads model, then runs [`run_chat_with_engine`]).
pub fn run_chat(opts: &CliRunOpts) -> Result<Engine, String> {
    let tty = io::stdout().is_terminal();
    eprintln!(
        "{}inkwell{} loading {} (ctx={})...",
        color(tty, BOLD),
        color(tty, RESET),
        opts.model,
        opts.ctx
    );
    let mut engine = Engine::load(&opts.model, opts.ctx)?;
    run_chat_with_engine(&mut engine, opts)?;
    Ok(engine)
}

/// Interactive multi-turn terminal chat REPL using an already-loaded [`Engine`].
pub fn run_chat_with_engine(engine: &mut Engine, opts: &CliRunOpts) -> Result<(), String> {
    let mut effort = parse_effort(Some(&opts.effort))?;
    let mut effort_label = opts.effort.clone();
    let mut temperature = opts.temperature;
    let tty = io::stdout().is_terminal();

    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &opts.system {
        history.push(ChatMessage {
            role: "system".to_owned(),
            content: sys.clone(),
            reasoning_content: None,
        });
    }

    println!(
        "\n{}{}inkwell chat{}  {}model={} · ctx={} · effort={} · temp={:.2}{}",
        color(tty, BOLD),
        color(tty, CYAN),
        color(tty, RESET),
        color(tty, DIM),
        engine.model_id(),
        engine.max_ctx(),
        effort_label,
        temperature,
        color(tty, RESET)
    );
    println!(
        "{}commands: /clear  /effort <none|low|medium|high>  /temp <f32>  /system <text>  /exit{}\n",
        color(tty, DIM),
        color(tty, RESET)
    );

    loop {
        print!("{}{}you ›{} ", color(tty, BOLD), color(tty, GREEN), color(tty, RESET));
        let _ = io::stdout().flush();
        let Some(line) = read_chat_turn(tty)? else {
            println!();
            break;
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if let Some(cmd) = input.strip_prefix('/') {
            let mut parts = cmd.splitn(2, ' ');
            let name = parts.next().unwrap_or("");
            let arg = parts.next().map(str::trim).unwrap_or("");
            match name {
                "exit" | "quit" | "q" => break,
                "clear" | "reset" => {
                    history.retain(|m| m.role == "system");
                    println!("{}history cleared.{}\n", color(tty, DIM), color(tty, RESET));
                }
                "effort" => match parse_effort(Some(arg)) {
                    Ok(v) => {
                        effort = v;
                        effort_label = arg.to_owned();
                        println!(
                            "{}effort set to {effort_label} ({effort}).{}\n",
                            color(tty, DIM),
                            color(tty, RESET)
                        );
                    }
                    Err(e) => println!("{}{e}{}\n", color(tty, YELLOW), color(tty, RESET)),
                },
                "temp" => match arg.parse::<f32>() {
                    Ok(v) if v >= 0.0 => {
                        temperature = v;
                        println!(
                            "{}temperature set to {temperature:.2}.{}\n",
                            color(tty, DIM),
                            color(tty, RESET)
                        );
                    }
                    _ => println!(
                        "{}usage: /temp <float >= 0.0>{}\n",
                        color(tty, YELLOW),
                        color(tty, RESET)
                    ),
                },
                "system" => {
                    history.retain(|m| m.role != "system");
                    if !arg.is_empty() {
                        history.insert(
                            0,
                            ChatMessage {
                                role: "system".to_owned(),
                                content: arg.to_owned(),
                                reasoning_content: None,
                            },
                        );
                    }
                    println!("{}system prompt updated.{}\n", color(tty, DIM), color(tty, RESET));
                }
                "help" => {
                    println!(
                        "{}/clear  /effort <none|low|medium|high>  /temp <f32>  /system <text>  /exit{}\n",
                        color(tty, DIM),
                        color(tty, RESET)
                    );
                }
                _ => println!(
                    "{}unknown command: /{name} (try /help){}\n",
                    color(tty, YELLOW),
                    color(tty, RESET)
                ),
            }
            continue;
        }

        history.push(ChatMessage {
            role: "user".to_owned(),
            content: input.to_owned(),
            reasoning_content: None,
        });

        let rendered = render_chat_prompt(&history, effort)?;
        let ids = engine.tokenizer().encode_with_special(&rendered);
        let params = GenParams {
            max_tokens: opts.max_tokens,
            temperature,
            top_p: opts.top_p,
            top_k: opts.top_k,
            chat_mode: true,
            ..GenParams::default()
        };

        let mut reply = String::new();
        let mut reasoning = String::new();
        let mut in_thinking = false;
        let mut printed_prefix = false;
        let mut out = io::stdout().lock();

        match engine.generate(&ids, &params, |chunk| {
            if !printed_prefix {
                let _ = write!(
                    out,
                    "{}{}inkwell ›{} ",
                    color(tty, BOLD),
                    color(tty, CYAN),
                    color(tty, RESET)
                );
                printed_prefix = true;
            }
            match chunk {
                GenChunk::Reasoning(s) => {
                    if !in_thinking {
                        let _ = write!(
                            out,
                            "\n{}{}[thinking] ",
                            color(tty, DIM),
                            color(tty, ITALIC)
                        );
                        in_thinking = true;
                    }
                    reasoning.push_str(&s);
                    let _ = write!(out, "{s}");
                }
                GenChunk::Content(s) => {
                    if in_thinking {
                        let _ = write!(out, "{}\n\n", color(tty, RESET));
                        in_thinking = false;
                    }
                    reply.push_str(&s);
                    let _ = write!(out, "{s}");
                }
            }
            let _ = out.flush();
            true
        }) {
            Ok(stats) => {
                if in_thinking {
                    let _ = write!(out, "{}", color(tty, RESET));
                }
                let _ = writeln!(out);
                drop(out);
                let trunc = if stats.finish_reason == "length" {
                    " · truncated (max_tokens)"
                } else {
                    ""
                };
                println!(
                    "{}prefill {} tok ({:.1} tok/s) · decode {} tok ({:.1} tok/s){trunc}{}\n",
                    color(tty, DIM),
                    stats.prompt_tokens,
                    stats.prefill_tps(),
                    stats.completion_tokens,
                    stats.decode_tps(),
                    color(tty, RESET)
                );
                history.push(ChatMessage {
                    role: "assistant".to_owned(),
                    content: reply,
                    reasoning_content: if reasoning.is_empty() {
                        None
                    } else {
                        Some(reasoning)
                    },
                });
            }
            Err(e) => {
                drop(out);
                history.pop();
                println!(
                    "\n{}error: {e}{}\n",
                    color(tty, YELLOW),
                    color(tty, RESET)
                );
            }
        }
    }
    Ok(())
}
