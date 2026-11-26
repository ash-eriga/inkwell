//! `inkwell` CLI: run, chat, serve (OpenAI-compatible), and benchmark Inkling-2 EXL3/GGUF models.

mod chat_template;
mod cli_ui;
mod engine;
mod models;
mod server;

use inkwell_core::InklingConfig;
use inkwell_gguf::GgufFile;
use inkwell_tokenizer::Tokenizer;

const HELP_TEXT: &str = "\
inkwell 0.1.0 — local EXL3/GGUF inference engine for Inkling-2

USAGE:
  inkwell serve [model] [OPTIONS]       Start OpenAI-compatible HTTP/SSE server (interactive picker if model omitted)
  inkwell chat  [model] [OPTIONS]       Interactive multi-turn terminal chat (interactive picker if model omitted)
  inkwell run   <model> -p <prompt>     One-shot streaming generation
  inkwell models                        List local EXL3 models and Hugging Face EXL3 presets
  inkwell <model> --run                 Legacy benchmark harness

SERVE OPTIONS:
  --host <addr>           Bind address (default: 127.0.0.1)
  --port <port>           Bind port (default: 8080)
  --ctx <tokens>          Max context length (default: 4096)
  --effort <level>        Default reasoning effort: none|minimal|low|medium|high|max (default: none)
  -v, --verbose           Show internal VRAM/cache/prefill diagnostics

CHAT / RUN OPTIONS:
  -p, --prompt <text>     Prompt text (required for `run`)
  --system <text>         System prompt
  --ctx <tokens>          Max context length (default: 4096)
  --max-tokens <n>        Max generated tokens (default: 512)
  --temp <f32>            Sampling temperature, 0.0 = greedy (default: 0.0)
  --top-p <f32>           Nucleus sampling threshold (default: 0.95)
  --top-k <n>             Top-k sampling cap (default: 40)
  --effort <level>        Reasoning effort: none|minimal|low|medium|high|max (default: none)
  --raw                   Skip chat template (`run` only)
  -v, --verbose           Show internal VRAM/cache/prefill diagnostics
";

fn run_on_big_stack<F>(f: F) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String> + Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(f)
        .map_err(|e| format!("failed to spawn worker thread: {e}"))?
        .join()
        .map_err(|_| "worker thread panicked".to_owned())?
}

fn next_val(args: &mut std::iter::Skip<std::env::Args>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

/// Run the interactive `inkwell` menu on the large-stack worker thread so a loaded
/// `Engine` can stay resident across `chat <-> serve` switches and quietly unload
/// after 4 seconds of idle time at the menu.
fn interactive_launcher() -> Result<(), String> {
    use std::io::{self, IsTerminal};
    if !io::stdin().is_terminal() {
        print!("{HELP_TEXT}");
        return Ok(());
    }
    run_on_big_stack(interactive_launcher_inner)
}

fn read_menu_line_with_idle_unload(
    warm: &mut Option<(String, engine::Engine)>,
    idle_limit: std::time::Duration,
) -> Result<Option<String>, String> {
    use std::io::{self, BufRead};
    use std::time::Instant;

    #[cfg(unix)]
    {
        let start = Instant::now();
        loop {
            if warm.is_some() && start.elapsed() >= idle_limit {
                *warm = None;
            }
            let mut pfd = libc::pollfd {
                fd: 0,
                events: libc::POLLIN,
                revents: 0,
            };
            let timeout = if warm.is_some() { 200 } else { -1 };
            let ready = unsafe { libc::poll(&mut pfd, 1, timeout) };
            if ready > 0 && (pfd.revents & libc::POLLIN) != 0 {
                let mut line = String::new();
                let n = io::stdin()
                    .lock()
                    .read_line(&mut line)
                    .map_err(|e| format!("failed to read input: {e}"))?;
                if n == 0 {
                    return Ok(None);
                }
                return Ok(Some(line));
            }
            if ready < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    return Ok(None);
                }
            }
        }
    }

    #[cfg(not(unix))]
    {
        let _ = (warm, idle_limit);
        let mut line = String::new();
        let n = io::stdin()
            .lock()
            .read_line(&mut line)
            .map_err(|e| format!("failed to read input: {e}"))?;
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(line))
    }
}

fn interactive_launcher_inner() -> Result<(), String> {
    use std::io::{self, IsTerminal, Write};
    use std::time::Duration;

    let tty = io::stdout().is_terminal();
    let (b, c, d, r) = if tty {
        ("\x1b[1m", "\x1b[36m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };

    let mut warm: Option<(String, engine::Engine)> = None;

    loop {
        println!("\n{b}{c}inkwell {}{r}  {d}EXL3 inference engine{r}\n", env!("CARGO_PKG_VERSION"));
        println!("  {c}[1]{r} {b}serve{r}    {d}OpenAI-compatible HTTP/SSE server (http://127.0.0.1:8080/v1){r}");
        println!("  {c}[2]{r} {b}chat{r}     {d}Interactive multi-turn terminal chat{r}");
        println!("  {c}[3]{r} {b}models{r}   {d}List local EXL3 models & Hugging Face presets{r}");
        println!("  {d}[q] quit{r}\n");

        print!("{b}Choose an action [1-3] (default 1, q=quit):{r} ");
        let _ = io::stdout().flush();

        let Some(line) = read_menu_line_with_idle_unload(&mut warm, Duration::from_secs(4))? else {
            return Ok(());
        };

        match line.trim().to_ascii_lowercase().as_str() {
            "" | "1" | "serve" => {
                inkwell_run::runner::set_verbose(std::env::var_os("INKWELL_VERBOSE").is_some());
                let (model_path, mut eng) = if let Some((path, eng)) = warm.take() {
                    (path, eng)
                } else {
                    let chosen = models::select_or_download_model()?;
                    if chosen.vision_enabled {
                        eprintln!("inkwell: vision tower = ON (soft flag; text-only inference active in v0.1)");
                    }
                    eprintln!("inkwell serve: loading {} (ctx=4096)...", chosen.path);
                    let eng = engine::Engine::load(&chosen.path, 4096)?;
                    (chosen.path, eng)
                };
                let serve_opts = server::ServeOpts {
                    model: model_path.clone(),
                    host: "127.0.0.1".to_owned(),
                    port: 8080,
                    ctx: 4096,
                    default_effort: "none".to_owned(),
                };
                server::run_server_with_engine(&mut eng, &serve_opts)?;
                warm = Some((model_path, eng));
            }
            "2" | "chat" => {
                inkwell_run::runner::set_verbose(std::env::var_os("INKWELL_VERBOSE").is_some());
                let (model_path, mut eng) = if let Some((path, eng)) = warm.take() {
                    (path, eng)
                } else {
                    let chosen = models::select_or_download_model()?;
                    if chosen.vision_enabled {
                        eprintln!("inkwell: vision tower = ON (soft flag; text-only inference active in v0.1)");
                    }
                    eprintln!("inkwell loading {} (ctx=4096)...", chosen.path);
                    let eng = engine::Engine::load(&chosen.path, 4096)?;
                    (chosen.path, eng)
                };
                let chat_opts = cli_ui::CliRunOpts {
                    model: model_path.clone(),
                    ..cli_ui::CliRunOpts::default()
                };
                cli_ui::run_chat_with_engine(&mut eng, &chat_opts)?;
                warm = Some((model_path, eng));
            }
            "3" | "models" => {
                models::print_model_catalog();
            }
            "q" | "quit" | "exit" => return Ok(()),
            _ => println!("Enter 1 (serve), 2 (chat), 3 (models), or q."),
        }
    }
}

fn parse_serve_and_run(mut args: std::iter::Skip<std::env::Args>) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut verbose = std::env::var_os("INKWELL_VERBOSE").is_some();
    let mut opts = server::ServeOpts {
        model: String::new(),
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: 4096,
        default_effort: "none".to_owned(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => opts.host = next_val(&mut args, "--host")?,
            "--port" => {
                let v = next_val(&mut args, "--port")?;
                opts.port = v.parse().map_err(|_| format!("invalid port: {v}"))?;
            }
            "--ctx" => {
                let v = next_val(&mut args, "--ctx")?;
                opts.ctx = v.parse().map_err(|_| format!("invalid ctx: {v}"))?;
            }
            "--effort" => opts.default_effort = next_val(&mut args, "--effort")?,
            "-v" | "--verbose" => verbose = true,
            "-h" | "--help" => {
                print!("{HELP_TEXT}");
                return Ok(());
            }
            other if !other.starts_with('-') && model.is_none() => {
                model = Some(other.to_owned());
            }
            other => return Err(format!("unknown option for serve: {other}")),
        }
    }
    inkwell_run::runner::set_verbose(verbose);
    opts.model = match model {
        Some(m) => m,
        None => {
            let chosen = models::select_or_download_model()?;
            if chosen.vision_enabled {
                eprintln!("inkwell: vision tower = ON (soft flag; text-only inference active in v0.1)");
            }
            chosen.path
        }
    };
    run_on_big_stack(move || server::run_server(&opts).map(|_| ()))
}

fn parse_cli_opts(
    mut args: std::iter::Skip<std::env::Args>,
    cmd: &str,
) -> Result<cli_ui::CliRunOpts, String> {
    let mut model: Option<String> = None;
    let mut verbose = std::env::var_os("INKWELL_VERBOSE").is_some();
    let mut opts = cli_ui::CliRunOpts::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-p" | "--prompt" => opts.prompt = Some(next_val(&mut args, &arg)?),
            "--system" => opts.system = Some(next_val(&mut args, "--system")?),
            "--ctx" => {
                let v = next_val(&mut args, "--ctx")?;
                opts.ctx = v.parse().map_err(|_| format!("invalid ctx: {v}"))?;
            }
            "--max-tokens" => {
                let v = next_val(&mut args, "--max-tokens")?;
                opts.max_tokens = v.parse().map_err(|_| format!("invalid max-tokens: {v}"))?;
            }
            "--temp" | "--temperature" => {
                let v = next_val(&mut args, &arg)?;
                opts.temperature = v.parse().map_err(|_| format!("invalid temp: {v}"))?;
            }
            "--top-p" => {
                let v = next_val(&mut args, "--top-p")?;
                opts.top_p = v.parse().map_err(|_| format!("invalid top-p: {v}"))?;
            }
            "--top-k" => {
                let v = next_val(&mut args, "--top-k")?;
                opts.top_k = v.parse().map_err(|_| format!("invalid top-k: {v}"))?;
            }
            "--effort" => opts.effort = next_val(&mut args, "--effort")?,
            "--raw" => opts.raw = true,
            "-v" | "--verbose" => verbose = true,
            "-h" | "--help" => {
                print!("{HELP_TEXT}");
                std::process::exit(0);
            }
            other if !other.starts_with('-') && model.is_none() => {
                model = Some(other.to_owned());
            }
            other => return Err(format!("unknown option for {cmd}: {other}")),
        }
    }
    inkwell_run::runner::set_verbose(verbose);
    opts.model = match model {
        Some(m) => m,
        None => {
            let chosen = models::select_or_download_model()?;
            if chosen.vision_enabled {
                eprintln!("inkwell: vision tower = ON (soft flag; text-only inference active in v0.1)");
            }
            chosen.path
        }
    };
    Ok(opts)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(first) = args.next() else {
        if let Err(e) = interactive_launcher() {
            if e != "cancelled" {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        return;
    };
    match first.as_str() {
        "-h" | "--help" | "help" => {
            print!("{HELP_TEXT}");
            return;
        }
        "-V" | "--version" => {
            println!("inkwell {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        "models" => {
            models::print_model_catalog();
            return;
        }
        "serve" => {
            if let Err(e) = parse_serve_and_run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
            return;
        }
        "chat" => {
            let res = parse_cli_opts(args, "chat")
                .and_then(|chat_opts| run_on_big_stack(move || cli_ui::run_chat(&chat_opts).map(|_| ())));
            if let Err(e) = res {
                if e != "cancelled" {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
            return;
        }
        "run" => {
            let res = parse_cli_opts(args, "run")
                .and_then(|opts| run_on_big_stack(move || cli_ui::run_oneshot(&opts)));
            if let Err(e) = res {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }

    let path = first;
    let st_dir = std::path::Path::new(&path).is_dir();
    let g = if st_dir {
        // tokenizer still lives in the GGUF sidecar
        let gg = std::env::var("INKWELL_TOK_GGUF")
            .unwrap_or_else(|_| format!("{path}/Inkling-2-Mini-UD-IQ2_XXS.gguf"));
        GgufFile::open(&gg).expect("open gguf (tokenizer sidecar)")
    } else {
        GgufFile::open(&path).expect("open gguf")
    };
    let cfg = if st_dir { InklingConfig::rexl3() } else { InklingConfig::from_gguf(&g) };
    println!("{cfg:?}");
    println!("visits/token: {}", cfg.n_visits());
    let mut total = 0u64;
    for (n, t) in &g.tensors {
        total += t.nbytes;
        let _ = n;
    }
    println!(
        "tensors: {}, weights {} MiB",
        g.tensors.len(),
        total / 1_048_576
    );

    let mode = args.next();
    if mode.as_deref() == Some("--encode-file") {
        let input = args.next().expect("--encode-file <text> <ids>");
        let output = args.next().expect("--encode-file <text> <ids>");
        let text = std::fs::read_to_string(input).expect("read prompt text");
        let ids = Tokenizer::from_gguf(&g).expect("tokenizer").encode(&text);
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for id in ids { bytes.extend_from_slice(&id.to_le_bytes()); }
        std::fs::write(output, bytes).expect("write prompt ids");
        return;
    }
    if mode.as_deref() == Some("--decode-file") {
        let input = args.next().expect("--decode-file <ids> <text>");
        let output = args.next().expect("--decode-file <ids> <text>");
        let ids = read_ids_file(std::ffi::OsStr::new(&input));
        let text = Tokenizer::from_gguf(&g).expect("tokenizer").decode(&ids).expect("decode ids");
        std::fs::write(output, text).expect("write prompt text");
        return;
    }
    if mode.as_deref() == Some("--run") {
        // deep ggml graphs (48 visits) overflow the default 8MB stack in
        // ggml_build_forward_expand's recursive DFS — run on a big-stack thread
        let cfg2 = cfg.clone();
        let p2 = path.clone();
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(move || run_e2e(&g, &cfg2, &p2))
            .expect("spawn")
            .join()
            .expect("run_e2e panicked");
        return;
    }
    if mode.as_deref() == Some("--types") {
        let mut census = std::collections::BTreeMap::new();
        for (name, t) in &g.tensors {
            let e = census.entry(t.ggml_type).or_insert((0usize, 0u64));
            e.0 += 1;
            e.1 += t.nbytes;
            let _ = name;
        }
        for (ty, (cnt, bytes)) in census {
            println!("type {ty}: {cnt} tensors, {} MiB", bytes / (1 << 20));
        }
        return;
    }
    if mode.as_deref() == Some("--spec") {
        bench_spec();
        return;
    }
    if mode.as_deref() == Some("--graph") {
        dump_graph(&cfg);
        return;
    }
    if mode.as_deref() == Some("--samplebench") {
        sample_bench(&cfg);
        return;
    }
    if mode.as_deref() == Some("--pager") {
        bench_pager(&g, &cfg);
        return;
    }
    if mode.as_deref() == Some("--kv") {
        dump_kv(&cfg);
        return;
    }
    if mode.as_deref() == Some("--model") {
        let g: &'static inkwell_gguf::GgufFile = Box::leak(Box::new(g));
        dump_model(g);
        return;
    }
    if mode.as_deref() == Some("--tokbench") {
        bench_tok(&g, 1000);
        return;
    }
    if mode.as_deref() == Some("--tok") {
        let text = args.next().expect("--tok <text>");
        let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
        let ids = tok.encode(&text);
        println!("n={} ids={ids:?}", ids.len());
        let back = tok.decode(&ids).expect("decode");
        println!("roundtrip: {back:?}");
    }
}

#[expect(clippy::cast_precision_loss, reason = "benchmark throughput counters")]
#[allow(dead_code)]
fn bench_tok(g: &inkwell_gguf::GgufFile, iters: u32) {
    use inkwell_tokenizer::Tokenizer;
    use std::time::Instant;
    let tok = Tokenizer::from_gguf(g).expect("tok");
    let text = "The quick brown fox jumps over the lazy dog. 42 + 17 = 59. def main():\n    print('hello world') # comment\n";
    let t = Instant::now();
    let mut n = 0;
    for _ in 0..iters { n += tok.encode(text).len(); }
    let d = t.elapsed();
    println!("encode: {} chars -> {} toks/iter, {:?} total, {:.0} tok/s", text.len(), n/(iters.max(1) as usize), d, n as f64/d.as_secs_f64());
    let ids = tok.encode(text);
    let t = Instant::now();
    for _ in 0..iters { std::hint::black_box(tok.decode(&ids).expect("d")); }
    let d = t.elapsed();
    println!("decode: {:?} total, {:.0} tok/s", d, (iters as usize*ids.len()) as f64/d.as_secs_f64());
}

use inkwell_kv::{KvPlan, KvType, Share};
fn dump_kv(cfg: &inkwell_core::InklingConfig) {
    println!("{:>10} {:>6} {:>7} {:>8} {:>10} {:>10} {:>6}", "ctx", "kv", "loop", "arenas", "swa MiB", "full MiB", "total");
    for ctx in [4_096usize, 32_768, 131_072, 262_144] {
        for (ty, tn) in [(KvType::F8E4m3, "fp8"), (KvType::Bf16, "bf16")] {
            for share in [Share::PerPass, Share::LoopShared] {
                let p = KvPlan::new(cfg, ctx, ty, share);
                println!("{:>10} {:>6} {:>7} {:>8} {:>10} {:>10} {:>6}",
                    ctx, tn, if share == Share::LoopShared {"shared"} else {"passes"}, p.arenas.len(),
                    p.swa_bytes()/1_048_576, p.full_bytes()/1_048_576, p.total_bytes()/1_048_576);
            }
        }
    }
}

#[expect(clippy::cast_precision_loss, reason = "MiB totals fit f64 mantissa")]
#[allow(dead_code)]
fn dump_model(g: &'static inkwell_gguf::GgufFile) {
    let m = inkwell_model::ModelWeights::from_gguf(g).expect("model");
    println!("layers: {} | global ok", m.layers.len());
    println!("total: {} MiB | experts: {} MiB ({:.1}%)", m.total_bytes()/1_048_576, m.expert_bytes()/1_048_576, 100.0 * m.expert_bytes() as f64 / m.total_bytes() as f64);
}

/// Simulated decode through the pager: each layer picks 8 experts,
/// each slot of the previous token's selection is reused with
/// probability RHO (temporal locality), else uniform. Draft-window
/// prefetch uses the same predictor as upstream (`prev_sel`).
const RHO: u64 = 55; // % chance an expert repeats from previous token
const N_TOK: u64 = 300;

#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss, reason = "simulation counters bounded below u32::MAX / f64 mantissa")]
fn bench_pager(g: &inkwell_gguf::GgufFile, cfg: &inkwell_core::InklingConfig) {
    use inkwell_model::ModelWeights;
    use inkwell_pager::{ExpertLayout, Pager};

    let m = ModelWeights::from_gguf(g).expect("model");
    let bytes_per_expert: Vec<usize> = m.layers.iter().map(|l| {
        let expert_bytes: usize = l.tensors.iter().zip(&l.names)
            .filter(|(_, n)| n.contains("exps"))
            .map(|(t, _)| t.nbytes as usize).sum();
        expert_bytes / cfg.n_expert as usize
    }).collect();
    let layout = ExpertLayout {
        bytes_per_expert: bytes_per_expert.clone(),
        n_expert: cfg.n_expert as usize,
        n_expert_used: cfg.n_expert_used as usize,
    };
    let slot_kb = layout.slot_bytes() / 1024;
    let per_token: usize = (0..cfg.loop_start).chain((0..cfg.n_loop).flat_map(|_| cfg.loop_start..cfg.loop_end)).chain(cfg.loop_end..cfg.n_layer)
        .map(|l| bytes_per_expert[l as usize] * cfg.n_expert_used as usize).sum();
    println!("expert size/layer: min {} KiB, max {} KiB | expert reads/token (44 visits x8): {} MiB",
        bytes_per_expert.iter().min().copied().unwrap_or(0)/1024, slot_kb, per_token / 1_048_576);

    // cheap deterministic "rng"
    let mut seed = 0x243f_6a88_85a3_08d3_u64;
    let mut rng = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };

    println!("{:>10} {:>8} {:>10} {:>12} {:>12}", "cache MiB", "slots", "hit %", "stream MB/tok", "t/s ceiling");
    for cache_mib in [256usize, 512, 1024, 2048, 3850] {
        let mut p = Pager::new(layout.clone(), cfg.n_layer as usize, cache_mib * 1_048_576);
        let mut prev: Vec<Vec<u32>> = vec![Vec::new(); cfg.n_layer as usize];
        let mut streamed = 0u64;
        let mut order: Vec<u32> = (0..cfg.loop_start).collect();
        for _ in 0..cfg.n_loop { order.extend(cfg.loop_start..cfg.loop_end); }
        order.extend(cfg.loop_end..cfg.n_layer);
        for _ in 0..N_TOK {
            for &l in &order {
                let l = l as usize;
                let mut sel = Vec::with_capacity(8);
                for k in 0..8u32 {
                    let keep = k < prev[l].len() as u32 && rng() % 100 < RHO;
                    sel.push(if keep { prev[l][k as usize] } else { (rng() % 128) as u32 });
                }
                // draft-window prefetch: predict with prev_sel, 4 MiB budget
                let mut budget = 4 * 1_048_576usize;
                p.prefetch(l as u32, &prev[l], &mut budget);
                let pre_fills = p.fills_len();
                let _slots = p.resolve(l as u32, &sel);
                let fills = p.take_fills();
                for (fl, _, _) in &fills[pre_fills.min(fills.len())..] {
                    streamed += layout.bytes_per_expert[*fl as usize] as u64;
                }
                prev[l] = sel;
            }
        }
        let hits = p.hits as f64 / (p.hits + p.misses) as f64 * 100.0;
        let mb_tok = streamed as f64 / N_TOK as f64 / 1_048_576.0;
        println!("{:>10} {:>8} {:>10.1} {:>12.2} {:>12.1}",
            cache_mib, p.n_slots(), hits, mb_tok, 10_000.0 / mb_tok);
    }
}

#[expect(clippy::cast_precision_loss, reason = "bench counters")]
fn sample_bench(cfg: &inkwell_core::InklingConfig) {
    use inkwell_sampler::{Rng, Sampler};
    use std::time::Instant;
    let nv = cfg.vocab as usize;
    let logits: Vec<f32> = (0..nv).map(|i| ((i * 2_654_435_761) % 1000) as f32 / 1000.0).collect();
    let mut rng = Rng::new(42);
    let smp = Sampler { top_k: 40, top_p: 0.95, min_p: 0.05, temperature: 0.8 };
    // warmup
    let mut acc = 0u64;
    for _ in 0..1000 { acc += u64::from(smp.sample(&logits, &mut rng)); }
    let t = Instant::now();
    let iters = 10_000;
    for _ in 0..iters { acc += u64::from(smp.sample(&logits, &mut rng)); }
    let d = t.elapsed();
    println!("sample(topk40,topp.95,minp.05,temp.8) x{} over {} vocab: {:?}/sample ({:.0}/s) [{}]",
        iters, nv, d / u32::try_from(iters).unwrap_or(1), f64::from(iters) / d.as_secs_f64(), acc % 97);
    let g = Sampler { temperature: 0.0, ..Sampler::default() };
    let t = Instant::now();
    for _ in 0..iters { acc += u64::from(g.sample(&logits, &mut rng)); }
    println!("greedy x{}: {:?}/sample [{acc}]", iters, t.elapsed() / u32::try_from(iters).unwrap_or(1));
}

fn dump_graph(cfg: &inkwell_core::InklingConfig) {
    let sched = inkwell_graph::build_schedule(cfg);
    let mut hist: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut total = 0usize;
    for v in &sched.visits {
        for op in &v.ops {
            *hist.entry(op_name(op)).or_default() += 1;
            total += 1;
        }
    }
    println!("visits: {} | ops total: {}", sched.visits.len(), total);
    let mut h: Vec<_> = hist.into_iter().collect();
    h.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (n, c) in h { println!("  {n:>14}: {c}"); }
    println!("per-visit ops: min {} max {}",
        sched.visits.iter().map(|v| v.ops.len()).min().unwrap_or(0),
        sched.visits.iter().map(|v| v.ops.len()).max().unwrap_or(0));
}

fn op_name(op: &inkwell_graph::Op) -> &'static str {
    use inkwell_graph::Op::{Attn, Clip, DepthMix, Embed, GateMul, Head, LoopControl,
        LoopMessage, MoeExperts, MoeRoute, Norm, Proj, PushBlock, RelBiasTable,
        Residual, ShortConv, TauScale, VNorm};
    match op {
        Embed => "embed", LoopControl { .. } => "loop_ctrl",
        DepthMix { .. } => "depth_mix", Norm { .. } => "norm", VNorm => "vnorm",
        Proj { .. } => "proj", Clip { .. } => "clip",
        ShortConv { .. } => "shortconv", TauScale => "tau",
        RelBiasTable => "rel_bias", Attn { .. } => "attn",
        GateMul { .. } => "gate", PushBlock => "push_blk",
        Residual => "residual", MoeRoute => "moe_route",
        MoeExperts => "moe_exps", LoopMessage => "loop_msg",
        Head => "head",
    }
}

/// Spec-decode projection: emitted tokens per weight read at varying
/// acceptance, crossed with the pager's stream/token.
fn bench_spec() {
    const OVERLAP: f64 = 1.3;
    println!("{:>14} {:>14} {:>14} {:>12}", "draft k", "acc/round", "tok/read", "t/s proj");
    // one weight read per round covers the whole draft (union of
    // expert selections ~1.3x a single pass at 63% pager hit ~325 MiB)
    for (k, acc) in [(4u32, 1.6f64), (4, 2.0), (8, 2.5), (8, 3.2), (8, 4.0)] {
        let tok_per_read = acc + 1.0; // accepted + bonus token
        let tps = tok_per_read * 10_000.0 / (325.0 * OVERLAP);
        println!("{k:>14} {acc:>14.1} {tok_per_read:>14.1} {tps:>12.1}");
    }
}

/// End-to-end decode: encode prompt tokens one at a time, then generate.
fn read_ids_file(path: &std::ffi::OsStr) -> Vec<u32> {
    std::fs::read(path).expect("read token id file")
        .chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().expect("u32"))).collect()
}

fn run_e2e(gg: &inkwell_gguf::GgufFile, cfg: &inkwell_core::InklingConfig, path: &str) {
    use inkwell_run::runner::Runner;
    use std::time::Instant;
    let max_ctx: i64 = std::env::var("INKWELL_MAX_CTX")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    let mut r = if std::path::Path::new(path).is_dir() {
        let repo = Box::leak(Box::new(inkwell_st::StRepo::open(std::path::Path::new(path)).expect("st repo")));
        Runner::load_st(repo, cfg, max_ctx)
    } else {
        Runner::load(gg, cfg, max_ctx)
    };
    eprintln!("runner loaded");
    // real prompt via our tokenizer (bit-identical to llama.cpp)
    let ids = if let Ok(path) = std::env::var("INKWELL_PROMPT_IDS") {
        read_ids_file(std::ffi::OsStr::new(&path))
    } else {
        inkwell_tokenizer::Tokenizer::from_gguf(gg)
            .expect("tokenizer")
            .encode(&std::env::var("INKWELL_PROMPT").unwrap_or_else(|_| "The capital of France is".into()))
    };
    // upstream adds no BOS for this model (add_bos=false)
    let t0 = Instant::now();
    let mut pos = 0i64;
    let mut logits = Vec::new();
    if let Some(state) = std::env::var_os("INKWELL_STATE_LOAD") {
        r.load_state(&state.to_string_lossy());
        pos = ids.len() as i64 - 1;
        eprintln!("state loaded; skipped encode of {} tok", ids.len() - 1);
    } else {
        let prompt = &ids[..ids.len() - 1];
        let use_prefill = prompt.len() >= 16 && std::env::var_os("INKWELL_NO_PREFILL").is_none();
        if use_prefill {
            let _ = r.prefill_chunked(prompt, 0);
            pos = prompt.len() as i64;
        } else {
            for &id in prompt {
                // Prompt rows before the final row never need logits, so run only
                // the state-update graph and omit lm_head/logits entirely.
                r.step_prompt_state(id, pos);
                pos += 1;
            }
            if !use_prefill { eprintln!("prompt lm_head/logits skipped for non-final encode rows"); }
        }
        eprintln!("encode {} tok in {:?} ({})", prompt.len(), t0.elapsed(), if use_prefill {"chunked"} else {"serial"});
        r.clear_step_cache();
    }
    if let Some(state) = std::env::var_os("INKWELL_STATE_SAVE") {
        r.save_state(&state.to_string_lossy());
        eprintln!("state saved");
    }
    // last prompt token → first generated
    let t1 = Instant::now();
    let mut id = ids[ids.len() - 1];
    let mut gen_ids = Vec::new();
    let n_gen: usize = std::env::var("INKWELL_GENS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    // INKWELL_FORCE_IDS=<u32le file>: teacher-force this generated-token sequence.
    // Row 0 is produced from the final prompt token; force[0] is then fed at row 1.
    let force_ids = std::env::var_os("INKWELL_FORCE_IDS").map(|p| read_ids_file(&p));
    if let Some(f) = &force_ids {
        assert!(f.len() >= n_gen, "INKWELL_FORCE_IDS has {} tokens, need {n_gen}", f.len());
    }
    // INKWELL_VTEST8: after every serial step, replay the 8 tokens just
    // processed as one step_block (conv state snapshotted 8 steps back) and
    // compare all 8 rows against the serial logits — covers deep rows, which
    // the single-row VTEST cannot reach.
    let vtest8 = std::env::var_os("INKWELL_VTEST8").is_some();
    let mut fed: Vec<u32> = Vec::new();
    let mut hist_logits: std::collections::VecDeque<Vec<f32>> = Default::default();
    let mut hist_snaps: std::collections::VecDeque<Vec<Vec<f32>>> = Default::default();
    let (mut vt8_rows, mut vt8_bad) = (0usize, 0usize);
    // INKWELL_STEPDBG: per-token stepwall/gen/top5/poststep prints (they cost
    // ~0.5ms of unbuffered stderr + scans on the token boundary); default off.
    let stepdbg = std::env::var_os("INKWELL_STEPDBG").is_some() || vtest8;
    let mut logit_dump = std::env::var_os("INKWELL_LOGIT_DUMP")
        .map(|path| std::fs::File::create(path).expect("logit dump file"));
    let dump_logits = logit_dump.is_some();
    // INKWELL_BTEST8: run one 8-token block at the current position and dump
    // its rows. Inputs are the final prompt token followed by force[0..7].
    if std::env::var_os("INKWELL_BTEST8").is_some() {
        let f = force_ids.as_ref().expect("INKWELL_BTEST8 needs INKWELL_FORCE_IDS");
        assert!(f.len() >= 7, "INKWELL_BTEST8 needs at least 7 forced ids");
        let mut blk = [0u32; 8];
        blk[0] = id;
        blk[1..].copy_from_slice(&f[..7]);
        let vl = r.step_block(&blk, pos);
        let nv = vl.len() / 8;
        if let Some(f) = &mut logit_dump {
            use std::io::Write as _;
            for j in 0..8 {
                f.write_all(&(nv as u32).to_le_bytes()).expect("logit dump write");
                for x in &vl[j * nv..(j + 1) * nv] { f.write_all(&x.to_le_bytes()).expect("logit dump write"); }
            }
        }
        eprintln!("BTEST8 base={pos} rows=8");
        return;
    }
    for g in 0..n_gen {
        if vtest8 { hist_snaps.push_back(r.snapshot_conv()); }
        let ts = Instant::now();
        let (best, fed_id);
        if stepdbg || dump_logits {
            logits = r.step(id, pos);
            if stepdbg { eprintln!("stepwall={:.1}ms", ts.elapsed().as_secs_f64()*1e3); }
            let ta = Instant::now();
            best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            fed_id = id;
            if let Some(f) = &mut logit_dump {
                use std::io::Write as _;
                f.write_all(&(logits.len() as u32).to_le_bytes()).expect("logit dump write");
                for x in &logits { f.write_all(&x.to_le_bytes()).expect("logit dump write"); }
            }
            if stepdbg {
                if g == 0 { eprintln!("gen0-logits sum={:.5} first3={:?}", logits.iter().sum::<f32>(), &logits[..3]); }
                eprintln!("gen {g}: id={id} next={best} top5={:?}", top(&logits, 5));
                eprintln!("poststep={:.1}ms", ta.elapsed().as_secs_f64()*1e3);
            }
        } else {
            best = r.step_id(id, pos) as usize;
            fed_id = id;
        }
        pos += 1;
        id = if let Some(f) = &force_ids { f[g] } else { best as u32 };
        gen_ids.push(id);
        if vtest8 {
            fed.push(fed_id);
            hist_logits.push_back(logits.clone());
            if hist_snaps.len() > 8 { hist_snaps.pop_front(); hist_logits.pop_front(); }
            if hist_snaps.len() == 8 {
                // steps g-7..g fed fed[g-7..=g] at pos-8..pos-1
                let mut blk = [0u32; 8];
                blk.copy_from_slice(&fed[g - 7..=g]);
                let base = pos - 8;
                let snap = hist_snaps.front().expect("8 conv snapshots").clone();
                r.restore_conv(&snap);
                let tb = Instant::now();
                let vl = r.step_block(&blk, base);
                let msb = tb.elapsed().as_secs_f64() * 1e3;
                let nv = logits.len();
                let mut line = String::new();
                for j in 0..8 {
                    let row = &vl[j * nv..(j + 1) * nv];
                    let ser = &hist_logits[j];
                    let d: f32 = row.iter().zip(ser.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                    let (ba, _) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
                    let (sa, _) = ser.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
                    // serial top1-top2 margin: a flip inside it is fp noise
                    let t2 = top(ser, 2);
                    let margin = t2[0].1 - t2[1].1;
                    vt8_rows += 1;
                    if ba != sa {
                        vt8_bad += 1;
                        eprintln!("VTEST8 MISMATCH g={g} base={base} j={j} block={ba} serial={sa} serial_margin={margin:.4} maxd={d:.5}");
                    }
                    line.push_str(&format!(" j{j}:{} d={d:.1e}", if ba == sa { "ok" } else { "BAD" }));
                }
                eprintln!("VTEST8 g={g} base={base} blk={msb:.0}ms{line}");
            }
        }
    }
    if vtest8 { eprintln!("VTEST8 total: rows={vt8_rows} mismatches={vt8_bad}"); }
    let dec_t = t1.elapsed();
    if let Some(path) = std::env::var_os("INKWELL_GEN_IDS_DUMP") {
        use std::io::Write as _;
        let mut f = std::fs::File::create(path).expect("create generated ids dump");
        for id in &gen_ids { f.write_all(&id.to_le_bytes()).expect("write generated id"); }
    }
    let text = inkwell_tokenizer::Tokenizer::from_gguf(gg)
        .expect("tokenizer")
        .decode(&gen_ids)
        .unwrap_or_else(|_| "<decode error>".into());
    eprintln!("generated text: {text:?}");
    eprintln!("decode 8 tok in {:?}", dec_t);
    inkwell_run::runner::stepprof_dump();

    if std::env::var_os("INKWELL_VTEST").is_some() {
        // verify path sanity: block of 8×(last id) at `pos` — row 0 logits must
        // equal what step() produced for that id at pos.
        // correct baseline: step() for `id` AT `pos` (not the stale decode
        // logits). Conv state is snapshotted + restored around it so the block
        // sees the same conv position; KV at pos is rewritten with identical
        // values by the block's token 0.
        let snap = r.snapshot_conv();
        let logits0 = r.step(id, pos);
        r.restore_conv(&snap);
        let blk = [id; 8];
        let t2 = Instant::now();
        let vl = r.step_block(&blk, pos);
        eprintln!("verify-8 in {:?}", t2.elapsed());
        let nv = logits0.len();
        let d: f32 = vl[..nv].iter().zip(logits0.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let (b0, _) = vl[..nv].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        let (br, _) = logits0.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        eprintln!("verify row0: max|d|={d:.6} argmax={b0} vs step argmax={br}");
        // also dump row7-vs-row0 sanity: rows should diverge (different KV)
    }

    if std::env::var("INKWELL_SPEC").as_deref() == Ok("mtp") {
        // DSpark draft: bootstrap feats via a throwaway block on `id` at pos,
        // then cycles of step_block(vtok) -> accept -> ctx_append -> propose.
        let blk0 = [id; 8];
        let _ = r.step_block(&blk0, pos);  // creates block_feats
        r.rollback_conv(1);
        let mut mtp = inkwell_run::draft::Mtp::new(path,
            r.unembed_desc().expect("unembed"), r.block_feats_ptr(), max_ctx as usize);
        // Prompt ctx: replay the prompt through step_block in NB-sized chunks so
        // draft ctx-K/V covers positions 0..pos (reference does this in prefill).
        {
            let t0p = Instant::now();
            // tokens for every committed position so far: prompt + serial gen
            let toks: Vec<u32> = ids.iter().map(|&t| t as u32)
                .chain(gen_ids.iter().copied()).collect();
            let mut bases: Vec<i64> = Vec::new();
            let mut b: i64 = 0;
            let last_full = (toks.len() as i64) - 8; // need 8 real tokens per block
            while b <= last_full { bases.push(b); b += 8; }
            if !bases.is_empty() && *bases.last().unwrap() != last_full {
                bases.push(last_full); // final overlapping chunk ends at pos-1
            }
            bases.dedup();
            for &bc in &bases {
                if bc < 0 { continue; }
                let mut blk = [0u32; 8];
                for j in 0..8 { blk[j] = toks[(bc as usize) + j]; }
                eprintln!("ctx replay base={bc}");
                let _ = r.step_block(&blk, bc);
                inkwell_run::draft::dev_sync();
                eprintln!("  block done");
                mtp.ctx_append(bc);
                inkwell_run::draft::dev_sync();
                eprintln!("  ctx appended (sync ok)");
            }
            eprintln!("prompt ctx replay: {} blocks in {:?}", bases.len(), t0p.elapsed());
        }
        // blk0 again: feats slab must hold rows for pos..pos+7 (replay overwrote it)
        let _ = r.step_block(&blk0, pos);
        r.rollback_conv(1);
        mtp.ctx_append(pos);
        // the anchor sits at `pos`: the verify block is [anchor, d1..d7] at
        // pos..pos+7 (same frame as VTEST / Mtp::cycle), not pos+1 — a +1 base
        // processes the anchor twice ("capital of of ...").
        let mut c = pos;
        let mut kp = 8usize;
        let mut vtok = mtp.propose(c, id, 0, &mut kp);
        let target: usize = std::env::var("INKWELL_SPEC_GENS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(32);
        let mut spec_ids: Vec<u32> = Vec::new();
        let (mut rounds, mut accepted) = (0u64, 0u64);
        let mut spec_rng: u64 = 0x243F6A8885A308D3;
        let t2 = Instant::now();
        while spec_ids.len() < target {
            let tv = Instant::now();
            inkwell_run::moe::SPEC_NSLOTS.store((kp * 8) as i32, std::sync::atomic::Ordering::Relaxed);
            let vl = r.step_block(&vtok, c);
            let ms = tv.elapsed().as_secs_f64() * 1e3;
            let nv = vl.len() / 8;
            let mut verify = [0u32; 8];
            for j in 0..8 {
                verify[j] = vl[j * nv..(j + 1) * nv].iter()
                    .enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            }
            // probabilistic acceptance (vLLM draft_sample_method=probabilistic):
            // accept draft d at row j w.p. min(1, p_target(d)/p_draft(d)); on
            // rejection emit a sample from the residual (p_t - p_d)+.
            let mut n = 0;
            let mut emitted: Vec<u32> = Vec::with_capacity(8);
            if std::env::var_os("INKWELL_SPEC_PROB").is_some() {
                for j in 0..kp - 1 {
                    let d = vtok[j + 1] as usize;
                    let vrow = &vl[j * nv..(j + 1) * nv];
                    let pt = softmax_prob(vrow, d);
                    let pd = softmax_prob(&mtp.dz[j + 1], d);
                    spec_rng ^= spec_rng << 13; spec_rng ^= spec_rng >> 7; spec_rng ^= spec_rng << 17;
                    let u = (spec_rng >> 11) as f64 / ((1u64 << 53) as f64);
                    if rounds <= 3 && j < 3 {
                        let t5v = top(vrow, 5);
                        let t5d = top(&mtp.dz[j + 1], 5);
                        eprintln!("      j{j} d={d} pt={pt:.4} pd={pd:.4} targ_top5={:?} draft_top5={:?}", t5v, t5d);
                    }
                    if u < (pt / pd.max(1e-300)).min(1.0) {
                        n += 1; emitted.push(d as u32);
                    } else {
                        emitted.push(residual_sample(vrow, &mtp.dz[j + 1], &mut spec_rng));
                        break;
                    }
                }
                if n == kp - 1 { emitted.push(verify[kp - 1]); }
            } else {
                while n < kp - 1 && vtok[n + 1] == verify[n] { n += 1; }
                emitted.extend_from_slice(&verify[..n + 1]);
            }
            let ne = emitted.len();
            spec_ids.extend_from_slice(&emitted);
            id = emitted[emitted.len() - 1];
            rounds += 1; accepted += n as u64;
            let c_old = c;
            c += ne as i64;
            eprintln!("spec r{rounds}: verify={ms:.1}ms accept={n} emit={ne} pos={c}");
            eprintln!("   vtok={:?}
   veri={:?}", &vtok, &verify);
            r.rollback_conv(ne);
            let td = Instant::now();
            mtp.ctx_append(c_old);
            vtok = mtp.propose(c, id, n, &mut kp);
            // fewer distinct tokens -> smaller expert union in verify: pad
            // rows past dlen with the anchor (they can never be emitted).
            // fewer distinct tokens -> smaller expert union in verify: pad
            // rows past INKWELL_SPEC_DLEN with the anchor (never emitted).
            let dlen: usize = std::env::var("INKWELL_SPEC_DLEN").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(7)
                .min(kp - 1);
            for j in dlen + 1..8 { vtok[j] = vtok[0]; }
            eprintln!("          draft={:.1}ms", td.elapsed().as_secs_f64() * 1e3);
        }
        let dt = t2.elapsed().as_secs_f64();
        let text = inkwell_tokenizer::Tokenizer::from_gguf(gg).expect("tokenizer")
            .decode(&spec_ids).unwrap_or_else(|_| "<decode error>".into());
        eprintln!("spec text: {text:?}");
        eprintln!("spec: {} tok in {dt:.1}s = {:.1} tok/s ({} rounds, accept {:.0}%, {:.2} tok/round)",
            spec_ids.len(), spec_ids.len() as f64 / dt, rounds,
            100.0 * accepted as f64 / (rounds.max(1) * 7) as f64,
            spec_ids.len() as f64 / rounds.max(1) as f64);
        return;
    }
    if let Ok(spec) = std::env::var("INKWELL_SPEC") {
        // spec decode: rounds of [committed id + <=7 drafts] through step_block;
        // accept the matching prefix, emit it + the bonus row, rewind conv.
        let k: usize = spec.parse().unwrap_or(7).min(7);
        let mut drafter = inkwell_spec::NgramDraft::new(4);
        use inkwell_spec::Draft;
        let mut hist: Vec<u32> = ids.clone();
        hist.extend_from_slice(&gen_ids);
        hist.push(id);
        let target: usize = std::env::var("INKWELL_SPEC_GENS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(32);
        let mut spec_ids: Vec<u32> = Vec::new();
        let (mut rounds, mut drafted, mut accepted) = (0u64, 0u64, 0u64);
        let t2 = Instant::now();
        while spec_ids.len() < target {
            let draft = drafter.propose(&hist, k);
            let nd = draft.len();
            let mut blk = [id; 8];
            for (i, &d) in draft.iter().enumerate() { blk[i + 1] = d; }
            let tv = Instant::now();
            let vl = r.step_block(&blk, pos);
            let ms = tv.elapsed().as_secs_f64() * 1e3;
            let nv = vl.len() / 8;
            let verify: Vec<u32> = (0..8).map(|j| vl[j * nv..(j + 1) * nv].iter()
                .enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32).collect();
            let mut n = 0;
            while n < nd && draft[n] == verify[n] { n += 1; }
            let mut emitted: Vec<u32> = draft[..n].to_vec();
            emitted.push(verify[n]);
            let ne = emitted.len();
            hist.extend_from_slice(&emitted);
            spec_ids.extend_from_slice(&emitted);
            id = *emitted.last().unwrap();
            pos += ne as i64;
            rounds += 1; drafted += nd as u64; accepted += n as u64;
            eprintln!("spec r{rounds}: verify={ms:.1}ms draft={nd} accept={n} emit={ne} pos={pos}");
            r.rollback_conv(ne);
        }
        let dt = t2.elapsed().as_secs_f64();
        let text = inkwell_tokenizer::Tokenizer::from_gguf(gg).expect("tokenizer")
            .decode(&spec_ids).unwrap_or_else(|_| "<decode error>".into());
        eprintln!("spec text: {text:?}");
        eprintln!("spec: {} tok in {dt:.1}s = {:.1} tok/s ({} rounds, accept {:.0}%, {:.2} tok/round)",
            spec_ids.len(), spec_ids.len() as f64 / dt, rounds,
            100.0 * accepted as f64 / drafted.max(1) as f64,
            spec_ids.len() as f64 / rounds.max(1) as f64);
    }
}

fn top(l: &[f32], n: usize) -> Vec<(usize, f32)> {
    // linear top-k: keep a sorted best-n, no big allocations
    let mut v: Vec<(usize, f32)> = Vec::with_capacity(n + 1);
    for (i, &x) in l.iter().enumerate() {
        if v.len() < n || x > v[n - 1].1 {
            v.push((i, x));
            let mut j = v.len() - 1;
            while j > 0 && v[j].1 > v[j - 1].1 { v.swap(j, j - 1); j -= 1; }
            v.truncate(n);
        }
    }
    v
}

fn softmax_prob(l: &[f32], d: usize) -> f64 {
    let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let (mut s_, mut pd) = (0.0f64, 0.0f64);
    for (i, &v) in l.iter().enumerate() {
        let e = ((v - m) as f64).exp(); s_ += e;
        if i == d { pd = e; }
    }
    pd / s_.max(1e-300)
}

fn residual_sample(vrow: &[f32], drow: &[f32], rng: &mut u64) -> u32 {
    let mv = vrow.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let md = drow.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut res = vec![0f64; vrow.len()];
    let (mut sv, mut sd) = (0.0f64, 0.0f64);
    for i in 0..vrow.len() {
        sv += ((vrow[i] - mv) as f64).exp();
        sd += ((drow[i] - md) as f64).exp();
    }
    let mut z = 0.0f64;
    for i in 0..vrow.len() {
        let r = (((vrow[i] - mv) as f64).exp() / sv - ((drow[i] - md) as f64).exp() / sd).max(0.0);
        res[i] = r; z += r;
    }
    if z <= 0.0 {
        return vrow.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    }
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    let mut u = (*rng >> 11) as f64 / ((1u64 << 53) as f64) * z;
    for (i, &r) in res.iter().enumerate() {
        u -= r;
        if u <= 0.0 { return i as u32; }
    }
    vrow.len() as u32 - 1
}
