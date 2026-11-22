//! `inkwell` CLI: inspect a GGUF and print the Inkling-2 model config.

use inkwell_core::InklingConfig;
use inkwell_gguf::GgufFile;
use inkwell_tokenizer::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: inkwell <model.gguf> [--tok <text>]");
    let g = GgufFile::open(&path).expect("open gguf");
    let cfg = InklingConfig::from_gguf(&g);
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

    if let Some("--pager") = args.next().as_deref() {
        bench_pager(&g, &cfg);
        return;
    }
    if let Some("--kv") = args.next().as_deref() {
        dump_kv(&cfg);
        return;
    }
    if let Some("--model") = args.next().as_deref() {
        let g: &'static inkwell_gguf::GgufFile = Box::leak(Box::new(g));
        dump_model(g);
        return;
    }
    if let Some("--tokbench") = args.next().as_deref() {
        bench_tok(&g, 1000);
        return;
    }
    if let Some("--tok") = args.next().as_deref() {
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
#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss, reason = "simulation counters bounded below u32::MAX / f64 mantissa")]
const RHO: u64 = 55; // % chance an expert repeats from previous token
const N_TOK: u64 = 300;

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
