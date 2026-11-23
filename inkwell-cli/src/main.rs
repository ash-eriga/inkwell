//! `inkwell` CLI: inspect a GGUF and print the Inkling-2 model config.

use inkwell_core::InklingConfig;
use inkwell_gguf::GgufFile;
use inkwell_tokenizer::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: inkwell <model.gguf> [--tok <text>]");
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

    if let Some("--run") = args.next().as_deref() {
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
    if let Some("--types") = args.next().as_deref() {
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
    if let Some("--spec") = args.next().as_deref() {
        bench_spec();
        return;
    }
    if let Some("--graph") = args.next().as_deref() {
        dump_graph(&cfg);
        return;
    }
    if let Some("--samplebench") = args.next().as_deref() {
        sample_bench(&cfg);
        return;
    }
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
    let mut ids = inkwell_tokenizer::Tokenizer::from_gguf(gg)
        .expect("tokenizer")
        .encode("The capital of France is");
    // upstream adds no BOS for this model (add_bos=false)
    let t0 = Instant::now();
    let mut pos = 0i64;
    let mut logits = Vec::new();
    for &id in &ids[..ids.len() - 1] {
        logits = r.step(id, pos);
        pos += 1;
    }
    eprintln!("encode {} tok in {:?}", ids.len() - 1, t0.elapsed());
    eprintln!("prompt-logits sum={:.5} first3={:?}", logits.iter().sum::<f32>(), &logits[..3]);
    // last prompt token → first generated
    let t1 = Instant::now();
    let mut id = ids[ids.len() - 1];
    let mut gen_ids = Vec::new();
    let n_gen: usize = std::env::var("INKWELL_GENS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    for g in 0..n_gen {
        let ts = Instant::now();
        logits = r.step(id, pos);
        eprintln!("stepwall={:.1}ms", ts.elapsed().as_secs_f64()*1e3);
        pos += 1;
        let ta = Instant::now();
        let (best, _) = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        if g == 0 { eprintln!("gen0-logits sum={:.5} first3={:?}", logits.iter().sum::<f32>(), &logits[..3]); }
        eprintln!("gen {g}: id={id} next={best} top5={:?}",
            top(&logits, 5));
        id = best as u32;
        gen_ids.push(id);
        eprintln!("poststep={:.1}ms", ta.elapsed().as_secs_f64()*1e3);
    }
    let dec_t = t1.elapsed();
    let text = inkwell_tokenizer::Tokenizer::from_gguf(gg)
        .expect("tokenizer")
        .decode(&gen_ids)
        .unwrap_or_else(|_| "<decode error>".into());
    eprintln!("generated text: {text:?}");
    eprintln!("decode 8 tok in {:?}", dec_t);

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
