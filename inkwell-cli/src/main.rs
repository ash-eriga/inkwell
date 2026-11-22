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
    println!("encode: {} chars -> {} toks/iter, {:?} total, {:.0} tok/s", text.len(), n/iters.max(1) as usize, d, n as f64/d.as_secs_f64());
    let ids = tok.encode(text);
    let t = Instant::now();
    for _ in 0..iters { std::hint::black_box(tok.decode(&ids).expect("d")); }
    let d = t.elapsed();
    println!("decode: {:?} total, {:.0} tok/s", d, (iters as usize*ids.len()) as f64/d.as_secs_f64());
}
