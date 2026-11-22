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

    if let Some("--tok") = args.next().as_deref() {
        let text = args.next().expect("--tok <text>");
        let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
        let ids = tok.encode(&text);
        println!("n={} ids={ids:?}", ids.len());
        let back = tok.decode(&ids).expect("decode");
        println!("roundtrip: {back:?}");
    }
}
