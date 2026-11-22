//! `inkwell` CLI: inspect a GGUF and print the Inkling-2 model config.

use inkwell_core::InklingConfig;
use inkwell_gguf::GgufFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inkwell <model.gguf>");
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
}
