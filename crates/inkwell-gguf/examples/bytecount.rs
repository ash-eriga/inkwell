//! byte accounting per token: dense vs expert vs lm_head, per layer
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let g = inkwell_gguf::GgufFile::open(path).expect("gguf");
    let mut v: Vec<_> = g.tensors.values().collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));

    let mut layers: BTreeMap<u32, (u64, u64, u64)> = BTreeMap::new(); // blk -> (expert, attn+dense, ctrl)
    let mut other: BTreeMap<String, u64> = BTreeMap::new();
    for t in &v {
        let name = &t.name;
        if let Some(rest) = name.strip_prefix("blk.") {
            let blk: u32 = rest.split('.').next().unwrap().parse().unwrap();
            let e = layers.entry(blk).or_default();
            if name.contains("_exps.") || name.contains("exp_probs") {
                e.0 += t.nbytes;
            } else if name.contains("loop_") {
                e.2 += t.nbytes;
            } else {
                e.1 += t.nbytes;
            }
        } else {
            *other.entry(name.clone()).or_default() += t.nbytes;
        }
    }

    let mut tot_expert = 0u64;
    let mut tot_dense = 0u64;
    let mut tot_ctrl = 0u64;
    for (blk, (e, d, c)) in &layers {
        println!("blk.{:02}  expert {:>8.1} MiB  dense {:>6.1} MiB  ctrl {:>5.1} MiB",
            blk, *e as f64 / 1048576.0, *d as f64 / 1048576.0, *c as f64 / 1048576.0);
        tot_expert += e; tot_dense += d; tot_ctrl += c;
    }
    for (k, n) in &other {
        println!("{:<40} {:>8.1} MiB", k, *n as f64 / 1048576.0);
    }
    println!("---");
    println!("experts total {:>9.1} MiB | dense-in-layer {:>6.1} MiB | ctrl {:>6.1} MiB",
        tot_expert as f64 / 1048576.0, tot_dense as f64 / 1048576.0, tot_ctrl as f64 / 1048576.0);
    let total: u64 = v.iter().map(|t| t.nbytes).sum();
    println!("file tensors total {:>9.1} MiB", total as f64 / 1048576.0);
}
