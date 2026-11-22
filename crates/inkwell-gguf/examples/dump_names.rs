//! Dump tensor names per layer for architecture inspection.
fn main() {
    let g = inkwell_gguf::GgufFile::open(std::env::args().nth(1).expect("path")).expect("gguf");
    for l in 0..24u32 {
        let p = format!("blk.{l}.");
        let mut names: Vec<_> = g.tensors.keys().filter(|n| n.starts_with(&p)).collect();
        names.sort();
        println!("== {p}");
        for n in names { println!("{n}"); }
    }
}
