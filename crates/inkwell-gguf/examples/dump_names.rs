//! Dump attention pattern metadata for architecture inspection.
fn main() {
    let g = inkwell_gguf::GgufFile::open(std::env::args().nth(1).expect("path")).expect("gguf");
    for (k, v) in &g.meta {
        if k.contains("sliding") || k.contains("swa") {
            println!("{k} = {v:?}");
        }
    }
}
