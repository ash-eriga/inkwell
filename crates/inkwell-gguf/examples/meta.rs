fn main() {
    let g = inkwell_gguf::GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    for (k, v) in &g.meta {
        if k.contains("sliding") || k.contains("swa") || k.contains("window") { println!("{k} = {v:?}"); }
    }
}
