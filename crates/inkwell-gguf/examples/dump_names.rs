//! Dump all architecture metadata keys.
fn main() {
    let g = inkwell_gguf::GgufFile::open(std::env::args().nth(1).expect("path")).expect("gguf");
    let mut keys: Vec<_> = g.meta.keys().collect();
    keys.sort();
    for k in keys {
        if k.starts_with("inkling") || k.starts_with("general") {
            println!("{k} = {:?}", g.meta[k]);
        }
    }
}
