//! dump name -> ggml type id for tensors matching a prefix
fn main() {
    let path = std::env::args().nth(1).expect("path");
    let prefix = std::env::args().nth(2).unwrap_or_default();
    let g = inkwell_gguf::GgufFile::open(path).expect("gguf");
    let mut v: Vec<_> = g.tensors.values().collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    for t in v {
        if t.name.starts_with(&prefix) || prefix.is_empty() {
            println!("{:>4} {:?} {}", t.ggml_type, t.dims, t.name);
        }
    }
}
