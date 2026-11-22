use inkwell_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    for (n, t) in &g.tensors {
        if n.contains("blk.6.")  {
            println!("{n:50} ty={} dims={:?} nb={}", t.ggml_type, t.dims, t.nbytes);
        }
    }
}
