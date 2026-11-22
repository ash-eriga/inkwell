use inkwell_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    let t = g.tensors.get("blk.0.attn_q.weight").unwrap();
    let lo = (g.data_start + t.offset) as usize;
    let d = &g.mmap[lo..lo + t.nbytes as usize];
    std::fs::write("/tmp/attn_q_q6k.bin", d).unwrap();
    println!("dims={:?} ty={} nbytes={}", t.dims, t.ggml_type, t.nbytes);
}
